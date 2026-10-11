"""A pretrained encoder with either a plain MLP or a small multi-gate MoE."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModel


def masked_mse_sum(predictions, labels):
    mask = torch.isfinite(labels)
    safe_labels = torch.where(mask, labels, predictions.detach())
    return ((predictions.float() - safe_labels.float()) ** 2 * mask).sum(), mask.sum()


class PlainHead(nn.Module):
    def __init__(self, size, targets, width=1376, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(size, width), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(width, targets))
        nn.init.normal_(self.network[-1].weight, std=0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, representation):
        return self.network(representation), None


class MultiGateHead(nn.Module):
    def __init__(self, size, targets, experts=4, width=256, dropout=0.1):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(size, width), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(width, width), nn.GELU()) for _ in range(experts)
        ])
        self.gates = nn.ModuleList([nn.Linear(size, experts) for _ in range(targets)])
        self.outputs = nn.ModuleList([nn.Linear(width, 1) for _ in range(targets)])
        for layer in self.outputs:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, representation):
        expert_features = torch.stack([expert(representation) for expert in self.experts], dim=1)
        weights = torch.stack([gate(representation).softmax(-1) for gate in self.gates], dim=1)
        mixed = torch.einsum("bte,bed->btd", weights, expert_features)
        scores = torch.cat([layer(mixed[:, i]) for i, layer in enumerate(self.outputs)], dim=-1)
        return scores, weights


class EssayScorer(nn.Module):
    def __init__(self, encoder, specification):
        super().__init__()
        self.encoder = encoder
        self.specification = specification
        count = len(specification["targets"])
        size = encoder.config.hidden_size
        self.head = (MultiGateHead(size, count, experts=4, width=256)
                     if specification["head"] == "moe" else PlainHead(size, count))
        low, high = specification["label_range"]
        self.midpoint = (low + high) / 2

    @classmethod
    def from_pretrained_encoder(cls, specification):
        encoder = AutoModel.from_pretrained(
            specification["model_id"], revision=specification["revision"],
            attn_implementation="sdpa", reference_compile=False,
        )
        return cls(encoder, specification)

    @classmethod
    def from_encoder_checkpoint(cls, specification, folder, expected_sha256):
        """Transfer only encoder weights; initialize a fresh target-task head."""
        folder = Path(folder)
        weights = folder / "model.safetensors"
        with weights.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected_sha256:
            raise ValueError("Transfer checkpoint checksum differs from the frozen source.")
        source = json.loads((folder / "scorer_config.json").read_text())
        for key in ("model_id", "revision"):
            if source[key] != specification[key]:
                raise ValueError(f"Transfer source differs in {key}.")
        config = AutoConfig.from_pretrained(folder / "encoder_config", local_files_only=True)
        config.reference_compile = False
        encoder = AutoModel.from_config(config, attn_implementation="sdpa")
        state = load_file(str(weights))
        encoder.load_state_dict({key.removeprefix("encoder."): value for key, value in state.items()
                                 if key.startswith("encoder.")}, strict=True)
        return cls(encoder, specification)

    def forward(self, input_ids, attention_mask, essay_mask=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # The essay tokens are contextualized by the question in the same encoder pass.
        mask = (attention_mask if essay_mask is None else essay_mask).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        logits, gates = self.head(pooled)
        return {"scores": logits + self.midpoint, "gates": gates}

    def save(self, folder, tokenizer):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        save_file({key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()},
                  str(folder / "model.safetensors"))
        self.encoder.config.save_pretrained(folder / "encoder_config")
        tokenizer.save_pretrained(folder)
        (folder / "scorer_config.json").write_text(json.dumps(self.specification, indent=2))

    @classmethod
    def load(cls, folder, device="cpu"):
        folder = Path(folder)
        specification = json.loads((folder / "scorer_config.json").read_text())
        config = AutoConfig.from_pretrained(folder / "encoder_config", local_files_only=True)
        config.reference_compile = False
        encoder = AutoModel.from_config(config, attn_implementation="sdpa")
        scorer = cls(encoder, specification)
        scorer.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
        return scorer.to(device).eval()
