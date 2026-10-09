import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModel, ModernBertConfig, PreTrainedTokenizerFast

from ielts_scoring.evaluate import regression_metrics
from ielts_scoring.models import EssayScorer, MultiGateHead, masked_mse_sum


def test_missing_targets_have_finite_zero_gradients():
    scores = torch.tensor([[2.0, 4.0], [3.0, 5.0]], requires_grad=True)
    labels = torch.tensor([[1.0, float("nan")], [3.0, 3.0]])
    loss_sum, observed = masked_mse_sum(scores, labels)
    assert int(observed) == 3
    assert float(loss_sum.detach()) == 5
    (loss_sum / observed).backward()
    torch.testing.assert_close(scores.grad, torch.tensor([[2 / 3, 0.0], [0.0, 4 / 3]]))


def test_expert_gates_are_independent_probability_distributions():
    head = MultiGateHead(32, 4)
    scores, gates = head(torch.randn(3, 32))
    assert scores.shape == (3, 4)
    assert gates.shape == (3, 4, 4)
    torch.testing.assert_close(gates.sum(-1), torch.ones(3, 4))
    scores.sum().backward()
    assert all(g.weight.grad is not None for g in head.gates)


@pytest.mark.parametrize("head", ["mlp", "moe"])
def test_real_modernbert_checkpoint_reloads_without_network(tmp_path, head):
    torch.manual_seed(42)
    config = ModernBertConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64,
                             local_attention=16, global_attn_every_n_layers=2, reference_compile=False,
                             pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2)
    model = EssayScorer(AutoModel.from_config(config, attn_implementation="sdpa"), {
        "head": head, "targets": ["TR", "CC", "LR", "GRA"], "label_range": [0, 9],
        "track": "ielts", "max_length": 32,
    }).eval()
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]")
    inputs = {"input_ids": torch.tensor([[1, 4, 2, 5, 6, 2, 0, 0]]),
              "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
              "essay_mask": torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0]])}
    before = model(**inputs)["scores"]
    model.save(tmp_path, tokenizer)
    restored = EssayScorer.load(tmp_path)
    after = restored(**inputs)["scores"]
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
    assert after.shape == (1, 4)
    assert restored.specification["targets"] == ["TR", "CC", "LR", "GRA"]


def test_metrics_use_native_units_and_observed_denominators():
    actual = np.array([[4.0, 3.0], [5.0, np.nan]])
    predictions = np.array([[4.5, 2.0], [6.0, 100.0]])
    result = regression_metrics(actual, predictions, ["a", "b"], ielts=True)
    assert result["a"]["mae"] == 0.75
    assert result["a"]["within_half_count"] == 1
    assert result["b"]["n"] == 1
    assert result["macro_mae"] == 0.875
