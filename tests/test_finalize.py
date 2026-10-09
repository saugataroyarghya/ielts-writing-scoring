import json
import shutil
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModel, ModernBertConfig, PreTrainedTokenizerFast

from ielts_scoring.data import sha256
from ielts_scoring.finalize import main
from ielts_scoring.models import EssayScorer


def test_finalization_selects_on_validation_and_stages_offline_release(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    shutil.copy(project / "pyproject.toml", tmp_path)
    shutil.copytree(project / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "configs").mkdir()
    data = tmp_path / "input" / "dataset"
    data.mkdir(parents=True)
    targets = ["TR", "CC", "LR", "GRA"]
    frame = pd.DataFrame({"id": ["a", "b"], "source": "ielts", "group_id": ["g1", "g2"],
                          "prompt_group": ["p1", "p2"], "prompt": ["Discuss cities", "Discuss schools"],
                          "essay": ["Cities offer jobs.\n\nThey attract people.", "Schools teach children to read."],
                          **{name: [4.0, 5.0] for name in targets}})
    checksums = {}
    for name in ["train", "validation", "test"]:
        frame.to_parquet(data / f"{name}.parquet", index=False)
        checksums[f"{name}.parquet"] = sha256((data / f"{name}.parquet").read_bytes())
    metadata = {"track": "ielts", "prepared_sha256": checksums,
                "source": {"source": "synthetic fixture", "license": "test fixture only"}}
    (data / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "configs/prepared_datasets.json").write_text(json.dumps({"ielts": metadata}))
    config = ModernBertConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128,
                             local_attention=16, global_attn_every_n_layers=2, reference_compile=False,
                             pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2)
    raw_tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3}, unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer, pad_token="[PAD]", unk_token="[UNK]")
    for head, mae in [("mlp", 0.1), ("moe", 0.2)]:
        run = tmp_path / "input" / head
        specification = {"head": head, "targets": targets, "label_range": [0, 9], "track": "ielts",
                         "max_length": 64, "model_id": "synthetic", "revision": "fixture", "seed": 42,
                         "dataset_checksums": checksums, "encoder_lr": 2e-5, "head_lr": 1e-4}
        model = EssayScorer(AutoModel.from_config(config, attn_implementation="sdpa"), specification)
        model.save(run / "best", tokenizer)
        (run / "run_config.json").write_text(json.dumps(specification))
        (run / "best_validation.json").write_text(json.dumps({"macro_mae": mae}))
        (run / "baselines_validation.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["finalize", "--track", "ielts", "--input", str(tmp_path / "input"),
                                    "--output", str(tmp_path / "output")])
    main()
    output = tmp_path / "output"
    assert json.loads((output / "selection.json").read_text())["selected_head"] == "mlp"
    assert json.loads((output / "cpu_inference_check.json").read_text())["passed"]
    assert json.loads((output / "test_metrics.json").read_text())["TR"]["n"] == 2
    assert (output / "release" / "model.safetensors").is_file()
    assert (output / "release" / "README.md").is_file()
