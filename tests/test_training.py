
import torch

from ielts_scoring.train import train_epoch


class TinyScorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2, bias=False)
        torch.nn.init.constant_(self.linear.weight, 0.1)

    def forward(self, input_ids, **kwargs):
        return {"scores": self.linear(input_ids.float()), "gates": None}


def test_accumulation_matches_full_batch_with_uneven_missing_labels():
    items = [
        {"input_ids": torch.tensor([[0.1]]), "labels": torch.tensor([[0.2, float("nan")]])},
        {"input_ids": torch.tensor([[0.2]]), "labels": torch.tensor([[0.1, 0.3]])},
        {"input_ids": torch.tensor([[0.3]]), "labels": torch.tensor([[float("nan"), 0.1]])},
    ]
    full = {key: torch.cat([item[key] for item in items]) for key in items[0]}
    accumulated, direct = TinyScorer(), TinyScorer()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    train_epoch(accumulated, items, torch.optim.SGD(accumulated.parameters(), lr=0.1), None,
                scaler, torch.device("cpu"), accumulation=8)
    train_epoch(direct, [full], torch.optim.SGD(direct.parameters(), lr=0.1), None,
                scaler, torch.device("cpu"), accumulation=1)
    torch.testing.assert_close(accumulated.linear.weight, direct.linear.weight)
