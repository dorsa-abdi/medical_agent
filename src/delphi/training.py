"""Minimal trainer for Delphi-compatible JSONL trajectories."""
import json
from pathlib import Path

from src.delphi.model import DelphiConfig, build_model


def train(jsonl: Path, vocabulary: Path, output: Path, epochs=10, learning_rate=6e-4):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset

    labels = json.loads(vocabulary.read_text(encoding="utf-8"))["labels"]
    token_to_id = {label.casefold(): i for i, label in enumerate(labels)}

    class Trajectories(Dataset):
        def __init__(self):
            self.rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        def __len__(self): return len(self.rows)
        def __getitem__(self, index):
            events = self.rows[index]["events"]
            ids = [token_to_id[e["label"].casefold()] for e in events if e["label"].casefold() in token_to_id]
            ages = [float(e["age_days"]) for e in events if e["label"].casefold() in token_to_id]
            return torch.tensor(ids), torch.tensor(ages)

    def collate(batch):
        ids, ages = zip(*[(x[:-1], a[:-1]) for x, a in batch if len(x) >= 2])
        targets, target_ages = zip(*[(x[1:], a[1:]) for x, a in batch if len(x) >= 2])
        return (pad_sequence(ids, batch_first=True, padding_value=0),
                pad_sequence(ages, batch_first=True),
                pad_sequence(targets, batch_first=True, padding_value=-1),
                pad_sequence(target_ages, batch_first=True))

    config = DelphiConfig(vocab_size=len(labels))
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.2)
    loader = DataLoader(Trajectories(), batch_size=32, shuffle=True, collate_fn=collate)
    model.train()
    for _ in range(epochs):
        for ids, ages, targets, target_ages in loader:
            optimizer.zero_grad(set_to_none=True)
            output_batch = model(ids, ages, targets, target_ages)
            output_batch["loss"].backward()
            optimizer.step()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.as_dict(), "model": model.state_dict()}, output)
