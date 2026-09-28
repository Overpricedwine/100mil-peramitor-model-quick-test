"""Supervised pretraining for the ~100M scene-function Transformer.

Dataset JSONL format: {"context": "serialized conversation + scene", "target": "{...valid JSON function...}"}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from scene_agent.grammar import validate_generated_call
from scene_agent.tokenizer import SceneBPETokenizer
from scene_agent.transformer import SceneFunctionTransformer, TransformerConfig


class FunctionDataset(Dataset):
    def __init__(self, path: Path, tokenizer: SceneBPETokenizer, context_length: int) -> None:
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            target = json.dumps(validate_generated_call(json.loads(record["target"])), separators=(",", ":"))
            prefix = f"<scene>\n{record['context']}\n</scene>\n<function>\n"
            prefix_ids = tokenizer.encode(prefix, add_bos=True)
            target_ids = tokenizer.encode(target, add_eos=True)
            token_ids = (prefix_ids + target_ids)[-context_length:]
            labels = [-100] * min(len(prefix_ids), len(token_ids)) + target_ids
            labels = labels[-len(token_ids):]
            self.items.append((torch.tensor(token_ids), torch.tensor(labels)))
        if not self.items:
            raise ValueError("dataset contains no training records")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.items[index]


def collate(batch, pad_id: int):
    tokens, labels = zip(*batch)
    return (pad_sequence(tokens, batch_first=True, padding_value=pad_id),
            pad_sequence(labels, batch_first=True, padding_value=-100))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/transformer"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--tokenizer", type=Path)
    arguments = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arguments.output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = arguments.tokenizer or arguments.output / "scene_tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = SceneBPETokenizer.load(tokenizer_path)
    else:
        texts = [line for line in arguments.data.read_text(encoding="utf-8").splitlines() if line.strip()]
        tokenizer = SceneBPETokenizer.train(texts, tokenizer_path)
    if tokenizer.vocab_size != 16_384:
        raise ValueError(f"tokenizer produced {tokenizer.vocab_size} tokens; provide enough data for 16,384")
    config = TransformerConfig(vocab_size=tokenizer.vocab_size)
    model = SceneFunctionTransformer(config).to(device)
    dataset = FunctionDataset(arguments.data, tokenizer, config.context_length)
    loader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=True,
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    print(f"device={device}; parameters={model.parameter_count:,}; examples={len(dataset):,}")
    model.train()
    iterator = iter(loader)
    for step in range(1, arguments.steps + 1):
        try:
            token_ids, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            token_ids, labels = next(iterator)
        result = model(token_ids.to(device), labels.to(device))
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"step={step} loss={result['loss'].item():.4f}")
        if step % 1_000 == 0 or step == arguments.steps:
            torch.save({"config": config, "model": model.state_dict(), "step": step},
                       arguments.output / f"checkpoint-{step}.pt")


if __name__ == "__main__":
    main()
