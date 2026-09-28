"""Offline reward fine-tuning for approved 100M scene-model trajectories.

This is reward-weighted behaviour cloning (RFT), the safe first RL stage.
It improves the actual token Transformer using rendered/grader-scored actions
while a KL term keeps it close to the supervised checkpoint.  It is not a
replacement for human/Studio review and is deliberately not labelled PPO.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from scene_agent.grammar import validate_generated_call
from scene_agent.prompt_templates import FUNCTION_GENERATOR_SYSTEM_PROMPT
from scene_agent.tokenizer import SceneBPETokenizer
from scene_agent.transformer import SceneFunctionTransformer


class RewardTrajectoryDataset(Dataset):
    def __init__(self, path: Path, tokenizer: SceneBPETokenizer, context_length: int,
                 min_score: float, include_unreviewed: bool) -> None:
        pending: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            grade = record.get("grade", {})
            score = float(grade.get("score", -1))
            if score < min_score or grade.get("status") == "repair":
                continue
            if not include_unreviewed and not record.get("candidate_for_review", False):
                continue
            try:
                target = json.dumps(validate_generated_call(json.loads(record["target"])), separators=(",", ":"))
                context = record["context"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid trajectory at line {line_number}: {error}") from error
            prefix = f"<scene>\n{FUNCTION_GENERATOR_SYSTEM_PROMPT}\n{context}\n</scene>\n<function>\n"
            prefix_ids = tokenizer.encode(prefix, add_bos=True)
            target_ids = tokenizer.encode(target, add_eos=True)
            token_ids = (prefix_ids + target_ids)[-context_length:]
            labels = ([-100] * min(len(prefix_ids), len(token_ids)) + target_ids)[-len(token_ids):]
            # Final quality matters most; score delta discourages regressions.
            reward = score / 100 + max(-0.5, min(0.5, float(record.get("score_delta", 0)) / 100))
            pending.append((torch.tensor(token_ids), torch.tensor(labels), reward))
        if not pending:
            raise ValueError("no accepted trajectories; lower --min-score or review more cycles")
        baseline = sum(item[2] for item in pending) / len(pending)
        self.items = []
        for tokens, labels, reward in pending:
            # Exponential advantage weighting, clipped so a single grader
            # outlier cannot dominate the entire fine-tune.
            weight = max(0.25, min(4.0, math.exp((reward - baseline) / 0.15)))
            self.items.append((tokens, labels, torch.tensor(weight, dtype=torch.float32)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def collate(batch, pad_id: int):
    tokens, labels, weights = zip(*batch)
    return (pad_sequence(tokens, batch_first=True, padding_value=pad_id),
            pad_sequence(labels, batch_first=True, padding_value=-100), torch.stack(weights))


def token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:], ignore_index=-100, reduction="none")
    mask = labels[:, 1:].ne(-100)
    return (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward-fine-tune a supervised scene Transformer from graded traces.")
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Supervised checkpoint used as actor and KL reference")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/rft"))
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--min-score", type=float, default=80.0)
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="Use all high-scoring traces instead of only candidate_for_review records")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("--steps and --batch-size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = SceneBPETokenizer.load(args.tokenizer)
    model = SceneFunctionTransformer(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    reference = SceneFunctionTransformer(checkpoint["config"]).to(device)
    reference.load_state_dict(checkpoint["model"])
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    dataset = RewardTrajectoryDataset(
        args.trajectories, tokenizer, model.config.context_length, args.min_score, args.include_unreviewed
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)
    model.train()
    iterator = iter(loader)
    print(f"device={device}; accepted_trajectories={len(dataset):,}; kl_beta={args.kl_beta}")
    for step in range(1, args.steps + 1):
        try:
            token_ids, labels, weights = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            token_ids, labels, weights = next(iterator)
        token_ids, labels, weights = token_ids.to(device), labels.to(device), weights.to(device)
        outputs = model(token_ids)["logits"]
        nll = token_nll(outputs, labels)
        with torch.no_grad():
            reference_logits = reference(token_ids)["logits"][:, :-1]
        mask = labels[:, 1:].ne(-100)
        actor_log_probs = F.log_softmax(outputs[:, :-1], dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)
        kl_per_token = (reference_probs * (F.log_softmax(reference_logits, dim=-1) - actor_log_probs)).sum(dim=-1)
        kl = (kl_per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        loss = (weights * (nll + args.kl_beta * kl)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(f"step={step} loss={loss.item():.4f} nll={nll.mean().item():.4f} kl={kl.mean().item():.4f}")
        if step % 500 == 0 or step == args.steps:
            torch.save({"config": model.config, "model": model.state_dict(), "step": step,
                        "source_checkpoint": str(args.checkpoint)}, args.output / f"checkpoint-rft-{step}.pt")


if __name__ == "__main__":
    main()
