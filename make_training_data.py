"""Create known-correct JSONL examples for the 100M Transformer."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scene_agent.request_author import LocalQwenRequestAuthor
from scene_agent.training_data import deterministic_request, make_seed_scene, records_for_scene, split_for_scene


STYLE_HINTS = (
    "plain builder request", "compact design brief", "casual game-creator wording",
    "clear asset request", "short production note", "architectural asset brief",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/scene_functions.jsonl"))
    parser.add_argument("--scenes", type=int, default=1_000)
    parser.add_argument("--request-author", choices=("deterministic", "qwen"), default="deterministic")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="train",
                        help="Keep whole template families held out by default.")
    args = parser.parse_args()
    if args.scenes < 1:
        raise ValueError("--scenes must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    author = LocalQwenRequestAuthor(args.model_id) if args.request_author == "qwen" else None
    variation_rng = random.Random(args.seed)
    count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for index in range(args.scenes):
            if args.split != "all" and split_for_scene(index) != args.split:
                continue
            spec, functions = make_seed_scene(index)
            variation_seed = variation_rng.randrange(1, 2**31)
            detail_level = "specific" if index % 2 else "minimal"
            if author is None:
                request = deterministic_request(spec)
            else:
                request = author.write(
                    spec, detail_level=detail_level, variation_seed=variation_seed,
                    style_hint=STYLE_HINTS[index % len(STYLE_HINTS)],
                )
            for record in records_for_scene(spec, functions, request, scene_id=index):
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                count += 1
    print(f"wrote {count:,} examples from {args.scenes:,} known-correct scenes to {args.output}")


if __name__ == "__main__":
    main()
