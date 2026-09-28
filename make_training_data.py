"""Create known-correct JSONL examples for the 100M Transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene_agent.training_data import deterministic_request, make_seed_scene, records_for_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/scene_functions.jsonl"))
    parser.add_argument("--scenes", type=int, default=1_000)
    args = parser.parse_args()
    if args.scenes < 1:
        raise ValueError("--scenes must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for index in range(args.scenes):
            spec, functions = make_seed_scene(index)
            for record in records_for_scene(spec, functions, deterministic_request(spec)):
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                count += 1
    print(f"wrote {count:,} examples from {args.scenes:,} known-correct scenes to {args.output}")


if __name__ == "__main__":
    main()
