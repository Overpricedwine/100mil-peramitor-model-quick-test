"""Upload stable training checkpoints to a private Hugging Face model repo.

The Hugging Face access token is read only from the `HF_TOKEN` environment
variable. Never put that token in source control or a command-line argument.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi


def stable_files(directory: Path, minimum_age: int) -> list[Path]:
    """Return completed snapshots, never a checkpoint still being written."""
    now = time.time()
    checkpoints = [path for path in directory.glob("checkpoint-*.pt") if path.is_file()]
    candidates = ([max(checkpoints, key=lambda path: path.stat().st_mtime)] if checkpoints else [])
    candidates += [directory / "scene_tokenizer.json", directory / "train.log"]
    return [path for path in candidates if path.is_file() and now - path.stat().st_mtime >= minimum_age]


def signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize finished scene-agent checkpoints to Hugging Face")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--minimum-age-seconds", type=int, default=90)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required and must not be placed in source code")
    if args.interval_seconds < 60:
        raise ValueError("interval must be at least 60 seconds")

    api = HfApi(token=token)
    uploaded: dict[str, tuple[int, int]] = {}
    print(f"syncing stable files from {args.source} to {args.repo_id} every {args.interval_seconds}s", flush=True)
    while True:
        for path in stable_files(args.source, args.minimum_age_seconds):
            file_signature = signature(path)
            remote_name = "checkpoint-latest.pt" if path.suffix == ".pt" else path.name
            if uploaded.get(remote_name) == file_signature:
                continue
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=remote_name,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"checkpoint sync: {path.name}",
            )
            uploaded[remote_name] = file_signature
            print(f"uploaded {remote_name} from {path.name} ({file_signature[0]:,} bytes)", flush=True)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
