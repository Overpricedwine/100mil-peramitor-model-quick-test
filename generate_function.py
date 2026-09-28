"""Generate one validated scene function from a trained 100M checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scene_agent.grammar import parse_generated_json
from scene_agent.tokenizer import SceneBPETokenizer
from scene_agent.transformer import SceneFunctionTransformer


def extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object was generated")
    return text[start:end + 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--context", required=True, help="Serialized scene request/history for the model")
    parser.add_argument("--attempts", type=int, default=8)
    arguments = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    model = SceneFunctionTransformer(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = SceneBPETokenizer.load(arguments.tokenizer)
    prefix = f"<scene>\n{arguments.context}\n</scene>\n<function>\n"
    prefix_ids = tokenizer.encode(prefix, add_bos=True)[-model.config.context_length:]
    prompt = torch.tensor([prefix_ids], device=device)
    failures: list[str] = []
    for _ in range(arguments.attempts):
        output = model.generate(prompt, max_new_tokens=256, temperature=0.7, top_k=50,
                                eos_token_id=tokenizer.eos_id)
        generated = tokenizer.decode(output[0, len(prefix_ids):].tolist())
        try:
            function = parse_generated_json(extract_json(generated))
            print(json.dumps(function, indent=2))
            return
        except (ValueError, json.JSONDecodeError) as error:
            failures.append(str(error))
    raise SystemExit("No grammar-valid function after retries:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
