"""Generate one validated scene function from a trained 100M checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from scene_agent.constrained_decoding import GrammarLogitsProcessor
from scene_agent.grammar import parse_generated_json
from scene_agent.preset_calls import NumericPresetLogitsProcessor, decode_preset_call
from scene_agent.prompt_templates import FUNCTION_GENERATOR_SYSTEM_PROMPT, PRESET_FUNCTION_GENERATOR_SYSTEM_PROMPT
from scene_agent.tokenizer import SceneBPETokenizer
from scene_agent.transformer import SceneFunctionTransformer


def extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object was generated")
    return text[start:end + 1]


def load_generator(checkpoint_path: Path, tokenizer_path: Path, device: torch.device) -> tuple[SceneFunctionTransformer, SceneBPETokenizer]:
    """Load the trained function model once for a multi-step scene cycle."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SceneFunctionTransformer(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, SceneBPETokenizer.load(tokenizer_path)


def generate_valid_function(
    model: SceneFunctionTransformer, tokenizer: SceneBPETokenizer, context: dict[str, Any] | str, *,
    attempts: int = 8, temperature: float = 0.0,
) -> dict[str, Any]:
    """Return exactly one schema-valid function for the supplied saved context.

    Scene construction defaults to greedy decoding.  Sampling is useful only
    after a checkpoint has passed held-out multi-function scene tests; for an
    early checkpoint it turns near-correct JSON into invalid key/value mixes.
    """
    serialized_context = context if isinstance(context, str) else json.dumps(context, separators=(",", ":"))
    prefix = f"<scene>\n{FUNCTION_GENERATOR_SYSTEM_PROMPT}\n{serialized_context}\n</scene>\n<function>\n"
    prefix_ids = tokenizer.encode(prefix, add_bos=True)[-model.config.context_length:]
    prompt = torch.tensor([prefix_ids], device=model.token_embedding.weight.device)
    constraint = GrammarLogitsProcessor(tokenizer, prompt_length=len(prefix_ids))
    failures: list[str] = []
    for _ in range(attempts):
        generated = ""
        try:
            output = model.generate(prompt, max_new_tokens=256, temperature=temperature, top_k=50,
                                    eos_token_id=tokenizer.eos_id, logits_processor=constraint)
            generated = tokenizer.decode(output[0, len(prefix_ids):].tolist())
            return parse_generated_json(extract_json(generated))
        except (ValueError, json.JSONDecodeError) as error:
            preview = generated[:160].replace("\n", "\\n")
            failures.append(f"{error} (output={preview!r})")
    raise RuntimeError("No grammar-valid function after retries:\n- " + "\n- ".join(failures))


def generate_valid_preset_function(
    model: SceneFunctionTransformer, tokenizer: SceneBPETokenizer, context: dict[str, Any] | str, *,
    attempts: int = 1,
) -> dict[str, Any]:
    """Compile a learned numeric-only call into one validated scene function."""
    serialized_context = context if isinstance(context, str) else json.dumps(context, separators=(",", ":"))
    prefix = f"<scene>\n{PRESET_FUNCTION_GENERATOR_SYSTEM_PROMPT}\n{serialized_context}\n</scene>\n<function>\n"
    prefix_ids = tokenizer.encode(prefix, add_bos=True)[-model.config.context_length:]
    prompt = torch.tensor([prefix_ids], device=model.token_embedding.weight.device)
    constraint = NumericPresetLogitsProcessor(tokenizer, len(prefix_ids), serialized_context)
    failures: list[str] = []
    for _ in range(attempts):
        generated = ""
        try:
            output = model.generate(prompt, max_new_tokens=128, temperature=0.0,
                                    eos_token_id=tokenizer.eos_id, logits_processor=constraint)
            generated = tokenizer.decode(output[0, len(prefix_ids):].tolist())
            return decode_preset_call(generated, serialized_context)
        except (ValueError, json.JSONDecodeError) as error:
            failures.append(f"{error} (output={generated[:160]!r})")
    raise RuntimeError("No valid numeric preset after retries:\n- " + "\n- ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--context", required=True, help="Serialized scene request/history for the model")
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--format", choices=("preset", "json"), default="preset")
    arguments = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_generator(arguments.checkpoint, arguments.tokenizer, device)
    generator = generate_valid_preset_function if arguments.format == "preset" else generate_valid_function
    function = generator(model, tokenizer, arguments.context, attempts=arguments.attempts)
    print(json.dumps(function, indent=2))


if __name__ == "__main__":
    main()
