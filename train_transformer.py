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
from scene_agent.preset_calls import encode_preset_call
from scene_agent.prompt_templates import FUNCTION_GENERATOR_SYSTEM_PROMPT, PRESET_FUNCTION_GENERATOR_SYSTEM_PROMPT
from scene_agent.tokenizer import SceneBPETokenizer
from scene_agent.transformer import SceneFunctionTransformer, TransformerConfig


class FunctionDataset(Dataset):
    def __init__(self, path: Path, tokenizer: SceneBPETokenizer, context_length: int, *, target_format: str) -> None:
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            call = validate_generated_call(json.loads(record["target"]))
            target = (encode_preset_call(call, record["context"]) if target_format == "preset"
                      else json.dumps(call, separators=(",", ":")))
            system_prompt = PRESET_FUNCTION_GENERATOR_SYSTEM_PROMPT if target_format == "preset" else FUNCTION_GENERATOR_SYSTEM_PROMPT
            prefix = f"<scene>\n{system_prompt}\n{record['context']}\n</scene>\n<function>\n"
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
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch; keep at 1 on a 12 GB GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                        help="Micro-batches per optimizer update; default effective batch is 8")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--target-format", choices=("preset", "json"), default="preset",
                        help="Train compact preset ID + numeric calls (recommended) or legacy JSON.")
    parser.add_argument("--resume", type=Path,
                        help="Checkpoint to resume. Includes the model and AdamW optimizer state.")
    parser.add_argument("--checkpoint-every", type=int, default=1_000,
                        help="Save a resumable checkpoint after this many optimizer updates.")
    arguments = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arguments.batch_size < 1 or arguments.gradient_accumulation < 1:
        raise ValueError("--batch-size and --gradient-accumulation must be positive")
    if arguments.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    arguments.output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = arguments.tokenizer or arguments.output / "scene_tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = SceneBPETokenizer.load(tokenizer_path)
    else:
        texts = [line for line in arguments.data.read_text(encoding="utf-8").splitlines() if line.strip()]
        tokenizer = SceneBPETokenizer.train(texts, tokenizer_path)
    # The constrained scene language is deliberately compact, so a BPE trainer
    # may finish with fewer than the requested 16,384 merge entries even with a
    # large generated dataset.  That is not an error: keep the transformer's
    # fixed 16,384-token output space (and therefore its ~100M parameter
    # architecture) while decoding only IDs the tokenizer can represent.
    if tokenizer.vocab_size > 16_384:
        raise ValueError(f"tokenizer produced {tokenizer.vocab_size} tokens; maximum is 16,384")
    if tokenizer.vocab_size < 16_384:
        print(f"tokenizer vocabulary={tokenizer.vocab_size:,}; model vocabulary remains 16,384")
    # Keep the architecture at its intended ~100M parameter size.  The
    # tokenizer may contain fewer entries, but all emitted IDs remain within
    # this fixed model vocabulary.
    config = TransformerConfig()
    model = SceneFunctionTransformer(config).to(device)
    dataset = FunctionDataset(arguments.data, tokenizer, config.context_length,
                              target_format=arguments.target_format)
    loader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=True,
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_id), pin_memory=device.type == "cuda")
    if device.type == "cuda":
        # Tensor Core matmuls and PyTorch's fused causal SDPA/Flash path make
        # the RTX 5090 useful for this 2k-context Transformer, not just large.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    start_step = 0
    if arguments.resume:
        checkpoint = torch.load(arguments.resume, map_location=device, weights_only=False)
        checkpoint_config = checkpoint.get("config")
        if checkpoint_config is not None and checkpoint_config != config:
            raise ValueError(f"checkpoint architecture does not match this trainer: {checkpoint_config}")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        print(f"resumed checkpoint={arguments.resume}; step={start_step:,}")
    if arguments.precision == "auto":
        precision = "bf16" if device.type == "cuda" else "fp32"
    else:
        precision = arguments.precision
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("bf16/fp16 require CUDA")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16")
    effective_batch = arguments.batch_size * arguments.gradient_accumulation
    print(f"device={device}; precision={precision}; target_format={arguments.target_format}; "
          f"parameters={model.parameter_count:,}; examples={len(dataset):,}; "
          f"micro_batch={arguments.batch_size}; effective_batch={effective_batch}")
    model.train()
    iterator = iter(loader)
    if start_step >= arguments.steps:
        raise ValueError(f"resume checkpoint is already at step {start_step:,}; --steps must be greater")
    for step in range(start_step + 1, arguments.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        reported_loss = 0.0
        for _ in range(arguments.gradient_accumulation):
            try:
                token_ids, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                token_ids, labels = next(iterator)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                result = model(token_ids.to(device, non_blocking=True), labels.to(device, non_blocking=True))
                loss = result["loss"] / arguments.gradient_accumulation
            reported_loss += loss.detach().item()
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 50 == 0:
            print(f"step={step} loss={reported_loss:.4f}")
        if step % arguments.checkpoint_every == 0 or step == arguments.steps:
            torch.save({"config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                       arguments.output / f"checkpoint-{step}.pt")


if __name__ == "__main__":
    main()
