# Vast.ai setup

Use one **on-demand, verified 24 GB or larger GPU**. A 32 GB RTX 3090/4090/5090
is the practical minimum for the 100M model, CUDA beauty renderer, and local
Qwen 3B visual grader on one instance. Use a 50 GB disk minimum and two open
ports (SSH and Jupyter). Prefer the Vast **PyTorch** template; the CUDA base
image also works because `vast_setup.sh` installs the Python dependencies.

## Batch size by VRAM

Train the 100M Transformer and Qwen grader in separate runs on a small GPU.
The generator trainer now uses mixed precision and gradient accumulation:

| VRAM | `--batch-size` | `--gradient-accumulation` | Effective batch |
| --- | ---: | ---: | ---: |
| 12 GB | 1 | 8 | 8 |
| 16 GB | 2 | 8 | 16 |
| 24 GB | 4 | 8 | 32 |
| 32 GB | 8 | 4 | 32 |

Start with the 12 GB row if unsure. Lower the micro-batch first if CUDA runs
out of memory; do not lower accumulation unless a smaller effective batch is
intentional.

For an RTX 5090, use the 32 GB row (`--batch-size 8
--gradient-accumulation 4 --precision bf16`). The trainer enables pinned memory
transfers, TensorFloat-32 matrix operations where they apply, and PyTorch fused
causal attention automatically. `vast_setup.sh` installs a CUDA 12.8 PyTorch
wheel because Blackwell/sm_120 support requires CUDA 12.8 or newer.

Do not create an instance until the current project files are in GitHub. On
the instance, clone the repository using your own GitHub credential method
(deploy key or fine-grained read-only token); do not put that credential in
this repository or in a command history shared with others.

```bash
git clone https://github.com/Overpricedwine/100mil-peramitor-model-quick-test.git
cd 100mil-peramitor-model-quick-test
chmod +x vast_setup.sh
./vast_setup.sh
source .venv/bin/activate
```

First smoke test:

```bash
python make_training_data.py --scenes 1000 --split train --output data/train.jsonl
python train_transformer.py --data data/train.jsonl --output runs/transformer --steps 100
```

Then download/load the local Qwen checkpoint and run a short scene cycle:

```bash
python run_scene_cycle.py --checkpoint runs/transformer/checkpoint-100.pt \
  --tokenizer runs/transformer/scene_tokenizer.json \
  --session runs/smoke --goal "Build a low-poly street lamp." --max-steps 4
```

Stop or destroy the instance as soon as you are finished. Persist checkpoints,
datasets, and accepted trajectories outside ephemeral instance storage before
destroying it.
