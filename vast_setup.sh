#!/usr/bin/env bash
# Run on a Vast PyTorch/CUDA instance after cloning this repository.
# It never contains a GitHub token, SSH key, or other credential.
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# RTX 5090 (Blackwell / sm_120) needs a PyTorch binary built with CUDA 12.8
# or newer. Install that wheel before the rest of the requirements.
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA was not exposed to PyTorch"
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
print("CUDA arch list:", torch.cuda.get_arch_list())
assert torch.version.cuda and tuple(map(int, torch.version.cuda.split(".")[:2])) >= (12, 8), "RTX 5090 needs a CUDA 12.8+ PyTorch wheel"
PY

mkdir -p data runs
echo "Environment ready. Next: generate data, train the 100M model, then run run_scene_cycle.py."
