#!/usr/bin/env bash
# Phase B set-up on the VM, from the repository root:  bash scripts/vm_setup.sh
# Creates .venv, installs CUDA PyTorch (cu128 wheels: driver 570 supports CUDA 12.8) and requirements, runs the tests.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q -r requirements.txt
# lpips pulls torchvision from PyPI, which may not match the cu128 torch ("torchvision::nms does not exist")
pip install -q --force-reinstall --no-deps torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torchvision, transformers.modeling_utils"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q -p no:cacheprovider
echo "SETUP_DONE $(date -u '+%F %T')"
