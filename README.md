# UniDriveVLA

This repository is a fork of <https://github.com/xiaomi-research/unidrivevla>.

## Bench2Drive Closed-Loop Evaluation Setup

These steps reproduce a working closed-loop evaluation environment for the Bench2Drive agent on Linux + RTX 5090 (sm_120, Blackwell). They diverge substantially from `docs/installation.md` because the upstream instructions target Python 3.8 + older CUDA / older Torch, none of which support sm_120.

### Prerequisites

- Ubuntu 22.04
- NVIDIA driver supporting CUDA 12.8+ (sm_120 capable)
- `uv` (>= 0.11) for Python environment management
- ~30 GB free disk for CUDA toolkit, CARLA, model weights

### 1. Install CUDA Toolkit 12.8 (system, toolkit only)

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

Add to your shell rc file (`~/.bashrc`) and re-source / reboot:

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
```

`nvcc --version` should report 12.8.

All shell commands below assume the current working directory is the repository root unless otherwise noted.

### 2. Install CARLA 0.9.16

Download from <https://github.com/carla-simulator/carla/releases/tag/0.9.16> and extract to `../CARLA_0.9.16` (one level above the repo root). [pyproject.toml](pyproject.toml) references the cp310 wheel via the relative path `../../CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl`, so the layout matters.

### 3. Create the Python Environment

```bash
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 \
    uv sync --no-build-isolation
```

The env vars are required because `mmcv-full==1.7.2` is built from source against torch 2.7 (no openmmlab prebuilt wheel exists for torch >= 2.2). The build takes 2-3 minutes. The custom op `deformable-aggregation-ext` under [Bench2Drive/projects/mmdet3d_plugin/ops](Bench2Drive/projects/mmdet3d_plugin/ops) is also compiled in this step.

### 4. Apply the Qwen3-VL transformers overlay

```bash
cp qwenvl3/transformers_replace/models/qwen3_vl/__init__.py \
   .venv/lib/python3.10/site-packages/transformers/models/qwen3_vl/__init__.py
cp qwenvl3/transformers_replace/models/qwen3_vl/modeling_qwen3_vl.py \
   .venv/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py
```

This needs to be re-applied if the venv is recreated.

### 5. Download model weights (HuggingFace cache)

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('owl10/UniDriveVLA_B2D_Base_Stage3')
snapshot_download('Qwen/Qwen3-VL-2B-Instruct')
"
```

### 6. Download kmeans anchor priors

The Bench2Drive config expects anchor files under `Bench2Drive/data/kmeans/`. They are not bundled in the repo but are compatible with HiP-AD's published priors:

```bash
mkdir -p Bench2Drive/data/kmeans
for f in b2d_det_900.npy b2d_map_100.npy b2d_motion_6.npy \
         b2d_plan_spat_6x8_2m.npy b2d_plan_spat_6x8_5m.npy; do
    curl -sL -o "Bench2Drive/data/kmeans/$f" \
        "https://raw.githubusercontent.com/nullmax-vision/HiP-AD/main/data/kmeans/$f"
done
```

### 7. Run the evaluation

A wrapper is provided at [Bench2Drive/scripts/eval_dev10_single.sh](Bench2Drive/scripts/eval_dev10_single.sh). By default it evaluates a single-route XML; override variables to point at other routes / save paths.

```bash
Bench2Drive/scripts/eval_dev10_single.sh
```

Overridable variables: `CHECKPOINT`, `VLM_PRETRAINED_PATH`, `ROUTES`, `SAVE_PATH`, `CARLA_ROOT`, `PORT`, `TM_PORT`, `GPU_RANK`, `CONFIG_NAME`, `PID_ABLATION`, `IS_VISUALIZE`.

The leaderboard auto-spawns CARLA on `carla-rpc-port=2003`; a manually launched CARLA on the default port is not used.

After completion, compute aggregate metrics:

```bash
uv run python Bench2Drive/bench2drive/tools/statistic_route_json.py \
    --route_dir Bench2Drive/evaluation/dev10_single
```
