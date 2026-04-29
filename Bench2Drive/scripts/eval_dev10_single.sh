#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${CHECKPOINT:-$HOME/.cache/huggingface/hub/models--owl10--UniDriveVLA_B2D_Base_Stage3/snapshots/b5ac33e9483f6f61633c53353a8b53fbd96e2bc5/UniDriveVLA_Stage3_Bench2drive_2B.pt}"
VLM_PRETRAINED_PATH="${VLM_PRETRAINED_PATH:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203}"
ROUTES="${ROUTES:-$HOME/work/vla_streaming_rl/external/Bench2Drive/leaderboard/data/dev10_single.xml}"
SAVE_PATH="${SAVE_PATH:-evaluation/dev10_single}"
CARLA_ROOT="${CARLA_ROOT:-$HOME/CARLA_0.9.16}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
GPU_RANK="${GPU_RANK:-0}"
CONFIG_NAME="${CONFIG_NAME:-unidrivevla_b2d_stage2_unified_2b}"

mkdir -p "$SAVE_PATH"

export PROJECT_DIR="$REPO_ROOT"
export WORK_DIR="$REPO_ROOT"
export CARLA_ROOT
export VLM_PRETRAINED_PATH
export PID_ABLATION="${PID_ABLATION:-v19b}"
export IS_VISUALIZE="${IS_VISUALIZE:-1}"

TEAM_AGENT=bench2drive/leaderboard/team_code/unidrivevla_b2d_agent.py
TEAM_CONFIG="${REPO_ROOT}/projects/configs/${CONFIG_NAME}.py+${CHECKPOINT}"
CHECKPOINT_ENDPOINT="${SAVE_PATH}/result.json"

uv run bash bench2drive/leaderboard/scripts/run_evaluation.sh \
    "$PORT" "$TM_PORT" True \
    "$ROUTES" \
    "$TEAM_AGENT" \
    "$TEAM_CONFIG" \
    "$CHECKPOINT_ENDPOINT" \
    "$SAVE_PATH" \
    traj "$GPU_RANK"
