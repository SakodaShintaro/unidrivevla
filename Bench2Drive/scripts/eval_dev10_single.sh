#!/usr/bin/env bash
set -euo pipefail

ROUTES=$(readlink -f "$1")

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SAVE_PATH=evaluation/$(basename "$ROUTES" .xml)_$(date +%Y%m%d_%H%M%S)
CHECKPOINT=$HOME/.cache/huggingface/hub/models--owl10--UniDriveVLA_B2D_Base_Stage3/snapshots/b5ac33e9483f6f61633c53353a8b53fbd96e2bc5/UniDriveVLA_Stage3_Bench2drive_2B.pt

mkdir -p "$SAVE_PATH"

export PROJECT_DIR=$PWD
export WORK_DIR=$PWD
export CARLA_ROOT=$HOME/CARLA_0.9.16
export VLM_PRETRAINED_PATH=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
export PID_ABLATION=v19b
export IS_VISUALIZE=1

uv run bash bench2drive/leaderboard/scripts/run_evaluation.sh \
    2000 8000 True \
    "$ROUTES" \
    bench2drive/leaderboard/team_code/unidrivevla_b2d_agent.py \
    "$PWD/projects/configs/unidrivevla_b2d_stage2_unified_2b.py+$CHECKPOINT" \
    "$SAVE_PATH/result.json" \
    "$SAVE_PATH" \
    traj 0
