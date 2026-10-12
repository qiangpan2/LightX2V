#!/bin/bash

# Dense: configs/platforms/amd_rocm/minimax_h3_t2av_sage2.json
# SLA:   configs/platforms/amd_rocm/minimax_h3_t2av_sla_sage2.json
# Requires aiter.ops.flydsl.sageattn2 and block_sparse_sageattn2.

lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3
config_json="${lightx2v_path}/configs/platforms/amd_rocm/minimax_h3_t2av_sage2.json"

export PLATFORM=amd_rocm
export CUDA_VISIBLE_DEVICES=0

source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

python -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant fl2av \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt 'integrated_multimodal_description: A cinematic fox walks through a snowy pine forest at dawn. overall_soundscape: Soft wind, crunching snow, and distant birds. non_diegetic_music: Quiet warm strings.' \
  --save_result_path "${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_sage2.mp4" \
  --seed 42
