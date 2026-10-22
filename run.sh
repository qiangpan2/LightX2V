#!/bin/bash
# MiniMax-H3 ref2av, official 4-step Turbo LoRA, two-GPU torch Ulysses.
# DiT default: sage1_aiter, sparsity 0.85.

set -euo pipefail

lightx2v_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$lightx2v_path"

nproc="${NPROC_PER_NODE:-2}"
src_config="configs/platforms/amd_rocm/minimax_h3_t2av_sla_sage1.json"
runtime_config="save_results/minimax_h3_ref2av_sla_sage1.runtime.json"
prompt_file="${PROMPT_FILE:-${lightx2v_path}/assets/inputs/minimax_h3/rooftop_15s/prompt.txt}"
image_path="${IMAGE_PATH:-${lightx2v_path}/assets/inputs/minimax_h3/rooftop_15s/Image_Gen-1.png,${lightx2v_path}/assets/inputs/minimax_h3/rooftop_15s/RYAN-2.png}"
out="${SAVE_RESULT_PATH:-save_results/output_lightx2v_minimax_h3_ref2av_sla_rooftop.mp4}"

export PLATFORM=amd_rocm
export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:$PYTHONPATH}"
export PROFILING_DEBUG_LEVEL="${PROFILING_DEBUG_LEVEL:-2}"

mapfile -t resolved < <(uv run --active --no-sync python download_weight.py "$src_config" "$runtime_config" "$nproc")
model_path="${resolved[0]}"

cache_dir="${HOME}/.cache/lightx2v/adaln/minimax_h3/ref2av_04steps_shift_6.0_3.0"
if [[ ! -f "${cache_dir}/manifest.json" ]]; then
  rm -rf "$cache_dir"
  uv run --active --no-sync python tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py \
    --model_path "$model_path" \
    --config_json "$runtime_config" \
    --model_cls minimax_h3 \
    --model-variant ref2av
fi

uv run --active --no-sync torchrun --standalone --nproc_per_node="${nproc}" -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant ref2av \
  --task ref2av \
  --model_path "$model_path" \
  --config_json "$runtime_config" \
  --prompt "$(cat "$prompt_file")" \
  --image_path "$image_path" \
  --save_result_path "$out" \
  --seed 42
echo "saved $out"
