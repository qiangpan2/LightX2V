#!/usr/bin/env python3
"""Resolve MiniMax-H3 and the official Turbo LoRA, then write the ref2av runtime config."""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache

HF_LOCAL = Path(os.environ.get("HF_LOCAL", "/home/nfs_share/huggingface"))
MODEL_REPO = os.environ.get("MODEL_REPO", "MiniMaxAI/MiniMax-H3")
LORA_REPO = os.environ.get("LORA_REPO", "lightx2v/Minimax-h3-Turbo")
LORA_FILENAME = os.environ.get("LORA_FILENAME", "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors")


def model_path():
    path = Path(os.environ.get("MODEL_PATH", HF_LOCAL / "MiniMaxAI" / "MiniMax-H3"))
    if (path / "model_index.json").is_file():
        return path
    return Path(snapshot_download(MODEL_REPO))


def lora_path():
    path = Path(os.environ.get("LORA_PATH", HF_LOCAL / "lightx2v" / "Minimax-h3-Turbo" / LORA_FILENAME))
    if path.is_file():
        return path
    cached = try_to_load_from_cache(repo_id=LORA_REPO, filename=LORA_FILENAME)
    if isinstance(cached, str):
        return Path(cached)
    return Path(hf_hub_download(repo_id=LORA_REPO, filename=LORA_FILENAME))


def write_runtime(src, dst, lora, nproc):
    cfg = json.loads(Path(src).read_text())
    cfg["lora_configs"][0]["path"] = str(lora)
    cfg["reference_image_resize_mode"] = "match"
    cfg["vae_decode_parallel"] = nproc > 1
    cfg["attn_type"] = "dynamic_sparse_attn"
    cfg["dynamic_sparse_attn_setting"] = {"sparsity_ratio": 0.85, "operator": "sage1_aiter"}
    cfg["parallel"] = {
        "seq_p_size": nproc,
        "seq_p_attn_type": "ulysses",
        "seq_p_prepost_backend": "torch",
        "seq_p_a2a_backend": "torch",
        "seq_p_tensor_fusion": False,
        "seq_p_head_parallel": False,
    }
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    src, dst, nproc = sys.argv[1:4]
    model = model_path()
    lora = lora_path()
    write_runtime(src, dst, lora, int(nproc))
    print(model)
    print(lora)
