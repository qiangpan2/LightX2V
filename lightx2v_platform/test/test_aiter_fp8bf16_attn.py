import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from lightx2v_platform.ops.attn.amd_rocm import flash_attn as m
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/platforms/amd_rocm/minimax_h3_t2av_sp.json"


@pytest.fixture
def captured():
    state = {}

    def _fake_quant(x, quant_dtype=None):
        state.setdefault("quant", []).append({"shape": tuple(x.shape), "dtype": x.dtype, "quant_dtype": quant_dtype})
        return x.to(torch.float8_e4m3fn), torch.ones(1, dtype=torch.float32)

    def _fake_bshd(q, k, v, q_s, k_s, v_s, causal=False):
        state["bshd"] = {"q_shape": tuple(q.shape), "q_dtype": q.dtype, "causal": causal}
        return torch.ones(q.shape, dtype=torch.bfloat16)

    def _fake_varlen(q, k, v, q_s, k_s, v_s, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False, **kwargs):
        state["varlen"] = {
            "q_shape": tuple(q.shape),
            "q_dtype": q.dtype,
            "cu": tuple(cu_seqlens_q.tolist()),
            "max_seqlen_q": max_seqlen_q,
            "causal": causal,
        }
        return torch.ones(q.shape, dtype=torch.bfloat16)

    with (
        patch.object(m, "AITER_FP8BF16_AVAILABLE", True),
        patch.object(m, "IS_AMD_ROCM", True),
        patch.object(m, "aiter_fp8_dtypes", type("D", (), {"fp8": torch.float8_e4m3fn})()),
        patch.object(m, "aiter_per_tensor_quant", side_effect=_fake_quant),
        patch.object(m, "aiter_flash_attn_fp8_pertensor_func", side_effect=_fake_bshd),
        patch.object(m, "aiter_flash_attn_varlen_fp8_pertensor_func", side_effect=_fake_varlen),
    ):
        yield state


def test_register_name():
    assert PLATFORM_ATTN_WEIGHT_REGISTER["aiter_fp8bf16_attn"] is m.AiterFp8Bf16AttnWeight


def test_3d_varlen_quantizes_and_forwards_causal(captured):
    attn = m.AiterFp8Bf16AttnWeight()
    q = torch.randn(8, 16, 72, dtype=torch.bfloat16)
    out = attn.apply(q, q, q, cu_seqlens_q=None, max_seqlen_q=8, causal=True, scheduler=object(), block_idx=3)
    assert out.shape == (8, 16 * 72)
    assert out.dtype == torch.bfloat16
    assert captured["varlen"]["q_shape"] == (8, 16, 72)
    assert captured["varlen"]["q_dtype"] == torch.float8_e4m3fn
    assert captured["varlen"]["cu"] == (0, 8)
    assert captured["varlen"]["causal"] is True
    assert len(captured["quant"]) == 3
    assert all(item["dtype"] == torch.bfloat16 for item in captured["quant"])
    assert all(item["quant_dtype"] == torch.float8_e4m3fn for item in captured["quant"])
    assert "bshd" not in captured


def test_causal_defaults_false(captured):
    attn = m.AiterFp8Bf16AttnWeight()
    q = torch.randn(4, 2, 128, dtype=torch.bfloat16)
    attn.apply(q, q, q)
    assert captured["varlen"]["causal"] is False
    assert captured["varlen"]["cu"] == (0, 4)


def test_3d_pair_cu_seqlens_uses_varlen(captured):
    attn = m.AiterFp8Bf16AttnWeight()
    q = torch.randn(4, 2, 128, dtype=torch.bfloat16)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    out = attn.apply(q, q, q, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=4, max_seqlen_kv=4)
    assert out.shape == (4, 256)
    assert captured["varlen"]["q_shape"] == (4, 2, 128)
    assert captured["varlen"]["cu"] == (0, 4)
    assert "bshd" not in captured


def test_3d_packed_cu_seqlens_uses_varlen(captured):
    attn = m.AiterFp8Bf16AttnWeight()
    q = torch.randn(7, 4, 8, dtype=torch.bfloat16)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    out = attn.apply(q, q, q, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=4, max_seqlen_kv=4)
    assert out.shape == (7, 32)
    assert captured["varlen"]["q_shape"] == (7, 4, 8)
    assert captured["varlen"]["cu"] == (0, 3, 7)
    assert captured["varlen"]["max_seqlen_q"] == 4
    assert "bshd" not in captured


def test_4d_uses_bshd_not_flattened(captured):
    attn = m.AiterFp8Bf16AttnWeight()
    q = torch.randn(2, 4, 2, 128, dtype=torch.bfloat16)
    out = attn.apply(q, q, q)
    assert captured["bshd"]["q_shape"] == (2, 4, 2, 128)
    assert captured["bshd"]["q_dtype"] == torch.float8_e4m3fn
    assert out.dtype == torch.bfloat16
    assert out.shape == (8, 256)
    assert "varlen" not in captured


def test_amd_h3_config_routes_h3_thd_to_fp8():
    cfg = json.loads(CONFIG_PATH.read_text())
    assert cfg["attn_type"] == "aiter_fp8bf16_attn"
    assert cfg["refiner_attn_type"] == "aiter_fp8bf16_attn"
    assert cfg["qwen3vl_attn_type"] == "aiter_fp8bf16_attn"
    assert cfg["qwen3vl_vision_attn_type"] == "aiter_fp8bf16_attn"
    assert "vae_attn_type" not in cfg
    assert "dit_quant_scheme" not in cfg


def _gpu_fp8_ready():
    if not torch.cuda.is_available() or not m.AITER_FP8BF16_AVAILABLE:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _gpu_thd(s, h, d):
    torch.manual_seed(0)
    return tuple(torch.rand(s, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))


@pytest.mark.skipif(not _gpu_fp8_ready(), reason="needs HIP + aiter fp8bf16")
def test_gpu_none_matches_pair_cu_seqlens():
    s, h, d = 128, 28, 128
    q, k, v = _gpu_thd(s, h, d)
    attn = m.AiterFp8Bf16AttnWeight()
    cu = torch.tensor([0, s], dtype=torch.int32, device="cuda")
    out_pair = attn.apply(q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=s, max_seqlen_kv=s)
    out_none = attn.apply(q, k, v, cu_seqlens_q=None, max_seqlen_q=s)
    assert out_pair.shape == (s, h * d)
    assert out_pair.dtype == torch.bfloat16
    assert torch.isfinite(out_pair).all()
    assert (out_pair.float() - out_none.float()).abs().max().item() == 0.0


@pytest.mark.skipif(not _gpu_fp8_ready(), reason="needs HIP + aiter fp8bf16")
@pytest.mark.parametrize("h,d,causal", [(16, 72, False), (64, 128, True), (56, 128, False)])
def test_gpu_h3_thd_finite(h, d, causal):
    s = 128
    q, k, v = _gpu_thd(s, h, d)
    out = m.AiterFp8Bf16AttnWeight().apply(q, k, v, causal=causal)
    assert out.shape == (s, h * d)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not _gpu_fp8_ready(), reason="needs HIP + aiter fp8bf16")
def test_gpu_packed_varlen_finite():
    s1, s2, h, d = 48, 80, 16, 72
    s = s1 + s2
    q, k, v = _gpu_thd(s, h, d)
    cu = torch.tensor([0, s1, s], dtype=torch.int32, device="cuda")
    out = m.AiterFp8Bf16AttnWeight().apply(q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=s2, max_seqlen_kv=s2, causal=False)
    assert out.shape == (s, h * d)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
