from unittest.mock import patch

import torch

from lightx2v_platform.ops.attn.amd_rocm import flash_attn as m


def test_ulysses_none_cu_seqlens_is_synthesized():
    captured = {}

    def fake_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
        captured["cu"] = None if cu_seqlens_q is None else tuple(cu_seqlens_q.tolist())
        captured["cu_kv"] = None if cu_seqlens_kv is None else tuple(cu_seqlens_kv.tolist())
        captured["max_q"] = max_seqlen_q
        captured["q_shape"] = tuple(q.shape)
        return torch.ones(q.shape, dtype=torch.bfloat16)

    with (
        patch.object(m, "IS_AMD_ROCM", True),
        patch.object(m, "AITER_AVAILABLE", True),
        patch.object(m, "aiter_flash_attn_varlen_func", fake_varlen),
    ):
        attn = m.AiterAttnWeight()
        q = torch.randn(8, 28, 128, dtype=torch.bfloat16)
        out = attn.apply(q, q, q, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=8, max_seqlen_kv=8)

    assert captured["cu"] == (0, 8)
    assert captured["cu_kv"] == (0, 8)
    assert captured["max_q"] == 8
    assert captured["q_shape"] == (8, 28, 128)
    assert out.shape == (8, 28 * 128)
    assert out.dtype == torch.bfloat16
