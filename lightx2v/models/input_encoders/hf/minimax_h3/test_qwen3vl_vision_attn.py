import torch

from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl import MiniMaxH3Qwen3VLTextEncoder
from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_vision import _VisionAttention
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER


class _DummyPackedAttn:
    def __init__(self):
        self.calls = []

    def apply(self, q, k, v, cu_seqlens_q=None, cu_seqlens_kv=None, max_seqlen_q=None, max_seqlen_kv=None, causal=False, **kwargs):
        self.calls.append(
            {
                "q_shape": tuple(q.shape),
                "cu": tuple(cu_seqlens_q.tolist()),
                "max_seqlen_q": max_seqlen_q,
                "causal": causal,
            }
        )
        return torch.zeros(q.shape[0], q.shape[1] * q.shape[2], dtype=q.dtype, device=q.device)


def test_resolve_vision_attn_type_ignores_text_key(monkeypatch):
    monkeypatch.setitem(ATTN_WEIGHT_REGISTER, "dummy_packed_attn", lambda: None)
    assert MiniMaxH3Qwen3VLTextEncoder._resolve_vision_attn_type({"qwen3vl_attn_type": "flash_attn2"}) == "torch_sdpa"
    assert MiniMaxH3Qwen3VLTextEncoder._resolve_vision_attn_type({"qwen3vl_vision_attn_type": "dummy_packed_attn", "qwen3vl_attn_type": "flash_attn2"}) == "dummy_packed_attn"


def test_vision_apply_packed_varlen_once(monkeypatch):
    dummy = _DummyPackedAttn()
    monkeypatch.setitem(ATTN_WEIGHT_REGISTER, "dummy_packed_attn", lambda: dummy)
    vis = _VisionAttention({"hidden_size": 32, "num_heads": 4, "attn_type": "dummy_packed_attn"})
    tokens = 7
    hidden = torch.randn(tokens, 32)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    cos = torch.zeros(tokens, 8)
    sin = torch.zeros(tokens, 8)
    out = vis.forward(hidden, cu, cos, sin)
    assert out.shape == (tokens, 32)
    assert len(dummy.calls) == 1
    assert dummy.calls[0]["q_shape"] == (7, 4, 8)
    assert dummy.calls[0]["cu"] == (0, 3, 7)
    assert dummy.calls[0]["max_seqlen_q"] == 4
    assert dummy.calls[0]["causal"] is False
