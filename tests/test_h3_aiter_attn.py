import importlib.util
import sys
import types
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]


def _preload(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules.setdefault(name, module)


def _load_sage_module():
    path = _ROOT / "lightx2v_platform" / "ops" / "attn" / "amd_rocm" / "sage_attn.py"
    spec = importlib.util.spec_from_file_location("h3_aiter_sage_attn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_sageattn(q, k, v, tensor_layout="HND", **kwargs):
    assert tensor_layout == "NHD"
    _fake_sageattn.seen = q.shape
    return torch.empty_like(q)


def test_dense_dit_and_vae_layouts():
    cls = _load_sage_module().AiterSageAttn2Weight
    attn = cls.__new__(cls)
    attn.sageattn = _fake_sageattn

    dit_q = torch.randn(5, 2, 4)
    dit_out = attn.apply(dit_q, dit_q, dit_q, causal=False, scheduler=None)
    assert _fake_sageattn.seen == (1, 5, 2, 4)
    assert dit_out.shape == (5, 8)

    vae_q = torch.randn(2, 3, 2, 4)
    vae_out = attn.apply(vae_q, vae_q, vae_q)
    assert _fake_sageattn.seen == (2, 3, 2, 4)
    assert vae_out.shape == (6, 8)


def test_sla_sage2_aiter_passes_64_block_mask(monkeypatch):
    triton = types.ModuleType("triton")

    def jit(*args, **kwargs):
        def deco(fn):
            return fn

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return deco

    triton.jit = jit
    language = types.ModuleType("triton.language")
    language.constexpr = object()
    sys.modules.setdefault("triton", triton)
    sys.modules.setdefault("triton.language", language)
    _preload("lightx2v", _ROOT / "lightx2v")
    _preload("lightx2v.common", _ROOT / "lightx2v" / "common")
    _preload("lightx2v.common.ops", _ROOT / "lightx2v" / "common" / "ops")
    _preload("lightx2v.common.ops.attn", _ROOT / "lightx2v" / "common" / "ops" / "attn")
    _preload("lightx2v.utils", _ROOT / "lightx2v" / "utils")

    from lightx2v.common.ops.attn.dynamic_sparse_attn import DynamicSparseAttnWeight

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    weight = DynamicSparseAttnWeight({"operator": "sage2_aiter", "sparsity_ratio": 0.85})
    assert (weight.BLKQ, weight.BLKK) == (64, 64)
    assert abs(weight.topk - 0.15) < 1e-12

    captured = {}

    def fake_block_map(q, k, topk_ratio, BLKQ, BLKK):
        nq = (q.shape[2] + BLKQ - 1) // BLKQ
        nk = (k.shape[2] + BLKK - 1) // BLKK
        return torch.ones(q.shape[0], q.shape[1], nq, nk, dtype=torch.int8), None, 1

    def fake_sparse(q, k, v, mask, tensor_layout="HND", **kwargs):
        captured["q"] = q.shape
        captured["mask"] = mask.shape
        captured["layout"] = tensor_layout
        return torch.zeros_like(q)

    weight._get_block_map = fake_block_map
    flydsl = types.ModuleType("aiter.ops.flydsl")
    flydsl.block_sparse_sageattn2 = fake_sparse
    sys.modules["aiter"] = types.ModuleType("aiter")
    sys.modules["aiter.ops"] = types.ModuleType("aiter.ops")
    sys.modules["aiter.ops.flydsl"] = flydsl

    tokens, heads, dim = 80, 2, 64
    q = torch.randn(tokens, heads, dim)
    out = weight.apply(q, q, q)
    assert captured == {"q": (1, heads, tokens, dim), "mask": (1, heads, 2, 2), "layout": "HND"}
    assert out.shape == (tokens, heads * dim)
