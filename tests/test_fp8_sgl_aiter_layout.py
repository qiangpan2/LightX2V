import os

import triton  # before torch; the reverse segfaults in this ROCm build

os.environ.pop("PLATFORM", None)

import torch

from lightx2v.common.ops.mm.mm_weight import MMWeightWfp8channelAfp8channelF16Accum, MMWeightWfp8channelAfp8channeldynamicSgl
from lightx2v.models.input_encoders.hf.q_linear import SglQuantLinearFp8
from lightx2v_platform.base import global_var


def _quant(x):
    return x, torch.empty(x.shape[0], 1)


def test_rocm_fp8_sgl_passes_checkpoint_layout(monkeypatch):
    global_var.PLATFORM = "amd_rocm"
    seen = {}

    def gemm(x, weight, x_scale, w_scale, dtype, bias=None):
        seen["weight"] = tuple(weight.shape)
        return torch.empty(x.shape[0], weight.shape[0])

    monkeypatch.setattr("lightx2v.common.ops.mm.mm_weight.aiter_fp8_w8a8", gemm)
    monkeypatch.setattr("lightx2v.models.input_encoders.hf.q_linear.aiter_fp8_w8a8", gemm)
    layer = MMWeightWfp8channelAfp8channeldynamicSgl("blk.weight", None)
    layer.act_quant_func = _quant
    layer.weight = torch.empty(16, 32)
    layer.weight_scale = torch.empty(16, 1)
    layer.bias = None
    layer.infer_dtype = torch.bfloat16

    assert layer.use_aiter_w8a8
    assert layer.weight_need_transpose is False
    layer.apply(torch.empty(4, 32))
    assert seen["weight"] == (16, 32)

    linear = SglQuantLinearFp8(32, 16, bias=False)
    linear.act_quant_func = _quant
    linear.weight = torch.empty(16, 32)
    linear.weight_scale = torch.empty(16, 1)
    linear(torch.empty(1, 4, 32))
    assert seen["weight"] == (16, 32)


def test_rocm_f16_accum_keeps_transposed_storage(monkeypatch):
    global_var.PLATFORM = "amd_rocm"
    seen = {}

    def gemm(x, weight, x_scale, w_scale, dtype, bias=None):
        seen["weight"] = tuple(weight.shape)
        return torch.empty(x.shape[0], weight.shape[0])

    monkeypatch.setattr("lightx2v.common.ops.mm.mm_weight.aiter_fp8_w8a8", gemm)
    layer = MMWeightWfp8channelAfp8channelF16Accum("blk.weight", None)
    layer.act_quant_func = _quant
    layer.weight = torch.empty(32, 16)
    layer.weight_scale = torch.empty(16, 1)
    layer.bias = None
    layer.infer_dtype = torch.bfloat16
    layer.fp8_activation_qmax = None

    assert layer.use_aiter_w8a8
    assert layer.weight_need_transpose is True
    layer.apply(torch.empty(4, 32))
    assert seen["weight"] == (16, 32)


def test_cuda_fp8_sgl_keeps_sgl_layout(monkeypatch):
    global_var.PLATFORM = "cuda"
    seen = {}

    def sgl(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
        seen["weight"] = tuple(mat_b.shape)
        return torch.empty(mat_a.shape[0], mat_b.shape[1])

    monkeypatch.setattr("lightx2v.common.ops.mm.mm_weight.sgl_fp8_scaled_mm", sgl)
    monkeypatch.setattr("lightx2v.models.input_encoders.hf.q_linear.sgl_fp8_scaled_mm", sgl)
    layer = MMWeightWfp8channelAfp8channeldynamicSgl("blk.weight", None)
    layer.act_quant_func = _quant
    layer.weight = torch.empty(32, 16)
    layer.weight_scale = torch.empty(16, 1)
    layer.bias = None
    layer.infer_dtype = torch.bfloat16

    assert layer.use_aiter_w8a8 is False
    assert layer.weight_need_transpose is True
    layer.apply(torch.empty(4, 32))
    assert seen["weight"] == (32, 16)

    linear = SglQuantLinearFp8(32, 16, bias=False)
    linear.act_quant_func = _quant
    linear.weight = torch.empty(16, 32)
    linear.weight_scale = torch.empty(16, 1)
    linear(torch.empty(1, 4, 32))
    assert seen["weight"] == (32, 16)
