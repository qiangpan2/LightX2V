"""
AMD ROCm optimized attention using aiter library.
Provides significantly faster attention computation on AMD GPUs (2.5x-6x speedup).
Internally uses FA3 (fmha_v3) when conditions are met.
"""

import torch
from loguru import logger

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

# Detect AMD ROCm platform
IS_AMD_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None

# aiter installation info
AITER_REPO = "https://github.com/ROCm/aiter.git"
AITER_COMMIT = "a7d3bf8cd47afbaf6a6133c1f12e3b01d2c27b0e"
AITER_INSTALL_CMD = f"""
# One-line install command for aiter (AMD ROCm optimized kernels):
git clone {AITER_REPO} /tmp/aiter && \\
cd /tmp/aiter && \\
git checkout {AITER_COMMIT} && \\
pip install -e .
"""

# Try to import aiter (AMD ROCm optimized)
aiter_flash_attn_varlen_func = None
AITER_AVAILABLE = False
AITER_IMPORT_ERROR = None

try:
    from aiter import flash_attn_varlen_func as aiter_flash_attn_varlen_func

    AITER_AVAILABLE = True
    logger.info("aiter flash_attn_varlen_func found (AMD ROCm optimized)")
except ImportError as e:
    AITER_IMPORT_ERROR = str(e)
    if IS_AMD_ROCM:
        logger.warning(f"aiter not found on AMD ROCm platform. For optimal performance, please install aiter:\n{AITER_INSTALL_CMD}")
    else:
        logger.debug("aiter not found (only available on AMD ROCm platform)")


@PLATFORM_ATTN_WEIGHT_REGISTER("aiter_attn")
class AiterAttnWeight(AttnWeightTemplate):
    """
    AMD ROCm optimized attention using aiter library.

    Performance:
        - 2.5x-6x faster than flash_attn package on AMD GPUs
        - Automatically uses FA3 (fmha_v3) when conditions are met

    Requirements:
        - aiter library (AMD ROCm)
        - AMD GPU with ROCm support
    """

    def __init__(self):
        self.config = {}

        # Check platform first
        if not IS_AMD_ROCM:
            raise RuntimeError(
                "aiter_attn is only available on AMD ROCm platform.\n"
                "Current platform is not AMD ROCm (torch.version.hip is not set).\n"
                "For NVIDIA GPUs, please use 'flash_attn2' or 'flash_attn3' instead."
            )

        # Check aiter availability
        if not AITER_AVAILABLE:
            raise ImportError(f"aiter is not installed on AMD ROCm platform.\nImport error: {AITER_IMPORT_ERROR}\nPlease install aiter for optimal performance:\n{AITER_INSTALL_CMD}")

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        if len(q.shape) == 3:
            bs = 1
        elif len(q.shape) == 4:
            bs = q.shape[0]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])

        # Callers may build cu_seqlens on CPU; the varlen kernel requires them on
        # the compute device, same as the flash_attn2/flash_attn3 paths do.
        if cu_seqlens_q is not None and cu_seqlens_q.is_cpu:
            cu_seqlens_q = cu_seqlens_q.to(q.device, non_blocking=True)
        if cu_seqlens_kv is not None and cu_seqlens_kv.is_cpu:
            cu_seqlens_kv = cu_seqlens_kv.to(k.device, non_blocking=True)

        x = aiter_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        ).reshape(bs * max_seqlen_q, -1)
        return x


aiter_fp8_dtypes = None
aiter_per_tensor_quant = None
aiter_flash_attn_fp8_pertensor_func = None
aiter_flash_attn_varlen_fp8_pertensor_func = None
AITER_FP8BF16_AVAILABLE = False
AITER_FP8BF16_IMPORT_ERROR = None

try:
    from aiter import dtypes as aiter_fp8_dtypes
    from aiter import per_tensor_quant as aiter_per_tensor_quant
    from aiter.ops.mha import flash_attn_fp8_pertensor_func as aiter_flash_attn_fp8_pertensor_func
    from aiter.ops.mha import flash_attn_varlen_fp8_pertensor_func as aiter_flash_attn_varlen_fp8_pertensor_func

    AITER_FP8BF16_AVAILABLE = True
    logger.info("aiter fp8bf16 pertensor kernels found")
except ImportError as e:
    AITER_FP8BF16_IMPORT_ERROR = str(e)
    if IS_AMD_ROCM:
        logger.warning(f"aiter fp8bf16 kernels not importable: {e}")


@PLATFORM_ATTN_WEIGHT_REGISTER("aiter_fp8bf16_attn")
class AiterFp8Bf16AttnWeight(AttnWeightTemplate):
    """aiter CK fp8 in / bf16 out. 3D THD uses varlen; 4D uses BSHD.

    Differs from aiter_attn: forwards causal, does not flatten 4D, returns
    out.shape[0] (packed-safe), synthesizes cu_seqlens=[0, S] when None.
    """

    def __init__(self):
        self.config = {}
        if not IS_AMD_ROCM:
            raise RuntimeError("aiter_fp8bf16_attn is only available on AMD ROCm platform.\nCurrent platform is not AMD ROCm (torch.version.hip is not set).")
        if not AITER_FP8BF16_AVAILABLE:
            raise ImportError(f"aiter_fp8bf16_attn requires aiter fp8bf16 kernels.\nImport error: {AITER_FP8BF16_IMPORT_ERROR}")

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        causal=False,
        **kwargs,
    ):
        # softmax_scale is ignored; the kernel uses 1/sqrt(D), which matches H3.
        q_q, q_s = aiter_per_tensor_quant(q, quant_dtype=aiter_fp8_dtypes.fp8)
        k_q, k_s = aiter_per_tensor_quant(k, quant_dtype=aiter_fp8_dtypes.fp8)
        v_q, v_s = aiter_per_tensor_quant(v, quant_dtype=aiter_fp8_dtypes.fp8)
        if q.ndim == 4:
            out = aiter_flash_attn_fp8_pertensor_func(q_q, k_q, v_q, q_s, k_s, v_s, causal=causal)
            return out.reshape(out.shape[0] * out.shape[1], -1)

        def _cu(cu, seq, device):
            if cu is None:
                return torch.tensor((0, seq), dtype=torch.int32, device=device)
            return cu.to(device, non_blocking=True) if cu.is_cpu else cu

        out = aiter_flash_attn_varlen_fp8_pertensor_func(
            q_q,
            k_q,
            v_q,
            q_s,
            k_s,
            v_s,
            _cu(cu_seqlens_q, q.shape[0], q.device),
            _cu(cu_seqlens_kv, k.shape[0], k.device),
            q.shape[0] if max_seqlen_q is None else max_seqlen_q,
            k.shape[0] if max_seqlen_kv is None else max_seqlen_kv,
            causal=causal,
        )
        return out.reshape(out.shape[0], -1)
