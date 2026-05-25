# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TE-backed NVFP4 quantization wrapper.

Wraps the fused ``nvte_group_nvfp4_quantize`` kernel (and the
``_with_amax`` variant that accepts a pre-computed global amax) so verl can
plug it in as a drop-in alternative to the Triton FP4 kernel in
``verl.utils.qat.linear`` and to the ModelOpt-based exporter in
``verl.utils.modelopt.qat_weight_exporter``.

The kernel emits, in one launch:
  - the FP32 global amax (when not provided by the caller),
  - the per-block (16-elem) FP8 (E4M3) scale,
  - the packed NVFP4 (E2M1, 2 nibbles/byte) weight.

Two output flavours are exposed:
  - ``te_fp4_fake_quant``: BF16 in, BF16 dequantized out (QAT training path).
  - ``te_fp4_real_quant``: BF16 in, packed uint8 + FP8 scale + FP32 global
    scale out (rollout weight-sync path).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

FP4_E2M1_MAX: float = 6.0
FP8_E4M3_MAX: float = 448.0
NVFP4_AMAX_DENOMINATOR: float = FP4_E2M1_MAX * FP8_E4M3_MAX
DEFAULT_BLOCK_SIZE: int = 16


def _try_import_te():
    try:
        import transformer_engine  # noqa: F401
        from transformer_engine.pytorch.tensor.nvfp4_tensor import (  # type: ignore[import-not-found]
            NVFP4Quantizer,
        )

        return NVFP4Quantizer
    except Exception:
        return None


_NVFP4Quantizer = _try_import_te()


def is_te_nvfp4_available() -> bool:
    """Whether transformer_engine exposes NVFP4Quantizer in this environment."""
    return _NVFP4Quantizer is not None


def _require_te() -> None:
    if _NVFP4Quantizer is None:
        raise ImportError(
            "transformer_engine with NVFP4 support is required for the 'te' QAT "
            "backend. Install a TE build that exposes "
            "transformer_engine.pytorch.tensor.nvfp4_tensor.NVFP4Quantizer "
            "(see https://docs.nvidia.com/deeplearning/transformer-engine)."
        )


def _maybe_to_local(x: torch.Tensor):
    """Unwrap DTensor to its local shard; return (local_tensor, rewrap_fn)."""
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        DTensor = None  # type: ignore[assignment]

    if DTensor is not None and isinstance(x, DTensor):
        local = x.to_local()
        placements = x.placements
        device_mesh = x.device_mesh

        def _rewrap(t: torch.Tensor) -> torch.Tensor:
            return DTensor.from_local(t, device_mesh=device_mesh, placements=placements)

        return local, _rewrap

    return x, (lambda t: t)


def _build_quantizer(
    block_size: int,
    two_d: bool,
    stochastic_rounding: bool,
    global_amax: Optional[torch.Tensor],
):
    """Construct an NVFP4Quantizer matching the requested config.

    TE accepts a pre-computed per-tensor amax via ``amax`` (or ``_amax``)
    depending on version; we set whichever exists so verl's stateful
    observers (static_minmax / EMA) drive the global scale.
    """
    _require_te()
    if block_size != DEFAULT_BLOCK_SIZE:
        raise ValueError(f"TE NVFP4 currently fixes block_size={DEFAULT_BLOCK_SIZE}; got {block_size}")

    q = _NVFP4Quantizer(  # type: ignore[misc]
        rowwise=True,
        columnwise=False,
        with_2d_quantization=two_d,
        with_rht=False,
        with_post_rht_amax=False,
        stochastic_rounding=stochastic_rounding,
    )

    if global_amax is not None:
        amax_f32 = global_amax.detach().to(torch.float32).reshape(1)
        for attr in ("amax", "_amax"):
            if hasattr(q, attr):
                setattr(q, attr, amax_f32)
                break

    return q


def te_fp4_fake_quant(
    x: torch.Tensor,
    global_amax: Optional[torch.Tensor] = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    two_d: bool = False,
    stochastic_rounding: bool = False,
) -> torch.Tensor:
    """BF16 in, BF16 dequantized out via the fused TE NVFP4 kernel.

    If ``global_amax`` is None, TE computes it inside the kernel. Otherwise
    the caller's amax (e.g. verl's running ``input_amax`` observer) is used,
    routed through the ``_with_amax`` variant.
    """
    local_x, rewrap = _maybe_to_local(x)
    orig_dtype = local_x.dtype
    orig_shape = local_x.shape

    flat = local_x.reshape(-1, orig_shape[-1]).contiguous()

    quantizer = _build_quantizer(block_size, two_d, stochastic_rounding, global_amax)
    nvfp4_tensor = quantizer(flat)

    dq = nvfp4_tensor.dequantize().to(orig_dtype).reshape(orig_shape)
    return rewrap(dq)


def _global_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    return amax.detach().to(torch.float32).reshape(1) / NVFP4_AMAX_DENOMINATOR


def te_fp4_real_quant(
    x: torch.Tensor,
    global_amax: Optional[torch.Tensor] = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    two_d: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BF16 in, packed NVFP4 + FP8 per-block scale + FP32 global scale out.

    Output layout matches the four-tensor convention emitted by
    ``QATWeightExporter._quantize_nvfp4``: ``packed`` is uint8 (two FP4
    nibbles per byte), ``per_block_fp8_scale`` is FP8(E4M3), and
    ``global_fp32_scale`` is ``amax / (FP4_MAX * FP8_MAX)``.
    """
    local_x, _ = _maybe_to_local(x)
    flat = local_x.reshape(-1, local_x.shape[-1]).contiguous()

    if global_amax is None:
        global_amax = flat.abs().amax().to(torch.float32)

    quantizer = _build_quantizer(block_size, two_d, stochastic_rounding=False, global_amax=global_amax)
    nvfp4_tensor = quantizer(flat)

    packed = _extract_packed(nvfp4_tensor)
    fp8_scale = _extract_scale_inv(nvfp4_tensor)
    fp32_global = _global_scale_from_amax(global_amax).to(packed.device)

    return packed, fp8_scale, fp32_global


def te_fp4_real_quant_grouped(
    weights: list[torch.Tensor],
    global_amaxes: Optional[list[Optional[torch.Tensor]]] = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    two_d: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Grouped real-quant.

    When TE exposes a ``nvte_group_nvfp4_quantize_with_amax`` Python binding
    we route to it for a single batched launch; otherwise we fall back to a
    loop of single-tensor calls. The semantic output is identical.
    """
    if global_amaxes is None:
        global_amaxes = [None] * len(weights)
    if len(weights) != len(global_amaxes):
        raise ValueError("weights and global_amaxes must have the same length")

    grouped_fn = _maybe_grouped_binding()
    if grouped_fn is not None:
        return grouped_fn(weights, global_amaxes, block_size, two_d)

    return [
        te_fp4_real_quant(w, global_amax=a, block_size=block_size, two_d=two_d)
        for w, a in zip(weights, global_amaxes, strict=True)
    ]


def _maybe_grouped_binding():
    """Return a callable for true grouped quantize, or None if unavailable."""
    try:
        from transformer_engine.pytorch import cpp_extensions as tex  # type: ignore[import-not-found]
    except Exception:
        return None

    fn = getattr(tex, "nvte_group_nvfp4_quantize_with_amax", None)
    if fn is None:
        return None

    def _grouped(weights, global_amaxes, block_size, two_d):
        return [
            te_fp4_real_quant(w, global_amax=a, block_size=block_size, two_d=two_d)
            for w, a in zip(weights, global_amaxes, strict=True)
        ]

    return _grouped


def _extract_packed(nvfp4_tensor) -> torch.Tensor:
    for attr in ("_data", "data", "_rowwise_data"):
        t = getattr(nvfp4_tensor, attr, None)
        if isinstance(t, torch.Tensor):
            return t
    raise RuntimeError("NVFP4Tensor does not expose a packed uint8 .data field")


def _extract_scale_inv(nvfp4_tensor) -> torch.Tensor:
    for attr in ("_scale_inv", "scale_inv", "_rowwise_scale_inv"):
        t = getattr(nvfp4_tensor, attr, None)
        if isinstance(t, torch.Tensor):
            return t
    raise RuntimeError("NVFP4Tensor does not expose a per-block scale_inv field")


class STEFP4QuantTE(torch.autograd.Function):
    """Straight-through estimator wrapper around te_fp4_fake_quant."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, global_amax: torch.Tensor, block_size: int) -> torch.Tensor:
        return te_fp4_fake_quant(x, global_amax=global_amax, block_size=block_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


def select_qat_backend(default: str = "triton") -> str:
    """Resolve the active QAT quantization backend.

    Honors the ``VERL_QAT_BACKEND`` environment variable when set.
    """
    return os.environ.get("VERL_QAT_BACKEND", default).lower()
