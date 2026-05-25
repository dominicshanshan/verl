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

"""GPU parity tests for the TE NVFP4 backend.

These tests are gated on CUDA + a TE build that exposes NVFP4Quantizer. When
either is missing the entire module is skipped at collection time.
"""

import pytest
import torch

from verl.utils.qat import te_fp4

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for TE NVFP4 parity tests", allow_module_level=True)
if not te_fp4.is_te_nvfp4_available():
    pytest.skip("transformer_engine with NVFP4 is not installed", allow_module_level=True)

from verl.utils.qat.linear import (  # noqa: E402  imports gated on TE availability
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    QATLinear,
    QATMode,
    fp4_fake_quant_weight,
)


@pytest.fixture
def bf16_weight():
    torch.manual_seed(0)
    return torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")


def test_triton_vs_te_dequantized_match(bf16_weight):
    """With a shared global amax both backends must agree on the FP4 grid."""
    amax = bf16_weight.abs().amax().to(torch.float32)
    triton_out = fp4_fake_quant_weight(bf16_weight, global_amax=amax, block_size=16)
    te_out = te_fp4.te_fp4_fake_quant(bf16_weight, global_amax=amax, block_size=16)
    assert triton_out.dtype == te_out.dtype == torch.bfloat16
    diff = (triton_out.float() - te_out.float()).abs()
    rel = diff / triton_out.float().abs().clamp_min(1e-3)
    assert rel.mean().item() < 5e-3, f"mean rel diff too large: {rel.mean().item()}"


def test_real_quant_returns_three_tensors(bf16_weight):
    packed, fp8_scale, fp32_global = te_fp4.te_fp4_real_quant(bf16_weight, block_size=16)
    assert packed.dtype == torch.uint8
    # Packed has two FP4 nibbles per uint8.
    expected_packed = bf16_weight.numel() // 2
    assert packed.numel() == expected_packed
    # global = amax / (FP4_MAX * FP8_MAX) = amax / 2688
    expected = bf16_weight.abs().amax().float() / (FP4_E2M1_MAX * FP8_E4M3_MAX)
    assert torch.allclose(fp32_global.float().view(()), expected.view(()).cuda(), rtol=0, atol=1e-6)
    # Per-block FP8 scale: one byte per 16 input values.
    assert fp8_scale.numel() == bf16_weight.numel() // 16


def test_grouped_matches_single_tensor():
    torch.manual_seed(1)
    ws = [torch.randn(32, 64, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    amaxes = [w.abs().amax().to(torch.float32) for w in ws]
    grouped = te_fp4.te_fp4_real_quant_grouped(ws, amaxes, block_size=16)
    singles = [te_fp4.te_fp4_real_quant(w, global_amax=a, block_size=16) for w, a in zip(ws, amaxes, strict=True)]
    assert len(grouped) == len(singles)
    for (gp, gs, gg), (sp, ss, sg) in zip(grouped, singles, strict=True):
        assert torch.equal(gp, sp)
        assert torch.equal(gs, ss)
        assert torch.allclose(gg, sg, rtol=0, atol=0)


def test_observer_state_matches_across_backends():
    """static_minmax should keep input_amax in lock-step between Triton and TE."""
    torch.manual_seed(2)
    layer_kwargs = dict(
        in_features=64,
        out_features=64,
        bias=False,
        mode=QATMode.W4A4,
        group_size=16,
        activation_observer="static_minmax",
        device="cuda",
        dtype=torch.bfloat16,
    )
    triton_layer = QATLinear(quant_backend="triton", **layer_kwargs)
    te_layer = QATLinear(quant_backend="te", **layer_kwargs)
    te_layer.load_state_dict(triton_layer.state_dict())

    triton_layer.train()
    te_layer.train()

    for _ in range(4):
        x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
        triton_layer(x)
        te_layer(x)

    assert torch.allclose(triton_layer.input_amax, te_layer.input_amax, rtol=0, atol=0)
