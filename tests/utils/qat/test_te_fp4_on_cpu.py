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

"""CPU sanity tests for the TE-backed NVFP4 QAT wrapper."""

import pytest
import torch

from verl.utils.qat import te_fp4


def test_select_qat_backend_default():
    assert te_fp4.select_qat_backend("triton") == "triton"
    assert te_fp4.select_qat_backend("te") == "te"


def test_select_qat_backend_env(monkeypatch):
    monkeypatch.setenv("VERL_QAT_BACKEND", "te")
    assert te_fp4.select_qat_backend("triton") == "te"

    monkeypatch.setenv("VERL_QAT_BACKEND", "Triton")
    assert te_fp4.select_qat_backend("te") == "triton"


def test_is_te_nvfp4_available_returns_bool():
    assert isinstance(te_fp4.is_te_nvfp4_available(), bool)


def test_te_fake_quant_without_te_raises_cleanly():
    if te_fp4.is_te_nvfp4_available():
        pytest.skip("transformer_engine is installed; the import-guard path can't be exercised")
    x = torch.zeros(16, 32, dtype=torch.bfloat16)
    with pytest.raises(ImportError, match="transformer_engine"):
        te_fp4.te_fp4_fake_quant(x)


def test_te_real_quant_grouped_length_mismatch():
    if te_fp4.is_te_nvfp4_available():
        pytest.skip("hits TE; this CPU test only validates the input check")
    with pytest.raises(ValueError, match="same length"):
        te_fp4.te_fp4_real_quant_grouped(
            [torch.zeros(16, 16)],
            global_amaxes=[None, None],
        )


def test_constants_match_published_recipe():
    assert te_fp4.FP4_E2M1_MAX == 6.0
    assert te_fp4.FP8_E4M3_MAX == 448.0
    assert te_fp4.NVFP4_AMAX_DENOMINATOR == 6.0 * 448.0
    assert te_fp4.DEFAULT_BLOCK_SIZE == 16
