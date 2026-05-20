# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl is a flexible, efficient, production-ready reinforcement learning training library for large language models, originating from the *HybridFlow: A Flexible and Efficient RLHF Framework* paper (ByteDance Seed). It supports a wide set of post-training algorithms (PPO, GRPO, DPO, DAPO, GSPO, ReMax, RLOO, REINFORCE++, PRIME, …) on top of pluggable training backends (FSDP, FSDP2, Megatron-LM, TorchTitan, VeOmni) and rollout backends (vLLM, SGLang, TensorRT-LLM, HF Transformers). It scales from small models to 671B-parameter MoEs across hundreds of GPUs.

Upstream AI-contribution policy and commit conventions live in [`AGENTS.md`](AGENTS.md) and apply unchanged here — this file extends them with architecture and command guidance specific to Claude Code sessions.

## Common Commands

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

uv pip install pre-commit hydra-core
pre-commit install
```

Python `>=3.10` is required (see `pyproject.toml`).

### Install verl

Pick one rollout backend and (optionally) one training backend:

```bash
uv pip install -e ".[test]"        # tests + pre-commit deps
uv pip install -e ".[vllm]"        # vLLM rollout (>= 0.8.5; avoid 0.7.x)
uv pip install -e ".[sglang]"      # SGLang rollout
uv pip install -e ".[trtllm]"      # TensorRT-LLM rollout (>= 1.2.0rc6)
uv pip install -e ".[mcore]"       # Megatron-LM training backend
```

Other extras defined in `pyproject.toml`: `trl`, `prime`, `geo`, `gpu`, `math`. Backend-specific install helpers also live in `scripts/install_vllm_sglang_mcore.sh` and `scripts/install_sglang_mcore_npu.sh`.

### Tests

Tests are split by accelerator requirement (see `tests/README.md`):

- Files named `*_on_cpu.py` run on CPU.
- Other test files require GPU(s).
- Directories prefixed `special_` are special suites: `special_distributed` (multi-GPU), `special_e2e` (end-to-end), `special_npu`, `special_sanity` (quick checks), `special_standalone` (dedicated env).

```bash
# Run a single test (typical local dev loop)
pytest tests/path/to/test_file.py::TestClass::test_method -v

# Reproduce the CPU-only CI job (cpu_unit_tests.yml)
printf '[pytest]\npython_files = *_on_cpu.py\n' > pytest.ini
pytest -s -x --asyncio-mode=auto tests/

# GPU unit-test CI job (gpu_unit_tests.yml) uses a long
# --ignore-glob list — read the workflow file for the exact command.
```

### Lint, format, type-check

`pre-commit` is the canonical entry point and runs ruff, ruff-format, mypy, and several custom sanity scripts (license headers, docstring coverage, device-API usage, `DataProto` usage, naming conventions, structure validation, plus `scripts/generate_trainer_config.sh` which regenerates auto-generated trainer configs):

```bash
pre-commit run --all-files      # full sanity (run before pushing)
pre-commit run                  # staged files only

# Or invoke the underlying tools directly:
ruff check --fix .              # line-length=120
ruff format .
mypy .
```

If the `autogen-trainer-cfg` hook reports changes, commit the regenerated `verl/trainer/config/_generated_*.yaml` files — don't hand-edit them; edit the source dataclasses and rerun the hook.

### Docs

```bash
cd docs && pip install -r requirements-docs.txt && make html
```

## Architecture

verl follows a **hybrid-controller programming model** (per the HybridFlow paper): a single Ray driver orchestrates pools of Ray-actor workers, and batches travel between them as `DataProto` (TensorDict) objects.

### Data flow

```
                   ┌──────────────────────────────────┐
                   │   Ray driver (RayPPOTrainer)     │
                   │   verl/trainer/ppo/ray_trainer.py│
                   └────────────┬─────────────────────┘
                                │ DataProto batches
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
  Rollout workers         Training workers         Reward workers
  (vllm/sglang/           (EngineWorker over       (reward_manager/)
   trtllm/hf)              fsdp/megatron/...)
  verl/workers/rollout    verl/workers/engine_workers.py
```

### Major packages

- **`verl/trainer/`** — RL algorithm entry points. `main_ppo_sync.py` is the current PPO entry; `main_ppo.py` is deprecated. Both are Hydra apps rooted at `verl/trainer/config/ppo_trainer.yaml`. Core math (advantage estimation, KL control, loss) is in `verl/trainer/ppo/core_algos.py`. SFT entry points are `sft_trainer.py` (torchrun) and `sft_trainer_ray.py` (Ray).
- **`verl/workers/`** — three Ray-actor worker roles:
  - **Rollout** (generation): `rollout/vllm_rollout/`, `rollout/sglang_rollout/`, `rollout/trtllm_rollout/`, `rollout/hf_rollout/`, `rollout/naive/`. `rollout/replica.py` manages model replication/versioning when policy weights update.
  - **Training**: `engine_workers.py` wraps `EngineWorker` over a backend implementation under `workers/engine/` (`fsdp/`, `megatron/`, `torchtitan/`, `veomni/`, `automodel/`).
  - **Reward**: `reward_manager/` computes rewards (function- or model-based).
- **`verl/single_controller/`** — the Ray-orchestration layer. `base/worker.py` (`Worker`) and `base/worker_group.py` (`WorkerGroup`) plus role specializations (`actor.py`, `critic.py`, `rollout.py`, `engine.py`). The newer `ray/` subpackage holds the current single-controller mode; older multi-controller code is being phased out.
- **`verl/protocol.py`** — `DataProto`, the TensorDict-based wire format every component speaks. Handles padding, validation, and cross-device tensor sync. **Read this before adding fields that flow between trainer ↔ workers.**
- **`verl/base_config.py`** — frozen-dataclass `BaseConfig` that backs every config object (mapping-style access + immutability + type safety). Hydra YAMLs deserialize into these.
- **`verl/models/`** — model loading/registry, weight management, and device placement for the various backends (`mcore`, `transformers`, …).
- **`verl/utils/`** — large utility set: `fsdp_utils.py`, `megatron_utils.py`, profiling/memory tools, SGLang/vLLM integration helpers, sequence-packing/Ulysses helpers.
- **`verl/checkpoint_engine/`** — multi-backend checkpoint save/load (NCCL, HCCL, Kimi, Mooncake, Nixon) coordinating platform-specific all-reduce during checkpointing.

### Recipes and examples

- **`recipe/`** — algorithm recipes (GRPO, GSPO, DPPO, ReMax, RLOO, DAPO, distillation, profiling, tuning, …) live in the separate `verl-project/verl-recipe` repo and are pulled in as a submodule; each recipe pins a verl version via its `REQUIRED_VERL.txt`.
- **`examples/`** — runnable end-to-end examples organized by algorithm (`grpo/`, `gspo/`, `mtp/`, `ppo/`, …), each with data prep, configs, and multi-GPU launch scripts.

### Configuration

verl is Hydra-driven. The composition root is `verl/trainer/config/ppo_trainer.yaml`, assembled from modular sub-configs under `verl/trainer/config/`: `actor/`, `critic/`, `reward/`, `ref/`, `rollout/`, `engine/`, `model_engine/`, `data/`, `optim/`, `profiler/`, `distillation/`. Several YAMLs in this tree are auto-generated from `BaseConfig` dataclasses by `scripts/generate_trainer_config.sh` (run as a pre-commit hook) — edit the source dataclass, not the generated file.

### Backends matrix

| Role     | Backends                                                          |
|----------|-------------------------------------------------------------------|
| Training | FSDP, FSDP2 (recommended), Megatron-LM, TorchTitan, VeOmni        |
| Rollout  | vLLM (>=0.8.5; avoid 0.7.x), SGLang, TensorRT-LLM, HF Transformers|
| Hardware | NVIDIA GPUs, AMD ROCm, Ascend NPU                                 |

## AI-Assisted Contributions

See [`AGENTS.md`](AGENTS.md) for the mandatory contribution policy (duplicate-work checks, no-busywork rule, accountability requirements, fail-closed behavior) and development workflow (commit trailers, handling agent-bot reviews). Those rules apply in full to Claude Code sessions in this repo.
