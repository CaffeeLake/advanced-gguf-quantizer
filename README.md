# advanced-gguf-quantizer

`advanced-gguf-quantizer` is a llama.cpp-derived GGUF quantization toolkit for
creating, inspecting, evaluating, repairing, and testing GGUF
models. The first focus is Blackwell NVFP4 and the local MXFP6_E2M3 GGUF
format, including mixed NVFP4/MXFP6 models.

This repository keeps a limited set of tools:

- `llama-quantize`: the primary command and preferred entry
  point for recipes, plans, runs, inspection, repair, candidate reports, and the
  interactive wizard.
- `advanced-gguf-quantizer`:  binary that backs those subcommands.
- `llama-imatrix`: calibration matrix generation.
- `llama-perplexity`: PPL and saved-logit KLD evidence.
- `llama-completion`: quick generation smoke tests for produced GGUFs.
- Optional: `llama-bench` and `llama-fit-params` for runtime and sizing
  evidence.

This tool requires input GGUF files to be present locally.
It does not download models, import Hugging Face checkpoints, or convert checkpoints.
Use llama.cpp to create a source GGUF if necessary, then use this tool to create
and evaluate NVFP4/MXFP6 GGUF models.

## Alpha  Notice

MXFP6_E2M3 is experimental and is not supported by NVIDIA or llama.cpp. If
official MXFP6 support appears later, the official format may change and GGUF
models created here may not work with future runtimes.

Feedback is requested. The latest full CUDA MXFP6 branch can be installed from
<https://github.com/michaelw9999/llama.cpp/tree/mxfp6-cuda>.
The initial llama.cpp MXFP6 CPU only PR is located at
<https://github.com/ggml-org/llama.cpp/pull/22671>.

## What This Does

- Creates NVFP4 GGUFs with proper tensor and input scales.
- Creates experimental MXFP6_E2M3 and mixed NVFP4/MXFP6 models.
- CUDA-accelerated quantization paths for NVFP4/MXFP6 tensor encoding.
- CUDA-backed evaluation workflows via `llama-perplexity`,
  `llama-imatrix`, and runtime patch evaluation.
- Activation-aware quantization using imatrix input scales.
- Recipe and project files for reproducible runs.
- Real run artifacts: locked recipe, run log, tensor assignment log, manifest,
  `checkpoint-key.json`, quantization report, and validation scripts.
- Checkpointed candidate search so interrupted long runs can resume safely.
- In-place repair and edit feature for existing GGUFs.
- p99 and p999 KLD gates in quality mode, visible in reports.
- Fused decision units.
- Best-candidate reporting for quality and budget choices.
- A text UI and wizard for guided recipe creation, monitoring, recovery, repair,
  and inspection.
- Agent-guided quantization in [SKILLS.md](SKILLS.md) for automated runs.

## Build

Use a fresh build directory:

```bash
cmake -S . -B build -DGGML_CUDA=ON
cmake --build build -j 20
```

Expected binaries are in `build/bin/`.

## Quick Start

Inspect the source GGUF:

```bash
./build/bin/llama-quantize inspect models/source-bf16.gguf
```

Create a starting recipe:

```bash
./build/bin/advanced-gguf-quantizer recipe init --profile nvfp4 --output recipes/model.toml
```

Edit the recipe fields for your files:

```toml
[io]
input = "models/source-bf16.gguf"
output = "runs/model/model-nvfp4-mxfp6.gguf"

[calibration]
corpus = "data/calibration.txt"
imatrix = "runs/model/imatrix.dat"

[evaluation]
bf16_reference = "models/source-bf16.gguf"
corpus = "data/eval.txt"
kld_base = "runs/model/bf16.kld"

[target]
precision_mode = "nvfp4"
target_bpw = 4.8
```

Generate an imatrix:

```bash
./build/bin/llama-imatrix \
  -m models/source-bf16.gguf \
  -f data/calibration.txt \
  -o runs/model/imatrix.dat
```

Generate a saved-logit KLD base:

```bash
./build/bin/advanced-gguf-quantizer kld-command recipes/model.toml
```

Inspect the planned command without writing a model:

```bash
./build/bin/advanced-gguf-quantizer recipe validate recipes/model.toml
./build/bin/advanced-gguf-quantizer plan recipes/model.toml
```

Run real quantization:

```bash
./build/bin/advanced-gguf-quantizer run recipes/model.toml --project runs/model --yes
```

Inspect and test the result:

```bash
./build/bin/llama-quantize inspect runs/model/model-nvfp4.gguf
./build/bin/llama-completion -m runs/model/model-nvfp4.gguf -p "The model is"
```

## Example Paths

Compact Qwen-style dense or MTP model:

```bash
./build/bin/advanced-gguf-quantizer recipe init --profile nvfp4_mxfp6 --output recipes/qwen.toml
./build/bin/advanced-gguf-quantizer run recipes/qwen.toml --project runs/qwen --yes
```

Quality-first MoE model:

```bash
./build/bin/advanced-gguf-quantizer recipe init --profile mxfp6-primary --output recipes/moe.toml
./build/bin/advanced-gguf-quantizer run recipes/moe.toml --project runs/moe --yes
```

Repair or edit an existing GGUF:

```bash
./build/bin/advanced-gguf-quantizer recipe init --profile repair --output recipes/repair.toml
./build/bin/advanced-gguf-quantizer run recipes/repair.toml --project runs/repair --yes
```

Use `plan` to inspect a recipe. For model production, launch `run`; it writes
real artifacts and does not waste time on fake quantization passes.

## Choosing NVFP4, MXFP6, Or Mixed

`NVFP4` when smaller size size and Blackwell speed matter most. It is the smallest
advanced mode but has more quantization error, so quality mode should rely on
imatrix, KLD evidence, p99/p999 gates, and tensor exceptions.

`MXFP6_E2M3` when quality matters more than file size. It is larger than
NVFP4 but useful for sensitive tensors, MoE experts, output/head tensors,
or models where NVFP4-only error is high.  MXFP6 is still experimental.

`NVFP4_MXFP6` for a balanced model. The allocator can either:

- use MXFP6 to improve an NVFP4-first model while keeping the model compact;
- use MXFP6 as the primary target and demote selected tensors to NVFP4 to increase speed
and reduce model size, when measured quality loss is acceptable.

Fallback types such as `MXFP6_E2M3`, `Q4_K`, `Q6_K`, `Q8_0`, `BF16`, and
`F16` remain available for tensor policy, speed-aware candidate assignment,
output/head handling, exclusions, and repair.

## Documentation

- [Documentation index](docs/README.md)
- [Notices and attribution](NOTICE.md)
- [Flag and workflow guide](docs/advanced-gguf-quantizer-flags.md)
- [Layer policy guide](docs/advanced-gguf-quantizer-layer-policy.md)
- [CUDA acceleration guide](docs/advanced-gguf-quantizer-cuda-acceleration.md)
- [Imatrix and KLD guide](docs/advanced-gguf-quantizer-imatrix-kld.md)
- [NVFP4 GGUF contract](docs/advanced-gguf-quantizer-nvfp4-contract.md)
- [AI agent guide](SKILLS.md)
