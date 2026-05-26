# advanced-gguf-quantizer

`advanced-gguf-quantizer` is a llama.cpp-derived GGUF quantization toolkit for
creating, inspecting, evaluating, repairing, and smoke-testing advanced GGUF
models. Its first focus is Blackwell-class NVFP4 and the local MXFP6_E2M3 GGUF
format, including mixed NVFP4/MXFP6 models.

The repository keeps a focused llama.cpp tool surface:

- `llama-quantize`: the main artifact-writing command and the preferred entry
  point for recipes, plans, runs, inspection, repair, candidate reports, and the
  interactive wizard.
- `advanced-gguf-quantizer`: the helper binary that backs those advanced
  subcommands while they are folded into `llama-quantize`.
- `llama-imatrix`: calibration matrix generation.
- `llama-perplexity`: PPL and saved-logit KLD evidence.
- `llama-completion`: quick generation smoke tests for produced GGUFs.
- Optional: `llama-bench` and `llama-fit-params` for runtime and sizing
  evidence.

This tool expects local GGUF files. It does not download models, import
Hugging Face checkpoints, or convert source checkpoints. Use an upstream
llama.cpp checkout to create a source GGUF, then use this repository to create
and evaluate NVFP4/MXFP6 GGUF outputs.

## Alpha Compatibility Notice

MXFP6_E2M3 is experimental and is not supported by NVIDIA or llama.cpp. If
official MXFP6 support appears later, the official format may change and GGUF
models created here may not work with future runtimes.

Feedback is requested. The current MXFP6-capable branch can be installed from
<https://github.com/michaelw9999/llama.cpp/tree/mxfp6-cuda>.

## What It Adds

- NVFP4 GGUF writing with the auxiliary weight, tensor-scale, and input-scale
  tensors needed by the runtime.
- MXFP6_E2M3 GGUF writing, plus mixed NVFP4/MXFP6 model creation.
- CUDA-accelerated quantization paths for NVFP4/MXFP6 tensor encoding.
- CUDA-backed evaluation workflows through retained `llama-perplexity`,
  `llama-imatrix`, and runtime patch evaluation.
- Activation-aware quantization using imatrix input scales.
- Recipe and project files for reproducible runs.
- Real run artifacts: locked recipe, run log, tensor assignment log, manifest,
  `checkpoint-key.json`, quantization report, and validation smoke script.
- Checkpointed candidate search so interrupted long runs can resume safely.
- In-place repair and edit flows for existing GGUFs.
- p99 and p999 KLD gates in quality mode, visible in reports.
- Fused decision units for coherent q/k/v, gate/up, expert-pair, MTP, and
  output/head policy decisions.
- Best-candidate reporting for non-dominated quality and budget
  choices.
- A text UI and wizard for guided recipe creation, monitoring, recovery, repair,
  and inspection.
- Agent guidance in [SKILLS.md](SKILLS.md) for automated quantization runs.

## Build

Use a fresh build directory:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 20
```

The focused quantizer build is enabled by default:

```bash
-DADVANCED_GGUF_QUANTIZER_STANDALONE=ON
```

Optional helper tools:

```bash
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DADVANCED_GGUF_QUANTIZER_BUILD_BENCH=ON \
  -DADVANCED_GGUF_QUANTIZER_BUILD_FIT_PARAMS=ON \
  -DCMAKE_BUILD_TYPE=Release
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
./build/bin/llama-quantize recipe init --profile nvfp4_mxfp6 --output recipes/model.toml
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
precision_mode = "nvfp4_mxfp6"
target_bpw = 4.8
```

Generate an imatrix:

```bash
./build/bin/llama-imatrix \
  -m models/source-bf16.gguf \
  -f data/calibration.txt \
  -o runs/model/imatrix.dat
```

Generate the saved-logit KLD base:

```bash
./build/bin/llama-quantize kld-command recipes/model.toml
```

Run the printed command as-is.

Inspect the planned command without writing a model:

```bash
./build/bin/llama-quantize recipe validate recipes/model.toml
./build/bin/llama-quantize plan recipes/model.toml
```

Run the real quantization:

```bash
./build/bin/llama-quantize run recipes/model.toml --project runs/model --yes
```

Inspect and smoke-test the result:

```bash
./build/bin/llama-quantize inspect runs/model/model-nvfp4-mxfp6.gguf
./build/bin/llama-completion -m runs/model/model-nvfp4-mxfp6.gguf -p "The model is"
```

## Public Happy Paths

Compact Qwen-style dense or MTP model:

```bash
./build/bin/llama-quantize recipe init --profile nvfp4_mxfp6 --output recipes/qwen.toml
./build/bin/llama-quantize run recipes/qwen.toml --project runs/qwen --yes
```

Quality-first Mistral or MoE model:

```bash
./build/bin/llama-quantize recipe init --profile mxfp6-primary --output recipes/moe.toml
./build/bin/llama-quantize run recipes/moe.toml --project runs/moe --yes
```

Repair or edit an existing GGUF:

```bash
./build/bin/llama-quantize recipe init --profile repair --output recipes/repair.toml
./build/bin/llama-quantize run recipes/repair.toml --project runs/repair --yes
```

Use `plan` to inspect a recipe. For model production, launch `run`; it writes
real artifacts and does not waste time on fake quantization passes.

## Choosing NVFP4, MXFP6, Or Mixed

Use `NVFP4` when size and Blackwell speed matter most. It is the smallest
advanced mode but has more quantization error, so quality mode should rely on
imatrix, KLD evidence, p99/p999 gates, and tensor exceptions.

Use `MXFP6_E2M3` when quality matters more than file size. It is larger than
NVFP4 but often useful for sensitive tensors, MoE experts, output/head tensors,
or models where NVFP4-only error is too high.

Use `NVFP4_MXFP6` for balanced artifacts. The allocator can either:

- use MXFP6 to improve an NVFP4-first model while keeping the artifact compact;
- use MXFP6 as the primary quality path and demote selected tensors to NVFP4
  where measured quality loss is acceptable.

Fallback types such as `Q8_0`, `Q6_K`, `Q4_0`, `BF16`, and `F16` remain
available for tensor policy, output/head handling, exclusions, and repair.

## Evidence Standard

A useful model report should keep:

- source GGUF and locked recipe;
- calibration corpus and imatrix;
- evaluation corpus and KLD base;
- code commit and CUDA device;
- PPL, mean KLD, p95/p99/p999 KLD, KLD tail mean, max KLD, RMS probability
  delta, same-top rate, top-flip weight, entropy drift, and non-finite counts;
- output size, BPW, tensor mix, MTP status, and scale tensor counts;
- at least one `llama-completion` smoke test.

Do not compare models using a proxy score, one first chunk, a tiny diagnostic
KLD sample, or mismatched calibration/evaluation provenance.

## Documentation

- [Documentation index](docs/README.md)
- [Notices and attribution](NOTICE.md)
- [Flag and workflow guide](docs/advanced-gguf-quantizer-flags.md)
- [Layer policy guide](docs/advanced-gguf-quantizer-layer-policy.md)
- [CUDA acceleration guide](docs/advanced-gguf-quantizer-cuda-acceleration.md)
- [Imatrix and KLD guide](docs/advanced-gguf-quantizer-imatrix-kld.md)
- [NVFP4 GGUF contract](docs/advanced-gguf-quantizer-nvfp4-contract.md)
- [AI agent guide](SKILLS.md)
