# CUDA Acceleration Guide

NVFP4 and MXFP6_E2M3 quantization are CUDA-first in this repository. CPU threads
coordinate file IO, GGUF writing, policy search, and metric reduction, while
CUDA performs the heavy tensor encoding and runtime model evaluation.

## Accelerated Paths

- NVFP4 tensor encoding uses CUDA kernels in `ggml/src/ggml-cuda/nvfp4-adv.cu`.
- MXFP6_E2M3 tensor encoding uses the same CUDA helper area.
- Full-tensor NVFP4 runtime tensor writes can repack compact GGUF bytes into the
  resident Blackwell layout on device.
- Measured candidate evaluation loads a runtime model once, patches candidate
  tensors, and runs `llama-perplexity` style decode on CUDA.
- `llama-imatrix` can run model execution on CUDA and then collect activation
  statistics on the host.

## Where CPU Threads Still Matter

CPU workers still handle:

- source tensor sampling and conversion;
- imatrix lookup and activation-aware scale setup;
- policy preparation and candidate bookkeeping;
- GGUF writer output;
- JSONL reports and manifests;
- PPL/KLD reductions after logits are available;
- project recovery and checkpoint management.

Use a conservative run profile on large models and shared machines. Leave CPU,
GPU memory, CUDA runtime, and file-cache headroom. Public docs should describe
resource intent instead of stream counts, row chunk sizes, selector workers, or
logits workspace caps as stable knobs.

## CUDA Encoding

CUDA tensor encoding is automatic. The recipe names the model goal and evidence
budget; the implementation chooses bounded chunks and workers for the run.

## Measured Candidate Evaluation

Quality mode uses runtime patch evaluation:

1. Quantize candidate tensor bytes with the selected policy.
2. Patch the resident runtime tensor.
3. Run CUDA decode on the evaluation window.
4. Reduce PPL/KLD metrics.
5. Restore or patch the next candidate.

This keeps measured evidence close to the final runtime path and avoids judging
quality only from proxy tensor error.

## Memory-Bounded Mode

For large models:

- choose a conservative run profile;
- use a run directory with checkpoints;
- prefer fewer measured candidates per pass;
- use `plan` to inspect the recipe before starting the expensive run.

Layer-by-layer streaming means the run should stage and patch the tensor groups
it needs instead of keeping every candidate resident at once.

## Validation

After CUDA or runtime changes, build and run cheap checks first:

```bash
cmake --build build --target llama-quantize advanced-gguf-quantizer llama-imatrix llama-perplexity -j 20
./build/bin/llama-quantize inspect models/source-bf16.gguf
./build/bin/llama-quantize plan recipes/model.toml
```

For quality claims, follow with a real run, `llama-completion` smoke test, and
matched PPL/KLD evaluation.
