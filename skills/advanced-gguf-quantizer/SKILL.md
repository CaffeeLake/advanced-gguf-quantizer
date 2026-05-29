---
name: advanced-gguf-quantizer
description: Use advanced-gguf-quantizer to create, inspect, evaluate, repair, and report local GGUF quantization runs for NVFP4, MXFP6_E2M3, and mixed NVFP4/MXFP6 models.
---

# advanced-gguf-quantizer

Use this skill when a user asks an agent to make or evaluate an advanced GGUF
quantized model in this repository.

## Core Rules

- Use local GGUF files as inputs.
- Use `advanced-gguf-quantizer` for recipe, project, plan, run, candidates,
  best-report, imatrix-command, and kld-command workflows.
- Use `llama-quantize` as the direct model-writing engine and for GGUF
  inspection.
- Use recipes and project directories for serious runs.
- Use `plan` and `recipe validate` for inspection.
- Use `run` for artifact-producing requests.
- Do not add model download, checkpoint import, or remote conversion workflow.
- Keep `llama-completion`, `llama-imatrix`, and `llama-perplexity` in the same
  build so produced artifacts can be tested with the same codebase.

## Production Run Procedure

1. Identify the source GGUF, output path, precision profile, project directory,
   calibration corpus, imatrix, evaluation corpus, KLD base, BPW/VRAM target,
   and expected MTP/NextN status.
2. Inspect the source:

   ```bash
   ./build/bin/llama-quantize inspect models/source-bf16.gguf
   ```

3. Create a recipe if one is missing:

   ```bash
   ./build/bin/advanced-gguf-quantizer recipe init --profile nvfp4_mxfp6 --output recipes/model.toml
   ```

4. Generate missing imatrix and KLD commands from the recipe:

   ```bash
   ./build/bin/advanced-gguf-quantizer imatrix-command recipes/model.toml
   ./build/bin/advanced-gguf-quantizer kld-command recipes/model.toml
   ```

   For saved-logit KLD bases, run the printed KLD command as-is. Do not add
   context, batch, stride, or runtime scheduling overrides; the defaults are
   the supported path.

5. Validate:

   ```bash
   ./build/bin/advanced-gguf-quantizer recipe validate recipes/model.toml
   ./build/bin/advanced-gguf-quantizer plan recipes/model.toml
   ```

6. Start the real run:

   ```bash
   ./build/bin/advanced-gguf-quantizer run recipes/model.toml --project runs/model --yes
   ```

7. Inspect and smoke-test:

   ```bash
   ./build/bin/llama-quantize inspect runs/model/output.gguf
   ./build/bin/llama-completion -m runs/model/output.gguf -p "The model is"
   ```

8. Evaluate and record metrics when quality matters:

   ```bash
   ./build/bin/llama-perplexity -m runs/model/output.gguf -f data/eval.txt
   ```

## Quality Mode

Quality mode should use:

- imatrix input scales;
- BF16 or trusted-reference KLD base;
- real PPL/KLD measured evaluation;
- p99 and p999 KLD gates;
- holdout chunks when available;
- fused decision units for q/k/v, gate/up, expert pairs, MTP heads, and output
  policy;
- a best-candidate report for non-dominated candidates.

Useful command:

```bash
./build/bin/advanced-gguf-quantizer best runs/model/metrics.jsonl \
  --real-ppl-kld \
  --quality-only \
  --report runs/model/best-report.json
```

## Mixed Precision Guidance

- Use `nvfp4` for compact Blackwell-oriented artifacts.
- Use `mxfp6` for quality-first local MXFP6_E2M3 artifacts.
- Use `nvfp4_mxfp6` when MXFP6 should improve selected NVFP4 tensors.
- Use `mxfp6-primary` when MXFP6 is the default and NVFP4 is used only where
  measured loss is acceptable.
- Keep fallback types explicit: `MXFP6_E2M3`, `Q4_K`, `Q6_K`, `Q8_0`, `BF16`,
  and `F16`.
- For NVFP4 quality mode, RSF is tried first. Speed-aware tensor type
  candidates are capped to the worst NVFP4-error tensors, defaulting to the
  worst 10% with `--nvfp4-selector-candidate-fraction 0.10`.
- Do not run benchmarks inside quantization. Use static speed penalties during
  candidate search and benchmark final artifacts separately.

MXFP6_E2M3 is experimental and unsupported by NVIDIA and llama.cpp. If official
MXFP6 support appears later, the official format may differ and GGUFs made here
may not remain compatible. Record the exact branch/runtime in reports and model
cards.

## Memory-Bounded Runs

For large models:

- keep CPU worker counts below physical cores;
- set `runtime.cuda_chunk_mb`;
- set `selector.max_logits_gib`;
- keep measured candidate batches modest;
- resume from the project/checkpoint instead of starting over;
- reduce worker counts before reducing quality evidence.

## Evidence Standard

Do not claim a model is better unless source, recipe, calibration corpus,
imatrix, evaluation corpus, KLD base, CUDA device, and commit are comparable.
Report PPL, mean KLD, p99/p999 KLD, tail KLD, same-top rate, top-flip weight,
BPW, tensor mix, and smoke verdict.

## Artifacts To Check

- `recipe.lock.toml`
- `run.jsonl`
- `assignment.jsonl`
- `run-manifest.json`
- `checkpoint-key.json`
- `quantization-report.md`
- `validate-output-smoke.sh`
- final GGUF

## Helpful References

- `README.md`
- `NOTICE.md`
- `docs/advanced-gguf-quantizer-flags.md`
- `docs/advanced-gguf-quantizer-layer-policy.md`
- `docs/advanced-gguf-quantizer-cuda-acceleration.md`
- `docs/advanced-gguf-quantizer-imatrix-kld.md`
- `docs/advanced-gguf-quantizer-nvfp4-contract.md`
