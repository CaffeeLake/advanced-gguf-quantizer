# Imatrix And KLD Guide

Imatrix and KLD artifacts make NVFP4/MXFP6 quantization activation-aware and
measurable. Keep them next to the recipe and run artifacts so another user or
agent can reproduce the model.

## Imatrix

An imatrix records activation statistics from a calibration corpus. NVFP4 and
MXFP6 candidate search uses it for input-scale and activation-aware decisions.

Generate one with:

```bash
./build/bin/llama-imatrix \
  -m models/source-bf16.gguf \
  -f data/calibration.txt \
  -o runs/model/imatrix.dat \
  -t 16 \
  -tb 16 \
  -b 2048 \
  -ub 1024
```

Recipe fields:

- `calibration.corpus`
- `calibration.imatrix`
- `calibration.ctx_size`
- `calibration.batch_size`
- `calibration.ubatch_size`
- `calibration.n_gpu_layers`

You can print the matching command from a recipe:

```bash
./build/bin/llama-quantize imatrix-command recipes/model.toml
```

## Saved-Logit KLD Base

A KLD base stores reference logits from a trusted BF16 or high-quality source
model. Quality mode compares candidate runtime logits against this base.

Print the matching command:

```bash
./build/bin/llama-quantize kld-command recipes/model.toml
```

Equivalent direct form:

```bash
./build/bin/llama-perplexity \
  -m models/source-bf16.gguf \
  -f data/eval.txt \
  --save-all-logits runs/model/bf16.kld
```

Important recipe fields:

- `evaluation.bf16_reference`
- `evaluation.corpus`
- `evaluation.kld_base`

Use the generated command as-is for saved-logit KLD bases. Do not add context,
batch, stride, or runtime scheduling overrides unless you are deliberately
running a separate diagnostic experiment and labeling it as such.

## Quality Metrics

Record at least:

- PPL and log-PPL ratio;
- mean KLD;
- p95, p99, and p999 KLD;
- KLD tail mean;
- max KLD;
- RMS probability delta;
- same-top rate;
- top-flip weight;
- teacher-top probability RMSE;
- entropy RMSE;
- non-finite counts.

Tiny KLD samples are loader checks only. Do not use them as quality evidence.

## Reuse Rules

Reuse an imatrix or KLD base only when the source model, tokenizer, corpus, and
recipe intent match. Runs write keys for the source, imatrix, KLD base, recipe,
selector flags, and commit so stale work is visible before it contaminates a
resumed run.
