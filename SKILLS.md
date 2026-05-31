# AI Agent Guide

This guide is for humans and AI agents running `advanced-gguf-quantizer`. It
focuses on producing good GGUF artifacts, recording evidence, and avoiding
common mistakes.

## Operating Principles

- Use `advanced-gguf-quantizer` for recipe, project, plan, run, candidates,
  best-report, imatrix-command, and kld-command workflows.
- Use `llama-quantize` as the direct model-writing engine and for GGUF
  inspection.
- Start from local GGUF files.
- Put every serious run in a project directory.
- Prefer recipes over ad hoc command lines.
- Use `plan` and `recipe validate` for inspection.
- Use `run` when the user asked for a model artifact.
- Keep imatrix, KLD base, recipe lock, report, and final GGUF together.

## Minimum Production Workflow

1. Confirm the source GGUF exists.
2. Inspect it:

   ```bash
   ./build/bin/llama-quantize inspect models/source-bf16.gguf
   ```

3. Create or load a recipe:

   ```bash
   ./build/bin/advanced-gguf-quantizer recipe init --profile nvfp4_mxfp6 --output recipes/model.toml
   ```

4. Fill in source, output, calibration, imatrix, evaluation, KLD, target BPW,
   and precision mode fields.
5. Generate missing calibration artifacts with `llama-imatrix`.
6. Generate missing KLD evidence with the command from:

   ```bash
   ./build/bin/advanced-gguf-quantizer kld-command recipes/model.toml
   ```

   Use that command exactly for saved-logit KLD bases. It should only name the
   reference model, evaluation corpus, and output KLD base; leave
   `llama-perplexity` runtime defaults alone.

7. Validate and inspect:

   ```bash
   ./build/bin/advanced-gguf-quantizer recipe validate recipes/model.toml
   ./build/bin/advanced-gguf-quantizer plan recipes/model.toml
   ```

8. Run:

   ```bash
   ./build/bin/advanced-gguf-quantizer run recipes/model.toml --project runs/model --yes
   ```

9. Inspect and smoke-test the output:

   ```bash
   ./build/bin/llama-quantize inspect runs/model/output.gguf
   ./build/bin/llama-completion -m runs/model/output.gguf -p "The model is"
   ```

10. Record PPL/KLD, BPW, tensor mix, and smoke verdict in the project report.

## Long-Running Run Discipline

Large dense models can run for hours. Treat an in-progress production run as
valuable state:

- start runs with a log file and a project directory;
- keep the terminal/session attached or record the process ID;
- monitor the log, checkpoint/output file size, process state, CPU/RSS, and GPU
  memory/utilization;
- do not stop a run just because one status field is quiet or shows
  `output=0.0 MiB`;
- do not run overlapping GPU benchmarks or evaluations;
- if a run fails, preserve the project directory and checkpoints, diagnose the
  first error in the log, fix the cause, rebuild, and resume when possible;
- do not restart a broad search from scratch when a checkpoint can be reused.

Useful monitoring commands are intentionally ordinary:

```bash
pgrep -af 'llama-quantize|advanced-gguf-quantizer'
nvidia-smi
tail -f runs/model/logs/run.log
ls -lh runs/model/*.gguf runs/model/*.gguf.bwq-checkpoint.gguf
```

## Recipe Checklist

For a high-quality run, verify:

- `io.input`
- `io.output`
- `target.precision_mode`
- `target.target_bpw` or another explicit size goal
- `calibration.corpus`
- `calibration.imatrix`
- `evaluation.bf16_reference`
- `evaluation.corpus`
- `evaluation.kld_base`
- `base.threads`
- CPU/GPU budget and memory headroom
- evaluation effort and measured-candidate breadth
- checkpoint/resume policy
- output/head policy
- MTP preservation policy

## Choosing A Profile

- `nvfp4`: smallest advanced mode; best when speed and size dominate.
- `nvfp4_mxfp6`: default balanced mode; starts compact and spends MXFP6 on
  sensitive tensors.
- `mxfp6-primary`: quality-first mode; starts with MXFP6 and uses NVFP4 where
  measured loss is acceptable.
- `mxfp6`: pure MXFP6_E2M3.
- `repair`: edit or repair an existing GGUF.

MXFP6_E2M3 is experimental and unsupported by NVIDIA and llama.cpp. Future
official support may use a different format, so keep model cards and reports
clear about the exact branch/runtime used. Feedback is requested at
<https://github.com/michaelw9999/llama.cpp/tree/mxfp6-cuda>.

## Candidate Search Expectations

Quality mode may build a quantized checkpoint before candidate search. That
checkpoint is the measured seed model used for runtime patch evaluation; it is
not necessarily the final artifact. The selector can then load the checkpoint
once, patch candidate tensors or tensor scales, measure KLD/PPL, and restore
the original bytes without rebuilding the whole model from BF16 for every
candidate.

Do not confuse the search stages:

- proxy ranking uses tensor-local evidence and can reject obviously redundant
  candidates quickly;
- the experimental assignment-ledger planner fields (`selector.ledger`,
  `selector.search`, `selector.local_top_k`, `selector.group_units`,
  `selector.beam_width`, `selector.exact_budget`, and `selector.delta_mode`)
  are search bookkeeping, not release evidence;
- measured ranking uses runtime model output against the KLD base and is the
  quality evidence that matters;
- holdout KLD is validation evidence, not the same subset used to choose
  candidates;
- the final artifact should be based on the exact measured checkpoint or exact
  accepted patches, not on a fresh unmeasured rewrite when avoidable.

For pure MXFP6_E2M3 runs, the implementation may still pass through shared
NVFP4/MXFP6 selector plumbing. NVFP4 policy names with `tensors=0` and
`skipped=no_nvfp4_tensors` are bookkeeping noise, not active NVFP4 cap tuning.
The active MXFP6-only refinement is tensor-scale selection over
`mxfp6.selector_scale_candidates`; these are positive scale multipliers for the
MXFP6 tensor scale around the E8M0 block-scale representation, not NVFP4
four-over-six or FP8 cap policies.

For NVFP4 quality mode, RSF is part of the normal candidate path by default.
The main selector may also consider speed-aware tensor type candidates for the
worst NVFP4-error tensors. This is not a separate repair pass. NVFP4/RSF
tensor-local repair is tried first; fallback types such as `Q4_K`, `Q6_K`,
`Q8_0`, and, in mixed runs, `MXFP6_E2M3` should only be applied when NVFP4
still has high tensor-local error. The default fast path caps this at the worst
10% of NVFP4 candidate tensors via `--nvfp4-selector-candidate-fraction 0.10`.
Do not run benchmarks during quantization; benchmark final artifacts separately.

## Quality Rules

- Treat PPL as a guardrail, not the whole objective.
- Use mean KLD, p99 KLD, p999 KLD, KLD tail mean, max KLD, RMS probability
  delta, same-top rate, top-flip weight, top-probability RMSE, and entropy RMSE.
- Make p99 and p999 gate settings visible in reports.
- Use holdout chunks when available.
- Do not compare candidates that used different source models, imatrices,
  corpora, KLD bases, or commits.
- Do not claim quality from proxy scores, first chunks, tiny KLD samples, or
  mismatched baselines.

## Final Evaluation And Benchmarks

After a serious artifact run, evaluate the final GGUF with the same reference
model, KLD base, corpus, and code commit used for comparable models. Prefer a
plain command shape for final evidence:

```bash
./build/bin/llama-perplexity \
  -m runs/model/output.gguf \
  -f data/wiki.test.raw \
  --kl-divergence \
  --kl-divergence-base runs/model/reference.kld
```

Record PPL(Q), PPL(base), mean KLD, p95/p99/p999 KLD, max KLD, RMS probability
delta, same-top rate, top-flip weight, entropy drift/RMSE, non-finite counts,
elapsed time, and maximum RSS. Then run `llama-bench` separately on a quiet GPU:

```bash
./build/bin/llama-bench -m runs/model/output.gguf
```

When comparing two GGUFs, also report exact byte size, BPW, tensor mix, and MTP
status. Do not compare one model using full KLD evidence against another model
using a proxy, a first chunk, or a different KLD base.

## Best Candidate Reports

Use best-candidate reports to compare non-dominated candidates. The report uses
Pareto-style selection internally:

```bash
./build/bin/advanced-gguf-quantizer best runs/model/metrics.jsonl \
  --real-ppl-kld \
  --quality-only \
  --report runs/model/best-report.json
```

Use `--quality-only` for quality decisions. Treat BPW, size, VRAM, and speed as
constraints unless the user requested an explicit tradeoff report.

## Advanced Knobs

Leave low-level knobs empty unless the run is an expert reproduction or a
deliberate optimizer study. The useful expert controls are MXFP6 input-scale
denom/quantile, MXFP6 tensor-scale sample/step effort, mixed-format proxy sample
breadth, and mixed-format imatrix weight shaping. Prefer recipe fields over ad
hoc process settings.

Do not add runtime-shape, batch, fit, or scheduling overrides to final PPL/KLD
commands unless the comparison explicitly requires them. If a knob is needed for
quantization quality, put it in the recipe so it is visible in `plan`, the
recipe lock, and the run manifest.

## Memory Discipline

Large models can freeze a workstation when too many CPU workers and logits
buffers are active. Use these defaults unless the machine has headroom:

- leave several CPU cores idle;
- use conservative evaluation effort and measured-candidate breadth;
- prefer checkpoint/resume over wide restarts;
- resume from checkpoints instead of restarting wider searches.

## Artifact Expectations

Every serious run should produce:

- final GGUF;
- `recipe.lock.toml`;
- `run.jsonl`;
- `assignment.jsonl`;
- `run-manifest.json`;
- `checkpoint-key.json`;
- `quantization-report.md`;
- `validate-output-smoke.sh`;
- metrics JSONL when evaluation was run.

## Terms To Use

- `candidate search`, not internal selector jargon in user docs.
- `quality repair`, not rescue in user docs.
- `four-over-six`, not low-level choose names.
- `mixed NVFP4/MXFP6`, not abbreviated internal names.
- `reuse base` or `edit base`, not implementation shorthand.

## Boundaries

- This tool quantizes local GGUFs.
- It does not import checkpoints from Hugging Face.
- It does not download models.
- MXFP6_E2M3 is a local GGUF quantization format produced here.
- Use another llama.cpp checkout for source GGUF conversion, then use this tool
  for quantization and evaluation.

## References

- [README](README.md)
- [Notices and attribution](NOTICE.md)
- [Flag and workflow guide](docs/advanced-gguf-quantizer-flags.md)
- [Layer policy guide](docs/advanced-gguf-quantizer-layer-policy.md)
- [CUDA acceleration guide](docs/advanced-gguf-quantizer-cuda-acceleration.md)
- [Imatrix and KLD guide](docs/advanced-gguf-quantizer-imatrix-kld.md)
- [NVFP4 GGUF contract](docs/advanced-gguf-quantizer-nvfp4-contract.md)
