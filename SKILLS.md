# AI Agent Guide

This guide is for humans and AI agents running `advanced-gguf-quantizer`. It
focuses on producing good GGUF artifacts, recording evidence, and avoiding
common mistakes.

## Operating Principles

- Use `llama-quantize` as the main command.
- Use `advanced-gguf-quantizer` only when calling an advanced helper subcommand
  directly.
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
   ./build/bin/llama-quantize recipe init --profile nvfp4_mxfp6 --output recipes/model.toml
   ```

4. Fill in source, output, calibration, imatrix, evaluation, KLD, target BPW,
   and precision mode fields.
5. Generate missing calibration artifacts with `llama-imatrix`.
6. Generate missing KLD evidence with the command from:

   ```bash
   ./build/bin/llama-quantize kld-command recipes/model.toml
   ```

   Use that command exactly for saved-logit KLD bases. It should only name the
   reference model, evaluation corpus, and output KLD base; leave
   `llama-perplexity` runtime defaults alone.

7. Validate and inspect:

   ```bash
   ./build/bin/llama-quantize recipe validate recipes/model.toml
   ./build/bin/llama-quantize plan recipes/model.toml
   ```

8. Run:

   ```bash
   ./build/bin/llama-quantize run recipes/model.toml --project runs/model --yes
   ```

9. Inspect and smoke-test the output:

   ```bash
   ./build/bin/llama-quantize inspect runs/model/output.gguf
   ./build/bin/llama-completion -m runs/model/output.gguf -p "The model is"
   ```

10. Record PPL/KLD, BPW, tensor mix, and smoke verdict in the project report.

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

## Best Candidate Reports

Use best-candidate reports to compare non-dominated candidates. The report uses
Pareto-style selection internally:

```bash
./build/bin/llama-quantize best runs/model/metrics.jsonl \
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
