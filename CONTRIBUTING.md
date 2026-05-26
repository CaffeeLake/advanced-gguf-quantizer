# Contributing

`advanced-gguf-quantizer` is a focused llama.cpp-derived GGUF quantization
toolkit. Contributions should improve quantization, calibration, evaluation,
GGUF inspection, repair/edit workflows, documentation, or the retained smoke
test tools.

## Good Contribution Areas

- NVFP4, MXFP6_E2M3, and mixed NVFP4/MXFP6 quantization.
- GGUF auxiliary scale tensor handling.
- CUDA acceleration and runtime patch evaluation.
- Recipe, project, checkpoint, report, and text UI workflows.
- `llama-quantize`, `advanced-gguf-quantizer`, `llama-imatrix`,
  `llama-perplexity`, `llama-completion`, `llama-bench`, and
  `llama-fit-params`.
- Tests and docs that make high-quality quantization easier to reproduce.

## Workflow

- Keep changes reviewable and commit frequently.
- Build the retained targets after C++ or CMake edits.
- Use recipes and projects for behavior examples.
- Preserve MTP/NextN metadata and tensors unless a recipe explicitly changes the
  policy.
- Preserve `NOTICE.md`, including Michael Wand attribution and MIT HAN Lab Four
  Over Six attribution.
- Keep the MXFP6 experimental compatibility warning in user docs and the TUI.
- Do not add model download, checkpoint import, server, web UI, or unrelated app
  features without maintainer approval.
- Do not use fake quantization passes as a substitute for real artifact runs.

## Build Check

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-quantize advanced-gguf-quantizer llama-completion llama-imatrix llama-perplexity -j 20
```

Optional:

```bash
cmake --build build --target llama-bench llama-fit-params -j 20
```

## Evidence For Quality Changes

When claiming quality improvement, include:

- source GGUF;
- recipe lock;
- calibration corpus and imatrix;
- evaluation corpus and KLD base;
- PPL, mean KLD, p99/p999 KLD, tail KLD, same-top rate, top-flip weight, BPW,
  tensor mix, and smoke verdict;
- build commit and CUDA device.

## Style

- Prefer existing llama.cpp style and local helper APIs.
- Keep public vocabulary simple: candidate search, quality repair,
  four-over-six, mixed NVFP4/MXFP6, best-candidate report.
- Keep comments short and useful.
- Fix numeric failures at the source rather than hiding them behind quiet
  fallbacks.
