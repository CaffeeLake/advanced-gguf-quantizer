# Security Policy

`advanced-gguf-quantizer` is a local GGUF quantization and evaluation tool. It
loads large model files, calibration corpora, imatrix files, KLD/logit evidence,
and recipes that can direct long-running CUDA work. Treat untrusted artifacts as
untrusted input.

## Reporting

Please report suspected security issues privately through GitHub private
vulnerability reporting when it is enabled for this repository. If private
reporting is not available, contact the maintainer privately before opening a
public issue.

Include:

- affected commit or release,
- exact command or recipe,
- involved model/evidence file provenance,
- proof of concept or crash reproducer,
- expected and observed behavior.

## Covered Surface

Security-sensitive reports are in scope for:

- `src/**/*`
- `ggml/**/*`
- `common/**/*`
- `tools/quantize/**/*`
- `tools/imatrix/**/*`
- `tools/perplexity/**/*`
- `tools/completion/**/*`

The upstream server/UI/RPC/demo application surfaces are not part of the retained
product release surface in this repository.

## Safe Use

- Run unknown GGUFs, recipes, corpora, and sidecar metric files in an isolated
  environment.
- Confirm hashes for model and evidence files used in published quality claims.
- Avoid exposing local quantization or evaluation commands to untrusted network
  input.
- Serialize benchmark runs on a quiet GPU when publishing performance evidence.
- Keep CUDA drivers, toolchains, and dependencies current.

Quality failures, crashes from malformed local inputs, or denial-of-service from
intentionally huge models are usually handled as bugs unless they cross a trust
boundary or enable unintended code execution or data exposure.
