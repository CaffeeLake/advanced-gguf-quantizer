# llama-perplexity

`llama-perplexity` evaluates a GGUF model on a local corpus. In this repository
it is used for PPL checks and saved-logit KLD evidence for quantization.

## Perplexity

```bash
./build/bin/llama-perplexity \
  -m runs/model/output.gguf \
  -f data/eval.txt
```

Perplexity is lower when the model predicts the corpus better. Compare PPL only
when tokenizers, corpus, model provenance, and runtime conditions match.

## Saved-Logit KLD Base

Create a reference KLD base with a trusted BF16 or high-quality model:

```bash
./build/bin/llama-perplexity \
  -m models/source-bf16.gguf \
  -f data/eval.txt \
  --save-all-logits runs/model/source-bf16.kld
```

Evaluate a quantized model against that base:

```bash
./build/bin/llama-perplexity \
  -m runs/model/output.gguf \
  -f data/eval.txt \
  --kl-divergence-base runs/model/source-bf16.kld \
  --kl-divergence
```

KLD bases can be large. Keep them with the recipe, imatrix, evaluation corpus,
and run manifest so comparisons are reproducible.

## Metrics To Record

- PPL and log-PPL ratio.
- Mean KLD.
- p95, p99, and p999 KLD.
- KLD tail mean and max KLD.
- RMS probability delta.
- Same-top rate.
- Top-flip weight.
- Teacher-top probability RMSE.
- Entropy RMSE.
- Non-finite counts.

Tiny samples are useful for loader checks, but not for quality claims.
