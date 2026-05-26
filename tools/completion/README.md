# llama-completion

`llama-completion` is retained so finished GGUF artifacts can be smoke-tested
with the same runtime codebase that produced them.

## Basic Smoke Test

```bash
./build/bin/llama-completion \
  -m runs/model/output.gguf \
  -p "The model is"
```

Use a short prompt and verify that the model loads, produces finite output, and
does not obviously degrade relative to the same prompt on the source or baseline
model.

## Useful Options

- `-m, --model`: GGUF model path.
- `-p, --prompt`: prompt text.
- `-f, --file`: prompt file.
- `-n, --n-predict`: token count to generate.
- `-c, --ctx-size`: context size.
- `-b, --batch-size`: logical batch size.
- `-ub, --ubatch-size`: physical batch size.
- `--flash-attn`: Flash Attention mode.
- `--seed`: sampling seed.
- `--temp`, `--top-k`, `--top-p`, `--min-p`: sampling controls.
- `--log-file`: write logs to a file.

For full option details, run:

```bash
./build/bin/llama-completion --help
```

Completion smoke tests are not quality proof by themselves. Pair them with
`llama-quantize inspect` and matched PPL/KLD evaluation for release claims.
