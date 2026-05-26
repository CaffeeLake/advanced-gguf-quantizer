# llama-fit-params

`llama-fit-params` is an optional helper for estimating runtime parameters that
fit a local GGUF model into available device memory.

Build it with:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DADVANCED_GGUF_QUANTIZER_BUILD_FIT_PARAMS=ON
cmake --build build --target llama-fit-params -j 16
```

Example:

```bash
./build/bin/llama-fit-params --model models/source.gguf
```

The tool prints suggested runtime arguments such as context size and tensor
placement overrides. Use those suggestions as sizing evidence for recipes and
smoke tests, not as a replacement for real PPL/KLD validation.

Typical follow-up:

```bash
./build/bin/llama-completion \
  --model runs/model/output.gguf \
  -p "quantizer smoke" \
  -n 32
```
