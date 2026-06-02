# llama-bench

`llama-bench` is an optional helper for measuring prompt-processing and
token-generation throughput of produced GGUF artifacts.

Build it with:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DADVANCED_GGUF_QUANTIZER_BUILD_BENCH=ON
cmake --build build --target llama-bench -j 16
```

Run benchmarks on a quiet GPU and serialize comparisons:

```bash
./build/bin/llama-bench -m runs/model/output.gguf -p 512 -n 128
```

Useful options:

- `-m, --model`: local GGUF model path.
- `-p, --n-prompt`: prompt tokens for prompt-processing tests.
- `-n, --n-gen`: generated tokens for token-generation tests.
- `-pg`: combined prompt-processing and token-generation test.
- `-r, --repetitions`: repetitions per case.
- `-b, --batch-size`: logical batch size.
- `-ub, --ubatch-size`: physical batch size.
- `-fa, --flash-attn`: Flash Attention setting, `on`, `off`, or `auto`.
- `-o, --output`: output format, such as `md`, `csv`, `json`, or `jsonl`.

Do not run competing benchmark jobs at the same time on the same GPU. If a run
overlaps another workload, discard the numbers and repeat on a quiet device.

Throughput is a tradeoff axis. Pair it with quality metrics, BPW, tensor mix,
and validation smoke output when comparing quantized models.
