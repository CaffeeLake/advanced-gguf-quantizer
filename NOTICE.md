# Notices And Attribution

This repository is derived from llama.cpp and ggml and is distributed under the
MIT license in [LICENSE](LICENSE). Keep this notice with redistributed copies,
forks, public model cards, and public benchmark writeups that rely on the
advanced GGUF quantization features in this repository.

## advanced-gguf-quantizer

The advanced GGUF quantization, recipe, project, repair/edit, text UI,
checkpoint-key, NVFP4/MXFP6, and reporting additions in this repository are
maintained by Michael Wand.

Copyright (c) 2026 Michael Wand and advanced-gguf-quantizer contributors.

Please cite `advanced-gguf-quantizer` when publishing models, measurements, or
derivative tools that rely on these additions.

Do not remove the Michael Wand attribution, this notice, or the citation
metadata from redistributed copies, forks, public model cards, benchmark
writeups, or derivative tools that rely on the advanced quantization features in
this repository.

## Four Over Six

The NVFP4 four-over-six controls and adaptive 4/6 policy are inspired by MIT
HAN Lab's Four Over Six work:

- Jack Cook, Junxian Guo, Guangxuan Xiao, Yujun Lin, and Song Han,
  "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling",
  2025, arXiv:2512.02010.
- Source project: <https://github.com/mit-han-lab/fouroversix>
- License notice: MIT License, Copyright (c) 2025 Jack Cook.

This repository does not vendor Four Over Six source code unless a future file
explicitly says so. The attribution above is for the published method and
open-source project that informed the local NVFP4 four-over-six controls.

Do not remove the Four Over Six attribution or license notice from this
repository when modifying the NVFP4 4/6 controls, presets, CUDA encoder paths,
candidate generation, or documentation.

## MXFP6 Compatibility Notice

MXFP6_E2M3 support in this repository is experimental and is not supported by
NVIDIA or llama.cpp. If official MXFP6 support appears later, the official
format may change and GGUF models created by this alpha tool may not work with
future runtimes.

Feedback is requested. The current MXFP6-capable branch can be installed from:
<https://github.com/michaelw9999/llama.cpp/tree/mxfp6-cuda>
