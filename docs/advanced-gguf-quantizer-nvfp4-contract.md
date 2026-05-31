# NVFP4 GGUF Contract

This document describes the GGUF contract that `advanced-gguf-quantizer` writes
for deployable NVFP4 models.

## Tensor Set

For a quantized weight tensor named `foo.weight`, NVFP4 output may include:

- `foo.weight`: packed NVFP4 weight data;
- `foo.scale`: tensor or expert weight scale data;
- `foo.input_scale`: activation/input scale data.

The `.scale` and `.input_scale` tensors are part of the model contract. A GGUF
that stores only an NVFP4 weight tensor without the required auxiliary scales is
not equivalent.

## Scale Meaning

Weight scale:

- restores the NVFP4 weight tensor to the intended range;
- participates in both dense and MoE matmul paths;
- must follow expert slices when the weight tensor is expert-indexed.

Input scale:

- records activation/input scaling used by activation-aware NVFP4 paths;
- is calculated from imatrix and recipe policy;
- must follow expert slices when activations route through expert tensors.

Do not treat weight scale and input scale as the same knob. They answer
different questions and can fail independently.

## Runtime Attachment

The loader attaches auxiliary tensors as derived data for matmul operations.
That is important for `mul_mat_id` and MoE routing because scale tensors must be
indexed the same way the quantized weight tensor is indexed.

The runtime path should see:

- the packed weight tensor;
- the weight scale tensor;
- the input scale tensor when the selected kernel needs it;
- tensor/expert indexing metadata that matches the original GGUF layout.

## Logical Block Layout

The logical NVFP4 block stores 64 values:

- four 16-value subblocks;
- one UE4M3 scale byte per subblock;
- packed FP4 E2M1 nibbles for the values.

The CUDA encoder may repack this logical format into a Blackwell-friendly
resident layout for runtime use, but the GGUF writer contract remains explicit:
weights plus auxiliary scale tensors.

## Four-Over-Six Policy

Four-over-six controls whether a subblock uses a wider M6-style envelope or a
tighter M4-style envelope. The policy can be adaptive or forced by recipe.
Because this changes quantization error and scale behavior, it must be recorded
in the recipe/report and evaluated with real PPL/KLD evidence for quality
claims.

The four-over-six controls are inspired by MIT HAN Lab's Four Over Six work by
Jack Cook, Junxian Guo, Guangxuan Xiao, Yujun Lin, and Song Han. Keep the
attribution and MIT license notice in [NOTICE](../NOTICE.md) when modifying this
policy or its implementation.

## Imatrix Input Scales

Input scales should be calculated from calibration evidence. Supported policy
families include max/RMS/tail-style statistics and activation-aware weighting.
Do not collapse input scales to identity when the recipe requests activation
aware quantization.

## Embedding And Output Policy

Pure NVFP4 defaults:

- token embeddings remain `NVFP4`;
- tied token embeddings follow the token embedding rule;
- a separate `output.weight` may use a stronger type such as `Q6_K`.

Mixed and repair recipes may override this, but the report should make the
decision visible.

## MTP And NextN

MTP and NextN metadata and tensors must be preserved unless a recipe explicitly
states a different policy. The inspect report should show whether MTP/NextN was
present in the source and present in the output. Use `--mtp-tensor-type` or
`base.mtp_tensor_type` for deliberate MTP conversion; release MTP blocks should
use source precision, `MXFP6_E2M3`, `Q8_0`, `BF16`, or `F16`, not `NVFP4`.

## Validation

Use:

```bash
./build/bin/llama-quantize inspect runs/model/output.gguf --tensors
./build/bin/llama-completion -m runs/model/output.gguf -p "The model is"
```

For quality claims, add matched PPL/KLD evidence from the same source,
calibration corpus, imatrix, evaluation corpus, KLD base, and commit.
