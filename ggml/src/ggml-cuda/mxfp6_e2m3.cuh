#pragma once

#include <cstdint>

#include "ggml-common.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(FP6_AVAILABLE) && \
        defined(__CUDACC__) && defined(__has_include)
#if __has_include(<cuda_fp6.h>)
#include <cuda_fp6.h>
#define FP6_AVAILABLE
#endif // __has_include(<cuda_fp6.h>)
#endif

#ifndef GGML_HD
#define GGML_HD __host__ __device__
#endif // GGML_HD

#define QK_MXFP6_E2M3 QK_MXFP6_SUB
#define QK_MXFP6_E2M3_SUB QK_MXFP6_SUB
#define QK_MXFP6_E2M3_FRAGS 1
#define QK_MXFP6_E2M3_PACKED_BYTES QK_MXFP6_PACKED_BYTES
#define QK_MXFP6_E2M3_PTX_BYTES QK_MXFP6_E2M3_SUB
#define QR_MXFP6_E2M3 1
#define QI_MXFP6_E2M3 (QK_MXFP6_E2M3 / (4 * QR_MXFP6_E2M3))

struct compact_mxfp6_k32 {
    uint8_t e[QK_MXFP6_E2M3_FRAGS];
    uint8_t qs[QK_MXFP6_E2M3_FRAGS][QK_MXFP6_PACKED_BYTES];
};
static_assert(sizeof(compact_mxfp6_k32) == 25, "unexpected mxfp6_e2m3 K32 compact block size");

#if defined(FP6_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_MXFP6_E2M3_NATIVE_FP6 1
using ggml_cuda_fp6_e2m3_storage_t  = __nv_fp6_storage_t;
using ggml_cuda_fp6x2_e2m3_storage_t = __nv_fp6x2_storage_t;
using ggml_cuda_fp6x4_e2m3_storage_t = __nv_fp6x4_storage_t;

static inline GGML_HD uint8_t ggml_cuda_mxfp6_e2m3_cvt_f32(float x) {
    const ggml_cuda_fp6_e2m3_storage_t raw = __nv_cvt_float_to_fp6(x, __NV_E2M3, cudaRoundNearest);
    return (uint8_t) (raw & 0x3Fu);
}

static inline GGML_HD uint32_t ggml_cuda_mxfp6_e2m3_cvt4_f32(float x0, float x1, float x2, float x3) {
    const float2 lo = { x0, x1 };
    const float2 hi = { x2, x3 };
    const ggml_cuda_fp6x2_e2m3_storage_t lo_raw = __nv_cvt_float2_to_fp6x2(lo, __NV_E2M3, cudaRoundNearest);
    const ggml_cuda_fp6x2_e2m3_storage_t hi_raw = __nv_cvt_float2_to_fp6x2(hi, __NV_E2M3, cudaRoundNearest);
    return (uint32_t(lo_raw) & 0x0000FFFFu) | ((uint32_t(hi_raw) & 0x0000FFFFu) << 16);
}

static inline GGML_HD float ggml_cuda_mxfp6_e2m3_to_f32(uint8_t code) {
    __nv_fp6_e2m3 x;
    x.__x = (ggml_cuda_fp6_e2m3_storage_t) (code & 0x3Fu);
    return (float) x;
}
#endif // defined(FP6_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static inline GGML_HD uint8_t ggml_cuda_mxfp6_e2m3_get_code(const compact_mxfp6_k32 & block, int frag, int i) {
    const int bit   = i * 6;
    const int byte  = bit >> 3;
    const int shift = bit & 7;
    uint32_t v = uint32_t(block.qs[frag][byte]) >> shift;
    if (byte + 1 < QK_MXFP6_PACKED_BYTES) {
        v |= uint32_t(block.qs[frag][byte + 1]) << (8 - shift);
    }
    return (uint8_t) (v & 0x3F);
}

static inline GGML_HD void ggml_cuda_mxfp6_e2m3_set_code(compact_mxfp6_k32 & block, int frag, int i, uint8_t code) {
    const int bit   = i * 6;
    const int byte  = bit >> 3;
    const int shift = bit & 7;
    uint32_t v = uint32_t(block.qs[frag][byte]);
    if (byte + 1 < QK_MXFP6_PACKED_BYTES) {
        v |= uint32_t(block.qs[frag][byte + 1]) << 8;
    }
    const uint32_t mask = uint32_t(0x3F) << shift;
    v = (v & ~mask) | ((uint32_t(code) & 0x3F) << shift);
    block.qs[frag][byte] = (uint8_t) (v & 0xFF);
    if (byte + 1 < QK_MXFP6_PACKED_BYTES) {
        block.qs[frag][byte + 1] = (uint8_t) ((v >> 8) & 0xFF);
    }
}

static inline GGML_HD uint32_t ggml_cuda_mxfp6_e2m3_pack4_u8containers(uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3) {
    return ((uint32_t) (x0 & 0x3f) <<  0) |
           ((uint32_t) (x1 & 0x3f) <<  8) |
           ((uint32_t) (x2 & 0x3f) << 16) |
           ((uint32_t) (x3 & 0x3f) << 24);
}

static inline GGML_HD uint32_t ggml_cuda_mxfp6_e2m3_unpack4_u8containers(uint32_t packed) {
    packed &= 0x00FFFFFFu;
    return ((packed & 0x00003Fu) << 0) |
           ((packed & 0x000FC0u) << 2) |
           ((packed & 0x03F000u) << 4) |
           ((packed & 0xFC0000u) << 6);
}

static inline GGML_HD uint32_t ggml_cuda_mxfp6_e2m3_get4_u8containers(const compact_mxfp6_k32 & block, int frag, int pack_idx) {
    const uint32_t packed =
        ((uint32_t) block.qs[frag][3 * pack_idx + 0] <<  0) |
        ((uint32_t) block.qs[frag][3 * pack_idx + 1] <<  8) |
        ((uint32_t) block.qs[frag][3 * pack_idx + 2] << 16);

    return ggml_cuda_mxfp6_e2m3_unpack4_u8containers(packed);
}

static inline GGML_HD void ggml_cuda_mxfp6_e2m3_set4_u8containers(compact_mxfp6_k32 & block, int frag, int pack_idx, uint32_t packed4) {
    for (int i = 0; i < 4; ++i) {
        ggml_cuda_mxfp6_e2m3_set_code(block, frag, 4 * pack_idx + i, (uint8_t) ((packed4 >> (8 * i)) & 0x3F));
    }
}
