#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ggml.h"
#include "ggml-common.h"
#include "mxfp6_e2m3.cuh"
#ifndef GGML_HD
#define GGML_HD __host__ __device__
#endif // GGML_HD

struct  __align__(16) block_nvfp4_blackwell_frag {
    uint32_t regs[32][4];
    uint32_t scales_u32[32];
};

struct  __align__(16) block_nvfp4_blackwell {
    block_nvfp4_blackwell_frag tiles[4];
};

struct  __align__(16) tile_mxfp6_e2m3_blackwell_frag {
    uint32_t regs[32][3];
    uint8_t  scales[32];
};

struct  __align__(16) tile_mxfp6_e2m3_blackwell {
    tile_mxfp6_e2m3_blackwell_frag tiles[QK_MXFP6_E2M3_FRAGS];
};

struct  __align__(16) tile_mxfp6_e2m3_blackwell_k64_pair {
    tile_mxfp6_e2m3_blackwell sub[2];
};

struct  __align__(16) tile_mxfp6_e2m3_blackwell_k96_triple {
    tile_mxfp6_e2m3_blackwell sub[3];
};

struct  __align__(16) tile_mxfp6_e2m3_blackwell_k128_quad {
    tile_mxfp6_e2m3_blackwell sub[4];
};

struct  __align__(16) tile_mxfp6_e2m3_blackwell_tensor {
    float         weight_scale;
    float         input_scale;
    const float * weight_scales; // For MOE per expert
    const float * input_scales;
    tile_mxfp6_e2m3_blackwell tiles[];
};

struct  __align__(16) block_nvfp4_blackwell_tensor {
    float         weight_scale;
    float         input_scale;
    const float * weight_scales; // For MOE per expert
    const float * input_scales;
    block_nvfp4_blackwell tiles[];
};

static_assert(sizeof(block_nvfp4_blackwell_frag) == 640, "unexpected nvfp4 blackwell fragment size");
static_assert(sizeof(block_nvfp4_blackwell) == 4 * sizeof(block_nvfp4_blackwell_frag), "unexpected nvfp4 blackwell size");
static_assert(sizeof(block_nvfp4_blackwell_tensor) == 32, "unexpected nvfp4 blackwell tensor header size");
static_assert(alignof(block_nvfp4_blackwell_frag) == 16, "nvfp4 blackwell fragment must be 16B aligned");
static_assert(alignof(block_nvfp4_blackwell) == 16, "nvfp4 blackwell must be 16B aligned");
static_assert(alignof(block_nvfp4_blackwell_tensor) == 16, "nvfp4 blackwell tensor must be 16B aligned");
static_assert(sizeof(tile_mxfp6_e2m3_blackwell_frag) == 416, "unexpected mxfp6_e2m3 blackwell fragment size");
static_assert(sizeof(tile_mxfp6_e2m3_blackwell) == QK_MXFP6_E2M3_FRAGS * sizeof(tile_mxfp6_e2m3_blackwell_frag), "unexpected mxfp6_e2m3 blackwell size");
static_assert(sizeof(tile_mxfp6_e2m3_blackwell_k64_pair) == 2 * sizeof(tile_mxfp6_e2m3_blackwell), "unexpected mxfp6_e2m3 K64 pair size");
static_assert(sizeof(tile_mxfp6_e2m3_blackwell_k96_triple) == 3 * sizeof(tile_mxfp6_e2m3_blackwell), "unexpected mxfp6_e2m3 K96 triple size");
static_assert(sizeof(tile_mxfp6_e2m3_blackwell_k128_quad) == 4 * sizeof(tile_mxfp6_e2m3_blackwell), "unexpected mxfp6_e2m3 K128 quad size");
static_assert(sizeof(tile_mxfp6_e2m3_blackwell_tensor) == 32, "unexpected mxfp6_e2m3 blackwell tensor header size");
static_assert(alignof(tile_mxfp6_e2m3_blackwell_frag) == 16, "mxfp6_e2m3 blackwell fragment must be 16B aligned");
static_assert(alignof(tile_mxfp6_e2m3_blackwell) == 16, "mxfp6_e2m3 blackwell must be 16B aligned");
static_assert(alignof(tile_mxfp6_e2m3_blackwell_k64_pair) == 16, "mxfp6_e2m3 K64 pair must be 16B aligned");
static_assert(alignof(tile_mxfp6_e2m3_blackwell_k96_triple) == 16, "mxfp6_e2m3 K96 triple must be 16B aligned");
static_assert(alignof(tile_mxfp6_e2m3_blackwell_k128_quad) == 16, "mxfp6_e2m3 K128 quad must be 16B aligned");
static_assert(alignof(tile_mxfp6_e2m3_blackwell_tensor) == 16, "mxfp6_e2m3 blackwell tensor must be 16B aligned");
static inline GGML_HD int64_t ggml_cuda_bw_div_up(int64_t n, int64_t d) {
        return (n + d - 1) / d;
}

static_assert(sizeof(compact_mxfp6_k32) == 25, "unexpected mxfp6_e2m3 K32 compact block size");

static inline GGML_HD uint32_t ggml_cuda_mxfp6_e2m3_tile_q_word(
        const compact_mxfp6_k32 & block, int frag_idx, int pack_idx) {
    return ggml_cuda_mxfp6_e2m3_get4_u8containers(block, frag_idx, pack_idx);
}

static inline GGML_HD uint8_t ggml_cuda_mxfp6_e2m3_tile_scale(
        const compact_mxfp6_k32 & block, int frag_idx) {
    return block.e[frag_idx];
}

static inline GGML_HD int64_t ggml_cuda_mxfp6_e2m3_blocks_per_row(int64_t ne0) {
    return ggml_cuda_bw_div_up(ne0, QK_MXFP6_E2M3);
}

static inline GGML_HD uint32_t ggml_cuda_mxfp6_e2m3_bw_tile_q_word(
    const tile_mxfp6_e2m3_blackwell & tile, int row_in_tile, int frag_idx, int pack_idx) {
        const int lane = ((row_in_tile & 7) * 4) + (pack_idx & 3);
        const uint32_t w0 = tile.tiles[frag_idx].regs[lane][0];
        const uint32_t w1 = tile.tiles[frag_idx].regs[lane][1];
        const uint32_t w2 = tile.tiles[frag_idx].regs[lane][2];
        return (row_in_tile & 8) ?
            ((pack_idx & 4) ? ggml_cuda_mxfp6_e2m3_unpack4_u8containers(w2 >> 8) :
                              ggml_cuda_mxfp6_e2m3_unpack4_u8containers((w0 >> 24) | (w1 << 8))) :
            ((pack_idx & 4) ? ggml_cuda_mxfp6_e2m3_unpack4_u8containers((w1 >> 16) | (w2 << 16)) :
                              ggml_cuda_mxfp6_e2m3_unpack4_u8containers(w0));
}

static inline GGML_HD uint8_t ggml_cuda_mxfp6_e2m3_bw_tile_scale(
    const tile_mxfp6_e2m3_blackwell & tile, int row_in_tile, int frag_idx) {
        const int lane = ((row_in_tile & 7) * 4) + (row_in_tile >> 3);
        return tile.tiles[frag_idx].scales[lane];
}

static inline size_t ggml_cuda_mxfp6_e2m3_plane_size(int64_t ne0, int64_t nrows) {
    return (size_t) ggml_cuda_bw_div_up(nrows, 16) *
           (size_t) ggml_cuda_mxfp6_e2m3_blocks_per_row(ne0) * sizeof(tile_mxfp6_e2m3_blackwell);
}

static inline size_t ggml_cuda_mxfp6_e2m3_tensor_size(int64_t ne0, int64_t ne1, int64_t nplanes) {
    return sizeof(tile_mxfp6_e2m3_blackwell_tensor) + (size_t) nplanes * ggml_cuda_mxfp6_e2m3_plane_size(ne0, ne1);
}

static inline size_t ggml_cuda_mxfp6_e2m3_tensor_alloc_size(int64_t ne0, int64_t ne1, int64_t nplanes) {
    return ggml_cuda_mxfp6_e2m3_tensor_size(ne0, ne1, nplanes);
}

static inline size_t ggml_cuda_mxfp6_e2m3_tensor_alloc_size(const ggml_tensor * tensor) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t nplanes = tensor->ne[2] * tensor->ne[3];
    return ggml_cuda_mxfp6_e2m3_tensor_alloc_size(ne0, tensor->ne[1], nplanes);
}

static inline GGML_HD uint32_t ggml_cuda_nvfp4_tile_q_word(
    const block_nvfp4_blackwell & tile, int row_in_tile, int frag_idx, int pack_idx) {
        const int lane = ((row_in_tile & 7) * 4) + (pack_idx & 3);
        const int reg  = (row_in_tile >> 3) + ((pack_idx >> 2) << 1);
        return tile.tiles[frag_idx].regs[lane][reg];
}

static inline GGML_HD uint32_t ggml_cuda_nvfp4_tile_scale_word(
    const block_nvfp4_blackwell & tile, int row_in_tile, int frag_idx) {
        const int lane = ((row_in_tile & 7) * 4) + (row_in_tile >> 3);
        return tile.tiles[frag_idx].scales_u32[lane];
}

static inline uint32_t ggml_cuda_bw_pack8(const uint8_t * p, int shift) {
    return
        (((uint32_t)((p[0] >> shift) & 0x0F)) <<  0) |
        (((uint32_t)((p[1] >> shift) & 0x0F)) <<  4) |
        (((uint32_t)((p[2] >> shift) & 0x0F)) <<  8) |
        (((uint32_t)((p[3] >> shift) & 0x0F)) << 12) |
        (((uint32_t)((p[4] >> shift) & 0x0F)) << 16) |
        (((uint32_t)((p[5] >> shift) & 0x0F)) << 20) |
        (((uint32_t)((p[6] >> shift) & 0x0F)) << 24) |
        (((uint32_t)((p[7] >> shift) & 0x0F)) << 28);
}

static inline GGML_HD int64_t ggml_cuda_nvfp4_blocks_per_row(int64_t ne0) {
    return ggml_cuda_bw_div_up(ne0, QK_K);
}

static inline size_t ggml_cuda_nvfp4_plane_size(int64_t ne0, int64_t nrows) {
    return (size_t) ggml_cuda_bw_div_up(nrows, 16) *
           (size_t) ggml_cuda_nvfp4_blocks_per_row(ne0) * sizeof(block_nvfp4_blackwell);
}

static inline size_t ggml_cuda_nvfp4_rows_size(int64_t ne0, int64_t ne1, int64_t nplanes) {
    return (size_t) nplanes * (size_t) ne1 * ggml_row_size(GGML_TYPE_NVFP4, ne0);
}

static inline size_t ggml_cuda_nvfp4_tensor_size(int64_t ne0, int64_t ne1, int64_t nplanes) {
    return sizeof(block_nvfp4_blackwell_tensor) + (size_t) nplanes * ggml_cuda_nvfp4_plane_size(ne0, ne1);
}

static inline size_t ggml_cuda_nvfp4_tensor_alloc_size(const ggml_tensor * tensor) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t nplanes = tensor->ne[2] * tensor->ne[3];
    return ggml_cuda_nvfp4_tensor_size(ne0, tensor->ne[1], nplanes);
}

static inline bool ggml_cuda_get_scalar_f32(const ggml_tensor * tensor, float * value) {
    if (tensor == nullptr || !ggml_is_scalar(tensor) || tensor->type != GGML_TYPE_F32 || tensor->data == nullptr) {
        return false;
    }
    if (tensor->buffer == nullptr || ggml_backend_buffer_is_host(tensor->buffer)) {
        memcpy(value, tensor->data, sizeof(*value));
        return true;
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    return cudaMemcpy(value, tensor->data, sizeof(*value), cudaMemcpyDeviceToHost) == cudaSuccess;
#else
    return false;
#endif
}

static inline void ggml_cuda_nvfp4_set_tensor_header(const ggml_tensor * tensor, block_nvfp4_blackwell_tensor * dst) {
    float weight_scale = 1.0f;
    float input_scale  = 1.0f;
    const ggml_tensor * weight_scale_t = tensor->src[0];
    const ggml_tensor * input_scale_t  = tensor->src[1];

    if (!ggml_cuda_get_scalar_f32(weight_scale_t, &weight_scale) || weight_scale <= 0.0f) {
        weight_scale = 1.0f;
    }
    if (!ggml_cuda_get_scalar_f32(input_scale_t, &input_scale) || input_scale <= 0.0f) {
        input_scale = 1.0f;
    }

    dst->weight_scale = weight_scale;
    dst->input_scale  = input_scale;
    dst->weight_scales = weight_scale_t != nullptr && !ggml_is_scalar(weight_scale_t) &&
        weight_scale_t->data != nullptr && weight_scale_t->buffer != nullptr &&
        !ggml_backend_buffer_is_host(weight_scale_t->buffer) ? (const float *) weight_scale_t->data : nullptr;
    dst->input_scales = input_scale_t != nullptr && !ggml_is_scalar(input_scale_t) &&
        input_scale_t->data != nullptr && input_scale_t->buffer != nullptr &&
        !ggml_backend_buffer_is_host(input_scale_t->buffer) ? (const float *) input_scale_t->data : nullptr;
}

static inline void ggml_cuda_mxfp6_e2m3_patch_tensor_header(const ggml_tensor * tensor, tile_mxfp6_e2m3_blackwell_tensor * dst) {
    const ggml_tensor * weight_scale_t = tensor->src[0];
    const ggml_tensor * input_scale_t  = tensor->src[1];
    float weight_scale = dst->weight_scale;
    float input_scale  = dst->input_scale;

    if (ggml_cuda_get_scalar_f32(weight_scale_t, &weight_scale) && weight_scale > 0.0f) {
        dst->weight_scale = weight_scale;
    }
    if (ggml_cuda_get_scalar_f32(input_scale_t, &input_scale) && input_scale > 0.0f) {
        dst->input_scale = input_scale;
    }

    dst->weight_scales = weight_scale_t != nullptr && !ggml_is_scalar(weight_scale_t) &&
        weight_scale_t->data != nullptr && weight_scale_t->buffer != nullptr &&
        !ggml_backend_buffer_is_host(weight_scale_t->buffer) ? (const float *) weight_scale_t->data : nullptr;
    dst->input_scales = input_scale_t != nullptr && !ggml_is_scalar(input_scale_t) &&
        input_scale_t->data != nullptr && input_scale_t->buffer != nullptr &&
        !ggml_backend_buffer_is_host(input_scale_t->buffer) ? (const float *) input_scale_t->data : nullptr;
}

static inline void ggml_cuda_repack_tiles_nvfp4(int64_t ne0, int64_t nrows, const void * src, void * dst) {
    GGML_ASSERT(ne0 % QK_NVFP4 == 0);

    const int64_t src_blocks_per_row = ggml_cuda_bw_div_up(ne0, QK_NVFP4);
    const int64_t dst_blocks_per_row = ggml_cuda_nvfp4_blocks_per_row(ne0);
    const int64_t tile_rows = ggml_cuda_bw_div_up(nrows, 16);
    const size_t src_row_size = ggml_row_size(GGML_TYPE_NVFP4, ne0);

    const uint8_t * src_bytes = (const uint8_t *) src;
    block_nvfp4_blackwell * dst_blocks = (block_nvfp4_blackwell *) dst;

    for (int64_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
        const int64_t row0 = tile_row * 16;
        const int rows_in_tile = (int) ((row0 + 16 <= nrows) ? 16 : (nrows - row0));

        for (int64_t block_col = 0; block_col < dst_blocks_per_row; ++block_col) {
            const int64_t src_block0 = block_col * 4;
            const int frags_in_block = (int) ((src_block0 + 4 <= src_blocks_per_row) ? 4 : (src_blocks_per_row - src_block0));

            block_nvfp4_blackwell & out = dst_blocks[tile_row * dst_blocks_per_row + block_col];
            if (rows_in_tile != 16 || frags_in_block != 4) {
                memset(&out, 0, sizeof(out));
            }

            for (int row_in_tile = 0; row_in_tile < rows_in_tile; ++row_in_tile) {
                const int64_t row = row0 + row_in_tile;
                const block_nvfp4 * src_row = (const block_nvfp4 *) (src_bytes + row * src_row_size);
                const int lane_base = (row_in_tile & 7) * 4;
                const int row_half = row_in_tile >> 3;
                const int scale_lane = lane_base + row_half;

                for (int frag = 0; frag < frags_in_block; ++frag) {
                    const block_nvfp4 & in = src_row[src_block0 + frag];
                    block_nvfp4_blackwell_frag & tile = out.tiles[frag];

                    const uint8_t * p0 = in.qs +  0;
                    const uint8_t * p1 = in.qs +  8;
                    const uint8_t * p2 = in.qs + 16;
                    const uint8_t * p3 = in.qs + 24;
                    tile.regs[lane_base + 0][row_half + 0] = ggml_cuda_bw_pack8(p0, 0);
                    tile.regs[lane_base + 1][row_half + 0] = ggml_cuda_bw_pack8(p0, 4);
                    tile.regs[lane_base + 2][row_half + 0] = ggml_cuda_bw_pack8(p1, 0);
                    tile.regs[lane_base + 3][row_half + 0] = ggml_cuda_bw_pack8(p1, 4);
                    tile.regs[lane_base + 0][row_half + 2] = ggml_cuda_bw_pack8(p2, 0);
                    tile.regs[lane_base + 1][row_half + 2] = ggml_cuda_bw_pack8(p2, 4);
                    tile.regs[lane_base + 2][row_half + 2] = ggml_cuda_bw_pack8(p3, 0);
                    tile.regs[lane_base + 3][row_half + 2] = ggml_cuda_bw_pack8(p3, 4);

                    uint32_t d = 0;
                    memcpy(&d, in.d, sizeof(d));
                    tile.scales_u32[scale_lane + 0] = d;
                    tile.scales_u32[scale_lane + 2] = d;
                }
            }
        }
    }
}

static inline void ggml_cuda_make_mxfp6_tiles_from_compact_for_microbench(int64_t ne0, int64_t nrows, const void * src, void * dst) {
    GGML_ASSERT(ne0 % QK_MXFP6_E2M3 == 0);

    const int64_t blocks_per_row = ggml_cuda_mxfp6_e2m3_blocks_per_row(ne0);
    const int64_t tile_rows = ggml_cuda_bw_div_up(nrows, 16);
    const size_t src_row_size = ggml_row_size(GGML_TYPE_MXFP6_E2M3, ne0);

    const uint8_t * src_bytes = (const uint8_t *) src;
    tile_mxfp6_e2m3_blackwell * dst_blocks = (tile_mxfp6_e2m3_blackwell *) dst;

    for (int64_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
        const int64_t row0 = tile_row * 16;
        const int rows_in_tile = (int) ((row0 + 16 <= nrows) ? 16 : (nrows - row0));

        for (int64_t block_col = 0; block_col < blocks_per_row; ++block_col) {
            tile_mxfp6_e2m3_blackwell & out = dst_blocks[tile_row * blocks_per_row + block_col];
            memset(&out, 0, sizeof(out));
            for (int frag = 0; frag < QK_MXFP6_E2M3_FRAGS; ++frag) {
                memset(out.tiles[frag].scales, 0x7F, sizeof(out.tiles[frag].scales));
            }

            for (int row_in_tile = 0; row_in_tile < rows_in_tile; ++row_in_tile) {
                const compact_mxfp6_k32 * src_row = (const compact_mxfp6_k32 *) (src_bytes + (row0 + row_in_tile) * src_row_size);
                const compact_mxfp6_k32 & in = src_row[block_col];
                const int lane_base = (row_in_tile & 7) * 4;
                const int row_half = row_in_tile >> 3;
                const int scale_lane = lane_base + row_half;

                for (int frag = 0; frag < QK_MXFP6_E2M3_FRAGS; ++frag) {
                    out.tiles[frag].scales[scale_lane] = in.e[frag];
                }
            }

            for (int frag = 0; frag < QK_MXFP6_E2M3_FRAGS; ++frag) {
                tile_mxfp6_e2m3_blackwell_frag & tile = out.tiles[frag];
                for (int lane = 0; lane < 32; ++lane) {
                    const int row_lo = lane >> 2;
                    const int row_hi = row_lo + 8;
                    const int word = lane & 3;
                    uint8_t packed[12] = {};
                    if (row_lo < rows_in_tile) {
                        const compact_mxfp6_k32 * src_row = (const compact_mxfp6_k32 *) (src_bytes + (row0 + row_lo) * src_row_size);
                        memcpy(packed + 0, &src_row[block_col].qs[frag][3 * (word + 0)], 3);
                        memcpy(packed + 6, &src_row[block_col].qs[frag][3 * (word + 4)], 3);
                    }
                    if (row_hi < rows_in_tile) {
                        const compact_mxfp6_k32 * src_row = (const compact_mxfp6_k32 *) (src_bytes + (row0 + row_hi) * src_row_size);
                        memcpy(packed + 3, &src_row[block_col].qs[frag][3 * (word + 0)], 3);
                        memcpy(packed + 9, &src_row[block_col].qs[frag][3 * (word + 4)], 3);
                    }
                    tile.regs[lane][0] = uint32_t(packed[0]) | (uint32_t(packed[1]) << 8) | (uint32_t(packed[2]) << 16) | (uint32_t(packed[3]) << 24);
                    tile.regs[lane][1] = uint32_t(packed[4]) | (uint32_t(packed[5]) << 8) | (uint32_t(packed[6]) << 16) | (uint32_t(packed[7]) << 24);
                    tile.regs[lane][2] = uint32_t(packed[8]) | (uint32_t(packed[9]) << 8) | (uint32_t(packed[10]) << 16) | (uint32_t(packed[11]) << 24);
                }
            }
        }
    }
}

static inline void ggml_cuda_repack_tensor_nvfp4(const ggml_tensor * tensor, const void * src, void * dst) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t nplanes = tensor->ne[2] * tensor->ne[3];
    const size_t src_plane_size = ggml_row_size(GGML_TYPE_NVFP4, ne0) * ne1;
    const size_t dst_plane_size = ggml_cuda_nvfp4_plane_size(ne0, ne1);
    block_nvfp4_blackwell_tensor * dst_tensor = (block_nvfp4_blackwell_tensor *) dst;

    ggml_cuda_nvfp4_set_tensor_header(tensor, dst_tensor);
    char * dst_tiles = (char *) dst_tensor->tiles;

    for (int64_t plane = 0; plane < nplanes; ++plane) {
        ggml_cuda_repack_tiles_nvfp4(ne0, ne1,
                (const char *) src + plane * src_plane_size,
                dst_tiles + plane * dst_plane_size);
    }
}

static inline void ggml_cuda_set_tensor_mxfp6_tiled(const ggml_tensor * tensor, const void * src, void * dst) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t nplanes = tensor->ne[2] * tensor->ne[3];
    const size_t src_plane_size = ggml_row_size(GGML_TYPE_MXFP6_E2M3, ne0) * ne1;
    const size_t dst_plane_size = ggml_cuda_mxfp6_e2m3_plane_size(ne0, ne1);
    tile_mxfp6_e2m3_blackwell_tensor * dst_tensor = (tile_mxfp6_e2m3_blackwell_tensor *) dst;

    memcpy(dst_tensor, src, MXFP6_HEADER_OFFSET);
    ggml_cuda_mxfp6_e2m3_patch_tensor_header(tensor, dst_tensor);
    char * dst_tiles = (char *) dst_tensor->tiles;
    const char * src_tiles = (const char *) src + MXFP6_HEADER_OFFSET;

    for (int64_t plane = 0; plane < nplanes; ++plane) {
        memcpy(dst_tiles + plane * dst_plane_size, src_tiles + plane * src_plane_size, src_plane_size);
        if (dst_plane_size > src_plane_size) {
            memset(dst_tiles + plane * dst_plane_size + src_plane_size, 0, dst_plane_size - src_plane_size);
        }
    }
}

static inline bool ggml_cuda_set_mxfp6_tile_rows(const ggml_tensor * tensor, int64_t row_low, int64_t nrows, const void * src, void * dst_tiles) {
    if (row_low < 0 || nrows <= 0 || row_low % 16 != 0 || tensor->ne[2] * tensor->ne[3] != 1) {
        return false;
    }

    const int64_t ne0 = tensor->ne[0];
    const size_t row_size = ggml_row_size(GGML_TYPE_MXFP6_E2M3, ne0);
    const size_t copy_size = (size_t) ggml_cuda_bw_div_up(nrows, 16) *
        (size_t) ggml_cuda_mxfp6_e2m3_blocks_per_row(ne0) * sizeof(tile_mxfp6_e2m3_blackwell);
    memcpy(dst_tiles, (const char *) src + (size_t) row_low * row_size, copy_size);
    return true;
}

static inline bool ggml_cuda_get_mxfp6_tile_rows(const ggml_tensor * tensor, int64_t row_low, int64_t nrows, const void * src_tiles, void * dst) {
    if (row_low < 0 || nrows <= 0 || row_low % 16 != 0 || tensor->ne[2] * tensor->ne[3] != 1) {
        return false;
    }

    const int64_t ne0 = tensor->ne[0];
    const size_t row_size = ggml_row_size(GGML_TYPE_MXFP6_E2M3, ne0);
    const size_t copy_size = (size_t) ggml_cuda_bw_div_up(nrows, 16) *
        (size_t) ggml_cuda_mxfp6_e2m3_blocks_per_row(ne0) * sizeof(tile_mxfp6_e2m3_blackwell);
    memcpy((char *) dst + (size_t) row_low * row_size, src_tiles, copy_size);
    return true;
}

static inline void ggml_cuda_unpack_tiles_nvfp4(int64_t ne0, int64_t nrows, const void * src, void * dst) {
    GGML_ASSERT(ne0 % QK_NVFP4 == 0);

    const int64_t src_blocks_per_row = ggml_cuda_nvfp4_blocks_per_row(ne0);
    const int64_t dst_blocks_per_row = ggml_cuda_bw_div_up(ne0, QK_NVFP4);
    const size_t dst_row_size = ggml_row_size(GGML_TYPE_NVFP4, ne0);

    const block_nvfp4_blackwell * src_blocks = (const block_nvfp4_blackwell *) src;

    for (int64_t row = 0; row < nrows; ++row) {
        block_nvfp4 * dst_row = (block_nvfp4 *) ((uint8_t *) dst + row * dst_row_size);
        const int64_t tile_row = row / 16;
        const int row_in_tile = (int) (row % 16);
        const int lane_base = (row_in_tile & 7) * 4;
        const int row_half = row_in_tile >> 3;
        const int scale_lane = lane_base + row_half;

        for (int64_t block_col = 0; block_col < src_blocks_per_row; ++block_col) {
            const int64_t dst_block0 = block_col * 4;
            const int frags_in_block = (int) ((dst_block0 + 4 <= dst_blocks_per_row) ? 4 : (dst_blocks_per_row - dst_block0));
            const block_nvfp4_blackwell & in = src_blocks[tile_row * src_blocks_per_row + block_col];

            for (int frag = 0; frag < frags_in_block; ++frag) {
                const block_nvfp4_blackwell_frag & tile = in.tiles[frag];
                block_nvfp4 & out = dst_row[dst_block0 + frag];

                for (int g = 0; g < 4; ++g) {
                    const uint32_t lo = tile.regs[lane_base + 2*(g & 1) + 0][row_half + 2*(g >> 1)];
                    const uint32_t hi = tile.regs[lane_base + 2*(g & 1) + 1][row_half + 2*(g >> 1)];
                    uint8_t * p = out.qs + 8*g;
                    for (int i = 0; i < 8; ++i) {
                        p[i] = ((lo >> (4*i)) & 0x0F) | (((hi >> (4*i)) & 0x0F) << 4);
                    }
                }

                const uint32_t d = tile.scales_u32[scale_lane];
                memcpy(out.d, &d, sizeof(d));
            }
        }
    }
}

static inline void ggml_cuda_unpack_tensor_nvfp4(const ggml_tensor * tensor, const void * src, void * dst) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t nplanes = tensor->ne[2] * tensor->ne[3];
    const size_t src_plane_size = ggml_cuda_nvfp4_plane_size(ne0, ne1);
    const size_t dst_plane_size = ggml_row_size(GGML_TYPE_NVFP4, ne0) * ne1;
    const char * src_tiles = (const char *) ((const block_nvfp4_blackwell_tensor *) src)->tiles;

    for (int64_t plane = 0; plane < nplanes; ++plane) {
        ggml_cuda_unpack_tiles_nvfp4(ne0, ne1,
                src_tiles + plane * src_plane_size,
                (char *) dst + plane * dst_plane_size);
    }
}

static inline void ggml_cuda_get_tensor_mxfp6_tiled(const ggml_tensor * tensor, const void * src, void * dst) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t nplanes = tensor->ne[2] * tensor->ne[3];
    const size_t src_plane_size = ggml_cuda_mxfp6_e2m3_plane_size(ne0, ne1);
    const size_t dst_plane_size = ggml_row_size(GGML_TYPE_MXFP6_E2M3, ne0) * ne1;
    const char * src_tiles = (const char *) ((const tile_mxfp6_e2m3_blackwell_tensor *) src)->tiles;

    for (int64_t plane = 0; plane < nplanes; ++plane) {
        memcpy((char *) dst + plane * dst_plane_size, src_tiles + plane * src_plane_size, dst_plane_size);
    }
}
