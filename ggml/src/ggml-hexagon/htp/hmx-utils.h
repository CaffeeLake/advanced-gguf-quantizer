// HMX tile-level inline helpers (FP16 32x32 tile operations).
// Ported from htp-ops-lib/include/dsp/hmx_utils.h. (https://github.com/haozixu/htp-ops-lib)

#ifndef HMX_UTILS_H
#define HMX_UTILS_H

#include "hvx-base.h"

#include <assert.h>
#include <hexagon_types.h>
#include <stddef.h>

#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE   2048

#define HMX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

// Initialise aligned 256-byte area with scale vector + zero padding.
static inline void hmx_init_column_scales(void *out_scales, HVX_Vector v_scale) {
    volatile HVX_Vector *pv = (HVX_Vector *) out_scales;
    pv[0] = v_scale;
    pv[1] = Q6_V_vzero();
}

// --- VTCM sequential allocator (from htp-ops-lib/include/dsp/vtcm_mgr.h) ---

        // tb0 starts at tile (c0=0, r0); tb1 at the adjacent dim-tile (c0=1, r0).
        // Each c step (+= 64) advances both by 2 dim-tiles worth of fp16.
        __fp16 *     tb0     = tiles_out + (size_t) r0 * HMX_FP16_TILE_N_ELMS;
        __fp16 *     tb1     = tb0 + tile_stride_elms;
        const size_t tb_step = 2 * tile_stride_elms;

        if (pv_in1) {
            for (int c = 0; c < head_dim; c += 64) {
                HVX_Vector     v0             = *pv_in0++;
                HVX_Vector     v1             = *pv_in1++;
                HVX_VectorPair vp             = Q6_W_vshuff_VVR(v1, v0, -2);
                ((HVX_Vector *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_Vector *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_step;
                tb1 += tb_step;
            }
        } else {
            const HVX_Vector vzero = Q6_V_vzero();
            for (int c = 0; c < head_dim; c += 64) {
                HVX_Vector     v0             = *pv_in0++;
                HVX_VectorPair vp             = Q6_W_vshuff_VVR(vzero, v0, -2);
                ((HVX_Vector *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_Vector *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_step;
                tb1 += tb_step;
            }
        }
    }
}

#endif // HMX_UTILS_H
