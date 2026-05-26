#include "common.cuh"
#include "convert.cuh"
#include "ggml-cuda.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cub/cub.cuh>
#endif

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#ifndef QK_MXFP6_E2M3_TILE_FRAGS
#define QK_MXFP6_E2M3_TILE_FRAGS MXFP6_TILE_FRAGS
#endif
#ifndef QK_MXFP6_E2M3_TILE
#define QK_MXFP6_E2M3_TILE QK_MXFP6
#endif
#ifndef MXFP6_E2M3_TILE_ROWS
#define MXFP6_E2M3_TILE_ROWS MXFP6_TILE_ROWS
#endif
#ifndef MXFP6_E2M3_TILE_LANES
#define MXFP6_E2M3_TILE_LANES MXFP6_TILE_LANES
#endif

using block_mxfp6_e2m3 = compact_mxfp6_k32;
using tile_mxfp6_e2m3 = tile_mxfp6;
using tile_mxfp6_e2m3_frag = tile_mxfp6_frag;

extern "C" void ggml_cuda_nvfp4_register_autotune();
extern "C" void ggml_cuda_nvfp4_autotune(
        const float * x,
        const float * qw,
        int64_t n,
        float * best_a,
        float * best_b,
        cudaStream_t stream);
extern "C" bool ggml_cuda_nvfp4_quantize(
        const void * x,
        bool x_bf16,
        float x_scale,
        void * vy,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float a,
        float b,
        cudaStream_t stream);
extern "C" bool ggml_cuda_nvfp4_quantize_cfg(
        const void * x,
        bool x_bf16,
        float x_scale,
        void * vy,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float a,
        float b,
        const nvfp4_cuda_runtime_cfg * cfg,
        cudaStream_t stream);
extern "C" bool ggml_cuda_nvfp4_quantize_eval_cfg(
        const void * x,
        bool x_bf16,
        float x_scale,
        void * vy,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float a,
        float b,
        const nvfp4_cuda_runtime_cfg * cfg,
        nvfp4_cuda_eval_result * eval,
        cudaStream_t stream);
extern "C" bool ggml_cuda_nvfp4_quantize_eval_to_tensor_cfg(
        const void * x,
        bool x_bf16,
        float x_scale,
        ggml_tensor * tensor,
        int64_t plane_index,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float a,
        float b,
        const nvfp4_cuda_runtime_cfg * cfg,
        nvfp4_cuda_eval_result * eval,
        cudaStream_t stream);
extern "C" bool ggml_cuda_nvfp4_autotune_ex(
        const float * x,
        const float * qw,
        int64_t n,
        const nvfp4_cuda_runtime_cfg * cfg_hint,
        nvfp4_cuda_tune_result * result,
        cudaStream_t stream);
extern "C" bool ggml_cuda_nvfp4_sample_cache_create(
        const void * x,
        int32_t x_type,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        int64_t sample_nb,
        int64_t sample_phase,
        float tune_x_mul,
        void ** cache,
        const float ** x_device,
        const float ** tune_x_device,
        const float ** qw_device,
        int64_t * n_device,
        cudaStream_t stream);
extern "C" void ggml_cuda_nvfp4_sample_cache_free(void * cache);
extern "C" void ggml_cuda_nvfp4_set_autotune_threads(int32_t n_threads);
extern "C" void ggml_cuda_nvfp4_clear_thread_cache();
extern "C" bool ggml_cuda_mxfp6_e2m3_quantize_eval(
        const void * x,
        bool x_bf16,
        float x_scale,
        void * vy,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        nvfp4_cuda_eval_result * eval,
        cudaStream_t stream);
extern "C" bool ggml_cuda_mxfp6_e2m3_quantize_eval_to_tensor(
        const void * x,
        bool x_bf16,
        float x_scale,
        ggml_tensor * tensor,
        int64_t plane_index,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float header_weight_scale,
        float header_input_scale,
        nvfp4_cuda_eval_result * eval,
        cudaStream_t stream);
extern "C" bool ggml_cuda_tensor_set_host_impl(
        ggml_tensor * tensor,
        const void * src,
        size_t nbytes,
        cudaStream_t stream);
extern "C" bool ggml_cuda_nvfp4_tensor_set_header_scales_impl(
        ggml_tensor * tensor,
        float weight_scale,
        float input_scale,
        cudaStream_t stream);
extern "C" bool ggml_cuda_tensor_get_host_impl(
        const ggml_tensor * tensor,
        void * dst,
        size_t nbytes,
        cudaStream_t stream);
extern "C" bool ggml_cuda_tensor_snapshot_impl(
        const ggml_tensor * tensor,
        size_t nbytes,
        void ** snapshot,
        cudaStream_t stream);
extern "C" bool ggml_cuda_tensor_restore_impl(
        ggml_tensor * tensor,
        const void * snapshot,
        size_t nbytes,
        cudaStream_t stream);
extern "C" void ggml_cuda_tensor_snapshot_free_impl(void * snapshot);
extern "C" bool ggml_cuda_nvfp4_kld_reduce_tensor(
        const ggml_tensor * logits,
        const uint16_t * base_logp_u16,
        const int32_t * token_ids,
        int32_t logits_row_offset,
        int32_t n_eval,
        int32_t n_vocab,
        int32_t nv,
        nvfp4_cuda_kld_result * result,
        double * kld_values,
        cudaStream_t stream);

static constexpr int64_t NVFP4_AUTOTUNE_MAX_SAMPLE_BLOCKS = 8192;
static constexpr double NVFP4_AUTOTUNE_MIN_IMPROVE_FRAC = 0.00075;
static constexpr float NVFP4_TUNE_AB_STEP_SCALE = 1.0f;
static constexpr size_t NVFP4_CUDA_KLD_BASE_CACHE_LIMIT_BYTES = (size_t) 2 * 1024 * 1024 * 1024;
static constexpr int NVFP4_REFIT_ITERS = 8;
static constexpr int NVFP4_TUNE_REFIT_ITERS = 8;
static constexpr int NVFP4_TUNE_POOL_SIZE = 48;
static constexpr int NVFP4_COMPAND_TOPK = 6;
static constexpr float NVFP4_E2M1_MAX_VALUE = 6.0f;
static constexpr float NVFP4_TUNE_FIXED_POOL[] = {
    0.9918823242f,
    0.9864501953f,
    0.875f,
    0.9375f,
    0.96875f,
    1.0f,
    1.015625f,
    1.03125f,
    1.046875f,
    1.0625f,
    1.09375f,
    1.125f,
    1.1875f,
    1.25f,
};

static inline void nvfp4_host_apply_ab_caps(
        nvfp4_cuda_runtime_cfg & resolved,
        float a,
        float b);
static __global__ void ggml_cuda_nvfp4_quantize_blocks_16x4(
        const void * x,
        bool x_bf16,
        float x_scale,
        const float * qw,
        block_nvfp4 * y,
        int64_t row_blocks,
        int choose46_mode,
        int refit_iters,
        int use_compand_sat,
        float cap_m6,
        float cap_m4);
static __global__ void ggml_cuda_mxfp6_e2m3_quantize_blocks_32(
        const void * x,
        bool x_bf16,
        float x_scale,
        const float * qw,
        block_mxfp6_e2m3 * y,
        int64_t row_blocks);
static __global__ void ggml_cuda_mxfp6_e2m3_pack_tiles_832(
        const block_mxfp6_e2m3 * x,
        tile_mxfp6_e2m3 * y,
        int64_t nrow,
        int64_t row_blocks);

static bool nvfp4_cuda_trace_enabled() {
    static const bool enabled = []() {
        const char * trace = std::getenv("LLAMA_NVFP4_TRACE");
        if (!trace || trace[0] == '\0') {
            trace = std::getenv("LLAMA_TRACE");
        }
        return trace && trace[0] != '\0' && std::strcmp(trace, "0") != 0;
    }();
    return enabled;
}

static void nvfp4_cuda_log_failure(const char * stage, cudaError_t err) {
    if (nvfp4_cuda_trace_enabled() || err != cudaSuccess) {
        size_t free_mem = 0;
        size_t total_mem = 0;
        const cudaError_t mem_err = cudaMemGetInfo(&free_mem, &total_mem);
        if (mem_err == cudaSuccess) {
            GGML_LOG_WARN(
                "%s: stage=%s failed: %s (%d), cuda_mem_free=%.2f GiB cuda_mem_total=%.2f GiB\n",
                __func__,
                stage,
                cudaGetErrorString(err),
                (int) err,
                (double) free_mem / (1024.0 * 1024.0 * 1024.0),
                (double) total_mem / (1024.0 * 1024.0 * 1024.0));
        } else {
            GGML_LOG_WARN(
                "%s: stage=%s failed: %s (%d), cudaMemGetInfo failed: %s (%d)\n",
                __func__,
                stage,
                cudaGetErrorString(err),
                (int) err,
                cudaGetErrorString(mem_err),
                (int) mem_err);
        }
    }
}

static inline void nvfp4_cuda_resolve_cfg(
        nvfp4_cuda_runtime_cfg & resolved,
        float a,
        float b,
        const nvfp4_cuda_runtime_cfg * cfg) {
    resolved = {};
    resolved.choose46_mode = NVFP4_CUDA_CHOOSE46_ADAPTIVE;
    resolved.refit_iters = NVFP4_REFIT_ITERS;
    resolved.use_compand_sat = 1;
    resolved.reserved_i32 = 0;
    nvfp4_host_apply_ab_caps(resolved, a, b);
    if (cfg != nullptr) {
        resolved = *cfg;
        if (resolved.choose46_mode < NVFP4_CUDA_CHOOSE46_ADAPTIVE || resolved.choose46_mode > NVFP4_CUDA_CHOOSE46_FORCE_M4) {
            resolved.choose46_mode = NVFP4_CUDA_CHOOSE46_ADAPTIVE;
        }
        if (resolved.refit_iters < 0) {
            resolved.refit_iters = NVFP4_REFIT_ITERS;
        }
        if (resolved.refit_iters > 64) {
            resolved.refit_iters = 64;
        }
        if (resolved.use_compand_sat != 0 && resolved.use_compand_sat != 1) {
            resolved.use_compand_sat = 1;
        }
        if (!(isfinite(resolved.cap_m6) && resolved.cap_m6 > 0.0f)) {
            nvfp4_host_apply_ab_caps(resolved, a, b);
        }
        if (!(isfinite(resolved.cap_m4) && resolved.cap_m4 > 0.0f)) {
            nvfp4_host_apply_ab_caps(resolved, a, b);
        }
        if (resolved.cap_m4 > resolved.cap_m6) {
            resolved.cap_m4 = resolved.cap_m6;
        }
    }
}

static inline bool ggml_cuda_nvfp4_launch_kernel(
        const void * d_x,
        bool x_bf16,
        float x_scale,
        const float * d_qw,
        block_nvfp4 * d_y,
        int64_t row_blocks,
        int64_t nb_total,
        const nvfp4_cuda_runtime_cfg & resolved,
        cudaStream_t st) {
    ggml_cuda_nvfp4_quantize_blocks_16x4<<<(unsigned) nb_total, QK_NVFP4, 0, st>>>(
        d_x, x_bf16, x_scale, d_qw, d_y, row_blocks,
        resolved.choose46_mode, resolved.refit_iters, resolved.use_compand_sat, resolved.cap_m6, resolved.cap_m4);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("kernel", err);
        cudaGetLastError();
        return false;
    }
    return true;
}

static inline bool ggml_cuda_nvfp4_ensure_buf(void ** buf, size_t * cap, size_t need, const char * stage) {
    if (need == 0) {
        return true;
    }
    if (*cap >= need && *buf != nullptr) {
        return true;
    }
    if (*buf != nullptr) {
        cudaFree(*buf);
        *buf = nullptr;
        *cap = 0;
    }
    const cudaError_t err = cudaMalloc(buf, need);
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure(stage, err);
        cudaGetLastError();
        return false;
    }
    *cap = need;
    return true;
}

static inline void ggml_cuda_nvfp4_release_buf(void ** buf, size_t * cap) {
    if (*buf != nullptr) {
        const cudaError_t err = cudaFree(*buf);
        if (err != cudaSuccess) {
            nvfp4_cuda_log_failure("cudaFree(cache)", err);
            cudaGetLastError();
        }
        *buf = nullptr;
    }
    *cap = 0;
}

struct nvfp4_cuda_eval_accum;
struct nvfp4_cuda_tune_eval_accum;

static bool ggml_cuda_nvfp4_device_pointer(const void * ptr);
static bool ggml_cuda_nvfp4_tensor_active_data(ggml_tensor * tensor, void ** data, int * device);

template <typename T>
static inline void ggml_cuda_nvfp4_release_typed_buf(T ** buf, size_t * cap) {
    void * raw = (void *) *buf;
    ggml_cuda_nvfp4_release_buf(&raw, cap);
    *buf = nullptr;
}

struct nvfp4_cuda_stream_tls {
    cudaStream_t stream = nullptr;
    bool initialized = false;

    void reset() {
        if (stream != nullptr) {
            const cudaError_t err = cudaStreamDestroy(stream);
            if (err != cudaSuccess) {
                nvfp4_cuda_log_failure("autotune stream destroy", err);
                cudaGetLastError();
            }
            stream = nullptr;
        }
        initialized = false;
    }

    ~nvfp4_cuda_stream_tls() {
        reset();
    }
};

struct nvfp4_cuda_autotune_tls {
    float * d_x_sample_buf = nullptr;
    size_t d_x_sample_cap = 0;
    float * d_qw_sample_buf = nullptr;
    size_t d_qw_sample_cap = 0;
    float * d_x_sample_alt_buf = nullptr;
    size_t d_x_sample_alt_cap = 0;
    float * d_qw_sample_alt_buf = nullptr;
    size_t d_qw_sample_alt_cap = 0;
    block_nvfp4 * d_y_sample_buf = nullptr;
    size_t d_y_sample_cap = 0;
    nvfp4_cuda_runtime_cfg * d_eval_cfg_buf = nullptr;
    size_t d_eval_cfg_cap = 0;
    float * d_eval_xscale_buf = nullptr;
    size_t d_eval_xscale_cap = 0;
    double * d_rel_obj_buf = nullptr;
    size_t d_rel_obj_cap = 0;
    double * d_rel_obj_sorted_buf = nullptr;
    size_t d_rel_obj_sorted_cap = 0;
    void * d_sort_temp_buf = nullptr;
    size_t d_sort_temp_cap = 0;
    nvfp4_cuda_tune_eval_accum * d_tune_eval_buf = nullptr;
    size_t d_tune_eval_cap = 0;
    const float * loaded_x = nullptr;
    const float * loaded_qw = nullptr;
    int64_t loaded_nb = 0;

    void reset() {
        ggml_cuda_nvfp4_release_typed_buf(&d_x_sample_buf, &d_x_sample_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_qw_sample_buf, &d_qw_sample_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_x_sample_alt_buf, &d_x_sample_alt_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_qw_sample_alt_buf, &d_qw_sample_alt_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_y_sample_buf, &d_y_sample_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_eval_cfg_buf, &d_eval_cfg_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_eval_xscale_buf, &d_eval_xscale_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_rel_obj_buf, &d_rel_obj_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_rel_obj_sorted_buf, &d_rel_obj_sorted_cap);
        ggml_cuda_nvfp4_release_buf(&d_sort_temp_buf, &d_sort_temp_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_tune_eval_buf, &d_tune_eval_cap);
        loaded_x = nullptr;
        loaded_qw = nullptr;
        loaded_nb = 0;
    }

    ~nvfp4_cuda_autotune_tls() {
        reset();
    }
};

struct nvfp4_cuda_quant_tls {
    void * d_x_buf = nullptr;
    size_t d_x_cap = 0;
    float * d_qw_buf = nullptr;
    size_t d_qw_cap = 0;
    block_nvfp4 * d_y_buf = nullptr;
    size_t d_y_cap = 0;
    nvfp4_cuda_eval_accum * d_eval_buf = nullptr;
    size_t d_eval_cap = 0;

    void reset() {
        ggml_cuda_nvfp4_release_buf(&d_x_buf, &d_x_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_qw_buf, &d_qw_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_y_buf, &d_y_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_eval_buf, &d_eval_cap);
    }

    ~nvfp4_cuda_quant_tls() {
        reset();
    }
};

struct mxfp6_e2m3_cuda_quant_tls {
    void * d_x_buf = nullptr;
    size_t d_x_cap = 0;
    float * d_qw_buf = nullptr;
    size_t d_qw_cap = 0;
    block_mxfp6_e2m3 * d_compact_buf = nullptr;
    size_t d_compact_cap = 0;
    tile_mxfp6_e2m3 * d_y_buf = nullptr;
    size_t d_y_cap = 0;
    nvfp4_cuda_eval_accum * d_eval_buf = nullptr;
    size_t d_eval_cap = 0;

    void reset() {
        ggml_cuda_nvfp4_release_buf(&d_x_buf, &d_x_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_qw_buf, &d_qw_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_compact_buf, &d_compact_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_y_buf, &d_y_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_eval_buf, &d_eval_cap);
    }

    ~mxfp6_e2m3_cuda_quant_tls() {
        reset();
    }
};

struct nvfp4_cuda_kld_tls {
    struct base_cache_entry {
        const uint16_t * host = nullptr;
        uint16_t * device = nullptr;
        size_t bytes = 0;
        int device_id = -1;
        uint64_t last_use = 0;
    };

    uint16_t * d_base = nullptr;
    size_t d_base_cap = 0;
    int32_t * d_tokens = nullptr;
    size_t d_tokens_cap = 0;
    void * d_rows = nullptr;
    size_t d_rows_cap = 0;
    nvfp4_cuda_kld_result * d_result = nullptr;
    size_t d_result_cap = 0;
    double * d_kld_values = nullptr;
    size_t d_kld_values_cap = 0;
    std::vector<base_cache_entry> base_cache;
    size_t base_cache_bytes = 0;
    uint64_t base_cache_clock = 0;

    void reset() {
        ggml_cuda_nvfp4_release_typed_buf(&d_base, &d_base_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_tokens, &d_tokens_cap);
        ggml_cuda_nvfp4_release_buf(&d_rows, &d_rows_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_result, &d_result_cap);
        ggml_cuda_nvfp4_release_typed_buf(&d_kld_values, &d_kld_values_cap);
        for (auto & entry : base_cache) {
            void * raw = (void *) entry.device;
            size_t cap = entry.bytes;
            ggml_cuda_nvfp4_release_buf(&raw, &cap);
            entry.device = nullptr;
            entry.bytes = 0;
        }
        base_cache.clear();
        base_cache_bytes = 0;
        base_cache_clock = 0;
    }

    ~nvfp4_cuda_kld_tls() {
        reset();
    }
};

static thread_local nvfp4_cuda_stream_tls      g_nvfp4_cuda_stream_tls;
static thread_local nvfp4_cuda_autotune_tls   g_nvfp4_cuda_autotune_tls;
static thread_local nvfp4_cuda_quant_tls      g_nvfp4_cuda_quant_tls;
static thread_local mxfp6_e2m3_cuda_quant_tls g_mxfp6_e2m3_cuda_quant_tls;
static thread_local nvfp4_cuda_kld_tls        g_nvfp4_cuda_kld_tls;
static std::atomic<int> g_nvfp4_cuda_autotune_threads{0};

extern "C" void ggml_cuda_nvfp4_set_autotune_threads(int32_t n_threads) {
    g_nvfp4_cuda_autotune_threads.store(std::max<int32_t>(0, n_threads), std::memory_order_release);
}

extern "C" void ggml_cuda_nvfp4_clear_thread_cache() {
    g_nvfp4_cuda_autotune_tls.reset();
    g_nvfp4_cuda_quant_tls.reset();
    g_mxfp6_e2m3_cuda_quant_tls.reset();
    g_nvfp4_cuda_kld_tls.reset();
    g_nvfp4_cuda_stream_tls.reset();
}

struct nvfp4_cuda_eval_accum {
    double sum_sq;
    double sum_abs;
    unsigned int max_abs_bits;
    unsigned long long count;
};

struct nvfp4_cuda_tune_eval_accum {
    double sum_obj;
    double sum_x2;
    double sum_abs_err;
    double sum_abs_x;
    double p95_rel_obj;
    double sum_tail2;
    double max_rel_obj;
    unsigned long long tail_count;
};

static __device__ __forceinline__ float nvfp4_cuda_kvalue(const int i) {
    static const float values[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return values[i & 0x0F];
}

static __device__ __forceinline__ uint8_t nvfp4_cuda_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) {
        return 0;
    }
    if (x > 448.0f) {
        x = 448.0f;
    }

    const uint32_t bits = __float_as_uint(x);
    const int fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    const int fp32_man  = (bits >> 20) & 0x7;
    const int ue4m3_exp = fp32_exp + 7;

    if (ue4m3_exp <= 0) {
        int man = (int) (x * 512.0f + 0.5f);
        if (man > 7) {
            man = 7;
        }
        if (man < 1) {
            return 0;
        }
        return (uint8_t) man;
    }
    if (ue4m3_exp >= 15) {
        return 0x7E;
    }

    int round_bit = (bits >> 19) & 1;
    int ue4m3_man = fp32_man + round_bit;
    int ue4m3_exp_adj = ue4m3_exp;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ++ue4m3_exp_adj;
        if (ue4m3_exp_adj >= 15) {
            return 0x7E;
        }
    }

    return (uint8_t) ((ue4m3_exp_adj << 3) | ue4m3_man);
}

static __device__ __forceinline__ float nvfp4_cuda_ue4m3_to_fp32(uint8_t x) {
    return ggml_cuda_ue4m3_to_fp32(x);
}

static __device__ __forceinline__ uint8_t nvfp4_cuda_best_index(const float v, const float s) {
    if (!(s > 0.0f) || !isfinite(s) || !isfinite(v)) {
        return 0;
    }

    float best_err = FLT_MAX;
    uint8_t best = 0;
    for (int i = 0; i < 16; ++i) {
        const float q = s * nvfp4_cuda_kvalue(i);
        const float err = (v - q) * (v - q);
        if (err < best_err) {
            best_err = err;
            best = (uint8_t) i;
        }
    }

    return best;
}

static __device__ __forceinline__ float nvfp4_cuda_topk_abs_threshold(const float * x, int n, int topk) {
    if (x == nullptr || n <= 0 || topk <= 0) {
        return 0.0f;
    }

    const int k = topk > NVFP4_COMPAND_TOPK ? NVFP4_COMPAND_TOPK : topk;
    float top[NVFP4_COMPAND_TOPK] = { 0.0f };
    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        int pos = k;
        for (int j = 0; j < k; ++j) {
            if (ax > top[j]) {
                pos = j;
                break;
            }
        }
        if (pos >= k) {
            continue;
        }
        for (int j = k - 1; j > pos; --j) {
            top[j] = top[j - 1];
        }
        top[pos] = ax;
    }

    return top[k - 1];
}

static __device__ __forceinline__ float nvfp4_cuda_qw16(const float * qw16, const int j) {
    if (!qw16) {
        return 1.0f;
    }

    const float w = qw16[j];
    return (isfinite(w) && w > 0.0f) ? w : 0.0f;
}

static __device__ __forceinline__ float nvfp4_cuda_fake_quantize_positive_e2m1(float x) {
    if (x <= 0.0f) {
        return 0.0f;
    }

    float step1 = roundf(2.0f * x) * 0.5f;
    float step2 = roundf(x);
    float step3 = 2.0f * roundf(x * 0.5f);

    if (step3 > 6.0f) {
        step3 = 6.0f;
    }

    if (x < 2.0f) {
        return step1;
    }
    if (x < 4.0f) {
        return step2;
    }
    return step3;
}

static __device__ __forceinline__ float nvfp4_cuda_fake_quantize_signed_e2m1(const float x) {
    const float ax = fabsf(x);
    const float q = nvfp4_cuda_fake_quantize_positive_e2m1(ax);
    return copysignf(q, x);
}

static __device__ __forceinline__ float nvfp4_cuda_mse_16_for_scale_mit_w(
        const float * x16,
        const float * qw16,
        const float scale) {
    if (!(scale > 0.0f) || !isfinite(scale)) {
        return FLT_MAX;
    }

    float sse = 0.0f;
    for (int i = 0; i < QK_NVFP4_SUB; ++i) {
        const float w = nvfp4_cuda_qw16(qw16, i);
        if (w == 0.0f) {
            continue;
        }

        const float v = x16[i];
        const float q = nvfp4_cuda_fake_quantize_signed_e2m1(v / scale);
        const float e = q * scale - v;
        sse += w * e * e;
    }

    return sse;
}

static __device__ __forceinline__ uint8_t nvfp4_cuda_adaptive_block_scale_4_or_6(
        const float * x16,
        const float * qw16,
        const float d_fp32) {
    const float eps        = 1e-20f;
    const float m6_anchor  = 6.00f;
    const float m4_anchor  = 4.00f;
    const float u_guard    = 6.34f;
    const float tail_thr   = 0.95f;
    const int   tail_min   = 2;
    const float gap_lo     = 4.8f;
    const float gap_hi     = 5.20f;
    const float gap_lambda = 0.00f;
    const float top2_thr   = 0.92f;

    if (!(d_fp32 > 0.0f) || !isfinite(d_fp32)) {
        return 0;
    }

    float max_abs = 0.0f;
    for (int i = 0; i < QK_NVFP4_SUB; ++i) {
        max_abs = fmaxf(max_abs, fabsf(x16[i]));
    }
    if (!(max_abs > 0.0f)) {
        return 0;
    }

    const float sb6_ideal = (max_abs / m6_anchor) / d_fp32;
    const float sb4_ideal = (max_abs / m4_anchor) / d_fp32;

    const uint8_t b6 = nvfp4_cuda_fp32_to_ue4m3(sb6_ideal);
    const uint8_t b4 = nvfp4_cuda_fp32_to_ue4m3(sb4_ideal);
    if (b6 == b4) {
        return b4;
    }

    const float s6u = 2.0f * nvfp4_cuda_ue4m3_to_fp32(b6);
    const float s4u = 2.0f * nvfp4_cuda_ue4m3_to_fp32(b4);
    if (s6u == 0.0f) {
        return b4;
    }
    if (s4u == 0.0f) {
        return b6;
    }

    const float scale6 = d_fp32 * s6u;
    const float scale4 = d_fp32 * s4u;
    const float u_max6 = max_abs / (scale6 + eps);

    float max1 = 0.0f;
    float max2 = 0.0f;
    for (int i = 0; i < QK_NVFP4_SUB; ++i) {
        const float ax = fabsf(x16[i]);
        if (ax > max1) {
            max2 = max1;
            max1 = ax;
        } else if (ax > max2) {
            max2 = ax;
        }
    }

    int tail_cnt = 0;
    const float tail_cut = tail_thr * max_abs;
    for (int i = 0; i < QK_NVFP4_SUB; ++i) {
        if (fabsf(x16[i]) >= tail_cut) {
            ++tail_cnt;
        }
    }

    float l6 = nvfp4_cuda_mse_16_for_scale_mit_w(x16, qw16, scale6);
    const float l4 = nvfp4_cuda_mse_16_for_scale_mit_w(x16, qw16, scale4);

    if (max2 >= top2_thr * max1) {
        const float u6_2 = max2 / (scale6 + eps);
        if (u6_2 >= gap_lo && u6_2 <= gap_hi) {
            l6 += gap_lambda * (scale6 * scale6);
        }
    }
    if (u_max6 > u_guard) {
        l6 += gap_lambda * (scale6 * scale6);
    }

    if (tail_cnt >= tail_min) {
        for (int i = 0; i < QK_NVFP4_SUB; ++i) {
            const float ua = fabsf(x16[i] / (scale6 + eps));
            if (ua >= gap_lo && ua <= gap_hi) {
                l6 += gap_lambda * (scale6 * scale6);
            }
        }
    }

    return (l4 < l6) ? b4 : b6;
}

static __device__ __forceinline__ float nvfp4_cuda_pick_subblock_fp8_cap(
        const float * x16,
        const float * qw16,
        const float d_fp32,
        const float max_fp8_high,
        const float max_fp8_low,
        uint8_t * picked_scale) {
    const float high = (isfinite(max_fp8_high) && max_fp8_high > 0.0f) ? max_fp8_high : 448.0f;
    const float low = (isfinite(max_fp8_low) && max_fp8_low > 0.0f) ? fminf(max_fp8_low, high) : high;

    if (!(d_fp32 > 0.0f) || !isfinite(d_fp32)) {
        if (picked_scale != nullptr) {
            *picked_scale = 0;
        }
        return high;
    }

    float max_abs = 0.0f;
    for (int j = 0; j < QK_NVFP4_SUB; ++j) {
        max_abs = fmaxf(max_abs, fabsf(x16[j]));
    }
    if (!(max_abs > 0.0f)) {
        if (picked_scale != nullptr) {
            *picked_scale = 0;
        }
        return high;
    }

    const float sb6_ideal = (max_abs / 6.0f) / d_fp32;
    const float sb4_ideal = (max_abs / 4.0f) / d_fp32;

    const uint8_t b6 = nvfp4_cuda_fp32_to_ue4m3(sb6_ideal);
    const uint8_t b4 = nvfp4_cuda_fp32_to_ue4m3(sb4_ideal);
    const uint8_t b = nvfp4_cuda_adaptive_block_scale_4_or_6(x16, qw16, d_fp32);
    if (picked_scale != nullptr) {
        *picked_scale = b;
    }

    return (b == b4 && b4 != b6) ? low : high;
}

static __device__ __forceinline__ float nvfp4_cuda_subblock_sse_w_best(
        const float * x16,
        const float * qw16,
        const float scale) {
    if (!(scale > 0.0f) || !isfinite(scale)) {
        float sse = 0.0f;
        for (int j = 0; j < QK_NVFP4_SUB; ++j) {
            const float w = nvfp4_cuda_qw16(qw16, j);
            if (w == 0.0f) {
                continue;
            }
            sse += w * x16[j] * x16[j];
        }
        return sse;
    }

    float sse = 0.0f;
    for (int j = 0; j < QK_NVFP4_SUB; ++j) {
        const float w = nvfp4_cuda_qw16(qw16, j);
        if (w == 0.0f) {
            continue;
        }

        const float v = x16[j];
        const uint8_t code = nvfp4_cuda_best_index(v, scale);
        const float q = scale * nvfp4_cuda_kvalue(code);
        const float e = v - q;
        sse += w * e * e;
    }

    return sse;
}

static __device__ __forceinline__ void nvfp4_cuda_add_unique_scale_candidate(
        uint8_t * candidates,
        int * n_candidates,
        const int max_candidates,
        const int b,
        const uint8_t cap_b) {
    if (b < 0 || b > (int) cap_b || *n_candidates >= max_candidates) {
        return;
    }
    const uint8_t ub = (uint8_t) b;
    for (int i = 0; i < *n_candidates; ++i) {
        if (candidates[i] == ub) {
            return;
        }
    }
    candidates[*n_candidates] = ub;
    ++(*n_candidates);
}

static __device__ __forceinline__ uint8_t nvfp4_cuda_refine_sbscale_fp8_rich_cap(
        const float * x16,
        const float * qw16,
        const float d,
        const uint8_t b_init,
        const uint8_t cap_b,
        const int refit_iters) {
    if (!(d > 0.0f) || !isfinite(d)) {
        return 0;
    }

    constexpr int max_candidates = 48;
    constexpr int topk = 4;
    uint8_t candidates[max_candidates];
    int n_candidates = 0;

    const uint8_t b0 = b_init > cap_b ? cap_b : b_init;
    const int local_radius = refit_iters > 0 ? ((2 + refit_iters) < 8 ? (2 + refit_iters) : 8) : 2;
    for (int delta = -local_radius; delta <= local_radius; ++delta) {
        nvfp4_cuda_add_unique_scale_candidate(candidates, &n_candidates, max_candidates, (int) b0 + delta, cap_b);
    }

    float top[topk] = { 0.0f };
    for (int i = 0; i < QK_NVFP4_SUB; ++i) {
        if (!isfinite(x16[i])) {
            continue;
        }
        const float a = fabsf(x16[i]);
        for (int j = 0; j < topk; ++j) {
            if (a > top[j]) {
                for (int k = topk - 1; k > j; --k) {
                    top[k] = top[k - 1];
                }
                top[j] = a;
                break;
            }
        }
    }

    static const float slots[] = { 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.5f, 1.0f };
    constexpr int slot_radius = 1;
    for (int t = 0; t < topk; ++t) {
        if (!(top[t] > 0.0f)) {
            break;
        }
        for (int s = 0; s < (int) (sizeof(slots) / sizeof(slots[0])); ++s) {
            const float anchor = top[t] / slots[s];
            if (!(anchor > 0.0f) || !isfinite(anchor)) {
                continue;
            }
            const uint8_t b = nvfp4_cuda_fp32_to_ue4m3(anchor);
            for (int delta = -slot_radius; delta <= slot_radius; ++delta) {
                nvfp4_cuda_add_unique_scale_candidate(candidates, &n_candidates, max_candidates, (int) b + delta, cap_b);
            }
        }
    }

    if (n_candidates <= 0) {
        return b0;
    }

    uint8_t best_b = candidates[0];
    float best_sse = nvfp4_cuda_subblock_sse_w_best(x16, qw16, d * (2.0f * nvfp4_cuda_ue4m3_to_fp32(best_b)));
    for (int i = 1; i < n_candidates; ++i) {
        const uint8_t b = candidates[i];
        const float sse = nvfp4_cuda_subblock_sse_w_best(x16, qw16, d * (2.0f * nvfp4_cuda_ue4m3_to_fp32(b)));
        if (sse < best_sse) {
            best_sse = sse;
            best_b = b;
        }
    }

    return best_b;
}

static __global__ void ggml_cuda_nvfp4_quantize_blocks_16x4(
        const void * x,
        bool x_bf16,
        float x_scale,
        const float * qw,
        block_nvfp4 * y,
        int64_t row_blocks,
        int choose46_mode,
        int refit_iters,
        int use_compand_sat,
        float cap_m6,
        float cap_m4) {
    const int64_t ib = (int64_t) blockIdx.x;
    const int tid = (int) threadIdx.x;

    if (tid >= QK_NVFP4) {
        return;
    }

    __shared__ float s_x[QK_NVFP4];
    __shared__ float s_qw[QK_NVFP4];
    __shared__ uint8_t s_scales[QK_NVFP4 / QK_NVFP4_SUB];
    __shared__ uint8_t s_qs[QK_NVFP4 / 2];

    const int64_t block_in_row = ib % row_blocks;
    const int64_t off = ib * QK_NVFP4 + tid;

    float xv;
    if (x_bf16) {
        const nv_bfloat16 * xb = (const nv_bfloat16 *) x;
        xv = ggml_cuda_cast<float>(xb[off]);
    } else {
        const float * xf = (const float *) x;
        xv = xf[off];
    }
    if (x_scale != 1.0f) {
        xv *= (1.0f / x_scale);
    }
    s_x[tid] = xv;

    if (qw != nullptr) {
        // Quantize-time imatrix weights are provided once per row shape, not once per tensor row.
        // Reuse the per-row-shape block weights across all rows, matching the CPU path.
        s_qw[tid] = qw[(size_t) block_in_row * QK_NVFP4 + (size_t) tid];
    }
    __syncthreads();

    const int sub = tid / QK_NVFP4_SUB;
    const int lane = tid % QK_NVFP4_SUB;

    if (lane == 0) {
        const float * x16 = s_x + sub * QK_NVFP4_SUB;
        const float * w16 = qw != nullptr ? (s_qw + sub * QK_NVFP4_SUB) : nullptr;

        float max_abs = 0.0f;
        for (int j = 0; j < QK_NVFP4_SUB; ++j) {
            max_abs = fmaxf(max_abs, fabsf(x16[j]));
        }

        if (max_abs < 1e-15f) {
            s_scales[sub] = 0;
        } else {
            float max_fp8 = (isfinite(cap_m6) && cap_m6 > 0.0f) ? cap_m6 : 448.0f;
            uint8_t ue = 0;
            float fallback_anchor = 6.0f;
            if (choose46_mode == NVFP4_CUDA_CHOOSE46_FORCE_M6) {
                fallback_anchor = 6.0f;
                ue = nvfp4_cuda_fp32_to_ue4m3(fminf(max_abs / 6.0f, max_fp8));
            } else if (choose46_mode == NVFP4_CUDA_CHOOSE46_FORCE_M4) {
                max_fp8 = (isfinite(cap_m4) && cap_m4 > 0.0f) ? fminf(cap_m4, max_fp8) : 256.0f;
                fallback_anchor = 4.0f;
                ue = nvfp4_cuda_fp32_to_ue4m3(fminf(max_abs / 4.0f, max_fp8));
            } else {
                max_fp8 = nvfp4_cuda_pick_subblock_fp8_cap(x16, w16, 1.0f, cap_m6, cap_m4, &ue);
                fallback_anchor = nvfp4_cuda_ue4m3_to_fp32(ue) <= 0.0f ? 6.0f : (max_abs / fmaxf(2.0f * nvfp4_cuda_ue4m3_to_fp32(ue), 1e-20f) > 5.0f ? 6.0f : 4.0f);
            }

            uint8_t cap_b = nvfp4_cuda_fp32_to_ue4m3(max_fp8);
            if (cap_b > 0x7E) {
                cap_b = 0x7E;
            }
            if (ue > cap_b) {
                ue = cap_b;
            }

            if (use_compand_sat != 0) {
                const float abs_q = nvfp4_cuda_topk_abs_threshold(x16, QK_NVFP4_SUB, NVFP4_COMPAND_TOPK);
                const float compand_scale = fminf(abs_q / NVFP4_E2M1_MAX_VALUE, max_fp8);
                if (isfinite(compand_scale) && compand_scale > 0.0f) {
                    const uint8_t compand_b = nvfp4_cuda_fp32_to_ue4m3(compand_scale);
                    if (compand_b > ue && compand_b <= cap_b) {
                        ue = compand_b;
                    }
                }
            }

            ue = nvfp4_cuda_refine_sbscale_fp8_rich_cap(x16, w16, 1.0f, ue, cap_b, refit_iters);

            const float d_eff = 2.0f * nvfp4_cuda_ue4m3_to_fp32(ue);
            if (!(d_eff > 0.0f) || !isfinite(d_eff)) {
                ue = nvfp4_cuda_fp32_to_ue4m3(fminf(max_abs / fallback_anchor, max_fp8));
            }

            s_scales[sub] = ue;
        }
    }
    __syncthreads();

    if (lane < QK_NVFP4_SUB / 2) {
        const uint8_t ue = s_scales[sub];
        uint8_t packed = 0;
        if (ue != 0) {
            const float d = 2.0f * nvfp4_cuda_ue4m3_to_fp32(ue);
            const float * x16 = s_x + sub * QK_NVFP4_SUB;
            const uint8_t x0 = nvfp4_cuda_best_index(x16[lane], d);
            const uint8_t x1 = nvfp4_cuda_best_index(x16[lane + QK_NVFP4_SUB / 2], d);
            packed = (uint8_t) (x0 | (x1 << 4));
        }
        s_qs[sub * (QK_NVFP4_SUB / 2) + lane] = packed;
    }
    __syncthreads();

    if (tid < QK_NVFP4 / QK_NVFP4_SUB) {
        y[ib].d[tid] = s_scales[tid];
    }
    if (tid < QK_NVFP4 / 2) {
        y[ib].qs[tid] = s_qs[tid];
    }
}

static __global__ void ggml_cuda_nvfp4_quantize_blocks_16x4_batched(
        const void * x,
        bool x_bf16,
        float x_scale,
        const float * x_scales,
        const float * qw,
        block_nvfp4 * y,
        int64_t row_blocks,
        int64_t nb_total,
        const nvfp4_cuda_runtime_cfg * cfgs,
        int n_cfgs) {
    const int64_t ib = (int64_t) blockIdx.x;
    const int cfg_idx = (int) blockIdx.y;
    const int tid = (int) threadIdx.x;

    if (tid >= QK_NVFP4 || cfg_idx >= n_cfgs) {
        return;
    }

    const nvfp4_cuda_runtime_cfg cfg = cfgs[cfg_idx];
    float x_scale_cfg = x_scale;
    if (x_scales != nullptr) {
        const float xs = x_scales[cfg_idx];
        if (isfinite(xs) && xs > 0.0f) {
            x_scale_cfg = xs;
        }
    }

    __shared__ float s_x[QK_NVFP4];
    __shared__ float s_qw[QK_NVFP4];
    __shared__ uint8_t s_scales[QK_NVFP4 / QK_NVFP4_SUB];
    __shared__ uint8_t s_qs[QK_NVFP4 / 2];

    const int64_t block_in_row = ib % row_blocks;
    const int64_t off = ib * QK_NVFP4 + tid;

    float xv;
    if (x_bf16) {
        const nv_bfloat16 * xb = (const nv_bfloat16 *) x;
        xv = ggml_cuda_cast<float>(xb[off]);
    } else {
        const float * xf = (const float *) x;
        xv = xf[off];
    }
    if (x_scale_cfg != 1.0f) {
        xv *= (1.0f / x_scale_cfg);
    }
    s_x[tid] = xv;

    if (qw != nullptr) {
        s_qw[tid] = qw[(size_t) block_in_row * QK_NVFP4 + (size_t) tid];
    }
    __syncthreads();

    const int sub = tid / QK_NVFP4_SUB;
    const int lane = tid % QK_NVFP4_SUB;

    if (lane == 0) {
        const float * x16 = s_x + sub * QK_NVFP4_SUB;
        const float * w16 = qw != nullptr ? (s_qw + sub * QK_NVFP4_SUB) : nullptr;

        float max_abs = 0.0f;
        for (int j = 0; j < QK_NVFP4_SUB; ++j) {
            max_abs = fmaxf(max_abs, fabsf(x16[j]));
        }

        if (max_abs < 1e-15f) {
            s_scales[sub] = 0;
        } else {
            float max_fp8 = (isfinite(cfg.cap_m6) && cfg.cap_m6 > 0.0f) ? cfg.cap_m6 : 448.0f;
            uint8_t ue = 0;
            float fallback_anchor = 6.0f;
            if (cfg.choose46_mode == NVFP4_CUDA_CHOOSE46_FORCE_M6) {
                fallback_anchor = 6.0f;
                ue = nvfp4_cuda_fp32_to_ue4m3(fminf(max_abs / 6.0f, max_fp8));
            } else if (cfg.choose46_mode == NVFP4_CUDA_CHOOSE46_FORCE_M4) {
                max_fp8 = (isfinite(cfg.cap_m4) && cfg.cap_m4 > 0.0f) ? fminf(cfg.cap_m4, max_fp8) : 256.0f;
                fallback_anchor = 4.0f;
                ue = nvfp4_cuda_fp32_to_ue4m3(fminf(max_abs / 4.0f, max_fp8));
            } else {
                max_fp8 = nvfp4_cuda_pick_subblock_fp8_cap(x16, w16, 1.0f, cfg.cap_m6, cfg.cap_m4, &ue);
                fallback_anchor = nvfp4_cuda_ue4m3_to_fp32(ue) <= 0.0f ? 6.0f : (max_abs / fmaxf(2.0f * nvfp4_cuda_ue4m3_to_fp32(ue), 1e-20f) > 5.0f ? 6.0f : 4.0f);
            }

            uint8_t cap_b = nvfp4_cuda_fp32_to_ue4m3(max_fp8);
            if (cap_b > 0x7E) {
                cap_b = 0x7E;
            }
            if (ue > cap_b) {
                ue = cap_b;
            }

            if (cfg.use_compand_sat != 0) {
                const float abs_q = nvfp4_cuda_topk_abs_threshold(x16, QK_NVFP4_SUB, NVFP4_COMPAND_TOPK);
                const float compand_scale = fminf(abs_q / NVFP4_E2M1_MAX_VALUE, max_fp8);
                if (isfinite(compand_scale) && compand_scale > 0.0f) {
                    const uint8_t compand_b = nvfp4_cuda_fp32_to_ue4m3(compand_scale);
                    if (compand_b > ue && compand_b <= cap_b) {
                        ue = compand_b;
                    }
                }
            }

            ue = nvfp4_cuda_refine_sbscale_fp8_rich_cap(x16, w16, 1.0f, ue, cap_b, cfg.refit_iters);

            const float d_eff = 2.0f * nvfp4_cuda_ue4m3_to_fp32(ue);
            if (!(d_eff > 0.0f) || !isfinite(d_eff)) {
                ue = nvfp4_cuda_fp32_to_ue4m3(fminf(max_abs / fallback_anchor, max_fp8));
            }

            s_scales[sub] = ue;
        }
    }
    __syncthreads();

    if (lane < QK_NVFP4_SUB / 2) {
        const uint8_t ue = s_scales[sub];
        uint8_t packed = 0;
        if (ue != 0) {
            const float d = 2.0f * nvfp4_cuda_ue4m3_to_fp32(ue);
            const float * x16 = s_x + sub * QK_NVFP4_SUB;
            const uint8_t x0 = nvfp4_cuda_best_index(x16[lane], d);
            const uint8_t x1 = nvfp4_cuda_best_index(x16[lane + QK_NVFP4_SUB / 2], d);
            packed = (uint8_t) (x0 | (x1 << 4));
        }
        s_qs[sub * (QK_NVFP4_SUB / 2) + lane] = packed;
    }
    __syncthreads();

    block_nvfp4 * y_cfg = y + (size_t) cfg_idx * (size_t) nb_total;
    if (tid < QK_NVFP4 / QK_NVFP4_SUB) {
        y_cfg[ib].d[tid] = s_scales[tid];
    }
    if (tid < QK_NVFP4 / 2) {
        y_cfg[ib].qs[tid] = s_qs[tid];
    }
}

static inline bool ggml_cuda_nvfp4_launch_kernel_batched(
        const void * d_x,
        bool x_bf16,
        float x_scale,
        const float * d_x_scales,
        const float * d_qw,
        block_nvfp4 * d_y,
        int64_t row_blocks,
        int64_t nb_total,
        const nvfp4_cuda_runtime_cfg * d_cfgs,
        int n_cfgs,
        cudaStream_t st) {
    if (d_cfgs == nullptr || n_cfgs <= 0) {
        return false;
    }

    const dim3 grid((unsigned int) nb_total, (unsigned int) n_cfgs, 1);
    ggml_cuda_nvfp4_quantize_blocks_16x4_batched<<<grid, QK_NVFP4, 0, st>>>(
        d_x, x_bf16, x_scale, d_x_scales, d_qw, d_y, row_blocks, nb_total, d_cfgs, n_cfgs);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("kernel batched", err);
        cudaGetLastError();
        return false;
    }
    return true;
}

static __device__ __forceinline__ float mxfp6_e2m3_cuda_ue8m0_to_fp32(uint8_t e) {
    return e == 0xFF ? 0.0f : ldexpf(1.0f, (int) e - 127);
}

static __device__ __forceinline__ uint8_t mxfp6_e2m3_cuda_fp32_to_ue8m0(float x) {
    if (!(x > 0.0f) || !isfinite(x)) {
        return 127;
    }

    int exp2 = 0;
    const float m = frexpf(x, &exp2);
    int unbiased = exp2 - 1;
    if (m >= 0.7071067811865475f) {
        ++unbiased;
    }

    int code = unbiased + 127;
    code = max(0, min(254, code));
    return (uint8_t) code;
}

static __device__ __forceinline__ float mxfp6_e2m3_cuda_value(uint8_t code) {
    code &= 0x3F;
    const int sign = code >> 5;
    const int exp  = (code >> 3) & 0x3;
    const int man  = code & 0x7;

    float v = 0.0f;
    if (exp == 0) {
        v = (float) man * 0.125f;
    } else {
        v = ldexpf(1.0f + (float) man * 0.125f, exp - 1);
    }
    return sign ? -v : v;
}

static __device__ __forceinline__ uint8_t mxfp6_e2m3_cuda_quant(float x, float inv_scale) {
    if (!isfinite(x) || x == 0.0f || !(inv_scale > 0.0f)) {
        return 0;
    }

    const int sign = x < 0.0f ? 0x20 : 0x00;
    const float ax = fabsf(x) * inv_scale;
    static const float bounds[31] = {
        0.0625f, 0.1875f, 0.3125f, 0.4375f, 0.5625f, 0.6875f, 0.8125f, 0.9375f,
        1.0625f, 1.1875f, 1.3125f, 1.4375f, 1.5625f, 1.6875f, 1.8125f, 1.9375f,
        2.125f, 2.375f, 2.625f, 2.875f, 3.125f, 3.375f, 3.625f, 3.875f,
        4.25f, 4.75f, 5.25f, 5.75f, 6.25f, 6.75f, 7.25f,
    };

#pragma unroll
    for (int i = 0; i < 31; ++i) {
        if (ax < bounds[i] || (ax == bounds[i] && (i & 1) == 0)) {
            return (uint8_t) (sign | i);
        }
    }

    return (uint8_t) (sign | 31);
}

static __device__ __forceinline__ void mxfp6_e2m3_cuda_add_candidate(uint8_t * candidates, int * n, int code) {
    code = max(0, min(254, code));
    const uint8_t c = (uint8_t) code;
#pragma unroll
    for (int i = 0; i < 256; ++i) {
        if (i >= *n) {
            break;
        }
        if (candidates[i] == c) {
            return;
        }
    }
    if (*n < 256) {
        candidates[(*n)++] = c;
    }
}

static __device__ __forceinline__ float mxfp6_e2m3_cuda_block_qw(
        const float * qw,
        int i,
        float mean) {
    if (qw == nullptr) {
        return 1.0f;
    }

    constexpr float blend = 0.35f;
    constexpr float power = 0.50f;
    constexpr float min_w = 0.25f;
    constexpr float max_w = 4.00f;

    float w = isfinite(qw[i]) && qw[i] > 0.0f ? qw[i] : mean;
    w = mean > 0.0f ? w / mean : 1.0f;
    w = fminf(max_w, fmaxf(min_w, w));
    if (power != 1.0f) {
        w = powf(w, power);
    }
    return (1.0f - blend) + blend * w;
}

static __device__ __forceinline__ float mxfp6_e2m3_cuda_sse32(
        const float * x,
        const float * qw,
        uint8_t scale_code) {
    const float scale = mxfp6_e2m3_cuda_ue8m0_to_fp32(scale_code);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

    float qw_mean = 1.0f;
    if (qw != nullptr) {
        float sum = 0.0f;
        int npos = 0;
#pragma unroll
        for (int i = 0; i < QK_MXFP6_E2M3_SUB; ++i) {
            if (isfinite(qw[i]) && qw[i] > 0.0f) {
                sum += qw[i];
                ++npos;
            }
        }
        if (npos > 0 && sum > 0.0f) {
            qw_mean = sum / npos;
        }
    }

    float sse = 0.0f;
#pragma unroll
    for (int i = 0; i < QK_MXFP6_E2M3_SUB; ++i) {
        const float xi = isfinite(x[i]) ? x[i] : 0.0f;
        const float wi = mxfp6_e2m3_cuda_block_qw(qw, i, qw_mean);
        const uint8_t q = mxfp6_e2m3_cuda_quant(xi, inv_scale);
        const float deq = scale * mxfp6_e2m3_cuda_value(q);
        const float err = xi - deq;
        sse += wi * err * err;
    }
    return sse;
}

static __device__ __forceinline__ uint8_t mxfp6_e2m3_cuda_best_scale_code32(
        const float * x,
        const float * qw) {
    float top[8] = { 0.0f };
#pragma unroll
    for (int i = 0; i < QK_MXFP6_E2M3_SUB; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }
        const float a = fabsf(x[i]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (a > top[j]) {
                for (int k = 7; k > j; --k) {
                    top[k] = top[k - 1];
                }
                top[j] = a;
                break;
            }
        }
    }

    if (!(top[0] > 0.0f)) {
        return 127;
    }

    uint8_t candidates[256];
    int n_candidates = 0;

    constexpr int radius = 8;
    const uint8_t base = mxfp6_e2m3_cuda_fp32_to_ue8m0(top[0] / 7.5f);
    for (int d = -radius; d <= radius; ++d) {
        mxfp6_e2m3_cuda_add_candidate(candidates, &n_candidates, (int) base + d);
    }

    static const float slots[] = {
        7.5f, 7.0f, 6.5f, 6.0f, 5.5f, 5.0f, 4.5f, 4.0f,
        3.75f, 3.5f, 3.25f, 3.0f, 2.75f, 2.5f, 2.25f, 2.0f,
        1.875f, 1.75f, 1.625f, 1.5f, 1.375f, 1.25f, 1.125f, 1.0f,
        0.875f, 0.75f, 0.625f, 0.5f, 0.375f, 0.25f, 0.125f,
    };
    constexpr int topk = 8;
    constexpr int slot_radius = 4;
    for (int t = 0; t < topk; ++t) {
        if (!(top[t] > 0.0f)) {
            break;
        }
        for (int s = 0; s < (int) (sizeof(slots) / sizeof(slots[0])); ++s) {
            const uint8_t c0 = mxfp6_e2m3_cuda_fp32_to_ue8m0(top[t] / slots[s]);
            for (int d = -slot_radius; d <= slot_radius; ++d) {
                mxfp6_e2m3_cuda_add_candidate(candidates, &n_candidates, (int) c0 + d);
            }
        }
    }

    uint8_t best = candidates[0];
    float best_sse = FLT_MAX;
    for (int i = 0; i < n_candidates; ++i) {
        const float sse = mxfp6_e2m3_cuda_sse32(x, qw, candidates[i]);
        if (sse < best_sse) {
            best_sse = sse;
            best = candidates[i];
        }
    }

    return best;
}

static __device__ __forceinline__ void mxfp6_e2m3_cuda_set_code(block_mxfp6_e2m3 * block, int i, uint8_t code) {
    const int bit   = i * 6;
    const int byte  = bit >> 3;
    const int shift = bit & 7;
    uint32_t v = (uint32_t) block->qs[0][byte];
    if (byte + 1 < QK_MXFP6_E2M3_PACKED_BYTES) {
        v |= (uint32_t) block->qs[0][byte + 1] << 8;
    }
    const uint32_t mask = (uint32_t) 0x3F << shift;
    v = (v & ~mask) | (((uint32_t) code & 0x3F) << shift);
    block->qs[0][byte] = (uint8_t) (v & 0xFF);
    if (byte + 1 < QK_MXFP6_E2M3_PACKED_BYTES) {
        block->qs[0][byte + 1] = (uint8_t) ((v >> 8) & 0xFF);
    }
}

static __device__ __forceinline__ uint8_t mxfp6_e2m3_cuda_get_code(const block_mxfp6_e2m3 * block, int i) {
    const int bit   = i * 6;
    const int byte  = bit >> 3;
    const int shift = bit & 7;
    uint32_t v = (uint32_t) block->qs[0][byte] >> shift;
    if (byte + 1 < QK_MXFP6_E2M3_PACKED_BYTES) {
        v |= (uint32_t) block->qs[0][byte + 1] << (8 - shift);
    }
    return (uint8_t) (v & 0x3F);
}

static __global__ void ggml_cuda_mxfp6_e2m3_quantize_blocks_32(
        const void * x,
        bool x_bf16,
        float x_scale,
        const float * qw,
        block_mxfp6_e2m3 * y,
        int64_t row_blocks) {
    const int64_t ib = (int64_t) blockIdx.x;
    const int tid = (int) threadIdx.x;
    if (tid >= QK_MXFP6_E2M3) {
        return;
    }

    __shared__ float s_x[QK_MXFP6_E2M3];
    __shared__ float s_qw[QK_MXFP6_E2M3];

    const int64_t block_in_row = ib % row_blocks;
    const int64_t off = ib * QK_MXFP6_E2M3 + tid;

    float xv;
    if (x_bf16) {
        const nv_bfloat16 * xb = (const nv_bfloat16 *) x;
        xv = ggml_cuda_cast<float>(xb[off]);
    } else {
        const float * xf = (const float *) x;
        xv = xf[off];
    }
    if (x_scale != 1.0f) {
        xv *= (1.0f / x_scale);
    }
    s_x[tid] = isfinite(xv) ? xv : 0.0f;

    if (qw != nullptr) {
        s_qw[tid] = qw[(size_t) block_in_row * QK_MXFP6_E2M3 + (size_t) tid];
    }
    __syncthreads();

    if (tid == 0) {
        block_mxfp6_e2m3 & out = y[ib];
        out.e[0] = mxfp6_e2m3_cuda_best_scale_code32(s_x, qw != nullptr ? s_qw : nullptr);
#pragma unroll
        for (int i = 0; i < QK_MXFP6_E2M3_PACKED_BYTES; ++i) {
            out.qs[0][i] = 0;
        }
        const float scale = mxfp6_e2m3_cuda_ue8m0_to_fp32(out.e[0]);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
#pragma unroll
        for (int i = 0; i < QK_MXFP6_E2M3; ++i) {
            mxfp6_e2m3_cuda_set_code(&out, i, mxfp6_e2m3_cuda_quant(s_x[i], inv_scale));
        }
    }
}

static __device__ __forceinline__ uint32_t mxfp6_e2m3_cuda_u32_from_bytes(
        const uint8_t * p,
        int i0) {
    return (uint32_t) p[i0 + 0] |
        ((uint32_t) p[i0 + 1] << 8) |
        ((uint32_t) p[i0 + 2] << 16) |
        ((uint32_t) p[i0 + 3] << 24);
}

static __global__ void ggml_cuda_mxfp6_e2m3_pack_tiles_832(
        const block_mxfp6_e2m3 * x,
        tile_mxfp6_e2m3 * y,
        int64_t nrow,
        int64_t row_blocks) {
    const int64_t tiles_per_row_group = row_blocks / QK_MXFP6_E2M3_TILE_FRAGS;
    const int64_t tile_frag = (int64_t) blockIdx.x;
    const int lane = (int) threadIdx.x;

    const int frag = (int) (tile_frag % QK_MXFP6_E2M3_TILE_FRAGS);
    const int64_t tile_idx = tile_frag / QK_MXFP6_E2M3_TILE_FRAGS;
    const int64_t tile_col = tile_idx % tiles_per_row_group;
    const int64_t row_group = tile_idx / tiles_per_row_group;
    const int row_lo = lane / 4;
    const int lane_base = lane & 3;
    const int row_hi = row_lo + 8;

    tile_mxfp6_e2m3_frag * out = &y[tile_idx].frag[frag];
    out->scale[lane] = 0x7f;
    __syncthreads();

    const int64_t block_col = tile_col * QK_MXFP6_E2M3_TILE_FRAGS + frag;
    const int64_t row0 = row_group * MXFP6_E2M3_TILE_ROWS;
    const block_mxfp6_e2m3 * lo = row0 + row_lo < nrow ? x + (row0 + row_lo) * row_blocks + block_col : nullptr;
    const block_mxfp6_e2m3 * hi = row0 + row_hi < nrow ? x + (row0 + row_hi) * row_blocks + block_col : nullptr;

    if (lane_base == 0) {
        out->scale[row_lo * 4 + 0] = lo ? lo->e[0] : 0x7f;
        out->scale[row_lo * 4 + 1] = hi ? hi->e[0] : 0x7f;
    }

    uint8_t packed[12] = {};
    if (lo) {
        const int lo0 = lane_base + 0;
        const int hi0 = lane_base + 4;
        packed[0] = lo->qs[0][3 * lo0 + 0];
        packed[1] = lo->qs[0][3 * lo0 + 1];
        packed[2] = lo->qs[0][3 * lo0 + 2];
        packed[6] = lo->qs[0][3 * hi0 + 0];
        packed[7] = lo->qs[0][3 * hi0 + 1];
        packed[8] = lo->qs[0][3 * hi0 + 2];
    }
    if (hi) {
        const int lo0 = lane_base + 0;
        const int hi0 = lane_base + 4;
        packed[3]  = hi->qs[0][3 * lo0 + 0];
        packed[4]  = hi->qs[0][3 * lo0 + 1];
        packed[5]  = hi->qs[0][3 * lo0 + 2];
        packed[9]  = hi->qs[0][3 * hi0 + 0];
        packed[10] = hi->qs[0][3 * hi0 + 1];
        packed[11] = hi->qs[0][3 * hi0 + 2];
    }

    out->lane[lane][0] = mxfp6_e2m3_cuda_u32_from_bytes(packed, 0);
    out->lane[lane][1] = mxfp6_e2m3_cuda_u32_from_bytes(packed, 4);
    out->lane[lane][2] = mxfp6_e2m3_cuda_u32_from_bytes(packed, 8);
}

static __global__ void ggml_cuda_mxfp6_e2m3_eval_quantized(
        const void * x,
        bool x_bf16,
        float x_scale,
        const float * qw,
        const block_mxfp6_e2m3 * y,
        int64_t nrow,
        int64_t n_per_row,
        nvfp4_cuda_eval_accum * out) {
    const int64_t n = nrow * n_per_row;
    const int64_t row_blocks = n_per_row / QK_MXFP6_E2M3;

    double local_sq = 0.0;
    double local_abs = 0.0;
    float local_max = 0.0f;
    unsigned long long local_count = 0;

    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
            idx < n;
            idx += (int64_t) gridDim.x * blockDim.x) {
        float xv;
        if (x_bf16) {
            const nv_bfloat16 * xb = (const nv_bfloat16 *) x;
            xv = ggml_cuda_cast<float>(xb[idx]);
        } else {
            const float * xf = (const float *) x;
            xv = xf[idx];
        }
        xv = isfinite(xv) ? xv : 0.0f;

        const int64_t ib = idx / QK_MXFP6_E2M3;
        const int elem = (int) (idx % QK_MXFP6_E2M3);
        const int64_t block_in_row = ib % row_blocks;
        const float w = qw != nullptr ? fmaxf(0.0f, qw[(size_t) block_in_row * QK_MXFP6_E2M3 + (size_t) elem]) : 1.0f;

        const block_mxfp6_e2m3 & block = y[ib];
        const float d = x_scale * mxfp6_e2m3_cuda_ue8m0_to_fp32(block.e[0]);
        const float qv = d * mxfp6_e2m3_cuda_value(mxfp6_e2m3_cuda_get_code(&block, elem));
        const float err = xv - qv;
        local_sq += (double) w * (double) err * (double) err;
        local_abs += (double) w * (double) fabsf(err);
        local_max = fmaxf(local_max, fabsf(err));
        ++local_count;
    }

    atomicAdd(&out->sum_sq, local_sq);
    atomicAdd(&out->sum_abs, local_abs);
    atomicAdd(&out->count, local_count);
    atomicMax(&out->max_abs_bits, __float_as_uint(local_max));
}

static bool ggml_cuda_nvfp4_quantize_impl(
        const void * x,
        bool x_bf16,
        float x_scale,
        void * vy,
        ggml_tensor * tensor,
        int64_t plane_index,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float a,
        float b,
        const nvfp4_cuda_runtime_cfg * cfg,
        nvfp4_cuda_eval_result * eval,
        cudaStream_t stream);

static inline void nvfp4_host_apply_ab_caps(
        nvfp4_cuda_runtime_cfg & resolved,
        float a,
        float b) {
    // The public 16x4 layout no longer has the MW tensor-global scale degree of freedom.
    // Reuse the A/B knobs as soft cap multipliers for the adaptive 6-anchor and 4-anchor paths.
    const float a_ratio = (isfinite(a) && a > 0.0f) ? (a / NVFP4_A0) : 1.0f;
    const float b_ratio = (isfinite(b) && b > 0.0f) ? (b / NVFP4_B0) : 1.0f;
    resolved.cap_m6 = std::clamp(448.0f * a_ratio, 32.0f, 448.0f);
    resolved.cap_m4 = std::clamp(256.0f * b_ratio, 16.0f, resolved.cap_m6);
}

struct nvfp4_tune_eval_stats {
    double obj_norm = DBL_MAX;
    double p95_rel_obj = DBL_MAX;
    double tail_rel_obj = DBL_MAX;
    double max_rel_obj = DBL_MAX;
    double abs_mean_err_rel = DBL_MAX;
};

static int64_t nvfp4_cuda_env_i64(const char * name, int64_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (end == value || (end != nullptr && *end != '\0') || parsed <= 0) {
        return fallback;
    }

    return (int64_t) parsed;
}

static bool ggml_cuda_nvfp4_kld_copy_base_to_scratch(
        nvfp4_cuda_kld_tls & tls,
        const uint16_t * base_host,
        size_t bytes_base,
        cudaStream_t stream,
        const uint16_t ** base_device) {
    if (!ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_base, &tls.d_base_cap, bytes_base, "kld cudaMalloc(base)")) {
        return false;
    }
    const cudaError_t err = cudaMemcpyAsync(tls.d_base, base_host, bytes_base, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("kld H2D(base)", err);
        cudaGetLastError();
        return false;
    }
    *base_device = tls.d_base;
    return true;
}

static bool ggml_cuda_nvfp4_kld_get_base_device(
        nvfp4_cuda_kld_tls & tls,
        const uint16_t * base_host,
        size_t bytes_base,
        cudaStream_t stream,
        const uint16_t ** base_device) {
    *base_device = nullptr;
    if (base_host == nullptr || bytes_base == 0) {
        return false;
    }

    const size_t cache_limit = NVFP4_CUDA_KLD_BASE_CACHE_LIMIT_BYTES;
    if (cache_limit < bytes_base) {
        return ggml_cuda_nvfp4_kld_copy_base_to_scratch(tls, base_host, bytes_base, stream, base_device);
    }

    int device_id = 0;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("kld cudaGetDevice", err);
        cudaGetLastError();
        return false;
    }

    for (auto & entry : tls.base_cache) {
        if (entry.host == base_host && entry.bytes == bytes_base && entry.device_id == device_id && entry.device != nullptr) {
            entry.last_use = ++tls.base_cache_clock;
            *base_device = entry.device;
            return true;
        }
    }

    while (tls.base_cache_bytes + bytes_base > cache_limit && !tls.base_cache.empty()) {
        auto victim = std::min_element(
            tls.base_cache.begin(),
            tls.base_cache.end(),
            [](const auto & a, const auto & b) {
                return a.last_use < b.last_use;
            });
        void * raw = (void *) victim->device;
        size_t cap = victim->bytes;
        ggml_cuda_nvfp4_release_buf(&raw, &cap);
        tls.base_cache_bytes -= victim->bytes;
        tls.base_cache.erase(victim);
    }

    auto clear_base_cache = [&]() {
        for (auto & entry : tls.base_cache) {
            void * raw = (void *) entry.device;
            size_t cap = entry.bytes;
            ggml_cuda_nvfp4_release_buf(&raw, &cap);
            entry.device = nullptr;
            entry.bytes = 0;
        }
        tls.base_cache.clear();
        tls.base_cache_bytes = 0;
    };

    uint16_t * d_base = nullptr;
    err = cudaMalloc((void **) &d_base, bytes_base);
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("kld cudaMalloc(base cache)", err);
        cudaGetLastError();
        clear_base_cache();
        return ggml_cuda_nvfp4_kld_copy_base_to_scratch(tls, base_host, bytes_base, stream, base_device);
    }
    err = cudaMemcpyAsync(d_base, base_host, bytes_base, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("kld H2D(base cache)", err);
        cudaFree(d_base);
        cudaGetLastError();
        return ggml_cuda_nvfp4_kld_copy_base_to_scratch(tls, base_host, bytes_base, stream, base_device);
    }

    nvfp4_cuda_kld_tls::base_cache_entry entry;
    entry.host = base_host;
    entry.device = d_base;
    entry.bytes = bytes_base;
    entry.device_id = device_id;
    entry.last_use = ++tls.base_cache_clock;
    tls.base_cache.push_back(entry);
    tls.base_cache_bytes += bytes_base;
    *base_device = d_base;
    return true;
}

static void nvfp4_host_prepare_block_sample(
        const float * src,
        int64_t nb_total,
        int64_t sample_nb,
        std::vector<float> & sampled,
        const float ** sampled_ptr,
        int64_t phase = 0) {
    sampled.clear();
    if (sampled_ptr == nullptr) {
        return;
    }

    *sampled_ptr = nullptr;
    if (src == nullptr || nb_total <= 0 || sample_nb <= 0) {
        return;
    }

    if (sample_nb >= nb_total) {
        *sampled_ptr = src;
        return;
    }

    sampled.resize((size_t) sample_nb * QK_NVFP4);
    for (int64_t is = 0; is < sample_nb; ++is) {
        const int64_t ib = sample_nb <= 1 ? 0 : (is * (nb_total - 1)) / (sample_nb - 1);
        const int64_t ib_phase = (ib + phase) % nb_total;
        std::memcpy(sampled.data() + is * QK_NVFP4, src + ib_phase * QK_NVFP4, QK_NVFP4 * sizeof(float));
    }

    *sampled_ptr = sampled.data();
}

static double nvfp4_host_robust_score(
        const nvfp4_tune_eval_stats & cand,
        const nvfp4_tune_eval_stats & base) {
    auto safe_ratio = [](double v, double b) {
        if (std::isfinite(v) && std::isfinite(b) && b > 0.0) {
            return v / b;
        }
        return v;
    };

    return
        0.40 * safe_ratio(cand.obj_norm, base.obj_norm) +
        0.22 * safe_ratio(cand.p95_rel_obj, base.p95_rel_obj) +
        0.22 * safe_ratio(cand.tail_rel_obj, base.tail_rel_obj) +
        0.12 * safe_ratio(cand.max_rel_obj, base.max_rel_obj) +
        0.04 * safe_ratio(cand.abs_mean_err_rel, base.abs_mean_err_rel);
}

static bool nvfp4_host_eval_better(
        const nvfp4_tune_eval_stats & cand,
        const nvfp4_tune_eval_stats & best) {
    auto nearly_eq = [](double a, double b) {
        const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
        return std::fabs(a - b) <= 1e-12 * scale;
    };

    if (cand.obj_norm < best.obj_norm && !nearly_eq(cand.obj_norm, best.obj_norm)) return true;
    if (nearly_eq(cand.obj_norm, best.obj_norm) && cand.p95_rel_obj < best.p95_rel_obj) return true;
    if (nearly_eq(cand.obj_norm, best.obj_norm) && nearly_eq(cand.p95_rel_obj, best.p95_rel_obj) && cand.tail_rel_obj < best.tail_rel_obj) return true;
    if (nearly_eq(cand.obj_norm, best.obj_norm) && nearly_eq(cand.p95_rel_obj, best.p95_rel_obj) && nearly_eq(cand.tail_rel_obj, best.tail_rel_obj) && cand.max_rel_obj < best.max_rel_obj) return true;
    if (nearly_eq(cand.obj_norm, best.obj_norm) && nearly_eq(cand.p95_rel_obj, best.p95_rel_obj) && nearly_eq(cand.tail_rel_obj, best.tail_rel_obj) && nearly_eq(cand.max_rel_obj, best.max_rel_obj) && cand.abs_mean_err_rel < best.abs_mean_err_rel) return true;
    return false;
}

struct nvfp4_cuda_eval_request {
    float a;
    float b;
    float scale_mul;
    nvfp4_cuda_runtime_cfg cfg;
};

static bool nvfp4_cuda_eval_stats_device_batch(
        nvfp4_cuda_autotune_tls & tls,
        const float * d_x,
        const float * d_qw,
        const block_nvfp4 * d_y,
        const float * d_x_scales,
        int64_t sample_nb,
        int n_batch,
        nvfp4_tune_eval_stats * stats,
        cudaStream_t stream);

static int nvfp4_cuda_autotune_worker_count(int64_t work_items) {
    if (work_items <= 1) {
        return 1;
    }

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int64_t configured = g_nvfp4_cuda_autotune_threads.load(std::memory_order_acquire);
    if (configured <= 0) {
        configured = std::min<int64_t>(4, (int64_t) hw);
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            const size_t gib = (size_t) 1024 * 1024 * 1024;
            if (total_bytes >= 24 * gib && free_bytes >= 8 * gib) {
                configured = std::max<int64_t>(configured, 4);
            } else if (total_bytes >= 16 * gib && free_bytes >= 6 * gib) {
                configured = std::max<int64_t>(configured, 2);
            }
        } else {
            cudaGetLastError();
        }
    }
    configured = std::min<int64_t>(configured, 4);
    return (int) std::max<int64_t>(1, std::min<int64_t>(work_items, configured));
}

static cudaStream_t nvfp4_cuda_worker_stream(cudaStream_t fallback_stream, bool private_stream) {
    if (!private_stream) {
        return fallback_stream ? fallback_stream : (cudaStream_t) 0;
    }

    if (!g_nvfp4_cuda_stream_tls.initialized) {
        g_nvfp4_cuda_stream_tls.initialized = true;
        const cudaError_t err = cudaStreamCreateWithFlags(&g_nvfp4_cuda_stream_tls.stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            nvfp4_cuda_log_failure("autotune stream create", err);
            cudaGetLastError();
            g_nvfp4_cuda_stream_tls.stream = nullptr;
        }
    }

    return g_nvfp4_cuda_stream_tls.stream ? g_nvfp4_cuda_stream_tls.stream : (fallback_stream ? fallback_stream : (cudaStream_t) 0);
}

struct nvfp4_cuda_sample_cache_handle {
    float * d_x = nullptr;
    float * d_tune_x = nullptr;
    float * d_qw = nullptr;
    int64_t n = 0;
};

static __device__ __forceinline__ int64_t nvfp4_cuda_selector_sample_block_index(
        int64_t is,
        int64_t sample_nb,
        int64_t nb_total,
        int64_t row_blocks,
        int64_t sample_phase) {
    if (nb_total <= 1 || sample_nb <= 1) {
        return 0;
    }
    const int64_t phase_shift = sample_phase > 0 ? (sample_phase % nb_total) : 0;
    if (row_blocks > 1 && nb_total > row_blocks && sample_nb >= 8) {
        const int64_t global_nb = sample_nb / 2;
        if (is < global_nb) {
            const int64_t base = (is * (nb_total - 1)) / max((int64_t) 1, global_nb - 1);
            return (base + phase_shift) % nb_total;
        }
        const int64_t nrows = nb_total / row_blocks;
        const int64_t rowwise_nb = sample_nb - global_nb;
        const int64_t row_group = max((int64_t) 1, (rowwise_nb + 3) / 4);
        const int64_t js = is - global_nb;
        const int64_t row_slot = js / 4;
        const int64_t phase = (js + phase_shift) % 4;
        const int64_t row_base = row_group <= 1 ? 0 : (row_slot * (nrows - 1)) / (row_group - 1);
        const int64_t row = nrows > 0 ? (row_base + (phase_shift % max((int64_t) 1, nrows))) % nrows : 0;

        int64_t block_in_row = 0;
        switch ((int) phase) {
            case 0: block_in_row = 0; break;
            case 1: block_in_row = row_blocks / 3; break;
            case 2: block_in_row = (2 * row_blocks) / 3; break;
            default: block_in_row = row_blocks - 1; break;
        }
        return min(nb_total - 1, row * row_blocks + min(row_blocks - 1, block_in_row));
    }
    return ((is * (nb_total - 1)) / (sample_nb - 1) + phase_shift) % nb_total;
}

static __device__ __forceinline__ int64_t nvfp4_cuda_linear_sample_block_index(
        int64_t is,
        int64_t sample_nb,
        int64_t nb_total,
        int64_t phase) {
    if (nb_total <= 1 || sample_nb <= 1) {
        return 0;
    }
    const int64_t ib = (is * (nb_total - 1)) / (sample_nb - 1);
    const int64_t phase_shift = phase > 0 ? (phase % nb_total) : 0;
    return (ib + phase_shift) % nb_total;
}

static __device__ __forceinline__ float nvfp4_cuda_load_sample_src(
        const void * src,
        int32_t src_type,
        int64_t idx) {
    if (src_type == GGML_TYPE_F32) {
        return ((const float *) src)[idx];
    }
    if (src_type == GGML_TYPE_F16) {
        return ggml_cuda_cast<float>(((const half *) src)[idx]);
    }
    if (src_type == GGML_TYPE_BF16) {
        return ggml_cuda_cast<float>(((const nv_bfloat16 *) src)[idx]);
    }
    return 0.0f;
}

static __global__ void ggml_cuda_nvfp4_selector_sample_gather_kernel(
        const void * __restrict__ src,
        int32_t src_type,
        const float * __restrict__ qw,
        int64_t n_per_row,
        int64_t nb_total,
        int64_t row_blocks,
        int64_t sample_nb,
        int64_t sample_phase,
        float tune_x_mul,
        float * __restrict__ out_x,
        float * __restrict__ out_tune_x,
        float * __restrict__ out_qw) {
    const int64_t is = (int64_t) blockIdx.x;
    const int lane = (int) threadIdx.x;
    if (is >= sample_nb || lane >= QK_NVFP4) {
        return;
    }
    const int64_t src_block = nvfp4_cuda_selector_sample_block_index(is, sample_nb, nb_total, row_blocks, sample_phase);
    const int64_t src_off = src_block * QK_NVFP4 + lane;
    const int64_t dst_off = is * QK_NVFP4 + lane;
    const float v = nvfp4_cuda_load_sample_src(src, src_type, src_off);
    out_x[dst_off] = v;
    if (out_tune_x != nullptr) {
        out_tune_x[dst_off] = v * tune_x_mul;
    }
    if (out_qw != nullptr && qw != nullptr) {
        const int64_t block_in_row = row_blocks > 0 ? src_block % row_blocks : 0;
        out_qw[dst_off] = qw[block_in_row * QK_NVFP4 + lane];
    }
    (void) n_per_row;
}

static __global__ void ggml_cuda_nvfp4_linear_sample_gather_kernel(
        const float * __restrict__ src,
        const float * __restrict__ qw,
        int64_t nb_total,
        int64_t sample_nb,
        int64_t phase,
        float * __restrict__ out_x,
        float * __restrict__ out_qw) {
    const int64_t is = (int64_t) blockIdx.x;
    const int lane = (int) threadIdx.x;
    if (is >= sample_nb || lane >= QK_NVFP4) {
        return;
    }
    const int64_t src_block = nvfp4_cuda_linear_sample_block_index(is, sample_nb, nb_total, phase);
    const int64_t src_off = src_block * QK_NVFP4 + lane;
    const int64_t dst_off = is * QK_NVFP4 + lane;
    out_x[dst_off] = src[src_off];
    if (out_qw != nullptr && qw != nullptr) {
        out_qw[dst_off] = qw[src_off];
    }
}

static size_t nvfp4_cuda_type_size(int32_t type) {
    switch ((ggml_type) type) {
        case GGML_TYPE_F32:  return sizeof(float);
        case GGML_TYPE_F16:  return sizeof(ggml_fp16_t);
        case GGML_TYPE_BF16: return sizeof(ggml_bf16_t);
        default:            return 0;
    }
}

static bool nvfp4_cuda_prepare_device_block_sample(
        nvfp4_cuda_autotune_tls & tls,
        const float * d_x,
        const float * d_qw,
        int64_t nb_total,
        int64_t sample_nb,
        int64_t phase,
        bool use_alt_buf,
        const float ** d_x_out,
        const float ** d_qw_out,
        cudaStream_t stream) {
    if (d_x_out == nullptr || d_qw_out == nullptr || d_x == nullptr || nb_total <= 0 || sample_nb <= 0) {
        return false;
    }
    if (sample_nb >= nb_total) {
        *d_x_out = d_x;
        *d_qw_out = d_qw;
        return true;
    }

    const size_t bytes_x = (size_t) sample_nb * QK_NVFP4 * sizeof(float);
    const size_t bytes_qw = d_qw != nullptr ? bytes_x : 0;
    float *& d_x_buf = use_alt_buf ? tls.d_x_sample_alt_buf : tls.d_x_sample_buf;
    size_t & d_x_cap = use_alt_buf ? tls.d_x_sample_alt_cap : tls.d_x_sample_cap;
    float *& d_qw_buf = use_alt_buf ? tls.d_qw_sample_alt_buf : tls.d_qw_sample_buf;
    size_t & d_qw_cap = use_alt_buf ? tls.d_qw_sample_alt_cap : tls.d_qw_sample_cap;
    if (!ggml_cuda_nvfp4_ensure_buf((void **) &d_x_buf, &d_x_cap, bytes_x, "autotune cudaMalloc(device sample x)") ||
        (bytes_qw != 0 && !ggml_cuda_nvfp4_ensure_buf((void **) &d_qw_buf, &d_qw_cap, bytes_qw, "autotune cudaMalloc(device sample qw)"))) {
        return false;
    }

    ggml_cuda_nvfp4_linear_sample_gather_kernel<<<(unsigned int) sample_nb, QK_NVFP4, 0, stream>>>(
        d_x, d_qw, nb_total, sample_nb, phase, d_x_buf, bytes_qw != 0 ? d_qw_buf : nullptr);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("autotune kernel(device sample gather)", err);
        cudaGetLastError();
        return false;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("autotune sync(device sample gather)", err);
        cudaGetLastError();
        return false;
    }

    *d_x_out = d_x_buf;
    *d_qw_out = bytes_qw != 0 ? d_qw_buf : nullptr;
    return true;
}

extern "C" bool ggml_cuda_nvfp4_sample_cache_create(
        const void * x,
        int32_t x_type,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        int64_t sample_nb,
        int64_t sample_phase,
        float tune_x_mul,
        void ** cache,
        const float ** x_device,
        const float ** tune_x_device,
        const float ** qw_device,
        int64_t * n_device,
        cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    (void) x; (void) x_type; (void) nrow; (void) n_per_row; (void) qw;
    (void) sample_nb; (void) sample_phase; (void) tune_x_mul; (void) cache;
    (void) x_device; (void) tune_x_device; (void) qw_device; (void) n_device; (void) stream;
    return false;
#else
    if (cache == nullptr || x_device == nullptr || tune_x_device == nullptr || qw_device == nullptr || n_device == nullptr ||
            x == nullptr || nrow <= 0 || n_per_row <= 0 || (n_per_row % QK_NVFP4) != 0 || sample_nb <= 0) {
        return false;
    }
    *cache = nullptr;
    *x_device = nullptr;
    *tune_x_device = nullptr;
    *qw_device = nullptr;
    *n_device = 0;

    const size_t type_size = nvfp4_cuda_type_size(x_type);
    if (type_size == 0) {
        return false;
    }
    const int64_t row_blocks = n_per_row / QK_NVFP4;
    const int64_t nb_total = nrow * row_blocks;
    if (sample_nb > nb_total) {
        return false;
    }

    cudaStream_t st = stream ? stream : 0;
    auto * handle = new nvfp4_cuda_sample_cache_handle();
    void * d_src = nullptr;
    float * d_qw_row = nullptr;

    auto cleanup = [&]() {
        if (d_src != nullptr) {
            cudaFree(d_src);
            d_src = nullptr;
        }
        if (d_qw_row != nullptr) {
            cudaFree(d_qw_row);
            d_qw_row = nullptr;
        }
        ggml_cuda_nvfp4_sample_cache_free(handle);
    };

    const size_t bytes_src = (size_t) nrow * (size_t) n_per_row * type_size;
    const size_t bytes_sample = (size_t) sample_nb * QK_NVFP4 * sizeof(float);
    cudaError_t err = cudaMalloc(&d_src, bytes_src);
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("sample cache cudaMalloc(src)", err);
        cudaGetLastError();
        cleanup();
        return false;
    }
    err = cudaMalloc((void **) &handle->d_x, bytes_sample);
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("sample cache cudaMalloc(x)", err);
        cudaGetLastError();
        cleanup();
        return false;
    }
    const bool need_tune_x = std::isfinite(tune_x_mul) && tune_x_mul > 0.0f && fabsf(tune_x_mul - 1.0f) > 1e-12f;
    if (need_tune_x) {
        err = cudaMalloc((void **) &handle->d_tune_x, bytes_sample);
        if (err != cudaSuccess) {
            nvfp4_cuda_log_failure("sample cache cudaMalloc(tune x)", err);
            cudaGetLastError();
            cleanup();
            return false;
        }
    }
    if (qw != nullptr) {
        err = cudaMalloc((void **) &d_qw_row, (size_t) n_per_row * sizeof(float));
        if (err != cudaSuccess) {
            nvfp4_cuda_log_failure("sample cache cudaMalloc(qw row)", err);
            cudaGetLastError();
            cleanup();
            return false;
        }
        err = cudaMalloc((void **) &handle->d_qw, bytes_sample);
        if (err != cudaSuccess) {
            nvfp4_cuda_log_failure("sample cache cudaMalloc(qw)", err);
            cudaGetLastError();
            cleanup();
            return false;
        }
    }

    err = cudaMemcpyAsync(d_src, x, bytes_src, cudaMemcpyHostToDevice, st);
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("sample cache H2D(src)", err);
        cudaGetLastError();
        cleanup();
        return false;
    }
    if (qw != nullptr) {
        err = cudaMemcpyAsync(d_qw_row, qw, (size_t) n_per_row * sizeof(float), cudaMemcpyHostToDevice, st);
        if (err != cudaSuccess) {
            nvfp4_cuda_log_failure("sample cache H2D(qw)", err);
            cudaGetLastError();
            cleanup();
            return false;
        }
    }

    ggml_cuda_nvfp4_selector_sample_gather_kernel<<<(unsigned int) sample_nb, QK_NVFP4, 0, st>>>(
        d_src, x_type, d_qw_row, n_per_row, nb_total, row_blocks, sample_nb, sample_phase,
        need_tune_x ? tune_x_mul : 1.0f,
        handle->d_x, need_tune_x ? handle->d_tune_x : nullptr, handle->d_qw);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("sample cache kernel(gather)", err);
        cudaGetLastError();
        cleanup();
        return false;
    }
    err = cudaStreamSynchronize(st);
    if (err != cudaSuccess) {
        nvfp4_cuda_log_failure("sample cache sync(gather)", err);
        cudaGetLastError();
        cleanup();
        return false;
    }

    cudaFree(d_src);
    d_src = nullptr;
    if (d_qw_row != nullptr) {
        cudaFree(d_qw_row);
        d_qw_row = nullptr;
    }

    handle->n = sample_nb * QK_NVFP4;
    *cache = handle;
    *x_device = handle->d_x;
    *tune_x_device = need_tune_x ? handle->d_tune_x : handle->d_x;
    *qw_device = handle->d_qw;
    *n_device = handle->n;
    return true;
#endif
}

extern "C" void ggml_cuda_nvfp4_sample_cache_free(void * cache) {
    auto * handle = (nvfp4_cuda_sample_cache_handle *) cache;
    if (handle == nullptr) {
        return;
    }
    if (handle->d_x != nullptr) {
        cudaFree(handle->d_x);
    }
    if (handle->d_tune_x != nullptr) {
        cudaFree(handle->d_tune_x);
    }
    if (handle->d_qw != nullptr) {
        cudaFree(handle->d_qw);
    }
    delete handle;
}

static bool nvfp4_cuda_eval_requests_parallel(
        const float * sample_x,
        const float * sample_qw,
        int64_t sample_nb,
        const nvfp4_cuda_eval_request * requests,
        int n_requests,
        std::vector<nvfp4_tune_eval_stats> & out_stats,
        std::vector<uint8_t> * out_ok,
        cudaStream_t fallback_stream) {
    if (sample_x == nullptr || sample_nb <= 0 || requests == nullptr || n_requests <= 0) {
        return false;
    }

    out_stats.assign((size_t) n_requests, nvfp4_tune_eval_stats{});
    if (out_ok != nullptr) {
        out_ok->assign((size_t) n_requests, 0);
    }

    constexpr int request_batch_cap = 4;
    const int workers = nvfp4_cuda_autotune_worker_count(n_requests);
    const int request_batch_size = std::max(1, std::min(request_batch_cap, (n_requests + workers - 1) / workers));
    const bool sample_x_device = ggml_cuda_nvfp4_device_pointer(sample_x);
    const bool sample_qw_device = sample_qw != nullptr && ggml_cuda_nvfp4_device_pointer(sample_qw);
    std::atomic<int> next_request { 0 };
    std::atomic<bool> cuda_failed { false };

    auto eval_worker = [&](bool private_stream) {
        auto & tls = g_nvfp4_cuda_autotune_tls;
        const size_t bytes_x = (size_t) sample_nb * QK_NVFP4 * sizeof(float);
        const size_t bytes_qw = sample_qw ? ((size_t) sample_nb * QK_NVFP4 * sizeof(float)) : 0;
        const int max_batch = std::min(request_batch_size, n_requests);
        const size_t bytes_y = (size_t) max_batch * (size_t) sample_nb * sizeof(block_nvfp4);
        const size_t bytes_cfg = (size_t) max_batch * sizeof(nvfp4_cuda_runtime_cfg);
        const size_t bytes_xscale = (size_t) max_batch * sizeof(float);
        const int64_t row_blocks = sample_nb;

        if ((!sample_x_device && !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_x_sample_buf, &tls.d_x_sample_cap, bytes_x, "autotune cudaMalloc(x)")) ||
            !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_y_sample_buf, &tls.d_y_sample_cap, bytes_y, "autotune cudaMalloc(y)") ||
            !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_eval_cfg_buf, &tls.d_eval_cfg_cap, bytes_cfg, "autotune cudaMalloc(cfg)") ||
            !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_eval_xscale_buf, &tls.d_eval_xscale_cap, bytes_xscale, "autotune cudaMalloc(xscale)") ||
            (bytes_qw != 0 && !sample_qw_device && !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_qw_sample_buf, &tls.d_qw_sample_cap, bytes_qw, "autotune cudaMalloc(qw)"))) {
            cuda_failed.store(true, std::memory_order_release);
            return;
        }

        cudaStream_t st = nvfp4_cuda_worker_stream(fallback_stream, private_stream);
        nvfp4_cuda_runtime_cfg cfg_batch[request_batch_cap];
        float xscale_batch[request_batch_cap];
        nvfp4_tune_eval_stats stats_batch[request_batch_cap];
        const float * d_x_sample = sample_x_device ? sample_x : tls.d_x_sample_buf;
        const float * d_qw_sample = sample_qw == nullptr ? nullptr : (sample_qw_device ? sample_qw : tls.d_qw_sample_buf);

        auto load_sample = [&]() -> bool {
            if (sample_x_device && (sample_qw == nullptr || sample_qw_device)) {
                return true;
            }
            if (tls.loaded_x == sample_x && tls.loaded_qw == sample_qw && tls.loaded_nb == sample_nb) {
                return true;
            }

            cudaError_t err = cudaSuccess;
            if (!sample_x_device) {
                err = cudaMemcpyAsync(tls.d_x_sample_buf, sample_x, bytes_x, cudaMemcpyHostToDevice, st);
                if (err != cudaSuccess) {
                    nvfp4_cuda_log_failure("autotune H2D(x)", err);
                    cudaGetLastError();
                    tls.loaded_x = nullptr;
                    tls.loaded_qw = nullptr;
                    tls.loaded_nb = 0;
                    cuda_failed.store(true, std::memory_order_release);
                    return false;
                }
            }

            if (sample_qw != nullptr && !sample_qw_device) {
                err = cudaMemcpyAsync(tls.d_qw_sample_buf, sample_qw, bytes_qw, cudaMemcpyHostToDevice, st);
                if (err != cudaSuccess) {
                    nvfp4_cuda_log_failure("autotune H2D(qw)", err);
                    cudaGetLastError();
                    tls.loaded_x = nullptr;
                    tls.loaded_qw = nullptr;
                    tls.loaded_nb = 0;
                    cuda_failed.store(true, std::memory_order_release);
                    return false;
                }
            }

            tls.loaded_x = sample_x;
            tls.loaded_qw = sample_qw;
            tls.loaded_nb = sample_nb;
            return true;
        };

        while (true) {
            if (cuda_failed.load(std::memory_order_acquire)) {
                break;
            }
            const int req_idx = next_request.fetch_add(max_batch, std::memory_order_relaxed);
            if (req_idx >= n_requests) {
                break;
            }
            const int n_batch = std::min(max_batch, n_requests - req_idx);

            if (!load_sample()) {
                continue;
            }

            for (int bi = 0; bi < n_batch; ++bi) {
                cfg_batch[bi] = requests[req_idx + bi].cfg;
                const float scale_mul = requests[req_idx + bi].scale_mul;
                xscale_batch[bi] = (std::isfinite(scale_mul) && scale_mul > 0.0f) ? scale_mul : 1.0f;
                stats_batch[bi] = nvfp4_tune_eval_stats{};
            }

            cudaError_t err = cudaMemcpyAsync(tls.d_eval_cfg_buf, cfg_batch,
                (size_t) n_batch * sizeof(nvfp4_cuda_runtime_cfg), cudaMemcpyHostToDevice, st);
            if (err != cudaSuccess) {
                nvfp4_cuda_log_failure("autotune H2D(cfg)", err);
                cudaGetLastError();
                tls.loaded_x = nullptr;
                tls.loaded_qw = nullptr;
                tls.loaded_nb = 0;
                cuda_failed.store(true, std::memory_order_release);
                continue;
            }
            err = cudaMemcpyAsync(tls.d_eval_xscale_buf, xscale_batch,
                (size_t) n_batch * sizeof(float), cudaMemcpyHostToDevice, st);
            if (err != cudaSuccess) {
                nvfp4_cuda_log_failure("autotune H2D(xscale)", err);
                cudaGetLastError();
                tls.loaded_x = nullptr;
                tls.loaded_qw = nullptr;
                tls.loaded_nb = 0;
                cuda_failed.store(true, std::memory_order_release);
                continue;
            }

            if (!ggml_cuda_nvfp4_launch_kernel_batched(
                    d_x_sample, false, 1.0f, tls.d_eval_xscale_buf, d_qw_sample, tls.d_y_sample_buf,
                    row_blocks, sample_nb, tls.d_eval_cfg_buf, n_batch, st)) {
                tls.loaded_x = nullptr;
                tls.loaded_qw = nullptr;
                tls.loaded_nb = 0;
                cuda_failed.store(true, std::memory_order_release);
                continue;
            }

            if (!nvfp4_cuda_eval_stats_device_batch(
                    tls, d_x_sample, d_qw_sample, tls.d_y_sample_buf, tls.d_eval_xscale_buf,
                    sample_nb, n_batch, stats_batch, st)) {
                tls.loaded_x = nullptr;
                tls.loaded_qw = nullptr;
                tls.loaded_nb = 0;
                cuda_failed.store(true, std::memory_order_release);
                continue;
            }

            for (int bi = 0; bi < n_batch; ++bi) {
                const nvfp4_tune_eval_stats & stats = stats_batch[bi];
                if (!std::isfinite(stats.obj_norm) || !std::isfinite(stats.p95_rel_obj) ||
                    !std::isfinite(stats.tail_rel_obj) || !std::isfinite(stats.max_rel_obj) ||
                    !std::isfinite(stats.abs_mean_err_rel)) {
                    continue;
                }

                const size_t out_idx = (size_t) req_idx + (size_t) bi;
                out_stats[out_idx] = stats;
                if (out_ok != nullptr) {
                    (*out_ok)[out_idx] = 1;
                }
            }
        }
    };

    if (workers <= 1) {
        eval_worker(false);
    } else {
        std::vector<std::thread> worker_threads;
        worker_threads.reserve((size_t) workers);
        for (int wi = 0; wi < workers; ++wi) {
            worker_threads.emplace_back(eval_worker, true);
        }
        for (auto & th : worker_threads) {
            th.join();
        }
    }

    return !cuda_failed.load(std::memory_order_acquire);
}

static __device__ __forceinline__ uint8_t nvfp4_cuda_code_at(const block_nvfp4 & block, int elem) {
    const int sub = elem / QK_NVFP4_SUB;
    const int lane = elem % QK_NVFP4_SUB;
    const uint8_t packed = block.qs[sub * (QK_NVFP4_SUB / 2) + (lane & 0x7)];
    return lane < (QK_NVFP4_SUB / 2) ? (packed & 0x0F) : (packed >> 4);
}

static __global__ void ggml_cuda_nvfp4_tune_eval_blocks(
        const float * __restrict__ x,
        const float * __restrict__ qw,
        const block_nvfp4 * __restrict__ y,
        const float * __restrict__ x_scales,
        int64_t nb,
        nvfp4_cuda_tune_eval_accum * __restrict__ out,
        double * __restrict__ rel_obj) {
    constexpr int THREADS = QK_NVFP4;
    __shared__ double s_obj[THREADS];
    __shared__ double s_x2[THREADS];
    __shared__ double s_abs_err[THREADS];
    __shared__ double s_abs_x[THREADS];

    const int64_t ib = (int64_t) blockIdx.x;
    const int tid = (int) threadIdx.x;
    double obj = 0.0;
    double x2 = 0.0;
    double abs_err = 0.0;
    double abs_x = 0.0;
    float x_scale = 1.0f;
    if (x_scales != nullptr) {
        const float xs = x_scales[0];
        if (isfinite(xs) && xs > 0.0f) {
            x_scale = xs;
        }
    }

    if (ib < nb && tid < QK_NVFP4) {
        const float xv_f = x[ib * QK_NVFP4 + tid];
        const double xv = (double) xv_f;
        const double w = qw != nullptr ? (double) fmaxf(0.0f, qw[ib * QK_NVFP4 + tid]) : 1.0;
        const block_nvfp4 block = y[ib];
        const int sub = tid / QK_NVFP4_SUB;
        const float d = nvfp4_cuda_ue4m3_to_fp32(block.d[sub]);
        const uint8_t code = nvfp4_cuda_code_at(block, tid);
        const double q = (double) d * (double) nvfp4_cuda_kvalue(code) * (double) x_scale;
        const double e = xv - q;
        obj = w * e * e;
        x2 = w * xv * xv;
        abs_err = w * fabs(e);
        abs_x = w * fabs(xv);
    }

    s_obj[tid] = obj;
    s_x2[tid] = x2;
    s_abs_err[tid] = abs_err;
    s_abs_x[tid] = abs_x;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_obj[tid] += s_obj[tid + stride];
            s_x2[tid] += s_x2[tid + stride];
            s_abs_err[tid] += s_abs_err[tid + stride];
            s_abs_x[tid] += s_abs_x[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && ib < nb) {
        constexpr double eps = 1e-30;
        const double rel = s_obj[0] / fmax(s_x2[0], eps);
        rel_obj[ib] = rel;
        atomicAdd(&out->sum_obj, s_obj[0]);
        atomicAdd(&out->sum_x2, s_x2[0]);
        atomicAdd(&out->sum_abs_err, s_abs_err[0]);
        atomicAdd(&out->sum_abs_x, s_abs_x[0]);
    }
}

static __global__ void ggml_cuda_nvfp4_tune_eval_blocks_batched(
        const float * __restrict__ x,
        const float * __restrict__ qw,
        const block_nvfp4 * __restrict__ y,
        const float * __restrict__ x_scales,
        int64_t nb,
        int n_batch,
        nvfp4_cuda_tune_eval_accum * __restrict__ out,
        double * __restrict__ rel_obj) {
    constexpr int THREADS = QK_NVFP4;
    __shared__ double s_obj[THREADS];
    __shared__ double s_x2[THREADS];
    __shared__ double s_abs_err[THREADS];
    __shared__ double s_abs_x[THREADS];

    const int64_t ib = (int64_t) blockIdx.x;
    const int cand = (int) blockIdx.y;
    const int tid = (int) threadIdx.x;
    double obj = 0.0;
    double x2 = 0.0;
    double abs_err = 0.0;
    double abs_x = 0.0;
    float x_scale = 1.0f;
    if (x_scales != nullptr && cand < n_batch) {
        const float xs = x_scales[cand];
        if (isfinite(xs) && xs > 0.0f) {
            x_scale = xs;
        }
    }

    if (cand < n_batch && ib < nb && tid < QK_NVFP4) {
        const float xv_f = x[ib * QK_NVFP4 + tid];
        const double xv = (double) xv_f;
        const double w = qw != nullptr ? (double) fmaxf(0.0f, qw[ib * QK_NVFP4 + tid]) : 1.0;
        const block_nvfp4 block = y[(size_t) cand * (size_t) nb + (size_t) ib];
        const int sub = tid / QK_NVFP4_SUB;
        const float d = nvfp4_cuda_ue4m3_to_fp32(block.d[sub]);
        const uint8_t code = nvfp4_cuda_code_at(block, tid);
        const double q = (double) d * (double) nvfp4_cuda_kvalue(code) * (double) x_scale;
        const double e = xv - q;
        obj = w * e * e;
        x2 = w * xv * xv;
        abs_err = w * fabs(e);
        abs_x = w * fabs(xv);
    }

    s_obj[tid] = obj;
    s_x2[tid] = x2;
    s_abs_err[tid] = abs_err;
    s_abs_x[tid] = abs_x;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_obj[tid] += s_obj[tid + stride];
            s_x2[tid] += s_x2[tid + stride];
            s_abs_err[tid] += s_abs_err[tid + stride];
            s_abs_x[tid] += s_abs_x[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && cand < n_batch && ib < nb) {
        constexpr double eps = 1e-30;
        const double rel = s_obj[0] / fmax(s_x2[0], eps);
        rel_obj[(size_t) cand * (size_t) nb + (size_t) ib] = rel;
        atomicAdd(&out[cand].sum_obj, s_obj[0]);
        atomicAdd(&out[cand].sum_x2, s_x2[0]);
        atomicAdd(&out[cand].sum_abs_err, s_abs_err[0]);
        atomicAdd(&out[cand].sum_abs_x, s_abs_x[0]);
    }
}

static __global__ void ggml_cuda_nvfp4_tune_eval_tail(
        const double * __restrict__ rel_sorted,
        int64_t nb,
        int64_t p95_idx,
        nvfp4_cuda_tune_eval_accum * __restrict__ out) {
    constexpr int THREADS = 256;
    __shared__ double s_tail2[THREADS];
    __shared__ unsigned long long s_tail_count[THREADS];

    const int tid = (int) threadIdx.x;
    const double p95 = rel_sorted[p95_idx];
    double tail2 = 0.0;
    unsigned long long tail_count = 0;
    for (int64_t i = tid; i < nb; i += THREADS) {
        const double rel = rel_sorted[i];
        if (rel >= p95) {
            tail2 += rel * rel;
            ++tail_count;
        }
    }

    s_tail2[tid] = tail2;
    s_tail_count[tid] = tail_count;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_tail2[tid] += s_tail2[tid + stride];
            s_tail_count[tid] += s_tail_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out->p95_rel_obj = p95;
        out->sum_tail2 = s_tail2[0];
        out->tail_count = s_tail_count[0];
        out->max_rel_obj = rel_sorted[nb - 1];
    }
}

static __global__ void ggml_cuda_nvfp4_tune_eval_tail_batched(
        const double * __restrict__ rel_sorted,
        int64_t nb,
        int64_t p95_idx,
        int n_batch,
        nvfp4_cuda_tune_eval_accum * __restrict__ out) {
    constexpr int THREADS = 256;
    __shared__ double s_tail2[THREADS];
    __shared__ unsigned long long s_tail_count[THREADS];

    const int cand = (int) blockIdx.x;
    const int tid = (int) threadIdx.x;
    if (cand >= n_batch) {
        return;
    }

    const double * rel = rel_sorted + (size_t) cand * (size_t) nb;
    const double p95 = rel[p95_idx];
    double tail2 = 0.0;
    unsigned long long tail_count = 0;
    for (int64_t i = tid; i < nb; i += THREADS) {
        const double v = rel[i];
        if (v >= p95) {
            tail2 += v * v;
            ++tail_count;
        }
    }

    s_tail2[tid] = tail2;
    s_tail_count[tid] = tail_count;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_tail2[tid] += s_tail2[tid + stride];
            s_tail_count[tid] += s_tail_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[cand].p95_rel_obj = p95;
        out[cand].sum_tail2 = s_tail2[0];
        out[cand].tail_count = s_tail_count[0];
        out[cand].max_rel_obj = rel[nb - 1];
    }
}

static bool nvfp4_cuda_eval_stats_device_batch(
        nvfp4_cuda_autotune_tls & tls,
        const float * d_x,
        const float * d_qw,
        const block_nvfp4 * d_y,
        const float * d_x_scales,
        int64_t sample_nb,
        int n_batch,
        nvfp4_tune_eval_stats * stats,
        cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    (void) tls; (void) d_x; (void) d_qw; (void) d_y; (void) d_x_scales; (void) sample_nb; (void) n_batch; (void) stats; (void) stream;
    return false;
#else
    if (d_x == nullptr || d_y == nullptr || stats == nullptr || sample_nb <= 0 || sample_nb > INT_MAX || n_batch <= 0) {
        return false;
    }

    const size_t bytes_rel = (size_t) n_batch * (size_t) sample_nb * sizeof(double);
    const size_t bytes_eval = (size_t) n_batch * sizeof(nvfp4_cuda_tune_eval_accum);
    if (!ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_rel_obj_buf, &tls.d_rel_obj_cap, bytes_rel, "autotune cudaMalloc(rel)") ||
        !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_rel_obj_sorted_buf, &tls.d_rel_obj_sorted_cap, bytes_rel, "autotune cudaMalloc(rel sorted)") ||
        !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_tune_eval_buf, &tls.d_tune_eval_cap, bytes_eval, "autotune cudaMalloc(eval stats)")) {
        return false;
    }

    auto fail_eval_stream = [&](const char * label, cudaError_t status) {
        nvfp4_cuda_log_failure(label, status);
        (void) cudaStreamSynchronize(stream);
        cudaGetLastError();
        return false;
    };

    cudaError_t err = cudaMemsetAsync(tls.d_tune_eval_buf, 0, bytes_eval, stream);
    if (err != cudaSuccess) {
        return fail_eval_stream("autotune memset(eval stats)", err);
    }

    if (n_batch == 1) {
        ggml_cuda_nvfp4_tune_eval_blocks<<<(unsigned int) sample_nb, QK_NVFP4, 0, stream>>>(
            d_x, d_qw, d_y, d_x_scales, sample_nb, tls.d_tune_eval_buf, tls.d_rel_obj_buf);
    } else {
        const dim3 eval_grid((unsigned int) sample_nb, (unsigned int) n_batch, 1);
        ggml_cuda_nvfp4_tune_eval_blocks_batched<<<eval_grid, QK_NVFP4, 0, stream>>>(
            d_x, d_qw, d_y, d_x_scales, sample_nb, n_batch, tls.d_tune_eval_buf, tls.d_rel_obj_buf);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail_eval_stream("autotune kernel(eval blocks)", err);
    }

    size_t sort_temp_bytes = 0;
    err = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_temp_bytes,
        tls.d_rel_obj_buf, tls.d_rel_obj_sorted_buf,
        (int) sample_nb,
        0, (int) (8 * sizeof(double)),
        stream);
    if (err != cudaSuccess) {
        return fail_eval_stream("autotune sort query", err);
    }
    if (!ggml_cuda_nvfp4_ensure_buf(&tls.d_sort_temp_buf, &tls.d_sort_temp_cap, sort_temp_bytes, "autotune cudaMalloc(sort temp)")) {
        return false;
    }

    for (int bi = 0; bi < n_batch; ++bi) {
        double * rel = tls.d_rel_obj_buf + (size_t) bi * (size_t) sample_nb;
        double * rel_sorted = tls.d_rel_obj_sorted_buf + (size_t) bi * (size_t) sample_nb;
        err = cub::DeviceRadixSort::SortKeys(
            tls.d_sort_temp_buf, sort_temp_bytes,
            rel, rel_sorted,
            (int) sample_nb,
            0, (int) (8 * sizeof(double)),
            stream);
        if (err != cudaSuccess) {
            return fail_eval_stream("autotune sort rel", err);
        }
    }

    const int64_t p95_idx = sample_nb > 1 ? (95 * (sample_nb - 1)) / 100 : 0;
    if (n_batch == 1) {
        ggml_cuda_nvfp4_tune_eval_tail<<<1, 256, 0, stream>>>(
            tls.d_rel_obj_sorted_buf, sample_nb, p95_idx, tls.d_tune_eval_buf);
    } else {
        ggml_cuda_nvfp4_tune_eval_tail_batched<<<(unsigned int) n_batch, 256, 0, stream>>>(
            tls.d_rel_obj_sorted_buf, sample_nb, p95_idx, n_batch, tls.d_tune_eval_buf);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail_eval_stream("autotune kernel(eval tail)", err);
    }

    std::vector<nvfp4_cuda_tune_eval_accum> eval_host((size_t) n_batch);
    err = cudaMemcpyAsync(eval_host.data(), tls.d_tune_eval_buf, bytes_eval, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return fail_eval_stream("autotune D2H(eval stats)", err);
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return fail_eval_stream("autotune sync(eval stats)", err);
    }

    constexpr double eps = 1e-30;
    for (int bi = 0; bi < n_batch; ++bi) {
        const nvfp4_cuda_tune_eval_accum & ev = eval_host[(size_t) bi];
        stats[bi].obj_norm = ev.sum_obj / std::max(ev.sum_x2, eps);
        stats[bi].abs_mean_err_rel = ev.sum_abs_err / std::max(ev.sum_abs_x, eps);
        stats[bi].p95_rel_obj = ev.p95_rel_obj;
        stats[bi].tail_rel_obj = ev.tail_count > 0
            ? std::sqrt(ev.sum_tail2 / (double) ev.tail_count)
            : ev.p95_rel_obj;
        stats[bi].max_rel_obj = ev.max_rel_obj;
    }
    return true;
#endif
}

static __global__ void ggml_cuda_nvfp4_eval_quantized(
        const void * x,
        bool x_bf16,
        float x_scale,
        const block_nvfp4 * y,
        int64_t nrow,
        int64_t n_per_row,
        nvfp4_cuda_eval_accum * out) {
    const int64_t total = nrow * n_per_row;
    const int64_t row_blocks = n_per_row / QK_NVFP4;
    const float x_scale_eff = (isfinite(x_scale) && x_scale > 0.0f) ? x_scale : 1.0f;

    double local_sq = 0.0;
    double local_abs = 0.0;
    float local_max = 0.0f;
    unsigned long long local_count = 0;

    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t) gridDim.x * blockDim.x) {
        const int64_t row = idx / n_per_row;
        const int col = (int) (idx - row * n_per_row);
        const int64_t block_idx = row * row_blocks + col / QK_NVFP4;
        const int elem = col % QK_NVFP4;
        const int sub = elem / QK_NVFP4_SUB;
        const block_nvfp4 block = y[block_idx];

        float xv;
        if (x_bf16) {
            const nv_bfloat16 * xb = (const nv_bfloat16 *) x;
            xv = ggml_cuda_cast<float>(xb[idx]);
        } else {
            const float * xf = (const float *) x;
            xv = xf[idx];
        }

        const float d = nvfp4_cuda_ue4m3_to_fp32(block.d[sub]);
        const uint8_t code = nvfp4_cuda_code_at(block, elem);
        const float q = d * nvfp4_cuda_kvalue(code) * x_scale_eff;
        const float err = q - xv;
        const float abs_err = fabsf(err);

        local_sq += (double) err * (double) err;
        local_abs += (double) abs_err;
        local_max = fmaxf(local_max, abs_err);
        ++local_count;
    }

    if (local_count > 0) {
        atomicAdd(&out->sum_sq, local_sq);
        atomicAdd(&out->sum_abs, local_abs);
        atomicAdd(&out->count, local_count);
        atomicMax(&out->max_abs_bits, __float_as_uint(local_max));
    }
}

static __device__ __forceinline__ uint32_t nvfp4_cuda_pack8_device(const uint8_t * p, int shift) {
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

static __global__ void ggml_cuda_nvfp4_repack_one_plane_kernel(
        const uint8_t * __restrict__ src,
        block_nvfp4_blackwell * __restrict__ dst_blocks,
        int64_t ne0,
        int64_t nrows) {
    const int64_t src_blocks_per_row = ggml_cuda_bw_div_up(ne0, QK_NVFP4);
    const int64_t dst_blocks_per_row = ggml_cuda_nvfp4_blocks_per_row(ne0);
    const int64_t out_index = (int64_t) blockIdx.x;
    const int64_t tile_row = out_index / dst_blocks_per_row;
    const int64_t block_col = out_index - tile_row * dst_blocks_per_row;
    const int64_t row0 = tile_row * 16;
    const int rows_in_tile = (int) ((row0 + 16 <= nrows) ? 16 : (nrows - row0));
    const int64_t src_block0 = block_col * 4;
    const int frags_in_block = (int) ((src_block0 + 4 <= src_blocks_per_row) ? 4 : (src_blocks_per_row - src_block0));
    if (rows_in_tile <= 0 || frags_in_block <= 0) {
        return;
    }

    const size_t src_row_size = (size_t) src_blocks_per_row * sizeof(block_nvfp4);
    block_nvfp4_blackwell * out = dst_blocks + out_index;
    uint32_t * out_words = reinterpret_cast<uint32_t *>(out);
    for (int i = threadIdx.x; i < (int) (sizeof(block_nvfp4_blackwell) / sizeof(uint32_t)); i += blockDim.x) {
        out_words[i] = 0;
    }
    __syncthreads();

    const int tasks = rows_in_tile * frags_in_block;
    for (int task = threadIdx.x; task < tasks; task += blockDim.x) {
        const int row_in_tile = task / frags_in_block;
        const int frag = task - row_in_tile * frags_in_block;
        const int64_t row = row0 + row_in_tile;
        const block_nvfp4 * src_row = reinterpret_cast<const block_nvfp4 *>(src + (size_t) row * src_row_size);
        const block_nvfp4 & in = src_row[src_block0 + frag];
        block_nvfp4_blackwell_frag & tile = out->tiles[frag];

        const int lane_base = (row_in_tile & 7) * 4;
        const int row_half = row_in_tile >> 3;
        const int scale_lane = lane_base + row_half;
        const uint8_t * p0 = in.qs +  0;
        const uint8_t * p1 = in.qs +  8;
        const uint8_t * p2 = in.qs + 16;
        const uint8_t * p3 = in.qs + 24;

        tile.regs[lane_base + 0][row_half + 0] = nvfp4_cuda_pack8_device(p0, 0);
        tile.regs[lane_base + 1][row_half + 0] = nvfp4_cuda_pack8_device(p0, 4);
        tile.regs[lane_base + 2][row_half + 0] = nvfp4_cuda_pack8_device(p1, 0);
        tile.regs[lane_base + 3][row_half + 0] = nvfp4_cuda_pack8_device(p1, 4);
        tile.regs[lane_base + 0][row_half + 2] = nvfp4_cuda_pack8_device(p2, 0);
        tile.regs[lane_base + 1][row_half + 2] = nvfp4_cuda_pack8_device(p2, 4);
        tile.regs[lane_base + 2][row_half + 2] = nvfp4_cuda_pack8_device(p3, 0);
        tile.regs[lane_base + 3][row_half + 2] = nvfp4_cuda_pack8_device(p3, 4);

        const uint32_t d =
            ((uint32_t) in.d[0] <<  0) |
            ((uint32_t) in.d[1] <<  8) |
            ((uint32_t) in.d[2] << 16) |
            ((uint32_t) in.d[3] << 24);
        tile.scales_u32[scale_lane + 0] = d;
        tile.scales_u32[scale_lane + 2] = d;
    }
}

struct nvfp4_cuda_kld_row_result {
    double nll;
    double nll2;
    double nll_base;
    double nll_base2;
    double nll_nll_base;
    double kld;
    double kld2;
    double p_diff2;
    double p_diff4;
    double entropy_diff2;
    double top_prob_diff2;
    double top_flip_weight;
    int same_top;
};

static __device__ __forceinline__ void nvfp4_cuda_kld_result_add_row(
        nvfp4_cuda_kld_result & acc,
        const nvfp4_cuda_kld_row_result & row) {
    acc.sum_nll += row.nll;
    acc.sum_nll2 += row.nll2;
    acc.sum_nll_base += row.nll_base;
    acc.sum_nll_base2 += row.nll_base2;
    acc.sum_nll_nll_base += row.nll_nll_base;
    acc.sum_kld += row.kld;
    acc.sum_kld2 += row.kld2;
    acc.max_kld = fmax(acc.max_kld, row.kld);
    acc.sum_p_diff2 += row.p_diff2;
    acc.sum_p_diff4 += row.p_diff4;
    acc.sum_entropy_diff2 += row.entropy_diff2;
    acc.sum_top_prob_diff2 += row.top_prob_diff2;
    acc.sum_top_flip_weight += row.top_flip_weight;
    acc.same_top += row.same_top ? 1 : 0;
    acc.count += 1;
}

static __device__ __forceinline__ void nvfp4_cuda_kld_result_merge(
        nvfp4_cuda_kld_result & acc,
        const nvfp4_cuda_kld_result & rhs) {
    acc.sum_nll += rhs.sum_nll;
    acc.sum_nll2 += rhs.sum_nll2;
    acc.sum_nll_base += rhs.sum_nll_base;
    acc.sum_nll_base2 += rhs.sum_nll_base2;
    acc.sum_nll_nll_base += rhs.sum_nll_nll_base;
    acc.sum_kld += rhs.sum_kld;
    acc.sum_kld2 += rhs.sum_kld2;
    acc.max_kld = fmax(acc.max_kld, rhs.max_kld);
    acc.sum_p_diff2 += rhs.sum_p_diff2;
    acc.sum_p_diff4 += rhs.sum_p_diff4;
    acc.sum_entropy_diff2 += rhs.sum_entropy_diff2;
    acc.sum_top_prob_diff2 += rhs.sum_top_prob_diff2;
    acc.sum_top_flip_weight += rhs.sum_top_flip_weight;
    acc.same_top += rhs.same_top;
    acc.count += rhs.count;
}

static __global__ void ggml_cuda_nvfp4_kld_reduce_rows_kernel(
        const nvfp4_cuda_kld_row_result * __restrict__ rows,
        int32_t n_eval,
        nvfp4_cuda_kld_result * __restrict__ result) {
    constexpr int KLD_REDUCE_THREADS = 256;
    __shared__ nvfp4_cuda_kld_result s_acc[KLD_REDUCE_THREADS];

    const int tid = (int) threadIdx.x;
    nvfp4_cuda_kld_result acc{};
    for (int i = tid; i < n_eval; i += KLD_REDUCE_THREADS) {
        nvfp4_cuda_kld_result_add_row(acc, rows[i]);
    }
    s_acc[tid] = acc;
    __syncthreads();

    for (int stride = KLD_REDUCE_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            nvfp4_cuda_kld_result_merge(s_acc[tid], s_acc[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = s_acc[0];
    }
}

static __device__ __forceinline__ float nvfp4_cuda_kld_u16_pair_to_float(const uint16_t * p) {
    const uint32_t bits = ((uint32_t) p[0]) | ((uint32_t) p[1] << 16);
    return __uint_as_float(bits);
}

static __global__ void ggml_cuda_nvfp4_kld_rows_kernel(
        const float * __restrict__ logits,
        int64_t logits_nb1,
        int32_t logits_row_offset,
        const uint16_t * __restrict__ base,
        const int32_t * __restrict__ token_ids,
        int32_t n_eval,
        int32_t n_vocab,
        int32_t nv,
        nvfp4_cuda_kld_row_result * __restrict__ rows,
        double * __restrict__ kld_values) {
    constexpr int KLD_THREADS = 256;
    __shared__ float s_max[KLD_THREADS];
    __shared__ int s_idx[KLD_THREADS];
    __shared__ float s_base_max[KLD_THREADS];
    __shared__ int s_base_idx[KLD_THREADS];
    __shared__ double s_sum_exp[KLD_THREADS];
    __shared__ double s_sum_exp_shifted_logit[KLD_THREADS];
    __shared__ double s_kld[KLD_THREADS];
    __shared__ double s_p_base_sum[KLD_THREADS];
    __shared__ double s_base_entropy[KLD_THREADS];

    const int row = (int) blockIdx.x;
    const int tid = (int) threadIdx.x;
    if (row >= n_eval || tid >= KLD_THREADS) {
        return;
    }

    const float * logits_row = reinterpret_cast<const float *>(
        reinterpret_cast<const char *>(logits) + (size_t) (logits_row_offset + row) * (size_t) logits_nb1);
    const uint16_t * base_row = base + (size_t) row * (size_t) nv;
    float scale = nvfp4_cuda_kld_u16_pair_to_float(base_row + 0);
    float min_log_prob = nvfp4_cuda_kld_u16_pair_to_float(base_row + 2);
    if (!(scale >= 0.0f) || !isfinite(scale)) {
        scale = 0.0f;
    }
    if (!isfinite(min_log_prob)) {
        min_log_prob = -16.0f;
    }
    const uint16_t * idx = base_row + 4;

    float local_max = -FLT_MAX;
    int local_idx = 0;
    float local_base_max = -FLT_MAX;
    int local_base_idx = 0;
    for (int i = tid; i < n_vocab; i += KLD_THREADS) {
        const float l = logits_row[i];
        if (l > local_max) {
            local_max = l;
            local_idx = i;
        }
        const float p_log_base = scale * idx[i] + min_log_prob;
        if (p_log_base > local_base_max) {
            local_base_max = p_log_base;
            local_base_idx = i;
        }
    }
    s_max[tid] = local_max;
    s_idx[tid] = local_idx;
    s_base_max[tid] = local_base_max;
    s_base_idx[tid] = local_base_idx;
    __syncthreads();

    for (int stride = KLD_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float cand_max = s_max[tid + stride];
            const int cand_idx = s_idx[tid + stride];
            if (cand_max > s_max[tid] || (cand_max == s_max[tid] && cand_idx < s_idx[tid])) {
                s_max[tid] = cand_max;
                s_idx[tid] = cand_idx;
            }
            const float cand_base_max = s_base_max[tid + stride];
            const int cand_base_idx = s_base_idx[tid + stride];
            if (cand_base_max > s_base_max[tid] || (cand_base_max == s_base_max[tid] && cand_base_idx < s_base_idx[tid])) {
                s_base_max[tid] = cand_base_max;
                s_base_idx[tid] = cand_base_idx;
            }
        }
        __syncthreads();
    }

    const float max_l = s_max[0];
    const int i_max = s_idx[0];
    const float p_log_base_max = s_base_max[0];
    const int i_max_base = s_base_idx[0];

    double local_sum_exp = 0.0;
    double local_sum_exp_shifted_logit = 0.0;
    double local_kld = 0.0;
    double local_p_base_sum = 0.0;
    double local_base_entropy = 0.0;
    for (int i = tid; i < n_vocab; i += KLD_THREADS) {
        const float l = logits_row[i];
        const double exp_shifted = (double) expf(l - max_l);
        local_sum_exp += exp_shifted;
        local_sum_exp_shifted_logit += exp_shifted * (double) (l - max_l);
        const float p_log_base = scale * idx[i] + min_log_prob;
        if (p_log_base > -16.0f) {
            const double p_base = (double) expf(p_log_base);
            local_p_base_sum += p_base;
            local_base_entropy -= p_base * (double) p_log_base;
            local_kld += p_base * ((double) p_log_base - (double) l + (double) max_l);
        }
    }

    s_sum_exp[tid] = local_sum_exp;
    s_sum_exp_shifted_logit[tid] = local_sum_exp_shifted_logit;
    s_kld[tid] = local_kld;
    s_p_base_sum[tid] = local_p_base_sum;
    s_base_entropy[tid] = local_base_entropy;
    __syncthreads();

    for (int stride = KLD_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum_exp[tid] += s_sum_exp[tid + stride];
            s_sum_exp_shifted_logit[tid] += s_sum_exp_shifted_logit[tid + stride];
            s_kld[tid] += s_kld[tid + stride];
            s_p_base_sum[tid] += s_p_base_sum[tid + stride];
            s_base_entropy[tid] += s_base_entropy[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int tok = token_ids[row];
        if (tok < 0 || tok >= n_vocab) {
            tok = 0;
        }
        const double nll_base = -((double) scale * (double) idx[tok] + (double) min_log_prob);
        const double sum_exp = s_sum_exp[0];
        const double log_sum_exp = log(fmax(sum_exp, 1e-300));
        const double kld = s_kld[0] + s_p_base_sum[0] * log_sum_exp;
        const double nll = (double) max_l + log_sum_exp - (double) logits_row[tok];
        const double p_q = exp(-nll);
        const double p_base_tok = exp(-nll_base);
        const double p_diff = p_q - p_base_tok;
        const double p_diff2 = p_diff * p_diff;
        const double base_norm = fmax(s_p_base_sum[0], 1e-300);
        const double q_entropy = log_sum_exp - s_sum_exp_shifted_logit[0] / fmax(sum_exp, 1e-300);
        const double base_entropy_norm = s_base_entropy[0] / base_norm + log(base_norm);
        const double entropy_diff = q_entropy - base_entropy_norm;
        const double q_prob_base_top = exp((double) logits_row[i_max_base] - (double) max_l - log_sum_exp);
        const double base_top_prob = exp((double) p_log_base_max) / base_norm;
        const double base_prob_q_top = exp((double) (scale * idx[i_max] + min_log_prob)) / base_norm;
        const double top_prob_diff = q_prob_base_top - base_top_prob;
        const double top_flip_weight = i_max == i_max_base ? 0.0 : fmax(0.0, base_top_prob - base_prob_q_top);

        nvfp4_cuda_kld_row_result r;
        r.nll = nll;
        r.nll2 = nll * nll;
        r.nll_base = nll_base;
        r.nll_base2 = nll_base * nll_base;
        r.nll_nll_base = nll * nll_base;
        r.kld = kld;
        r.kld2 = kld * kld;
        r.p_diff2 = p_diff2;
        r.p_diff4 = p_diff2 * p_diff2;
        r.entropy_diff2 = entropy_diff * entropy_diff;
        r.top_prob_diff2 = top_prob_diff * top_prob_diff;
        r.top_flip_weight = top_flip_weight;
        r.same_top = i_max == i_max_base;
        rows[row] = r;
        if (kld_values != nullptr) {
            kld_values[row] = kld;
        }
    }
}

extern "C" void ggml_cuda_nvfp4_register_autotune() {
}

extern "C" bool ggml_cuda_nvfp4_autotune_ex(
        const float * x,
        const float * qw,
        int64_t n,
        const nvfp4_cuda_runtime_cfg * cfg_hint,
        nvfp4_cuda_tune_result * result,
        cudaStream_t stream) {
    const bool trace = nvfp4_cuda_trace_enabled();

    if (result) {
        result->a = NVFP4_A0;
        result->b = NVFP4_B0;
        result->scale_mul = 1.0f;
        result->cfg = { NVFP4_CUDA_CHOOSE46_ADAPTIVE, NVFP4_TUNE_REFIT_ITERS, 1, 0, 448.0f, 256.0f };
        result->has_cfg = 0;
    }

    if (x == nullptr || n < QK_NVFP4 || (n % QK_NVFP4) != 0) {
        return false;
    }

    const int64_t nb_total = n / QK_NVFP4;
    const int64_t coarse_cap = std::max<int64_t>(32, nvfp4_cuda_env_i64("LLAMA_NVFP4_AUTOTUNE_COARSE_BLOCKS", 512));
    const int64_t refine_cap = std::max<int64_t>(
            coarse_cap,
            nvfp4_cuda_env_i64("LLAMA_NVFP4_AUTOTUNE_REFINE_BLOCKS",
                nvfp4_cuda_env_i64("LLAMA_NVFP4_AUTOTUNE_MAX_BLOCKS", NVFP4_AUTOTUNE_MAX_SAMPLE_BLOCKS)));
    const int coarse_topk = (int) std::clamp<int64_t>(nvfp4_cuda_env_i64("LLAMA_NVFP4_AUTOTUNE_TOPK", 24), 1, 48);

    const int64_t coarse_nb = std::min<int64_t>(nb_total, coarse_cap);
    const int64_t refine_nb = std::min<int64_t>(nb_total, refine_cap);
    if (coarse_nb <= 0) {
        return false;
    }

    cudaStream_t st = stream ? stream : 0;
    const bool input_device = ggml_cuda_nvfp4_device_pointer(x);
    const bool qw_device = qw != nullptr && ggml_cuda_nvfp4_device_pointer(qw);
    std::vector<float> x_coarse_sample;
    std::vector<float> x_refine_sample;
    std::vector<float> qw_coarse_sample;
    std::vector<float> qw_refine_sample;
    const float * x_coarse = nullptr;
    const float * x_refine = nullptr;
    const float * qw_coarse = nullptr;
    const float * qw_refine = nullptr;
    if (input_device && (qw == nullptr || qw_device)) {
        auto & tls = g_nvfp4_cuda_autotune_tls;
        if (!nvfp4_cuda_prepare_device_block_sample(tls, x, qw, nb_total, coarse_nb, 0, false, &x_coarse, &qw_coarse, st) ||
            !nvfp4_cuda_prepare_device_block_sample(tls, x, qw, nb_total, refine_nb, 0, true, &x_refine, &qw_refine, st)) {
            return false;
        }
    } else {
        nvfp4_host_prepare_block_sample(x, nb_total, coarse_nb, x_coarse_sample, &x_coarse);
        nvfp4_host_prepare_block_sample(x, nb_total, refine_nb, x_refine_sample, &x_refine);
        if (qw != nullptr) {
            nvfp4_host_prepare_block_sample(qw, nb_total, coarse_nb, qw_coarse_sample, &qw_coarse);
            nvfp4_host_prepare_block_sample(qw, nb_total, refine_nb, qw_refine_sample, &qw_refine);
        }
    }

    const float * a_candidates = NVFP4_TUNE_FIXED_POOL;
    const float * b_candidates = NVFP4_TUNE_FIXED_POOL;
    const int a_candidates_n = (int) (sizeof(NVFP4_TUNE_FIXED_POOL) / sizeof(NVFP4_TUNE_FIXED_POOL[0]));
    const int b_candidates_n = a_candidates_n;
    const bool enable_scale_mul_tune = cfg_hint == nullptr;

    struct coarse_candidate {
        float a;
        float b;
        float scale_mul;
        nvfp4_tune_eval_stats stats;
        double rank_score;
    };

    std::vector<coarse_candidate> top_candidates;
    top_candidates.reserve((size_t) coarse_topk);
    float best_a_now = NVFP4_A0;
    float best_b_now = NVFP4_B0;
    float best_scale_mul_now = 1.0f;
    nvfp4_tune_eval_stats best_coarse{};
    nvfp4_tune_eval_stats base_coarse{};
    nvfp4_cuda_runtime_cfg final_cfg = { NVFP4_CUDA_CHOOSE46_ADAPTIVE, NVFP4_TUNE_REFIT_ITERS, 1, 0, 448.0f, 256.0f };

    auto make_ab_request = [&](float cand_a, float cand_b, float cand_scale_mul = 1.0f) {
        nvfp4_cuda_eval_request req{};
        req.a = cand_a;
        req.b = cand_b;
        req.scale_mul = (std::isfinite(cand_scale_mul) && cand_scale_mul > 0.0f) ? cand_scale_mul : 1.0f;
        nvfp4_cuda_resolve_cfg(req.cfg, cand_a, cand_b, cfg_hint);
        return req;
    };

    auto insert_topk = [&](float cand_a, float cand_b, float cand_scale_mul, const nvfp4_tune_eval_stats & cand_stats, double rank_score) {
        coarse_candidate candidate{ cand_a, cand_b, cand_scale_mul, cand_stats, rank_score };
        auto it = top_candidates.begin();
        while (it != top_candidates.end() && it->rank_score <= rank_score) {
            ++it;
        }
        top_candidates.insert(it, candidate);
        if ((int) top_candidates.size() > coarse_topk) {
            top_candidates.resize((size_t) coarse_topk);
        }
    };

    std::vector<nvfp4_cuda_eval_request> coarse_requests;
    const size_t coarse_request_limit = 1 + (size_t) a_candidates_n * (size_t) b_candidates_n;
    coarse_requests.reserve(coarse_request_limit);
    coarse_requests.push_back(make_ab_request(NVFP4_A0, NVFP4_B0));
    if (enable_scale_mul_tune) {
        for (int si = 0; si < a_candidates_n && coarse_requests.size() < coarse_request_limit; ++si) {
            const float cand_scale = a_candidates[si];
            if (std::fabs(cand_scale - 1.0f) <= 1e-12f) {
                continue;
            }
            coarse_requests.push_back(make_ab_request(NVFP4_A0, NVFP4_B0, cand_scale));
        }
    }
    for (int ai = 0; ai < a_candidates_n; ++ai) {
        const float cand_a = a_candidates[ai];
        for (int bi = 0; bi < b_candidates_n; ++bi) {
            if (coarse_requests.size() >= coarse_request_limit) {
                break;
            }
            coarse_requests.push_back(make_ab_request(cand_a, b_candidates[bi]));
        }
        if (coarse_requests.size() >= coarse_request_limit) {
            break;
        }
    }

    std::vector<nvfp4_tune_eval_stats> coarse_results;
    std::vector<uint8_t> coarse_ok;
    if (!nvfp4_cuda_eval_requests_parallel(x_coarse, qw_coarse, coarse_nb,
            coarse_requests.data(), (int) coarse_requests.size(), coarse_results, &coarse_ok, st) ||
        coarse_ok.empty() || coarse_ok[0] == 0) {
        return false;
    }

    base_coarse = coarse_results[0];
    best_coarse = base_coarse;
    insert_topk(NVFP4_A0, NVFP4_B0, 1.0f, base_coarse, 1.0);

    for (size_t ci = 1; ci < coarse_requests.size(); ++ci) {
        if (coarse_ok[ci] == 0) {
            continue;
        }
        const nvfp4_tune_eval_stats & coarse_stats = coarse_results[ci];
        const double rank_score = nvfp4_host_robust_score(coarse_stats, base_coarse);
        insert_topk(coarse_requests[ci].a, coarse_requests[ci].b, coarse_requests[ci].scale_mul, coarse_stats, rank_score);
        if (nvfp4_host_eval_better(coarse_stats, best_coarse)) {
            best_coarse = coarse_stats;
            best_a_now = coarse_requests[ci].a;
            best_b_now = coarse_requests[ci].b;
            best_scale_mul_now = coarse_requests[ci].scale_mul;
        }
    }

    if (top_candidates.empty()) {
        return false;
    }

    nvfp4_tune_eval_stats final_stats = best_coarse;
    nvfp4_tune_eval_stats base_stats = base_coarse;

    if (refine_nb > coarse_nb) {
        nvfp4_tune_eval_stats base_refine{};
        bool base_refine_ready = false;
        nvfp4_tune_eval_stats best_refine{};
        double best_refine_score = DBL_MAX;

        std::vector<nvfp4_cuda_eval_request> refine_requests;
        refine_requests.reserve(top_candidates.size());
        for (const coarse_candidate & cand : top_candidates) {
            refine_requests.push_back(make_ab_request(cand.a, cand.b, cand.scale_mul));
        }

        std::vector<nvfp4_tune_eval_stats> refine_results;
        std::vector<uint8_t> refine_ok;
        if (!refine_requests.empty() &&
            nvfp4_cuda_eval_requests_parallel(x_refine, qw_refine, refine_nb,
                refine_requests.data(), (int) refine_requests.size(), refine_results, &refine_ok, st)) {
            for (size_t ri = 0; ri < refine_requests.size(); ++ri) {
                if (refine_ok[ri] == 0) {
                    continue;
                }
                const nvfp4_tune_eval_stats & refine_stats = refine_results[ri];
                if (refine_requests[ri].a == NVFP4_A0 &&
                    refine_requests[ri].b == NVFP4_B0 &&
                    std::fabs(refine_requests[ri].scale_mul - 1.0f) <= 1e-12f) {
                    base_refine = refine_stats;
                    base_refine_ready = true;
                }
                if (!base_refine_ready) {
                    base_refine = refine_stats;
                    base_refine_ready = true;
                }

                const double score = nvfp4_host_robust_score(refine_stats, base_refine);
                if (score < best_refine_score ||
                    (std::fabs(score - best_refine_score) <= 1e-12 * std::max(1.0, std::max(std::fabs(score), std::fabs(best_refine_score))) &&
                     nvfp4_host_eval_better(refine_stats, best_refine))) {
                    best_refine_score = score;
                    best_refine = refine_stats;
                    best_a_now = refine_requests[ri].a;
                    best_b_now = refine_requests[ri].b;
                    best_scale_mul_now = refine_requests[ri].scale_mul;
                }
            }
        }

        if (base_refine_ready && std::isfinite(best_refine.obj_norm)) {
            base_stats = base_refine;
            final_stats = best_refine;
        }
    }

    float final_a = best_a_now;
    float final_b = best_b_now;
    float final_scale_mul = best_scale_mul_now;

    struct tune_policy_cfg {
        const char * name;
        int choose46_mode;
        int refit_iters;
        int use_compand_sat;
        float cap_m6;
        float cap_m4;
    };

    auto normalize_policy = [](tune_policy_cfg cfg) -> tune_policy_cfg {
        if (cfg.choose46_mode < NVFP4_CUDA_CHOOSE46_ADAPTIVE || cfg.choose46_mode > NVFP4_CUDA_CHOOSE46_FORCE_M4) {
            cfg.choose46_mode = NVFP4_CUDA_CHOOSE46_ADAPTIVE;
        }
        if (cfg.refit_iters < 0) cfg.refit_iters = NVFP4_TUNE_REFIT_ITERS;
        if (cfg.refit_iters > 64) cfg.refit_iters = 64;
        if (cfg.use_compand_sat != 0 && cfg.use_compand_sat != 1) {
            cfg.use_compand_sat = 1;
        }
        if (!(isfinite(cfg.cap_m6) && cfg.cap_m6 > 0.0f)) {
            cfg.cap_m6 = NVFP4_E2M1_MAX_VALUE * 64.0f;
        }
        if (!(isfinite(cfg.cap_m4) && cfg.cap_m4 > 0.0f)) {
            cfg.cap_m4 = cfg.cap_m6;
        }
        if (cfg.cap_m4 > cfg.cap_m6) {
            cfg.cap_m4 = cfg.cap_m6;
        }
        if (!cfg.name) {
            cfg.name = "unnamed";
        }
        return cfg;
    };

    auto same_policy = [](const tune_policy_cfg & a, const tune_policy_cfg & b) -> bool {
        return a.choose46_mode == b.choose46_mode &&
               a.refit_iters == b.refit_iters &&
               a.use_compand_sat == b.use_compand_sat &&
               a.cap_m6 == b.cap_m6 &&
               a.cap_m4 == b.cap_m4;
    };

    auto selector_score = [](const nvfp4_tune_eval_stats & s) {
        return
            0.22 * s.obj_norm +
            0.24 * s.p95_rel_obj +
            0.20 * s.tail_rel_obj +
            0.30 * s.max_rel_obj +
            0.04 * s.abs_mean_err_rel;
    };

    auto selector_better = [](const nvfp4_tune_eval_stats & cand, double cand_score, const nvfp4_tune_eval_stats & best, double best_score) {
        const double eps = 1e-12;
        if (cand_score + eps < best_score) {
            return true;
        }
        if (std::fabs(cand_score - best_score) <= eps) {
            if (cand.max_rel_obj + eps < best.max_rel_obj) return true;
            if (std::fabs(cand.max_rel_obj - best.max_rel_obj) <= eps && cand.p95_rel_obj + eps < best.p95_rel_obj) return true;
            if (std::fabs(cand.max_rel_obj - best.max_rel_obj) <= eps && std::fabs(cand.p95_rel_obj - best.p95_rel_obj) <= eps && cand.obj_norm + eps < best.obj_norm) return true;
            if (std::fabs(cand.max_rel_obj - best.max_rel_obj) <= eps && std::fabs(cand.p95_rel_obj - best.p95_rel_obj) <= eps && std::fabs(cand.obj_norm - best.obj_norm) <= eps && cand.tail_rel_obj + eps < best.tail_rel_obj) return true;
        }
        return false;
    };

    auto make_policy_request = [&](float cand_a, float cand_b, const tune_policy_cfg & policy, float cand_scale_mul = 1.0f) {
        nvfp4_cuda_eval_request req{};
        req.a = cand_a;
        req.b = cand_b;
        req.scale_mul = (std::isfinite(cand_scale_mul) && cand_scale_mul > 0.0f) ? cand_scale_mul : 1.0f;
        nvfp4_cuda_resolve_cfg(req.cfg, cand_a, cand_b, nullptr);
        req.cfg.choose46_mode = policy.choose46_mode;
        req.cfg.refit_iters = policy.refit_iters;
        req.cfg.use_compand_sat = policy.use_compand_sat;
        req.cfg.cap_m6 = policy.cap_m6;
        req.cfg.cap_m4 = policy.cap_m4;
        if (req.cfg.cap_m4 > req.cfg.cap_m6) {
            req.cfg.cap_m4 = req.cfg.cap_m6;
        }
        return req;
    };

    const int64_t guard_nb = refine_nb;
    const bool run_selector = cfg_hint == nullptr && (nb_total >= 1024 || refine_nb >= 256);
    if (run_selector) {
#define NVFP4_SELECTOR_POLICY(name, choose46, refit, compand, cap6, cap4) \
            { name, choose46, refit, compand, cap6, cap4 },
        const tune_policy_cfg selector_policies_raw[] = {
            GGML_NVFP4_CUDA_PRESET_LIST(NVFP4_SELECTOR_POLICY)
        };
#undef NVFP4_SELECTOR_POLICY

        constexpr int k_sel_max = NVFP4_TUNE_POOL_SIZE;
        const float step = NVFP4_STEP * NVFP4_TUNE_AB_STEP_SCALE;
        const float half = 0.5f * step;
        float cand_a[k_sel_max];
        float cand_b[k_sel_max];
        int cand_n = 0;
        auto add_ab = [&](float aa, float bb) {
            if (cand_n >= k_sel_max) {
                return;
            }
            for (int i = 0; i < cand_n; ++i) {
                if (std::fabs(cand_a[i] - aa) <= 1e-12f && std::fabs(cand_b[i] - bb) <= 1e-12f) {
                    return;
                }
            }
            cand_a[cand_n] = aa;
            cand_b[cand_n] = bb;
            ++cand_n;
        };

        add_ab(final_a, final_b);
        add_ab(NVFP4_A0, NVFP4_B0);
        add_ab(final_a + step, final_b);
        add_ab(final_a - step, final_b);
        add_ab(final_a, final_b + step);
        add_ab(final_a, final_b - step);
        add_ab(final_a + step, final_b + step);
        add_ab(final_a + step, final_b - step);
        add_ab(final_a - step, final_b + step);
        add_ab(final_a - step, final_b - step);
        add_ab(final_a + 2.0f * step, final_b);
        add_ab(final_a - 2.0f * step, final_b);
        add_ab(final_a, final_b + 2.0f * step);
        add_ab(final_a, final_b - 2.0f * step);
        add_ab(final_a + 3.0f * step, final_b);
        add_ab(final_a - 3.0f * step, final_b);
        add_ab(final_a, final_b + 3.0f * step);
        add_ab(final_a, final_b - 3.0f * step);
        add_ab(final_a + half, final_b + half);
        add_ab(final_a - half, final_b - half);
        add_ab(final_a + half, final_b - half);
        add_ab(final_a - half, final_b + half);
        add_ab(final_a + 1.5f * step, final_b + half);
        add_ab(final_a - 1.5f * step, final_b - half);
        add_ab(final_a + half, final_b + 1.5f * step);
        add_ab(final_a - half, final_b - 1.5f * step);
        for (float pool_a : NVFP4_TUNE_FIXED_POOL) {
            add_ab(pool_a, final_b);
        }
        for (float pool_b : NVFP4_TUNE_FIXED_POOL) {
            add_ab(final_a, pool_b);
        }
        for (int i = 0; i < (int) (sizeof(NVFP4_TUNE_FIXED_POOL) / sizeof(NVFP4_TUNE_FIXED_POOL[0])); ++i) {
            add_ab(NVFP4_TUNE_FIXED_POOL[i], NVFP4_TUNE_FIXED_POOL[i]);
        }

        nvfp4_tune_eval_stats selector_best = final_stats;
        float selector_best_a = final_a;
        float selector_best_b = final_b;
        float selector_best_scale_mul = final_scale_mul;
        tune_policy_cfg selector_best_policy = normalize_policy(selector_policies_raw[0]);
        int selector_total = 0;
        int selector_accept = 0;

        for (const tune_policy_cfg & raw_policy : selector_policies_raw) {
            const tune_policy_cfg policy = normalize_policy(raw_policy);
            if (same_policy(policy, normalize_policy(selector_policies_raw[0])) && same_policy(policy, selector_best_policy)) {
                // keep the first policy as the baseline anchor without extra logs
            }

            ++selector_total;
            nvfp4_tune_eval_stats local_best_eval{};
            double local_best_score = DBL_MAX;
            float local_best_a = selector_best_a;
            float local_best_b = selector_best_b;
            float local_best_scale_mul = selector_best_scale_mul;
            bool local_best_ready = false;

            std::vector<nvfp4_cuda_eval_request> selector_requests;
            selector_requests.reserve((size_t) cand_n);
            for (int ci = 0; ci < cand_n; ++ci) {
                selector_requests.push_back(make_policy_request(cand_a[ci], cand_b[ci], policy, selector_best_scale_mul));
            }
            std::vector<nvfp4_tune_eval_stats> selector_results;
            std::vector<uint8_t> selector_ok;
            if (nvfp4_cuda_eval_requests_parallel(x_refine, qw_refine, refine_nb,
                    selector_requests.data(), (int) selector_requests.size(), selector_results, &selector_ok, st)) {
                for (size_t ci = 0; ci < selector_requests.size(); ++ci) {
                    if (selector_ok[ci] == 0) {
                        continue;
                    }
                    const nvfp4_tune_eval_stats & ce = selector_results[ci];
                    const double sc = selector_score(ce);
                    if (!local_best_ready || selector_better(ce, sc, local_best_eval, local_best_score)) {
                        local_best_score = sc;
                        local_best_eval = ce;
                        local_best_a = selector_requests[ci].a;
                        local_best_b = selector_requests[ci].b;
                        local_best_scale_mul = selector_requests[ci].scale_mul;
                        local_best_ready = true;
                    }
                }
            }

            if (local_best_ready) {
                constexpr int k_ref_max = 16;
                float ref_a[k_ref_max];
                float ref_b[k_ref_max];
                int ref_n = 0;
                auto add_ref = [&](float aa, float bb) {
                    if (ref_n >= k_ref_max || !(aa > 0.0f) || !(bb > 0.0f)) {
                        return;
                    }
                    for (int i = 0; i < ref_n; ++i) {
                        if (std::fabs(ref_a[i] - aa) <= 1e-12f && std::fabs(ref_b[i] - bb) <= 1e-12f) {
                            return;
                        }
                    }
                    ref_a[ref_n] = aa;
                    ref_b[ref_n] = bb;
                    ++ref_n;
                };

                add_ref(local_best_a + half, local_best_b);
                add_ref(local_best_a - half, local_best_b);
                add_ref(local_best_a, local_best_b + half);
                add_ref(local_best_a, local_best_b - half);
                add_ref(local_best_a + half, local_best_b + half);
                add_ref(local_best_a + half, local_best_b - half);
                add_ref(local_best_a - half, local_best_b + half);
                add_ref(local_best_a - half, local_best_b - half);
                add_ref(local_best_a + step, local_best_b);
                add_ref(local_best_a - step, local_best_b);
                add_ref(local_best_a, local_best_b + step);
                add_ref(local_best_a, local_best_b - step);

                std::vector<nvfp4_cuda_eval_request> selector_refine_requests;
                selector_refine_requests.reserve((size_t) ref_n);
                for (int ri = 0; ri < ref_n; ++ri) {
                    selector_refine_requests.push_back(make_policy_request(ref_a[ri], ref_b[ri], policy, local_best_scale_mul));
                }
                std::vector<nvfp4_tune_eval_stats> selector_refine_results;
                std::vector<uint8_t> selector_refine_ok;
                if (nvfp4_cuda_eval_requests_parallel(x_refine, qw_refine, refine_nb,
                        selector_refine_requests.data(), (int) selector_refine_requests.size(),
                        selector_refine_results, &selector_refine_ok, st)) {
                    for (size_t ri = 0; ri < selector_refine_requests.size(); ++ri) {
                        if (selector_refine_ok[ri] == 0) {
                            continue;
                        }
                        const nvfp4_tune_eval_stats & ce = selector_refine_results[ri];
                        const double sc = selector_score(ce);
                        if (!local_best_ready || selector_better(ce, sc, local_best_eval, local_best_score)) {
                            local_best_score = sc;
                            local_best_eval = ce;
                            local_best_a = selector_refine_requests[ri].a;
                            local_best_b = selector_refine_requests[ri].b;
                            local_best_scale_mul = selector_refine_requests[ri].scale_mul;
                            local_best_ready = true;
                        }
                    }
                }
            }

            const bool accepted = local_best_ready && std::isfinite(local_best_eval.obj_norm) &&
                selector_better(local_best_eval, local_best_score, selector_best, selector_score(selector_best));
            if (accepted) {
                ++selector_accept;
                selector_best = local_best_eval;
                selector_best_a = local_best_a;
                selector_best_b = local_best_b;
                selector_best_scale_mul = local_best_scale_mul;
                selector_best_policy = policy;
            }

            if (trace) {
                fprintf(stderr,
                    "NVFP4_TUNE (CUDA selector) cand=%d/%d policy=%s choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f score=%g obj_norm=%g p95_rel_obj=%g tail_rel_obj=%g max_rel_obj=%g accepted=%d\n",
                    selector_total,
                    (int) (sizeof(selector_policies_raw) / sizeof(selector_policies_raw[0])),
                    policy.name,
                    policy.choose46_mode,
                    policy.refit_iters,
                    policy.use_compand_sat,
                    (double) policy.cap_m6,
                    (double) policy.cap_m4,
                    local_best_score,
                    local_best_eval.obj_norm,
                    local_best_eval.p95_rel_obj,
                    local_best_eval.tail_rel_obj,
                    local_best_eval.max_rel_obj,
                    accepted ? 1 : 0);
            }
        }

        if (std::isfinite(selector_best.obj_norm)) {
            final_stats = selector_best;
            final_a = selector_best_a;
            final_b = selector_best_b;
            final_scale_mul = selector_best_scale_mul;
            final_cfg = {
                selector_best_policy.choose46_mode,
                selector_best_policy.refit_iters,
                selector_best_policy.use_compand_sat,
                0,
                selector_best_policy.cap_m6,
                selector_best_policy.cap_m4,
            };
            if (trace) {
                fprintf(stderr,
                    "NVFP4_TUNE (CUDA selector final) policy=%s score=%g obj_norm=%g p95_rel_obj=%g tail_rel_obj=%g max_rel_obj=%g accepts=%d/%d\n",
                    selector_best_policy.name,
                    selector_score(selector_best),
                    selector_best.obj_norm,
                    selector_best.p95_rel_obj,
                    selector_best.tail_rel_obj,
                    selector_best.max_rel_obj,
                    selector_accept,
                    selector_total);
            }
        }

        if (guard_nb > 0) {
            std::vector<float> x_guard_sample;
            std::vector<float> qw_guard_sample;
            const float * x_guard = nullptr;
            const float * qw_guard = nullptr;
            const int64_t guard_phase = std::max<int64_t>(1, nb_total / 3);
            if (input_device && (qw == nullptr || qw_device)) {
                auto & tls = g_nvfp4_cuda_autotune_tls;
                if (!nvfp4_cuda_prepare_device_block_sample(tls, x, qw, nb_total, guard_nb, guard_phase, false, &x_guard, &qw_guard, st)) {
                    return false;
                }
            } else {
                nvfp4_host_prepare_block_sample(x, nb_total, guard_nb, x_guard_sample, &x_guard, guard_phase);
                if (qw != nullptr) {
                    nvfp4_host_prepare_block_sample(qw, nb_total, guard_nb, qw_guard_sample, &qw_guard, guard_phase);
                }
            }
            const tune_policy_cfg baseline_policy = normalize_policy(selector_policies_raw[0]);
            nvfp4_tune_eval_stats selected_guard{};
            nvfp4_tune_eval_stats baseline_guard{};
            std::vector<nvfp4_cuda_eval_request> guard_requests;
            guard_requests.reserve(2);
            guard_requests.push_back(make_policy_request(final_a, final_b, selector_best_policy, final_scale_mul));
            guard_requests.push_back(make_policy_request(NVFP4_A0, NVFP4_B0, baseline_policy, 1.0f));
            std::vector<nvfp4_tune_eval_stats> guard_results;
            std::vector<uint8_t> guard_status;
            bool selected_guard_ready = false;
            bool baseline_guard_ready = false;
            if (nvfp4_cuda_eval_requests_parallel(x_guard, qw_guard, guard_nb,
                    guard_requests.data(), (int) guard_requests.size(), guard_results, &guard_status, st)) {
                if (!guard_results.empty() && guard_status[0] != 0) {
                    selected_guard = guard_results[0];
                    selected_guard_ready = true;
                }
                if (guard_results.size() > 1 && guard_status[1] != 0) {
                    baseline_guard = guard_results[1];
                    baseline_guard_ready = true;
                }
            }
            const double selected_guard_score = selector_score(selected_guard);
            const double baseline_guard_score = selector_score(baseline_guard);
            const bool guard_ok = selected_guard_ready && baseline_guard_ready && std::isfinite(selected_guard.obj_norm) &&
                (selector_better(selected_guard, selected_guard_score, baseline_guard, baseline_guard_score) ||
                 selected_guard_score < baseline_guard_score * (1.0 - NVFP4_AUTOTUNE_MIN_IMPROVE_FRAC));
            if (!guard_ok) {
                final_a = NVFP4_A0;
                final_b = NVFP4_B0;
                final_scale_mul = 1.0f;
                final_stats = baseline_guard;
                selector_best_policy = baseline_policy;
                final_cfg = {
                    baseline_policy.choose46_mode,
                    baseline_policy.refit_iters,
                    baseline_policy.use_compand_sat,
                    0,
                    baseline_policy.cap_m6,
                    baseline_policy.cap_m4,
                };
                if (trace) {
                    fprintf(stderr,
                        "NVFP4_TUNE (CUDA guard fallback) selected_score=%g baseline_score=%g -> baseline\n",
                        selected_guard_score, baseline_guard_score);
                }
            }
        }
    }

    if (std::isfinite(base_stats.obj_norm) && std::isfinite(final_stats.obj_norm)) {
        const double best_score = nvfp4_host_robust_score(final_stats, base_stats);
        if (!(best_score < (1.0 - NVFP4_AUTOTUNE_MIN_IMPROVE_FRAC))) {
            best_a_now = NVFP4_A0;
            best_b_now = NVFP4_B0;
            best_scale_mul_now = 1.0f;
        } else {
            best_a_now = final_a;
            best_b_now = final_b;
            best_scale_mul_now = final_scale_mul;
        }
    }

    if (cfg_hint != nullptr) {
        nvfp4_cuda_resolve_cfg(final_cfg, best_a_now, best_b_now, cfg_hint);
    } else if (!run_selector) {
        nvfp4_cuda_resolve_cfg(final_cfg, best_a_now, best_b_now, nullptr);
    }

    if (result) {
        result->a = best_a_now;
        result->b = best_b_now;
        result->scale_mul = enable_scale_mul_tune ? best_scale_mul_now : 1.0f;
        result->cfg = final_cfg;
        result->has_cfg = 1;
    }
    return true;
}

extern "C" void ggml_cuda_nvfp4_autotune(
        const float * x,
        const float * qw,
        int64_t n,
        float * best_a,
        float * best_b,
        cudaStream_t stream) {
    nvfp4_cuda_tune_result result = {
        NVFP4_A0,
        NVFP4_B0,
        1.0f,
        { NVFP4_CUDA_CHOOSE46_ADAPTIVE, NVFP4_TUNE_REFIT_ITERS, 1, 0, 448.0f, 256.0f },
        0,
    };
    if (!ggml_cuda_nvfp4_autotune_ex(x, qw, n, nullptr, &result, stream)) {
        if (best_a) {
            *best_a = NVFP4_A0;
        }
        if (best_b) {
            *best_b = NVFP4_B0;
        }
        return;
    }
    if (best_a) {
        *best_a = result.a;
    }
    if (best_b) {
        *best_b = result.b;
    }
}

static bool ggml_cuda_nvfp4_quantize_impl(
        const void * x,
        bool x_bf16,
        float x_scale,
        void * vy,
        ggml_tensor * tensor,
        int64_t plane_index,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float a,
        float b,
        const nvfp4_cuda_runtime_cfg * cfg,
        nvfp4_cuda_eval_result * eval,
        cudaStream_t stream) {
    if (nrow <= 0 || n_per_row <= 0 || (n_per_row % QK_NVFP4) != 0) {
        return false;
    }
    const bool direct_tensor = tensor != nullptr;
    if (!direct_tensor && vy == nullptr) {
        return false;
    }
    if (direct_tensor) {
        const int64_t tensor_planes = std::max<int64_t>(1, tensor->ne[2] * tensor->ne[3]);
        if (tensor->type != GGML_TYPE_NVFP4 || tensor->view_src != nullptr || tensor->data == nullptr ||
                tensor->ne[0] != n_per_row || tensor->ne[1] != nrow ||
                plane_index < 0 || plane_index >= tensor_planes) {
            return false;
        }
    }

    nvfp4_cuda_runtime_cfg resolved{};
    nvfp4_cuda_resolve_cfg(resolved, a, b, cfg);
    const float x_scale_eff = (isfinite(x_scale) && x_scale > 0.0f) ? x_scale : 1.0f;

    const int64_t row_blocks = n_per_row / QK_NVFP4;
    const int64_t nb_total = nrow * row_blocks;
    const size_t bytes_x = (size_t) nrow * (size_t) n_per_row * (x_bf16 ? sizeof(ggml_bf16_t) : sizeof(float));
    // Imatrix weights are stored once per row-shape and reused across rows.
    // Do not treat them as a full tensor-sized buffer.
    const size_t bytes_qw = qw ? (size_t) n_per_row * sizeof(float) : 0;
    const size_t bytes_y = (size_t) nrow * ggml_row_size(GGML_TYPE_NVFP4, n_per_row);
    const bool x_device = ggml_cuda_nvfp4_device_pointer(x);
    const bool qw_device = qw != nullptr && ggml_cuda_nvfp4_device_pointer(qw);

    auto & tls = g_nvfp4_cuda_quant_tls;
    if ((!x_device && !ggml_cuda_nvfp4_ensure_buf(&tls.d_x_buf, &tls.d_x_cap, bytes_x, "cudaMalloc(x)")) ||
        !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_y_buf, &tls.d_y_cap, bytes_y, "cudaMalloc(y)") ||
        (bytes_qw != 0 && !qw_device && !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_qw_buf, &tls.d_qw_cap, bytes_qw, "cudaMalloc(qw)")) ||
        (eval != nullptr && !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_eval_buf, &tls.d_eval_cap, sizeof(nvfp4_cuda_eval_accum), "cudaMalloc(eval)"))) {
        return false;
    }

    cudaStream_t st = stream ? stream : 0;
    auto fail_quant_stream = [&](const char * label, cudaError_t status) {
        nvfp4_cuda_log_failure(label, status);
        (void) cudaStreamSynchronize(st);
        cudaGetLastError();
        tls.reset();
        return false;
    };
    cudaError_t err = cudaSuccess;
    if (!x_device) {
        err = cudaMemcpyAsync(tls.d_x_buf, x, bytes_x, cudaMemcpyHostToDevice, st);
        if (err != cudaSuccess) {
            return fail_quant_stream("H2D(x)", err);
        }
    }
    if (bytes_qw != 0 && !qw_device) {
        err = cudaMemcpyAsync(tls.d_qw_buf, qw, bytes_qw, cudaMemcpyHostToDevice, st);
        if (err != cudaSuccess) {
            return fail_quant_stream("H2D(qw)", err);
        }
    }
    const void * d_x = x_device ? x : tls.d_x_buf;
    const float * d_qw = bytes_qw == 0 ? nullptr : (qw_device ? qw : tls.d_qw_buf);

    if (!ggml_cuda_nvfp4_launch_kernel(
            d_x, x_bf16, x_scale_eff, d_qw, tls.d_y_buf,
            row_blocks, nb_total, resolved, st)) {
        return false;
    }

    nvfp4_cuda_eval_accum eval_host{};
    if (eval != nullptr) {
        err = cudaMemsetAsync(tls.d_eval_buf, 0, sizeof(nvfp4_cuda_eval_accum), st);
        if (err != cudaSuccess) {
            return fail_quant_stream("memset(eval)", err);
        }

        const int threads = 256;
        const int blocks = (int) std::min<int64_t>((nb_total * QK_NVFP4 + threads - 1) / threads, 4096);
        ggml_cuda_nvfp4_eval_quantized<<<blocks, threads, 0, st>>>(
            d_x, x_bf16, x_scale_eff, tls.d_y_buf, nrow, n_per_row, tls.d_eval_buf);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return fail_quant_stream("kernel(eval)", err);
        }
        err = cudaMemcpyAsync(&eval_host, tls.d_eval_buf, sizeof(eval_host), cudaMemcpyDeviceToHost, st);
        if (err != cudaSuccess) {
            return fail_quant_stream("D2H(eval)", err);
        }
    }

    if (direct_tensor) {
        void * tensor_data = nullptr;
        int tensor_device = -1;
        if (!ggml_cuda_nvfp4_tensor_active_data(tensor, &tensor_data, &tensor_device)) {
            return false;
        }

        // The tensor is already loaded in the active CUDA layout. Preserve
        // its header scale/pointer state and only replace the packed tiles.
        const int64_t blocks_per_plane =
            ggml_cuda_bw_div_up(nrow, 16) * ggml_cuda_nvfp4_blocks_per_row(n_per_row);
        block_nvfp4_blackwell * dst_tiles =
            reinterpret_cast<block_nvfp4_blackwell *>((char *) tensor_data + sizeof(block_nvfp4_blackwell_tensor)) +
            (size_t) plane_index * (size_t) blocks_per_plane;
        ggml_cuda_nvfp4_repack_one_plane_kernel<<<(unsigned int) blocks_per_plane, 256, 0, st>>>(
            reinterpret_cast<const uint8_t *>(tls.d_y_buf), dst_tiles, n_per_row, nrow);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return fail_quant_stream("kernel(nvfp4 direct repack)", err);
        }
    } else {
        err = cudaMemcpyAsync(vy, tls.d_y_buf, bytes_y, cudaMemcpyDeviceToHost, st);
        if (err != cudaSuccess) {
            return fail_quant_stream("D2H(y)", err);
        }
    }
    err = cudaStreamSynchronize(st);
    if (err != cudaSuccess) {
        return fail_quant_stream("sync", err);
    }

    if (eval != nullptr) {
        float max_abs = 0.0f;
        std::memcpy(&max_abs, &eval_host.max_abs_bits, sizeof(max_abs));
        eval->sum_sq = eval_host.sum_sq;
        eval->sum_abs = eval_host.sum_abs;
        eval->max_abs = (double) max_abs;
        eval->count = (int64_t) eval_host.count;
    }

    return true;
}

extern "C" bool ggml_cuda_nvfp4_quantize(
        const void * x,
        bool x_bf16,
        float x_scale,
        void * vy,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float a,
        float b,
        cudaStream_t stream) {
    return ggml_cuda_nvfp4_quantize_impl(x, x_bf16, x_scale, vy, nullptr, 0, nrow, n_per_row, qw, a, b, nullptr, nullptr, stream);
}

extern "C" bool ggml_cuda_nvfp4_quantize_cfg(
        const void * x,
        bool x_bf16,
        float x_scale,
        void * vy,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float a,
        float b,
        const nvfp4_cuda_runtime_cfg * cfg,
        cudaStream_t stream) {
    return ggml_cuda_nvfp4_quantize_impl(x, x_bf16, x_scale, vy, nullptr, 0, nrow, n_per_row, qw, a, b, cfg, nullptr, stream);
}

extern "C" bool ggml_cuda_nvfp4_quantize_eval_cfg(
        const void * x,
        bool x_bf16,
        float x_scale,
        void * vy,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float a,
        float b,
        const nvfp4_cuda_runtime_cfg * cfg,
        nvfp4_cuda_eval_result * eval,
        cudaStream_t stream) {
    return ggml_cuda_nvfp4_quantize_impl(x, x_bf16, x_scale, vy, nullptr, 0, nrow, n_per_row, qw, a, b, cfg, eval, stream);
}

extern "C" bool ggml_cuda_nvfp4_quantize_eval_to_tensor_cfg(
        const void * x,
        bool x_bf16,
        float x_scale,
        ggml_tensor * tensor,
        int64_t plane_index,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float a,
        float b,
        const nvfp4_cuda_runtime_cfg * cfg,
        nvfp4_cuda_eval_result * eval,
        cudaStream_t stream) {
    return ggml_cuda_nvfp4_quantize_impl(x, x_bf16, x_scale, nullptr, tensor, plane_index, nrow, n_per_row, qw, a, b, cfg, eval, stream);
}

extern "C" bool ggml_cuda_nvfp4_kld_reduce_tensor(
        const ggml_tensor * logits,
        const uint16_t * base_logp_u16,
        const int32_t * token_ids,
        int32_t logits_row_offset,
        int32_t n_eval,
        int32_t n_vocab,
        int32_t nv,
        nvfp4_cuda_kld_result * result,
        double * kld_values,
        cudaStream_t stream) {
    if (result == nullptr) {
        return false;
    }
    *result = {};
    if (logits == nullptr || logits->type != GGML_TYPE_F32 || logits->data == nullptr ||
            base_logp_u16 == nullptr || token_ids == nullptr ||
            logits_row_offset < 0 ||
            n_eval <= 0 || n_vocab <= 0 || nv < n_vocab + 4 ||
            logits->ne[0] < n_vocab || logits->ne[1] < (int64_t) logits_row_offset + n_eval) {
        return false;
    }

    cudaPointerAttributes logits_attr{};
    const cudaError_t attr_err = cudaPointerGetAttributes(&logits_attr, logits->data);
    if (attr_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    if (logits_attr.type != cudaMemoryTypeDevice && logits_attr.type != cudaMemoryTypeManaged) {
        return false;
    }
#else
    if (logits_attr.memoryType != cudaMemoryTypeDevice) {
        return false;
    }
#endif

    auto & tls = g_nvfp4_cuda_kld_tls;
    const size_t bytes_base = (size_t) n_eval * (size_t) nv * sizeof(uint16_t);
    const size_t bytes_tokens = (size_t) n_eval * sizeof(int32_t);
    const size_t bytes_rows = (size_t) n_eval * sizeof(nvfp4_cuda_kld_row_result);
    const size_t bytes_kld_values = (size_t) n_eval * sizeof(double);
    if (!ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_tokens, &tls.d_tokens_cap, bytes_tokens, "kld cudaMalloc(tokens)") ||
        !ggml_cuda_nvfp4_ensure_buf(&tls.d_rows, &tls.d_rows_cap, bytes_rows, "kld cudaMalloc(rows)") ||
        !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_result, &tls.d_result_cap, sizeof(nvfp4_cuda_kld_result), "kld cudaMalloc(result)") ||
        (kld_values != nullptr && !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_kld_values, &tls.d_kld_values_cap, bytes_kld_values, "kld cudaMalloc(values)"))) {
        return false;
    }

    cudaStream_t st = stream ? stream : 0;
    auto fail_kld_stream = [&](const char * label, cudaError_t status) {
        nvfp4_cuda_log_failure(label, status);
        (void) cudaStreamSynchronize(st);
        cudaGetLastError();
        return false;
    };

    const uint16_t * d_base = nullptr;
    if (!ggml_cuda_nvfp4_kld_get_base_device(tls, base_logp_u16, bytes_base, st, &d_base)) {
        (void) cudaStreamSynchronize(st);
        cudaGetLastError();
        return false;
    }
    cudaError_t err = cudaMemcpyAsync(tls.d_tokens, token_ids, bytes_tokens, cudaMemcpyHostToDevice, st);
    if (err != cudaSuccess) {
        return fail_kld_stream("kld H2D(tokens)", err);
    }

    auto * d_rows = reinterpret_cast<nvfp4_cuda_kld_row_result *>(tls.d_rows);
    double * d_kld_values = kld_values != nullptr ? tls.d_kld_values : nullptr;
    ggml_cuda_nvfp4_kld_rows_kernel<<<(unsigned) n_eval, 256, 0, st>>>(
        reinterpret_cast<const float *>(logits->data),
        (int64_t) logits->nb[1],
        logits_row_offset,
        d_base,
        tls.d_tokens,
        n_eval,
        n_vocab,
        nv,
        d_rows,
        d_kld_values);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail_kld_stream("kernel(kld rows)", err);
    }

    ggml_cuda_nvfp4_kld_reduce_rows_kernel<<<1, 256, 0, st>>>(d_rows, n_eval, tls.d_result);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail_kld_stream("kernel(kld reduce rows)", err);
    }
    err = cudaMemcpyAsync(result, tls.d_result, sizeof(nvfp4_cuda_kld_result), cudaMemcpyDeviceToHost, st);
    if (err != cudaSuccess) {
        return fail_kld_stream("kld D2H(result)", err);
    }
    if (kld_values != nullptr) {
        err = cudaMemcpyAsync(kld_values, d_kld_values, bytes_kld_values, cudaMemcpyDeviceToHost, st);
        if (err != cudaSuccess) {
            return fail_kld_stream("kld D2H(values)", err);
        }
    }
    err = cudaStreamSynchronize(st);
    if (err != cudaSuccess) {
        return fail_kld_stream("kld sync", err);
    }
    return true;
}

static bool ggml_cuda_nvfp4_device_pointer_info(const void * ptr, int * device) {
    if (ptr == nullptr) {
        return false;
    }
    cudaPointerAttributes attr{};
    const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    bool ok = false;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
    ok = attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
#else
    ok = attr.memoryType == cudaMemoryTypeDevice;
#endif
    if (ok && device != nullptr) {
        *device = attr.device;
    }
    return ok;
}

static bool ggml_cuda_nvfp4_device_pointer(const void * ptr) {
    return ggml_cuda_nvfp4_device_pointer_info(ptr, nullptr);
}

static bool ggml_cuda_nvfp4_tensor_active_data(ggml_tensor * tensor, void ** data, int * device) {
    if (data == nullptr || tensor == nullptr) {
        return false;
    }
    *data = nullptr;
    if (device != nullptr) {
        *device = -1;
    }

    if (tensor->extra != nullptr) {
        auto * extra = (ggml_tensor_extra_gpu *) tensor->extra;
        void * found = nullptr;
        int found_device = -1;
        int found_count = 0;
        const int ndev = std::min<int>(ggml_backend_cuda_get_device_count(), GGML_CUDA_MAX_DEVICES);
        for (int id = 0; id < ndev; ++id) {
            void * ptr = extra->data_device[id];
            int ptr_device = -1;
            if (ptr == nullptr || !ggml_cuda_nvfp4_device_pointer_info(ptr, &ptr_device)) {
                continue;
            }
            found = ptr;
            found_device = ptr_device >= 0 ? ptr_device : id;
            ++found_count;
        }
        if (found_count == 1) {
            *data = found;
            if (device != nullptr) {
                *device = found_device;
            }
            return true;
        }
        if (found_count > 1) {
            return false;
        }
    }

    int ptr_device = -1;
    if (ggml_cuda_nvfp4_device_pointer_info(tensor->data, &ptr_device)) {
        *data = tensor->data;
        if (device != nullptr) {
            *device = ptr_device;
        }
        return true;
    }

    return false;
}

extern "C" bool ggml_cuda_tensor_set_host_impl(
        ggml_tensor * tensor,
        const void * src,
        size_t nbytes,
        cudaStream_t stream) {
    void * tensor_data = nullptr;
    if (tensor == nullptr || src == nullptr || nbytes == 0 ||
            !ggml_cuda_nvfp4_tensor_active_data(tensor, &tensor_data, nullptr)) {
        return false;
    }
    cudaStream_t st = stream ? stream : 0;
    cudaError_t err = cudaMemcpyAsync(tensor_data, src, nbytes, cudaMemcpyHostToDevice, st);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    err = cudaStreamSynchronize(st);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

extern "C" bool ggml_cuda_nvfp4_tensor_set_header_scales_impl(
        ggml_tensor * tensor,
        float weight_scale,
        float input_scale,
        cudaStream_t stream) {
    void * tensor_data = nullptr;
    if (tensor == nullptr || tensor->type != GGML_TYPE_NVFP4 ||
            !ggml_cuda_nvfp4_tensor_active_data(tensor, &tensor_data, nullptr)) {
        return false;
    }

    const float scales[2] = {
        (isfinite(weight_scale) && weight_scale > 0.0f) ? weight_scale : 1.0f,
        (isfinite(input_scale)  && input_scale  > 0.0f) ? input_scale  : 1.0f,
    };

    cudaStream_t st = stream ? stream : 0;
    cudaError_t err = cudaMemcpyAsync(tensor_data, scales, sizeof(scales), cudaMemcpyHostToDevice, st);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    err = cudaStreamSynchronize(st);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

extern "C" bool ggml_cuda_tensor_get_host_impl(
        const ggml_tensor * tensor,
        void * dst,
        size_t nbytes,
        cudaStream_t stream) {
    void * tensor_data = nullptr;
    if (tensor == nullptr || dst == nullptr || nbytes == 0 ||
            !ggml_cuda_nvfp4_tensor_active_data(const_cast<ggml_tensor *>(tensor), &tensor_data, nullptr)) {
        return false;
    }
    cudaStream_t st = stream ? stream : 0;
    cudaError_t err = cudaMemcpyAsync(dst, tensor_data, nbytes, cudaMemcpyDeviceToHost, st);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    err = cudaStreamSynchronize(st);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

extern "C" bool ggml_cuda_tensor_snapshot_impl(
        const ggml_tensor * tensor,
        size_t nbytes,
        void ** snapshot,
        cudaStream_t stream) {
    void * tensor_data = nullptr;
    if (tensor == nullptr || snapshot == nullptr || nbytes == 0 ||
            !ggml_cuda_nvfp4_tensor_active_data(const_cast<ggml_tensor *>(tensor), &tensor_data, nullptr)) {
        return false;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err == cudaSuccess) {
        const size_t reserve_bytes = std::max<size_t>(total_bytes / 8, (size_t) 512 * 1024 * 1024);
        if (free_bytes <= nbytes + reserve_bytes) {
            return false;
        }
    } else {
        cudaGetLastError();
    }

    void * tmp = nullptr;
    err = cudaMalloc(&tmp, nbytes);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    cudaStream_t st = stream ? stream : 0;
    err = cudaMemcpyAsync(tmp, tensor_data, nbytes, cudaMemcpyDeviceToDevice, st);
    if (err != cudaSuccess) {
        cudaGetLastError();
        cudaFree(tmp);
        return false;
    }
    err = cudaStreamSynchronize(st);
    if (err != cudaSuccess) {
        cudaGetLastError();
        cudaFree(tmp);
        return false;
    }

    *snapshot = tmp;
    return true;
}

extern "C" bool ggml_cuda_tensor_restore_impl(
        ggml_tensor * tensor,
        const void * snapshot,
        size_t nbytes,
        cudaStream_t stream) {
    void * tensor_data = nullptr;
    if (tensor == nullptr || snapshot == nullptr || nbytes == 0 ||
            !ggml_cuda_nvfp4_tensor_active_data(tensor, &tensor_data, nullptr) ||
            !ggml_cuda_nvfp4_device_pointer(snapshot)) {
        return false;
    }
    cudaStream_t st = stream ? stream : 0;
    cudaError_t err = cudaMemcpyAsync(tensor_data, snapshot, nbytes, cudaMemcpyDeviceToDevice, st);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    err = cudaStreamSynchronize(st);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

extern "C" void ggml_cuda_tensor_snapshot_free_impl(void * snapshot) {
    if (snapshot != nullptr) {
        cudaFree(snapshot);
    }
}

static bool ggml_cuda_mxfp6_e2m3_quantize_eval_impl(
        const void * x,
        bool x_bf16,
        float x_scale,
        void * vy,
        ggml_tensor * tensor,
        int64_t plane_index,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float header_weight_scale,
        float header_input_scale,
        nvfp4_cuda_eval_result * eval,
        cudaStream_t stream) {
    const bool direct_tensor = tensor != nullptr;
    if (nrow <= 0 || n_per_row <= 0 ||
            (nrow % MXFP6_E2M3_TILE_ROWS) != 0 ||
            (n_per_row % QK_MXFP6_E2M3_TILE) != 0) {
        return false;
    }
    if (!direct_tensor && vy == nullptr) {
        return false;
    }
    if (direct_tensor) {
        const int64_t nplanes = std::max<int64_t>(1, tensor->ne[2] * tensor->ne[3]);
        if (tensor->type != GGML_TYPE_MXFP6_E2M3 || tensor->data == nullptr ||
                tensor->view_src != nullptr ||
                tensor->ne[0] != n_per_row || tensor->ne[1] != nrow ||
                plane_index < 0 || plane_index >= nplanes) {
            return false;
        }
    }

    const float x_scale_eff = (isfinite(x_scale) && x_scale > 0.0f) ? x_scale : 1.0f;
    const int64_t row_blocks = n_per_row / QK_MXFP6_E2M3;
    const int64_t nb_total = nrow * row_blocks;
    const int64_t tiles_total = (nrow / MXFP6_E2M3_TILE_ROWS) * (row_blocks / QK_MXFP6_E2M3_TILE_FRAGS);
    const size_t bytes_x = (size_t) nrow * (size_t) n_per_row * (x_bf16 ? sizeof(ggml_bf16_t) : sizeof(float));
    const size_t bytes_qw = qw ? (size_t) n_per_row * sizeof(float) : 0;
    const size_t bytes_compact = (size_t) nb_total * sizeof(block_mxfp6_e2m3);
    const size_t bytes_y = (size_t) nrow * ggml_row_size(GGML_TYPE_MXFP6_E2M3, n_per_row);

    auto & tls = g_mxfp6_e2m3_cuda_quant_tls;
    if (!ggml_cuda_nvfp4_ensure_buf(&tls.d_x_buf, &tls.d_x_cap, bytes_x, "cudaMalloc(mx6 x)") ||
        !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_compact_buf, &tls.d_compact_cap, bytes_compact, "cudaMalloc(mx6 compact)") ||
        !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_y_buf, &tls.d_y_cap, bytes_y, "cudaMalloc(mx6 y)") ||
        (bytes_qw != 0 && !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_qw_buf, &tls.d_qw_cap, bytes_qw, "cudaMalloc(mx6 qw)")) ||
        (eval != nullptr && !ggml_cuda_nvfp4_ensure_buf((void **) &tls.d_eval_buf, &tls.d_eval_cap, sizeof(nvfp4_cuda_eval_accum), "cudaMalloc(mx6 eval)"))) {
        return false;
    }

    cudaStream_t st = stream ? stream : 0;
    auto fail_quant_stream = [&](const char * label, cudaError_t status) {
        nvfp4_cuda_log_failure(label, status);
        (void) cudaStreamSynchronize(st);
        cudaGetLastError();
        tls.reset();
        return false;
    };

    cudaError_t err = cudaMemcpyAsync(tls.d_x_buf, x, bytes_x, cudaMemcpyHostToDevice, st);
    if (err != cudaSuccess) {
        return fail_quant_stream("MXFP6 H2D(x)", err);
    }
    if (bytes_qw != 0) {
        err = cudaMemcpyAsync(tls.d_qw_buf, qw, bytes_qw, cudaMemcpyHostToDevice, st);
        if (err != cudaSuccess) {
            return fail_quant_stream("MXFP6 H2D(qw)", err);
        }
    }

    ggml_cuda_mxfp6_e2m3_quantize_blocks_32<<<(unsigned) nb_total, QK_MXFP6_E2M3, 0, st>>>(
        tls.d_x_buf, x_bf16, x_scale_eff, bytes_qw ? tls.d_qw_buf : nullptr, tls.d_compact_buf, row_blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail_quant_stream("MXFP6 kernel(quantize)", err);
    }
    ggml_cuda_mxfp6_e2m3_pack_tiles_832<<<(unsigned) (tiles_total * QK_MXFP6_E2M3_TILE_FRAGS), MXFP6_E2M3_TILE_LANES, 0, st>>>(
        tls.d_compact_buf, tls.d_y_buf, nrow, row_blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail_quant_stream("MXFP6 kernel(pack_tiles)", err);
    }

    nvfp4_cuda_eval_accum eval_host{};
    if (eval != nullptr) {
        err = cudaMemsetAsync(tls.d_eval_buf, 0, sizeof(nvfp4_cuda_eval_accum), st);
        if (err != cudaSuccess) {
            return fail_quant_stream("MXFP6 memset(eval)", err);
        }
        const int threads = 256;
        const int blocks = (int) std::min<int64_t>((nrow * n_per_row + threads - 1) / threads, 4096);
        ggml_cuda_mxfp6_e2m3_eval_quantized<<<blocks, threads, 0, st>>>(
            tls.d_x_buf, x_bf16, x_scale_eff, bytes_qw ? tls.d_qw_buf : nullptr, tls.d_compact_buf, nrow, n_per_row, tls.d_eval_buf);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return fail_quant_stream("MXFP6 kernel(eval)", err);
        }
        err = cudaMemcpyAsync(&eval_host, tls.d_eval_buf, sizeof(eval_host), cudaMemcpyDeviceToHost, st);
        if (err != cudaSuccess) {
            return fail_quant_stream("MXFP6 D2H(eval)", err);
        }
    }

    if (direct_tensor) {
        void * tensor_data = nullptr;
        int tensor_device = -1;
        if (!ggml_cuda_nvfp4_tensor_active_data(tensor, &tensor_data, &tensor_device)) {
            return false;
        }

        // Preserve the loaded runtime header pointer fields.  Direct selector
        // patching only needs to update scalar fallbacks; vector scales are
        // updated through their scale tensors.
        const ggml_tensor * weight_scale_t = tensor->src[0];
        const ggml_tensor * input_scale_t  = tensor->src[1];
        const bool weight_scale_vector = weight_scale_t != nullptr && !ggml_is_scalar(weight_scale_t);
        const bool input_scale_vector  = input_scale_t  != nullptr && !ggml_is_scalar(input_scale_t);
        if (!weight_scale_vector) {
            const float weight_scale =
                (isfinite(header_weight_scale) && header_weight_scale > 0.0f) ? header_weight_scale : x_scale_eff;
            err = cudaMemcpyAsync(
                (char *) tensor_data,
                &weight_scale,
                sizeof(weight_scale),
                cudaMemcpyHostToDevice,
                st);
            if (err != cudaSuccess) {
                return fail_quant_stream("MXFP6 H2D(weight_scale)", err);
            }
        }
        if (!input_scale_vector) {
            const float input_scale =
                (isfinite(header_input_scale) && header_input_scale > 0.0f) ? header_input_scale : 1.0f;
            err = cudaMemcpyAsync(
                (char *) tensor_data + sizeof(float),
                &input_scale,
                sizeof(input_scale),
                cudaMemcpyHostToDevice,
                st);
            if (err != cudaSuccess) {
                return fail_quant_stream("MXFP6 H2D(input_scale)", err);
            }
        }
        const size_t dst_plane_size = ggml_cuda_mxfp6_e2m3_plane_size(n_per_row, nrow);
        if (dst_plane_size < bytes_y) {
            return false;
        }
        char * dst_tiles = reinterpret_cast<char *>(tensor_data) + MXFP6_HEADER_OFFSET + (size_t) plane_index * dst_plane_size;
        err = cudaMemcpyAsync(dst_tiles, tls.d_y_buf, bytes_y, cudaMemcpyDeviceToDevice, st);
        if (err != cudaSuccess) {
            return fail_quant_stream("MXFP6 D2D(y)", err);
        }
        if (dst_plane_size > bytes_y) {
            err = cudaMemsetAsync(dst_tiles + bytes_y, 0, dst_plane_size - bytes_y, st);
            if (err != cudaSuccess) {
                return fail_quant_stream("MXFP6 memset(y pad)", err);
            }
        }
    } else {
        err = cudaMemcpyAsync(vy, tls.d_y_buf, bytes_y, cudaMemcpyDeviceToHost, st);
        if (err != cudaSuccess) {
            return fail_quant_stream("MXFP6 D2H(y)", err);
        }
    }
    err = cudaStreamSynchronize(st);
    if (err != cudaSuccess) {
        return fail_quant_stream("MXFP6 sync", err);
    }

    if (eval != nullptr) {
        float max_abs = 0.0f;
        std::memcpy(&max_abs, &eval_host.max_abs_bits, sizeof(max_abs));
        eval->sum_sq = eval_host.sum_sq;
        eval->sum_abs = eval_host.sum_abs;
        eval->max_abs = (double) max_abs;
        eval->count = (int64_t) eval_host.count;
    }

    return true;
}

extern "C" bool ggml_cuda_mxfp6_e2m3_quantize_eval(
        const void * x,
        bool x_bf16,
        float x_scale,
        void * vy,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        nvfp4_cuda_eval_result * eval,
        cudaStream_t stream) {
    return ggml_cuda_mxfp6_e2m3_quantize_eval_impl(
        x, x_bf16, x_scale, vy, nullptr, 0, nrow, n_per_row, qw,
        x_scale, 1.0f, eval, stream);
}

extern "C" bool ggml_cuda_mxfp6_e2m3_quantize_eval_to_tensor(
        const void * x,
        bool x_bf16,
        float x_scale,
        ggml_tensor * tensor,
        int64_t plane_index,
        int64_t nrow,
        int64_t n_per_row,
        const float * qw,
        float header_weight_scale,
        float header_input_scale,
        nvfp4_cuda_eval_result * eval,
        cudaStream_t stream) {
    return ggml_cuda_mxfp6_e2m3_quantize_eval_impl(
        x, x_bf16, x_scale, nullptr, tensor, plane_index, nrow, n_per_row, qw,
        header_weight_scale, header_input_scale, eval, stream);
}
