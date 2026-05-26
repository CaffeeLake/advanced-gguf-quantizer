#include "common.h"
#include "build-info.h"
#include "ggml-cuda.h"
#include "llama.h"
#include "quantize-imatrix.h"
#include "quantize-kld.h"
#include "quantize-mxfp6.h"
#include "quantize-options.h"
#include "quantize-selector-runtime.h"
#include "../../src/llama-context.h"
#include "../../src/llama-model.h"
#include "../../src/llama-model-loader.h"
#include "../../src/llama-quant.h"
#include "../../ggml/src/ggml-quants.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <cinttypes>
#include <cstdint>
#include <atomic>
#include <filesystem>
#include <thread>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#ifndef QK_MXFP6_E2M3
#define QK_MXFP6_E2M3 QK_MXFP6_SUB
#endif

static const char * const LLM_KV_QUANTIZE_IMATRIX_FILE       = "quantize.imatrix.file";
static const char * const LLM_KV_QUANTIZE_IMATRIX_DATASET    = "quantize.imatrix.dataset";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES  = "quantize.imatrix.entries_count";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS   = "quantize.imatrix.chunks_count";

// satisfies -Wmissing-declarations
int llama_quantize(int argc, char ** argv);

static constexpr int64_t NVFP4_SELECTOR_BLOCK_SIZE = 64;
static constexpr int     SELECTOR_CLASS_BUCKET_LIMIT = 2;
static constexpr int     SELECTOR_SPECIAL_LIMIT = 4;
static constexpr int     SELECTOR_EXCEPTION_LIMIT = 6;
static constexpr double  SELECTOR_FIRST_TENSOR_SOFT_FACTOR = 1.35;
static constexpr double  SELECTOR_FIRST_TENSOR_HARD_FACTOR = 1.85;
static constexpr double  SELECTOR_FIRST_TENSOR_PENALTY = 8.0;
static constexpr double  SELECTOR_HOLDOUT_WEIGHT = 0.35;
static constexpr double  SELECTOR_PPL_SIGMA = 2.0;
static constexpr double  SELECTOR_KLD_SIGMA = 2.0;
static constexpr double  SELECTOR_RMS_DP_SIGMA = 2.0;
static constexpr double  SELECTOR_SAME_TOP_SIGMA = 1.0;
static constexpr double  SELECTOR_LN_RATIO_ABS_FLOOR = 5e-4;
static constexpr double  SELECTOR_MEAN_KLD_ABS_FLOOR = 5e-5;
static constexpr double  SELECTOR_RMS_DP_ABS_FLOOR = 2e-4;
static constexpr double  SELECTOR_SAME_TOP_ABS_FLOOR = 2.5e-4;
static constexpr double  SELECTOR_ENTROPY_RMSE_ABS_FLOOR = 2e-4;
static constexpr double  SELECTOR_TOP_PROB_RMSE_ABS_FLOOR = 1.5e-4;
static constexpr double  SELECTOR_TOP_FLIP_WEIGHT_ABS_FLOOR = 5e-6;
static constexpr double  SELECTOR_P99_ABS_MARGIN = 7.5e-4;
static constexpr double  SELECTOR_P99_REL_MARGIN = 0.02;
static constexpr double  SELECTOR_P999_ABS_MARGIN = 1.5e-3;
static constexpr double  SELECTOR_P999_REL_MARGIN = 0.02;
static constexpr double  SELECTOR_MAX_KLD_ABS_MARGIN = 0.05;
static constexpr double  SELECTOR_MAX_KLD_REL_MARGIN = 0.05;
static constexpr double  SELECTOR_MXFP6_PPL_ABS_TOL = 2e-4;
static constexpr double  SELECTOR_MXFP6_PPL_REL_TOL = 0.01;
static constexpr double  SELECTOR_MXFP6_MEAN_KLD_ABS_TOL = 7.5e-5;
static constexpr double  SELECTOR_MXFP6_MEAN_KLD_REL_TOL = 0.01;
static constexpr int64_t SELECTOR_RESCUE_NVFP4_SAMPLE_MULT = 12;
static constexpr int64_t SELECTOR_RESCUE_NVFP4_MIN_BLOCKS = 8192;
static constexpr int64_t SELECTOR_RESCUE_NVFP4_REFINE_MULT = 3;
static constexpr int64_t SELECTOR_RESCUE_NVFP4_REFINE_MIN_BLOCKS = 16384;
static constexpr int64_t SELECTOR_RESCUE_NVFP4_GUARD_MULT = 2;
static constexpr int64_t SELECTOR_RESCUE_NVFP4_GUARD_MIN_BLOCKS = 32768;
static constexpr int     SELECTOR_RESCUE_NVFP4_POLICY_BUDGET = 72;
static constexpr double  SELECTOR_RESCUE_NVFP4_PHASE_TAIL_WEIGHT = 0.35;
static constexpr int     SELECTOR_RESCUE_NVFP4_REFINE_TOP = 12;
static constexpr int     SELECTOR_RESCUE_NVFP4_GUARD_TOP = 6;
static constexpr double  SELECTOR_RESCUE_NVFP4_MIN_GAIN = 5e-4;
static constexpr double  SELECTOR_RESCUE_NVFP4_MIN_REL_GAIN = 0.0025;
static constexpr double  SELECTOR_RESCUE_NVFP4_PREFER_GAIN = 9e-4;
static constexpr double  SELECTOR_RESCUE_NVFP4_PREFER_REL_GAIN = 0.005;
static constexpr double  SELECTOR_RESCUE_SPEED_WEIGHT = 1.0;
static constexpr double  SELECTOR_RESCUE_Q8_SPEED_PENALTY = 0.35;
static constexpr double  SELECTOR_RESCUE_BF16_SPEED_PENALTY = 1.5;
static constexpr double  SELECTOR_RESCUE_Q8_GAIN = 1.0;
static constexpr double  SELECTOR_RESCUE_BF16_GAIN = 1.35;
static constexpr double  SELECTOR_RESCUE_BF16_MARGIN = 1.20;
static constexpr int     MXFP6_SELECTOR_SCALE_POOL_PER_TENSOR = 2;
static constexpr int     MXFP6_SELECTOR_SCALE_POOL_TOP = 48;
static constexpr int     MXFP6_SELECTOR_SCALE_POOL_RESCORE_TOP = 24;
static constexpr int     MXFP6_SELECTOR_SCALE_POOL_APPLY_TOP = 0;
static constexpr double  MXFP6_SELECTOR_SCALE_POOL_SCORE_SLACK = 0.05;

static bool nvfp4_selector_reset_cuda_device(int device) {
#if defined(GGML_USE_CUDA) && (defined(__unix__) || defined(__APPLE__))
    nvfp4_clear_cuda_stream_cache();
    void * handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr) {
        handle = dlopen("libcudart.so.13", RTLD_LAZY | RTLD_NOLOAD);
    }
    if (handle == nullptr) {
        handle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_NOLOAD);
    }
    if (handle == nullptr) {
        handle = dlopen("libcudart.so", RTLD_LAZY);
    }
    if (handle == nullptr) {
        fprintf(stderr, "%s: selector stage-b CUDA reset skipped: libcudart is not available through dlopen\n", __func__);
        return false;
    }

    using cuda_set_device_fn = int (*)(int);
    using cuda_noarg_fn = int (*)();
    using cuda_get_error_string_fn = const char * (*)(int);
    auto cuda_set_device = reinterpret_cast<cuda_set_device_fn>(dlsym(handle, "cudaSetDevice"));
    auto cuda_device_synchronize = reinterpret_cast<cuda_noarg_fn>(dlsym(handle, "cudaDeviceSynchronize"));
    auto cuda_device_reset = reinterpret_cast<cuda_noarg_fn>(dlsym(handle, "cudaDeviceReset"));
    auto cuda_get_error_string = reinterpret_cast<cuda_get_error_string_fn>(dlsym(handle, "cudaGetErrorString"));
    if (cuda_set_device == nullptr || cuda_device_synchronize == nullptr || cuda_device_reset == nullptr) {
        fprintf(stderr, "%s: selector stage-b CUDA reset skipped: required libcudart symbols are unavailable\n", __func__);
        return false;
    }

    int err = cuda_set_device(device);
    if (err == 0) {
        err = cuda_device_synchronize();
    }
    if (err == 0) {
        err = cuda_device_reset();
    }
    if (err != 0) {
        fprintf(stderr,
            "%s: selector stage-b CUDA reset failed: %s (%d)\n",
            __func__,
            cuda_get_error_string != nullptr ? cuda_get_error_string(err) : "unknown CUDA error",
            err);
        return false;
    }
    nvfp4_clear_cuda_stream_cache();
    return true;
#else
    (void) device;
    return false;
#endif
}

struct selector_rank_config {
    struct mxfp6_config {
        double ppl_abs_tol = SELECTOR_MXFP6_PPL_ABS_TOL;
        double ppl_rel_tol = SELECTOR_MXFP6_PPL_REL_TOL;
        double mean_kld_abs_tol = SELECTOR_MXFP6_MEAN_KLD_ABS_TOL;
        double mean_kld_rel_tol = SELECTOR_MXFP6_MEAN_KLD_REL_TOL;
    };

    double kld_threshold = -1.0;
    double p99_threshold = -1.0;
    double p999_threshold = -1.0;
    double max_kld_threshold = -1.0;
    double kld_penalty = 0.0;
    double p99_penalty = 0.0;
    double p999_penalty = 0.0;
    double max_kld_penalty = 0.0;
    double holdout_weight = SELECTOR_HOLDOUT_WEIGHT;
    bool kld_hard_gate = false;
    bool p99_hard_gate = false;
    bool p999_hard_gate = false;
    bool max_kld_hard_gate = false;
    double ppl_sigma = SELECTOR_PPL_SIGMA;
    double kld_sigma = SELECTOR_KLD_SIGMA;
    double rms_dp_sigma = SELECTOR_RMS_DP_SIGMA;
    double same_top_sigma = SELECTOR_SAME_TOP_SIGMA;
    double ln_ratio_abs_floor = SELECTOR_LN_RATIO_ABS_FLOOR;
    double mean_kld_abs_floor = SELECTOR_MEAN_KLD_ABS_FLOOR;
    double rms_dp_abs_floor = SELECTOR_RMS_DP_ABS_FLOOR;
    double same_top_abs_floor = SELECTOR_SAME_TOP_ABS_FLOOR;
    double entropy_rmse_abs_floor = SELECTOR_ENTROPY_RMSE_ABS_FLOOR;
    double top_prob_rmse_abs_floor = SELECTOR_TOP_PROB_RMSE_ABS_FLOOR;
    double top_flip_weight_abs_floor = SELECTOR_TOP_FLIP_WEIGHT_ABS_FLOOR;
    double p99_abs_margin = SELECTOR_P99_ABS_MARGIN;
    double p99_rel_margin = SELECTOR_P99_REL_MARGIN;
    double p999_abs_margin = SELECTOR_P999_ABS_MARGIN;
    double p999_rel_margin = SELECTOR_P999_REL_MARGIN;
    double max_kld_abs_margin = SELECTOR_MAX_KLD_ABS_MARGIN;
    double max_kld_rel_margin = SELECTOR_MAX_KLD_REL_MARGIN;
    mxfp6_config mxfp6;
};

struct nvfp4_selector_policy {
    std::string name;
    nvfp4_cuda_runtime_cfg cfg = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    double proxy_score = std::numeric_limits<double>::infinity();
    double proxy_rmse = std::numeric_limits<double>::infinity();
    double proxy_abs_mean = std::numeric_limits<double>::infinity();
    double proxy_max_abs = std::numeric_limits<double>::infinity();
    double survey_proxy_score = std::numeric_limits<double>::infinity();
    double survey_proxy_rmse = std::numeric_limits<double>::infinity();
    double survey_proxy_abs_mean = std::numeric_limits<double>::infinity();
    double survey_proxy_max_abs = std::numeric_limits<double>::infinity();
    double first_tensor_rmse = std::numeric_limits<double>::infinity();
    double first_tensor_abs_mean = std::numeric_limits<double>::infinity();
    double first_tensor_max_abs = std::numeric_limits<double>::infinity();
    bool proxy_rejected = false;
    bool has_survey = false;
    nvfp4_selector_derived_metrics measured;
    nvfp4_selector_derived_metrics measured_holdout;
    bool has_holdout = false;
    bool measured_pass = false;
    double measured_score = std::numeric_limits<double>::infinity();
};

enum class nvfp4_selector_tensor_class : uint8_t {
    EMBEDDING_OUTPUT = 0,
    ATTN_QKV,
    ATTN_OUT,
    ROUTER,
    EXPERT_UP_GATE,
    EXPERT_DOWN,
    DENSE_MLP,
    SSM,
    OTHER,
};

struct nvfp4_selector_device_snapshot {
    void * data = nullptr;
    size_t nbytes = 0;

    nvfp4_selector_device_snapshot() = default;
    nvfp4_selector_device_snapshot(const nvfp4_selector_device_snapshot &) = delete;
    nvfp4_selector_device_snapshot & operator=(const nvfp4_selector_device_snapshot &) = delete;

    nvfp4_selector_device_snapshot(nvfp4_selector_device_snapshot && other) noexcept {
        data = other.data;
        nbytes = other.nbytes;
        other.data = nullptr;
        other.nbytes = 0;
    }

    nvfp4_selector_device_snapshot & operator=(nvfp4_selector_device_snapshot && other) noexcept {
        if (this != &other) {
            reset();
            data = other.data;
            nbytes = other.nbytes;
            other.data = nullptr;
            other.nbytes = 0;
        }
        return *this;
    }

    ~nvfp4_selector_device_snapshot() {
        reset();
    }

    void reset() {
        if (data != nullptr) {
            ggml_cuda_tensor_snapshot_free(data);
            data = nullptr;
            nbytes = 0;
        }
    }

    bool valid_for(size_t bytes) const {
        return data != nullptr && nbytes == bytes && bytes > 0;
    }

    bool capture(const ggml_tensor * tensor, size_t bytes) {
        reset();
        void * snapshot = nullptr;
        if (!ggml_cuda_tensor_snapshot(tensor, bytes, &snapshot, nullptr)) {
            return false;
        }
        data = snapshot;
        nbytes = bytes;
        return true;
    }

    bool restore(ggml_tensor * tensor, size_t bytes) const {
        return valid_for(bytes) && ggml_cuda_tensor_restore(tensor, data, bytes, nullptr);
    }
};

struct nvfp4_selector_device_sample_entry {
    int64_t slice_index = -1;
    int64_t sample_nb = 0;
    int64_t sample_phase = 0;
    int32_t source_type = GGML_TYPE_COUNT;
    const void * source_ptr = nullptr;
    const float * imatrix_ptr = nullptr;
    float tune_x_mul = 1.0f;
    void * cache = nullptr;
    const float * x_device = nullptr;
    const float * tune_x_device = nullptr;
    const float * qw_device = nullptr;
    int64_t n_device = 0;

    ~nvfp4_selector_device_sample_entry() {
        if (cache != nullptr) {
            nvfp4_sample_cache_cuda_free(cache);
        }
    }

    bool matches(
            int64_t slice,
            int64_t nb,
            int64_t phase,
            int32_t type,
            const void * src,
            const float * imat,
            float tune_mul) const {
        return
            slice_index == slice &&
            sample_nb == nb &&
            sample_phase == phase &&
            source_type == type &&
            source_ptr == src &&
            imatrix_ptr == imat &&
            std::fabs(tune_x_mul - tune_mul) <= 1e-12f * std::max(1.0f, std::max(std::fabs(tune_x_mul), std::fabs(tune_mul)));
    }
};

struct nvfp4_selector_device_sample_bank {
    std::mutex mutex;
    std::vector<std::unique_ptr<nvfp4_selector_device_sample_entry>> entries;
};

struct nvfp4_selector_device_sample_view {
    const float * x_device = nullptr;
    const float * tune_x_device = nullptr;
    const float * qw_device = nullptr;
    int64_t n_device = 0;
    float x_scale = 1.0f;
};

struct nvfp4_selector_binding {
    std::string name;
    nvfp4_selector_tensor_class cls = nvfp4_selector_tensor_class::OTHER;
    int bucket = -1;
    int32_t layer = -1;
    ggml_tensor * target = nullptr;
    ggml_tensor * target_scale = nullptr;
    ggml_tensor * target_input_scale = nullptr;
    ggml_tensor * source = nullptr;
    const float * imatrix_row = nullptr;
    float source_tensor_scale = 1.0f;
    std::vector<float> quant_tensor_scales;
    size_t source_nbytes = 0;
    size_t target_nbytes = 0;
    size_t target_scale_nbytes = 0;
    size_t target_input_scale_nbytes = 0;
    std::vector<uint8_t> original_target_bytes;
    std::vector<uint8_t> working_target_bytes;
    std::vector<uint8_t> original_scale_bytes;
    std::vector<uint8_t> working_scale_bytes;
    std::vector<uint8_t> original_input_scale_bytes;
    std::vector<uint8_t> working_input_scale_bytes;
    nvfp4_selector_device_snapshot original_target_device;
    nvfp4_selector_device_snapshot original_scale_device;
    nvfp4_selector_device_snapshot original_input_scale_device;
    std::shared_ptr<nvfp4_selector_device_sample_bank> device_samples;
};

struct nvfp4_selector_tensor_sensitivity {
    std::string name;
    nvfp4_selector_tensor_class cls = nvfp4_selector_tensor_class::OTHER;
    int bucket = -1;
    int32_t layer = -1;
    size_t binding_index = 0;
    size_t target_nbytes = 0;
    size_t q8_nbytes = 0;
    size_t bf16_nbytes = 0;
    double proxy_score = 0.0;
    double proxy_rmse = 0.0;
    double proxy_abs_mean = 0.0;
    double proxy_max_abs = 0.0;
    double q8_roi = 0.0;
    double bf16_roi = 0.0;
    size_t q8_delta_bytes = 0;
    size_t bf16_delta_bytes = 0;
    double q8_speed_penalty = 0.0;
    double bf16_speed_penalty = 0.0;
    ggml_type q8_type = GGML_TYPE_Q8_0;
    bool has_alt_nvfp4_cfg = false;
    nvfp4_cuda_runtime_cfg alt_nvfp4_cfg = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    int64_t alt_nvfp4_sample_blocks = 0;
    std::string alt_nvfp4_policy_name;
    double alt_nvfp4_proxy_score = std::numeric_limits<double>::infinity();
    double alt_nvfp4_gain = 0.0;
    double alt_nvfp4_gain_rel = 0.0;
    double suggested_priority = 0.0;
    ggml_type suggested_type = GGML_TYPE_COUNT;
};

static std::string format_rescue_suggested_type(const nvfp4_selector_tensor_sensitivity & s) {
    tensor_type_option opt;
    opt.type = s.suggested_type;
    if (s.suggested_type == GGML_TYPE_NVFP4 && s.has_alt_nvfp4_cfg) {
        opt.has_nvfp4_cfg = true;
        opt.nvfp4_cfg = s.alt_nvfp4_cfg;
        opt.nvfp4_sample_blocks = s.alt_nvfp4_sample_blocks;
        opt.nvfp4_policy_name = s.alt_nvfp4_policy_name;
    }
    return format_tensor_type_value(opt);
}

static int64_t nvfp4_selector_autotune_sample_blocks(int64_t nb_total, int64_t override_cap);

enum class quantize_stream_key_mode {
    HASHED_SLOT,
    LINEAR_SLOT,
};

struct quantize_cuda_eval_params_t {
    const void * x = nullptr;
    bool x_bf16 = false;
    uint8_t * out = nullptr;
    ggml_tensor * target = nullptr;
    int64_t slice_index = 0;
    int64_t nrow = 0;
    int64_t n_per_row = 0;
    const float * qw = nullptr;
    float x_scale = 1.0f;
    float a = 1.0f;
    float b = 1.0f;
    float weight_scale = 1.0f;
    float input_scale = 1.0f;
    const nvfp4_cuda_runtime_cfg * nvfp4_cfg = nullptr;
    nvfp4_cuda_eval_result * eval = nullptr;
    void * stream_key = nullptr;
};

template <ggml_type type>
struct ggml_quantize_type_traits;

template <>
struct ggml_quantize_type_traits<GGML_TYPE_NVFP4> {
    static constexpr ggml_type quant_type = GGML_TYPE_NVFP4;
    static constexpr int64_t block_size = NVFP4_SELECTOR_BLOCK_SIZE;
    static constexpr bool require_scale_bytes = false;
    static constexpr bool supports_sample_only = true;
    static constexpr const char * thread_control_key = "LLAMA_NVFP4_SELECTOR_THREADS";
    static constexpr quantize_stream_key_mode stream_key_mode = quantize_stream_key_mode::HASHED_SLOT;
    static constexpr uintptr_t single_stream_slot = 2;
    static constexpr uintptr_t worker_stream_slot = 16;
    static constexpr bool single_stream_slot_add_slice = false;

    static bool quantize_eval(const quantize_cuda_eval_params_t & p) {
        if (p.target != nullptr) {
            return nvfp4_quantize_cuda_ab_eval_to_tensor_cfg(
                p.x, p.x_bf16, p.target, p.slice_index, p.nrow, p.n_per_row, p.qw,
                p.x_scale, p.a, p.b, p.nvfp4_cfg, p.eval, p.stream_key);
        }
        return nvfp4_quantize_cuda_ab_eval_cfg(
            p.x, p.x_bf16, p.out, p.nrow, p.n_per_row, p.qw,
            p.x_scale, p.a, p.b, p.nvfp4_cfg, p.eval, p.stream_key);
    }
};

template <>
struct ggml_quantize_type_traits<GGML_TYPE_MXFP6_E2M3> {
    static constexpr ggml_type quant_type = GGML_TYPE_MXFP6_E2M3;
    static constexpr int64_t block_size = QK_MXFP6_E2M3;
    static constexpr bool require_scale_bytes = true;
    static constexpr bool supports_sample_only = false;
    static constexpr const char * thread_control_key = nullptr;
    static constexpr quantize_stream_key_mode stream_key_mode = quantize_stream_key_mode::LINEAR_SLOT;
    static constexpr uintptr_t single_stream_slot = 0x3000;
    static constexpr uintptr_t worker_stream_slot = 0x3000;
    static constexpr bool single_stream_slot_add_slice = true;

    static bool quantize_eval(const quantize_cuda_eval_params_t & p) {
        if (p.target != nullptr) {
            return mxfp6_e2m3_quantize_cuda_eval_to_tensor(
                p.x, p.x_bf16, p.target, p.slice_index,
                p.nrow, p.n_per_row, p.qw, p.weight_scale,
                p.weight_scale, p.input_scale,
                p.eval, p.stream_key);
        }
        return mxfp6_e2m3_quantize_cuda_eval(
            p.x, p.x_bf16, p.out,
            p.nrow, p.n_per_row, p.qw, p.weight_scale,
            p.eval, p.stream_key);
    }
};

struct quantize_binding_slice_accum_t {
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    double max_abs = 0.0;
    int64_t count = 0;
    bool ok = true;
};

struct quantize_binding_shape_t {
    int64_t n_per_row = 0;
    int64_t nrows = 0;
    int64_t n_slices = 0;
    int64_t nb_total = 0;
    bool sample_only = false;
    int64_t sample_nb_for_bytes = 0;
    int64_t out_nrow = 0;
    int64_t out_n_per_row = 0;
    size_t out_slice_bytes = 0;
};

template <ggml_type type>
static bool quantize_binding_prepare_shape(
    const nvfp4_selector_binding & binding,
    int64_t sample_blocks_override,
    int64_t sample_phase,
    const char * caller,
    quantize_binding_shape_t & shape) {
    using traits = ggml_quantize_type_traits<type>;
    shape.n_per_row = binding.source->ne[0];
    shape.nrows = binding.source->ne[1];
    shape.n_slices = std::max<int64_t>(1, binding.source->ne[2]);

    if (shape.n_per_row <= 0 || shape.nrows <= 0 || shape.n_per_row % traits::block_size != 0) {
        fprintf(stderr,
            "%s: invalid %s quantize shape for %s (rows=%" PRId64 " cols=%" PRId64 " block=%" PRId64 ")\n",
            caller,
            ggml_type_name(traits::quant_type),
            binding.name.c_str(),
            shape.nrows,
            shape.n_per_row,
            traits::block_size);
        return false;
    }

    shape.sample_only = sample_blocks_override > 0;
    if constexpr (!traits::supports_sample_only) {
        if (shape.sample_only) {
            fprintf(stderr,
                "%s: %s quantize binding does not support sampled output for %s\n",
                caller,
                ggml_type_name(traits::quant_type),
                binding.name.c_str());
            return false;
        }
    }

    shape.nb_total = (shape.nrows * shape.n_per_row) / traits::block_size;
    if constexpr (traits::supports_sample_only) {
        shape.sample_nb_for_bytes = shape.sample_only
            ? nvfp4_selector_autotune_sample_blocks(shape.nb_total, sample_blocks_override)
            : 0;
    } else {
        shape.sample_nb_for_bytes = 0;
    }
    if (shape.sample_only && shape.sample_nb_for_bytes <= 0) {
        fprintf(stderr,
            "%s: empty selector sample for %s (rows=%" PRId64 " cols=%" PRId64 " slices=%" PRId64 " blocks=%" PRId64 " requested=%" PRId64 " phase=%" PRId64 ")\n",
            caller,
            binding.name.c_str(),
            shape.nrows,
            shape.n_per_row,
            shape.n_slices,
            shape.nb_total,
            sample_blocks_override,
            sample_phase);
        return false;
    }

    shape.out_nrow = shape.sample_only ? 1 : shape.nrows;
    shape.out_n_per_row = shape.sample_only ? shape.sample_nb_for_bytes * traits::block_size : shape.n_per_row;
    shape.out_slice_bytes = (size_t) shape.out_nrow * ggml_row_size(traits::quant_type, shape.out_n_per_row);
    return true;
}

template <ggml_type type>
static int quantize_binding_thread_count(int64_t n_slices, int nthread) {
    using traits = ggml_quantize_type_traits<type>;
    int64_t requested = std::max(1, nthread);
    if constexpr (traits::thread_control_key != nullptr) {
        requested = quantize_control_i64(traits::thread_control_key, requested);
    }
    return (int) std::max<int64_t>(1, std::min<int64_t>(std::max<int64_t>(1, n_slices), requested));
}

template <ggml_type type>
static void * quantize_binding_stream_key(
    uintptr_t stream_seed,
    int worker_index,
    int64_t slice_index,
    bool single_worker) {
    using traits = ggml_quantize_type_traits<type>;
    const uintptr_t slot = single_worker
        ? traits::single_stream_slot + (traits::single_stream_slot_add_slice ? (uintptr_t) slice_index : 0)
        : traits::worker_stream_slot + (uintptr_t) worker_index;

    switch (traits::stream_key_mode) {
        case quantize_stream_key_mode::HASHED_SLOT:
            return reinterpret_cast<void *>(stream_seed ^ (slot * UINT64_C(0x9E3779B97F4A7C15)));
        case quantize_stream_key_mode::LINEAR_SLOT:
            return reinterpret_cast<void *>(slot);
    }

    GGML_ABORT("unreachable");
}

template <ggml_type type, typename SliceFn>
static bool quantize_run_binding_slices(
    int64_t n_slices,
    int nthread,
    uintptr_t stream_seed,
    SliceFn && run_slice,
    double & sum_sq,
    double & sum_abs,
    double & max_abs,
    int64_t & count) {
    static_assert(ggml_quantize_type_traits<type>::quant_type == type, "quantize type trait mismatch");

    sum_sq = 0.0;
    sum_abs = 0.0;
    max_abs = 0.0;
    count = 0;

    const int worker_count = (int) std::max<int64_t>(1, std::min<int64_t>(
        std::max<int64_t>(1, n_slices),
        nthread > 0 ? nthread : 1));
    if (worker_count <= 1) {
        quantize_binding_slice_accum_t acc;
        for (int64_t i03 = 0; i03 < n_slices; ++i03) {
            run_slice(i03, quantize_binding_stream_key<type>(stream_seed, 0, i03, true), acc);
            if (!acc.ok) {
                return false;
            }
        }
        sum_sq = acc.sum_sq;
        sum_abs = acc.sum_abs;
        max_abs = acc.max_abs;
        count = acc.count;
        return true;
    }

    std::atomic<int64_t> next_slice { 0 };
    std::vector<std::thread> workers;
    std::vector<quantize_binding_slice_accum_t> accs((size_t) worker_count);
    workers.reserve((size_t) worker_count);
    for (int wi = 0; wi < worker_count; ++wi) {
        workers.emplace_back([&, wi]() {
            void * stream_key = quantize_binding_stream_key<type>(stream_seed, wi, -1, false);
            while (accs[(size_t) wi].ok) {
                const int64_t i03 = next_slice.fetch_add(1, std::memory_order_relaxed);
                if (i03 >= n_slices) {
                    break;
                }
                run_slice(i03, stream_key, accs[(size_t) wi]);
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    for (const auto & acc : accs) {
        if (!acc.ok) {
            return false;
        }
        sum_sq += acc.sum_sq;
        sum_abs += acc.sum_abs;
        max_abs = std::max(max_abs, acc.max_abs);
        count += acc.count;
    }
    return true;
}

template <ggml_type type>
static bool quantize_binding_has_target(
    const nvfp4_selector_binding & binding,
    const char * caller) {
    using traits = ggml_quantize_type_traits<type>;

    if (binding.source != nullptr &&
            binding.target != nullptr &&
            binding.target->type == traits::quant_type &&
            (!traits::require_scale_bytes || binding.target_scale_nbytes != 0)) {
        return true;
    }

    fprintf(stderr,
        "%s: bad %s quantize binding for %s (source=%p target=%p target_type=%s scale=%p scale_bytes=%zu)\n",
        caller,
        ggml_type_name(traits::quant_type),
        binding.name.c_str(),
        (void *) binding.source,
        (void *) binding.target,
        binding.target ? ggml_type_name(binding.target->type) : "null",
        (void *) binding.target_scale,
        binding.target_scale_nbytes);
    return false;
}

template <ggml_type type, typename Binding, typename PrepareFn, typename SliceFn>
static bool quantize_binding_quantize(
    Binding & binding,
    int nthread,
    int64_t sample_blocks_override,
    int64_t sample_phase,
    uintptr_t stream_seed,
    const char * caller,
    PrepareFn && prepare,
    SliceFn && run_slice,
    double & sum_sq,
    double & sum_abs,
    double & max_abs,
    int64_t & count) {
    using traits = ggml_quantize_type_traits<type>;

    if (!quantize_binding_has_target<traits::quant_type>(binding, caller)) {
        return false;
    }

    quantize_binding_shape_t shape;
    if (!quantize_binding_prepare_shape<traits::quant_type>(
            binding, sample_blocks_override, sample_phase, caller, shape)) {
        return false;
    }

    if (binding.source->data == nullptr) {
        fprintf(stderr,
            "%s: %s source tensor %s has null data pointer (type=%s rows=%" PRId64 " cols=%" PRId64 " slices=%" PRId64 ")\n",
            caller,
            ggml_type_name(traits::quant_type),
            binding.name.c_str(),
            ggml_type_name(binding.source->type),
            shape.nrows,
            shape.n_per_row,
            shape.n_slices);
        return false;
    }

    if (!prepare(shape)) {
        return false;
    }

    const int selector_threads = quantize_binding_thread_count<traits::quant_type>(shape.n_slices, nthread);
    auto run_slice_with_shape = [&](int64_t i03, void * stream_key, quantize_binding_slice_accum_t & acc) {
        run_slice(i03, stream_key, shape, acc);
    };
    return quantize_run_binding_slices<traits::quant_type>(
        shape.n_slices,
        selector_threads,
        stream_seed,
        run_slice_with_shape,
        sum_sq,
        sum_abs,
        max_abs,
        count);
}

static int64_t nvfp4_selector_sample_phase_stride(
    int64_t nb_total,
    nvfp4_selector_tensor_class cls) {
    int64_t stride = std::max<int64_t>(1, nb_total / 11);
    switch (cls) {
        case nvfp4_selector_tensor_class::ATTN_QKV:
        case nvfp4_selector_tensor_class::ATTN_OUT:
        case nvfp4_selector_tensor_class::SSM:
            stride = std::max<int64_t>(stride, std::max<int64_t>(1, nb_total / 7));
            break;
        case nvfp4_selector_tensor_class::EXPERT_UP_GATE:
        case nvfp4_selector_tensor_class::EXPERT_DOWN:
            stride = std::max<int64_t>(stride, std::max<int64_t>(1, nb_total / 5));
            break;
        default:
            break;
    }
    return std::max<int64_t>(1, stride);
}

static float gguf_get_nvfp4_tensor_scale(const gguf_context * ctx_gguf, const char * tensor_name) {
    if (!ctx_gguf || !tensor_name) {
        return 1.0f;
    }

    float weight_scale = 1.0f;
    float weight_scale_2 = 1.0f;

    const std::string key = std::string(tensor_name) + ".weight_scale";
    const int key_id = gguf_find_key(ctx_gguf, key.c_str());
    if (key_id >= 0 && gguf_get_kv_type(ctx_gguf, key_id) == GGUF_TYPE_FLOAT32) {
        weight_scale = gguf_get_val_f32(ctx_gguf, key_id);
    }

    const std::string key_2 = std::string(tensor_name) + ".weight_scale_2";
    const int key_id_2 = gguf_find_key(ctx_gguf, key_2.c_str());
    if (key_id_2 >= 0 && gguf_get_kv_type(ctx_gguf, key_id_2) == GGUF_TYPE_FLOAT32) {
        weight_scale_2 = gguf_get_val_f32(ctx_gguf, key_id_2);
    }

    const float out = weight_scale * weight_scale_2;
    return (std::isfinite(out) && out > 0.0f) ? out : 1.0f;
}

static bool nvfp4_selector_gpu_kld_parity_ok(
        const nvfp4_selector_kld_metrics & cpu,
        const nvfp4_selector_kld_metrics & gpu,
        std::string * reason) {
    auto fail = [&](const std::string & why) {
        if (reason != nullptr) {
            *reason = why;
        }
        return false;
    };
    if (cpu.count != gpu.count) {
        return fail("count mismatch cpu=" + std::to_string(cpu.count) + " gpu=" + std::to_string(gpu.count));
    }
    if (cpu.count <= 0) {
        return fail("empty metric set");
    }

    auto close = [](double a, double b, double abs_tol, double rel_tol) {
        const double diff = std::fabs(a - b);
        const double scale = std::max({1.0, std::fabs(a), std::fabs(b)});
        return diff <= abs_tol || diff <= rel_tol * scale;
    };
    const double inv = 1.0 / (double) cpu.count;
    const double cpu_ln  = cpu.sum_nll * inv;
    const double gpu_ln  = gpu.sum_nll * inv;
    const double cpu_kld = cpu.sum_kld * inv;
    const double gpu_kld = gpu.sum_kld * inv;
    if (!close(cpu_ln, gpu_ln, 1e-4, 5e-4)) {
        char buf[192];
        snprintf(buf, sizeof(buf), "mean ln mismatch cpu=%.9f gpu=%.9f", cpu_ln, gpu_ln);
        return fail(buf);
    }
    if (!close(cpu_kld, gpu_kld, 1e-4, 5e-4)) {
        char buf[192];
        snprintf(buf, sizeof(buf), "mean kld mismatch cpu=%.9f gpu=%.9f", cpu_kld, gpu_kld);
        return fail(buf);
    }
    if (!close(cpu.max_kld, gpu.max_kld, 1e-3, 1e-3)) {
        char buf[192];
        snprintf(buf, sizeof(buf), "max kld mismatch cpu=%.9f gpu=%.9f", cpu.max_kld, gpu.max_kld);
        return fail(buf);
    }
    return true;
}

class nvfp4_selector_kld_worker_pool {
public:
    explicit nvfp4_selector_kld_worker_pool(int n_threads) {
        const int capped = std::max(0, n_threads);
        workers.reserve((size_t) capped);
        for (int i = 0; i < capped; ++i) {
            workers.emplace_back([this, i]() { worker_loop(i); });
        }
    }

    ~nvfp4_selector_kld_worker_pool() {
        {
            std::lock_guard<std::mutex> lock(mu);
            stopping = true;
            generation++;
        }
        cv.notify_all();
        for (std::thread & worker : workers) {
            worker.join();
        }
    }

    int capacity() const {
        return (int) workers.size();
    }

    void run(int n_tasks, std::function<void(int)> fn) {
        if (n_tasks <= 0) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mu);
            task = std::move(fn);
            active = std::min(n_tasks, capacity());
            remaining = active;
            generation++;
        }
        cv.notify_all();

        std::unique_lock<std::mutex> lock(mu);
        done_cv.wait(lock, [this]() { return remaining == 0; });
        task = {};
    }

private:
    void worker_loop(int worker_index) {
        uint64_t seen_generation = 0;
        for (;;) {
            std::function<void(int)> local_task;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [this, &seen_generation]() {
                    return stopping || generation != seen_generation;
                });
                if (stopping) {
                    return;
                }
                seen_generation = generation;
                if (worker_index >= active) {
                    continue;
                }
                local_task = task;
            }

            local_task(worker_index);

            {
                std::lock_guard<std::mutex> lock(mu);
                if (--remaining == 0) {
                    done_cv.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv;
    std::condition_variable done_cv;
    std::function<void(int)> task;
    int active = 0;
    int remaining = 0;
    uint64_t generation = 0;
    bool stopping = false;
};

static bool nvfp4_selector_eval_kld_subset(
    llama_context * ctx,
    const nvfp4_selector_kld_subset & kld,
    int32_t n_batch,
    nvfp4_selector_kld_metrics & out,
    bool collect_kld_distribution,
    const std::string & progress_label = "selector kld eval") {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    if (kld.n_vocab != llama_vocab_n_tokens(vocab)) {
        fprintf(stderr, "%s: selector vocab mismatch\n", __func__);
        return false;
    }
    if (kld.n_ctx > (int32_t) llama_n_ctx(ctx)) {
        fprintf(stderr, "%s: selector ctx too small\n", __func__);
        return false;
    }

    if (n_batch <= 0) {
        n_batch = kld.n_ctx;
    }
    int n_seq = std::max(1, n_batch / kld.n_ctx);
    n_seq = std::min(n_seq, (int) llama_n_seq_max(ctx));
    n_seq = std::min(n_seq, std::max(1, (int) llama_n_ctx(ctx) / kld.n_ctx));
    if (n_seq < 1) {
        n_seq = 1;
    }
    n_batch = std::max(1, n_batch);
    const int num_batches = (kld.n_ctx + n_batch - 1) / n_batch;
    out.collect_kld_values = collect_kld_distribution;
    if (collect_kld_distribution) {
        out.kld_values.clear();
    }
    const int64_t metric_thread_cap = std::max<int64_t>(1, kld.n_chunk * kld.n_score);
    int64_t metric_thread_request = nvfp4_selector_kld_threads_override();
    if (metric_thread_request <= 0) {
        metric_thread_request = std::max<unsigned int>(1, std::thread::hardware_concurrency());
    }
    const int metric_threads = (int) std::max<int64_t>(1, std::min<int64_t>(metric_thread_cap, metric_thread_request));
    const bool progress = true;
    const int64_t max_decode_tokens =
        (int64_t) std::max(1, std::min(n_batch, kld.n_ctx)) * (int64_t) n_seq;
    const bool gpu_kld_enabled = llama_n_ubatch(ctx) >= (uint32_t) max_decode_tokens;
    if (!gpu_kld_enabled && progress) {
        fprintf(stderr,
            "%s: selector GPU KLD disabled for this eval because n_ubatch=%u < decode_tokens=%" PRId64
            "; using CPU reduction to avoid raw-logits row mismatch\n",
            __func__,
            llama_n_ubatch(ctx),
            max_decode_tokens);
    }
    static std::atomic<int> gpu_kld_state { 0 };
    struct scoped_selector_gpu_kld_active {
        llama_context * ctx = nullptr;
        bool active = false;
        scoped_selector_gpu_kld_active(llama_context * ctx_, bool active_) : ctx(ctx_), active(active_) {
            if (active) {
                llama_internal_set_selector_gpu_kld_active(ctx, true);
            }
        }
        ~scoped_selector_gpu_kld_active() {
            if (active) {
                llama_internal_set_selector_gpu_kld_active(ctx, false);
            }
        }
    };

    std::unique_ptr<nvfp4_selector_kld_worker_pool> cpu_kld_pool;
    llama_batch batch = llama_batch_init(std::min(n_batch, kld.n_ctx * n_seq), 0, 1);
    const int64_t progress_total =
        (int64_t) ((kld.n_chunk + n_seq - 1) / n_seq) * (int64_t) std::max(1, num_batches);
    int64_t progress_done = 0;
    nvfp4_selector_progress_heartbeat heartbeat(progress_label, progress_total);
    {
        char detail[192];
        snprintf(detail, sizeof(detail),
            "chunks=%d n_seq=%d ctx=%d batch=%d splits=%d score_tokens=%d threads=%d",
            kld.n_chunk, n_seq, kld.n_ctx, n_batch, num_batches, kld.n_score, metric_threads);
        heartbeat.update(progress_done, progress_total, detail, true);
    }
    for (int ci = 0; ci < kld.n_chunk; ci += n_seq) {
        const int n_seq_batch = std::min(n_seq, kld.n_chunk - ci);
        if (progress) {
            fprintf(stderr,
                "selector kld eval chunk [%d/%d] n_seq=%d ctx=%d batch=%d splits=%d score_tokens=%d threads=%d\n",
                ci + 1,
                kld.n_chunk,
                n_seq_batch,
                kld.n_ctx,
                n_batch,
                num_batches,
                kld.n_score,
                metric_threads);
        }
        {
            char detail[160];
            snprintf(detail, sizeof(detail), "chunk [%d/%d] preparing", ci + 1, kld.n_chunk);
            heartbeat.detail(detail);
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<float> logits_all;
        if (num_batches > 1) {
            logits_all.reserve((size_t) n_seq_batch * (size_t) kld.n_score * (size_t) kld.n_vocab);
        }

        bool chunk_skipped_host_logits = false;
        bool chunk_gpu_complete = gpu_kld_enabled && gpu_kld_state.load(std::memory_order_relaxed) >= 0;
        bool chunk_gpu_failed = false;
        int chunk_gpu_rows = 0;
        nvfp4_selector_kld_metrics chunk_gpu_metrics;
        chunk_gpu_metrics.collect_kld_values = collect_kld_distribution;
        for (int jb = 0; jb < num_batches; ++jb) {
            const int batch_start = jb * n_batch;
            const int batch_size = std::min(kld.n_ctx - batch_start, n_batch);
            const int s_begin = std::max(0, batch_start - kld.first);
            const int s_end = std::min(kld.n_score, batch_start + batch_size - kld.first);
            const int score_count = std::max(0, s_end - s_begin);
            {
                char detail[192];
                snprintf(detail, sizeof(detail), "chunk [%d/%d] split [%d/%d] decoding", ci + 1, kld.n_chunk, jb + 1, num_batches);
                heartbeat.detail(detail);
            }
            common_batch_clear(batch);
            int n_outputs = 0;
            for (int seq = 0; seq < n_seq_batch; ++seq) {
                const llama_token * chunk_tokens = kld.tokens.data() + (size_t) (ci + seq) * (size_t) kld.n_ctx;
                for (int t = 0; t < batch_size; ++t) {
                    const int pos = batch_start + t;
                    llama_token tok = chunk_tokens[pos];
                    if (add_bos && jb == 0 && t == 0) {
                        tok = llama_vocab_bos(vocab);
                    }
                    const bool need_logits = pos >= kld.first && (pos - kld.first) < kld.n_score;
                    common_batch_add(batch, tok, pos, { seq }, need_logits);
                    n_outputs += need_logits;
                }
            }
            if (n_outputs != n_seq_batch * score_count) {
                fprintf(stderr,
                    "%s: selector logits row mismatch chunk=%d split=%d outputs=%d expected=%d\n",
                    __func__, ci, jb, n_outputs, n_seq_batch * score_count);
                llama_batch_free(batch);
                return false;
            }
            const bool skip_host_logits = gpu_kld_enabled && gpu_kld_state.load(std::memory_order_relaxed) > 0;
            if (skip_host_logits) {
                chunk_skipped_host_logits = true;
            }
            scoped_selector_gpu_kld_active gpu_kld_guard(ctx, skip_host_logits);
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s: selector llama_decode failed chunk=%d batch=%d\n", __func__, ci, jb);
                llama_batch_free(batch);
                return false;
            }
            ++progress_done;
            {
                char detail[192];
                snprintf(detail, sizeof(detail), "chunk [%d/%d] split [%d/%d] decoded", ci + 1, kld.n_chunk, jb + 1, num_batches);
                heartbeat.update(progress_done, progress_total, detail);
            }
            if (!skip_host_logits && num_batches > 1 && n_outputs > 0) {
                const float * l = llama_get_logits(ctx);
                if (l == nullptr) {
                    fprintf(stderr, "%s: selector missing split logits chunk=%d split=%d\n", __func__, ci, jb);
                    llama_batch_free(batch);
                    return false;
                }
                if (logits_all.empty()) {
                    logits_all.resize((size_t) n_seq_batch * (size_t) kld.n_score * (size_t) kld.n_vocab);
                }
                for (int seq = 0; seq < n_seq_batch; ++seq) {
                    const float * src = l + (size_t) seq * (size_t) score_count * (size_t) kld.n_vocab;
                    float * dst = logits_all.data() +
                        ((size_t) seq * (size_t) kld.n_score + (size_t) s_begin) * (size_t) kld.n_vocab;
                    memcpy(dst, src, (size_t) score_count * (size_t) kld.n_vocab * sizeof(float));
                }
            }
            if (chunk_gpu_complete && !chunk_gpu_failed && n_outputs > 0) {
                {
                    char detail[192];
                    snprintf(detail, sizeof(detail), "chunk [%d/%d] split [%d/%d] reducing GPU KLD", ci + 1, kld.n_chunk, jb + 1, num_batches);
                    heartbeat.detail(detail);
                }
                if (score_count > 0) {
                    ctx->synchronize();
                    const ggml_tensor * logits_tensor = llama_internal_get_logits_tensor(ctx);
                    for (int seq = 0; seq < n_seq_batch; ++seq) {
                        std::vector<int32_t> token_ids((size_t) score_count);
                        const llama_token * chunk_tokens = kld.tokens.data() + (size_t) (ci + seq) * (size_t) kld.n_ctx;
                        for (int s = 0; s < score_count; ++s) {
                            const int t = kld.first + s_begin + s;
                            token_ids[(size_t) s] = (int32_t) chunk_tokens[t + 1];
                        }
                        const uint16_t * base_seq =
                            nvfp4_selector_kld_log_probs_data(kld) +
                            ((size_t) (ci + seq) * (size_t) kld.n_score + (size_t) s_begin) * (size_t) kld.nv;
                        nvfp4_cuda_kld_result cuda_km{};
                        std::vector<double> cuda_kld_values;
                        if (collect_kld_distribution) {
                            cuda_kld_values.resize((size_t) score_count);
                        }
                        if (!nvfp4_kld_reduce_cuda_tensor(
                                logits_tensor,
                                base_seq,
                                token_ids.data(),
                                seq * score_count,
                                score_count,
                                kld.n_vocab,
                                kld.nv,
                                &cuda_km,
                                collect_kld_distribution ? cuda_kld_values.data() : nullptr,
                                nullptr)) {
                            chunk_gpu_failed = true;
                            chunk_gpu_complete = false;
                            if (chunk_skipped_host_logits) {
                                fprintf(stderr, "%s: selector GPU KLD failed after host-logits copy was skipped; CPU fallback is unavailable for this chunk\n", __func__);
                                llama_batch_free(batch);
                                return false;
                            }
                            gpu_kld_state.store(-1, std::memory_order_relaxed);
                            break;
                        }
                        nvfp4_selector_merge_cuda_kld_metrics(chunk_gpu_metrics, cuda_km, std::move(cuda_kld_values));
                        chunk_gpu_rows += score_count;
                    }
                }
            }
        }
        const int n_eval = n_seq_batch * kld.n_score;
        const bool chunk_gpu_ready = chunk_gpu_complete && !chunk_gpu_failed && chunk_gpu_rows == n_eval;
        const bool validate_gpu_chunk = chunk_gpu_ready && gpu_kld_state.load(std::memory_order_relaxed) == 0;
        if (chunk_gpu_ready && !validate_gpu_chunk) {
            {
                char detail[160];
                snprintf(detail, sizeof(detail), "chunk [%d/%d] complete via GPU KLD", ci + 1, kld.n_chunk);
                heartbeat.detail(detail);
            }
            gpu_kld_state.store(1, std::memory_order_relaxed);
            nvfp4_selector_merge_kld_metrics(out, std::move(chunk_gpu_metrics));
            continue;
        }
        if (chunk_skipped_host_logits) {
            fprintf(stderr, "%s: selector GPU KLD did not cover the full chunk after host-logits copy was skipped\n", __func__);
            llama_batch_free(batch);
            return false;
        }
        const float * logits_contiguous = nullptr;
        if (num_batches == 1) {
            // Synchronize exactly once per decoded chunk. Worker threads must not
            // call llama_get_logits_ith(), because that public API synchronizes
            // the whole context on every token row and collapses KLD reduction
            // back into a mostly single-threaded bottleneck.
            logits_contiguous = llama_get_logits(ctx);
            if (logits_contiguous == nullptr) {
                fprintf(stderr, "%s: selector missing contiguous logits chunk=%d\n", __func__, ci);
                llama_batch_free(batch);
                return false;
            }
        }

        {
            char detail[160];
            snprintf(detail, sizeof(detail), "chunk [%d/%d] reducing CPU KLD", ci + 1, kld.n_chunk);
            heartbeat.detail(detail);
        }
        auto eval_range = [&](int begin, int end, nvfp4_selector_kld_metrics & local) -> bool {
            local.collect_kld_values = collect_kld_distribution;
            if (collect_kld_distribution) {
                local.kld_values.reserve((size_t) std::max(0, end - begin));
            }
            for (int item = begin; item < end; ++item) {
                const int seq = item / kld.n_score;
                const int s = item - seq * kld.n_score;
                const int t = kld.first + s;
                const llama_token * chunk_tokens = kld.tokens.data() + (size_t) (ci + seq) * (size_t) kld.n_ctx;
                const uint16_t * base_chunk = nvfp4_selector_kld_log_probs_data(kld) + (size_t) (ci + seq) * (size_t) kld.n_score * (size_t) kld.nv;
                const float * logits_t = num_batches > 1
                    ? logits_all.data() + ((size_t) seq * (size_t) kld.n_score + (size_t) s) * (size_t) kld.n_vocab
                    : logits_contiguous + ((size_t) seq * (size_t) kld.n_score + (size_t) s) * (size_t) kld.n_vocab;
                const uint16_t * base_t = base_chunk + (size_t) s * (size_t) kld.nv;
                const llama_token tok = chunk_tokens[t + 1];
                nvfp4_selector_eval_one_token(kld.n_vocab, logits_t, base_t, tok, local);
            }
            return true;
        };

        nvfp4_selector_kld_metrics chunk_cpu_metrics;
        chunk_cpu_metrics.collect_kld_values = collect_kld_distribution;
        const int threads_for_chunk = std::min(metric_threads, std::max(1, n_eval));
        if (threads_for_chunk <= 1 || n_eval < 2) {
            nvfp4_selector_kld_metrics local;
            if (!eval_range(0, n_eval, local)) {
                fprintf(stderr, "%s: selector missing logits chunk=%d token=%d\n", __func__, ci, kld.first);
                llama_batch_free(batch);
                return false;
            }
            nvfp4_selector_merge_kld_metrics(chunk_cpu_metrics, std::move(local));
        } else {
            std::vector<nvfp4_selector_kld_metrics> locals((size_t) threads_for_chunk);
            if (!cpu_kld_pool || cpu_kld_pool->capacity() < threads_for_chunk) {
                cpu_kld_pool = std::make_unique<nvfp4_selector_kld_worker_pool>(threads_for_chunk);
            }
            std::atomic_bool ok{true};
            cpu_kld_pool->run(threads_for_chunk, [&](int ti) {
                const int begin = (n_eval * ti) / threads_for_chunk;
                const int end = (n_eval * (ti + 1)) / threads_for_chunk;
                if (!eval_range(begin, end, locals[(size_t) ti])) {
                    ok.store(false, std::memory_order_relaxed);
                }
            });
            if (!ok.load(std::memory_order_relaxed)) {
                fprintf(stderr, "%s: selector missing logits chunk=%d token=%d\n", __func__, ci, kld.first);
                llama_batch_free(batch);
                return false;
            }
            for (auto & local : locals) {
                nvfp4_selector_merge_kld_metrics(chunk_cpu_metrics, std::move(local));
            }
        }
        if (validate_gpu_chunk) {
            std::string parity_reason;
            if (nvfp4_selector_gpu_kld_parity_ok(chunk_cpu_metrics, chunk_gpu_metrics, &parity_reason)) {
                fprintf(stderr,
                    "%s: selector GPU KLD parity passed; enabling host-logits skip for later chunks\n",
                    __func__);
                gpu_kld_state.store(1, std::memory_order_relaxed);
                nvfp4_selector_merge_kld_metrics(out, std::move(chunk_gpu_metrics));
            } else {
                fprintf(stderr,
                    "%s: selector GPU KLD parity failed (%s); disabling CUDA reducer for this process\n",
                    __func__, parity_reason.c_str());
                gpu_kld_state.store(-1, std::memory_order_relaxed);
                nvfp4_selector_merge_kld_metrics(out, std::move(chunk_cpu_metrics));
            }
        } else {
            nvfp4_selector_merge_kld_metrics(out, std::move(chunk_cpu_metrics));
        }
        {
            char detail[160];
            snprintf(detail, sizeof(detail), "chunk [%d/%d] complete", ci + 1, kld.n_chunk);
            heartbeat.detail(detail);
        }
    }
    llama_batch_free(batch);
    heartbeat.finish("complete");

    return true;
}

static std::string nvfp4_selector_format_metrics(
        const char * label,
        const nvfp4_selector_derived_metrics & dm) {
    char buf[1024];
    snprintf(buf, sizeof(buf),
        "%s{ln=%.6f+-%.6f kld=%.6f+-%.6f p95=%.6f p99=%.6f p999=%.6f tail99=%.6f rms=%.6f+-%.6f max=%.6f top=%.4f+-%.4f top_flip_w=%.6f top_p_rmse=%.6f entropy_rmse=%.6f}",
        label,
        dm.ln_ratio,
        dm.ln_ratio_unc,
        dm.mean_kld,
        dm.mean_kld_unc,
        dm.kld_p95,
        dm.kld_p99,
        dm.kld_p999,
        dm.kld_tail_mean,
        dm.rms_dp,
        dm.rms_dp_unc,
        dm.max_kld,
        dm.same_top,
        dm.same_top_unc,
        dm.top_flip_weight,
        dm.top_prob_rmse,
        dm.entropy_rmse);
    return buf;
}

static std::string nvfp4_selector_json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out.push_back(c); break;
        }
    }
    return out;
}

static void nvfp4_selector_write_json_number(std::ostream & out, double value) {
    if (std::isfinite(value)) {
        out << value;
    } else {
        out << "null";
    }
}

static void nvfp4_selector_write_json_number_field(std::ostream & out, const char * name, double value) {
    out << ", \"" << name << "\": ";
    nvfp4_selector_write_json_number(out, value);
}

static void nvfp4_selector_write_metric_json_fields(
        std::ostream & out,
        const nvfp4_selector_derived_metrics & dm,
        const char * prefix) {
    out
        << ", \"" << prefix << "valid\": " << (dm.ok ? "true" : "false");
    out << ", \"" << prefix << "ppl\": "; nvfp4_selector_write_json_number(out, dm.ppl_q);
    out << ", \"" << prefix << "ppl_base\": "; nvfp4_selector_write_json_number(out, dm.ppl_base);
    out << ", \"" << prefix << "ln_ratio\": "; nvfp4_selector_write_json_number(out, dm.ln_ratio);
    out << ", \"" << prefix << "mean_kld\": "; nvfp4_selector_write_json_number(out, dm.mean_kld);
    out << ", \"" << prefix << "kld_p95\": "; nvfp4_selector_write_json_number(out, dm.kld_p95);
    out << ", \"" << prefix << "kld_p99\": "; nvfp4_selector_write_json_number(out, dm.kld_p99);
    out << ", \"" << prefix << "kld_p999\": "; nvfp4_selector_write_json_number(out, dm.kld_p999);
    out << ", \"" << prefix << "kld_tail_mean\": "; nvfp4_selector_write_json_number(out, dm.kld_tail_mean);
    out << ", \"" << prefix << "max_kld\": "; nvfp4_selector_write_json_number(out, dm.max_kld);
    out << ", \"" << prefix << "rms_dp\": "; nvfp4_selector_write_json_number(out, dm.rms_dp);
    out << ", \"" << prefix << "same_top\": "; nvfp4_selector_write_json_number(out, dm.same_top);
    out << ", \"" << prefix << "entropy_rmse\": "; nvfp4_selector_write_json_number(out, dm.entropy_rmse);
    out << ", \"" << prefix << "top_prob_rmse\": "; nvfp4_selector_write_json_number(out, dm.top_prob_rmse);
    out << ", \"" << prefix << "top_flip_weight\": "; nvfp4_selector_write_json_number(out, dm.top_flip_weight);
}

static void nvfp4_selector_write_delta_metric_json_fields(
        std::ostream & out,
        const nvfp4_selector_derived_metrics & after,
        const nvfp4_selector_derived_metrics & before) {
    auto write_delta = [&](const char * name, double value) {
        out << ", \"" << name << "\": ";
        if (after.ok && before.ok && std::isfinite(value)) {
            nvfp4_selector_write_json_number(out, value);
        } else {
            out << "null";
        }
    };
    write_delta("delta_ppl", after.ppl_q - before.ppl_q);
    write_delta("delta_ln_ratio", after.ln_ratio - before.ln_ratio);
    write_delta("delta_mean_kld", after.mean_kld - before.mean_kld);
    write_delta("delta_kld_p95", after.kld_p95 - before.kld_p95);
    write_delta("delta_kld_p99", after.kld_p99 - before.kld_p99);
    write_delta("delta_kld_p999", after.kld_p999 - before.kld_p999);
    write_delta("delta_kld_tail_mean", after.kld_tail_mean - before.kld_tail_mean);
    write_delta("delta_max_kld", after.max_kld - before.max_kld);
    write_delta("delta_rms_dp", after.rms_dp - before.rms_dp);
    write_delta("delta_same_top", after.same_top - before.same_top);
    write_delta("delta_entropy_rmse", after.entropy_rmse - before.entropy_rmse);
    write_delta("delta_top_prob_rmse", after.top_prob_rmse - before.top_prob_rmse);
    write_delta("delta_top_flip_weight", after.top_flip_weight - before.top_flip_weight);
}

static double nvfp4_selector_combined_unc(double ua, double ub) {
    return std::sqrt(std::max(0.0, ua * ua + ub * ub));
}

static int nvfp4_selector_compare_lower_noise(double a, double a_unc, double b, double b_unc, double sigma, double abs_floor) {
    const double margin = std::max(abs_floor, std::max(0.0, sigma) * nvfp4_selector_combined_unc(a_unc, b_unc));
    if (a + margin < b) {
        return -1;
    }
    if (b + margin < a) {
        return 1;
    }
    return 0;
}

static int nvfp4_selector_compare_higher_noise(double a, double a_unc, double b, double b_unc, double sigma, double abs_floor) {
    const double margin = std::max(abs_floor, std::max(0.0, sigma) * nvfp4_selector_combined_unc(a_unc, b_unc));
    if (a > b + margin) {
        return -1;
    }
    if (b > a + margin) {
        return 1;
    }
    return 0;
}

static int nvfp4_selector_compare_lower_margin(double a, double b, double abs_margin, double rel_margin) {
    const double margin = std::max(std::max(0.0, abs_margin), std::max(0.0, rel_margin) * std::max(std::fabs(a), std::fabs(b)));
    if (a + margin < b) {
        return -1;
    }
    if (b + margin < a) {
        return 1;
    }
    return 0;
}

static double nvfp4_selector_threshold_penalty(double value, double threshold, double penalty) {
    if (!(penalty > 0.0) || !std::isfinite(penalty)) {
        return 0.0;
    }
    const double excess = threshold > 0.0 && std::isfinite(threshold)
        ? std::max(0.0, value - threshold)
        : value;
    return penalty * std::max(0.0, excess);
}

static double mxfp6_selector_metric_tol(double base, double abs_floor, double rel_floor) {
    return std::max(std::max(0.0, abs_floor), std::max(0.0, rel_floor) * std::max(std::fabs(base), 1e-12));
}

static double mxfp6_selector_rel_excess(double value, double base, double abs_floor, double rel_floor) {
    const double excess = value - base - mxfp6_selector_metric_tol(base, abs_floor, rel_floor);
    if (!(excess > 0.0) || !std::isfinite(excess)) {
        return 0.0;
    }
    return excess / std::max(std::fabs(base), 1e-12);
}

enum class nvfp4_selector_metric_mode {
    POLICY,
    MXFP6_SCALE,
};

struct nvfp4_selector_metric_score {
    bool pass = false;
    double score = std::numeric_limits<double>::infinity();
};

static nvfp4_selector_metric_score nvfp4_selector_score_metrics(
    const nvfp4_selector_derived_metrics & dm,
    const nvfp4_selector_derived_metrics * base,
    const selector_rank_config & rank_cfg,
    nvfp4_selector_metric_mode mode) {
    nvfp4_selector_metric_score out;
    if (!dm.ok) {
        return out;
    }

    const bool have_base = base != nullptr && base->ok;
    bool pass = true;
    if (have_base) {
        const double kld_thr = rank_cfg.kld_threshold > 0.0 ? std::max(rank_cfg.kld_threshold, base->mean_kld) : base->mean_kld;
        const double p99_thr = rank_cfg.p99_threshold > 0.0 ? std::max(rank_cfg.p99_threshold, base->kld_p99) : base->kld_p99;
        const double p999_thr = rank_cfg.p999_threshold > 0.0 ? std::max(rank_cfg.p999_threshold, base->kld_p999) : base->kld_p999;
        const double max_kld_thr = rank_cfg.max_kld_threshold > 0.0 ? std::max(rank_cfg.max_kld_threshold, base->max_kld) : base->max_kld;

        pass =
            (!rank_cfg.kld_hard_gate || dm.mean_kld <= kld_thr) &&
            (!rank_cfg.p99_hard_gate || dm.kld_p99 <= p99_thr) &&
            (!rank_cfg.p999_hard_gate || dm.kld_p999 <= p999_thr) &&
            (!rank_cfg.max_kld_hard_gate || dm.max_kld <= max_kld_thr);
    }

    const bool scale_mode = mode == nvfp4_selector_metric_mode::MXFP6_SCALE;
    if (scale_mode && !have_base) {
        return out;
    }

    const double same_top_deficit = std::max(0.0, 1.0 - dm.same_top);
    out.score =
        4.00 * dm.mean_kld +
        (scale_mode ? 0.35 : 0.45) * dm.kld_p95 +
        0.90 * dm.kld_p99 +
        0.18 * dm.kld_p999 +
        (scale_mode ? 0.70 : 0.75) * dm.kld_tail_mean +
        1.25 * dm.rms_dp +
        (scale_mode ? 2.00 : 2.40) * same_top_deficit +
        (scale_mode ? 2.80 : 3.20) * dm.top_flip_weight +
        (scale_mode ? 1.10 : 1.25) * dm.top_prob_rmse +
        (scale_mode ? 0.45 : 0.50) * dm.entropy_rmse +
        0.003 * dm.max_kld +
        (scale_mode ? 0.050 : 0.005) * dm.ln_ratio;

    if (!scale_mode) {
        out.score +=
            nvfp4_selector_threshold_penalty(dm.mean_kld, rank_cfg.kld_threshold, rank_cfg.kld_penalty) +
            nvfp4_selector_threshold_penalty(dm.kld_p99, rank_cfg.p99_threshold, rank_cfg.p99_penalty) +
            nvfp4_selector_threshold_penalty(dm.kld_p999, rank_cfg.p999_threshold, rank_cfg.p999_penalty) +
            nvfp4_selector_threshold_penalty(dm.max_kld, rank_cfg.max_kld_threshold, rank_cfg.max_kld_penalty);
        out.pass = pass;
        return out;
    }

    // MXFP6_E2M3 scale refinement is a per-tensor surgical edit.  KLD and
    // token-distribution shape are the primary guards; PPL only prevents a
    // clearly worse teacher-token loss from slipping through.
    pass = pass &&
        dm.mean_kld <= base->mean_kld + mxfp6_selector_metric_tol(base->mean_kld, rank_cfg.mxfp6.mean_kld_abs_tol, rank_cfg.mxfp6.mean_kld_rel_tol) &&
        dm.kld_p99 <= base->kld_p99 + mxfp6_selector_metric_tol(base->kld_p99, 0.01, 0.15) &&
        dm.kld_p999 <= base->kld_p999 + mxfp6_selector_metric_tol(base->kld_p999, 0.50, 0.75) &&
        dm.max_kld <= base->max_kld + mxfp6_selector_metric_tol(base->max_kld, 2.0, 2.0) &&
        dm.rms_dp <= base->rms_dp + mxfp6_selector_metric_tol(base->rms_dp, 4e-4, 0.01) &&
        dm.same_top + mxfp6_selector_metric_tol(base->same_top, 3e-4, 0.0015) >= base->same_top &&
        dm.top_flip_weight <= base->top_flip_weight + mxfp6_selector_metric_tol(base->top_flip_weight, 1e-5, 0.005) &&
        dm.entropy_rmse <= base->entropy_rmse + mxfp6_selector_metric_tol(base->entropy_rmse, 5e-4, 0.02) &&
        dm.ln_ratio <= base->ln_ratio + mxfp6_selector_metric_tol(base->ln_ratio, rank_cfg.mxfp6.ppl_abs_tol, rank_cfg.mxfp6.ppl_rel_tol);

    out.score +=
        1.0  * mxfp6_selector_rel_excess(dm.mean_kld, base->mean_kld, rank_cfg.mxfp6.mean_kld_abs_tol, rank_cfg.mxfp6.mean_kld_rel_tol) +
        0.35 * mxfp6_selector_rel_excess(dm.kld_p99,  base->kld_p99,  0.01, 0.15) +
        0.20 * mxfp6_selector_rel_excess(dm.kld_p999, base->kld_p999, 0.50, 0.75) +
        0.05 * mxfp6_selector_rel_excess(dm.max_kld,  base->max_kld,  2.0, 2.0) +
        0.50 * mxfp6_selector_rel_excess(dm.rms_dp,   base->rms_dp,   4e-4, 0.01) +
        1.00 * mxfp6_selector_rel_excess(base->same_top, dm.same_top, 3e-4, 0.0015) +
        1.25 * mxfp6_selector_rel_excess(dm.top_flip_weight, base->top_flip_weight, 1e-5, 0.005) +
        0.40 * mxfp6_selector_rel_excess(dm.entropy_rmse, base->entropy_rmse, 5e-4, 0.02) +
        0.25 * mxfp6_selector_rel_excess(dm.ln_ratio, base->ln_ratio, rank_cfg.mxfp6.ppl_abs_tol, rank_cfg.mxfp6.ppl_rel_tol);
    out.pass = pass;
    return out;
}

struct nvfp4_selector_metric_rank {
    bool pass = false;
    double score = std::numeric_limits<double>::infinity();
};

static nvfp4_selector_metric_rank nvfp4_selector_rank_policy_metrics(
        const nvfp4_selector_derived_metrics & main_dm,
        const nvfp4_selector_derived_metrics * holdout_dm,
        const nvfp4_selector_derived_metrics & baseline_main,
        const nvfp4_selector_derived_metrics * baseline_holdout,
        const selector_rank_config & rank_cfg) {
    const nvfp4_selector_metric_score main_score =
        nvfp4_selector_score_metrics(main_dm, &baseline_main, rank_cfg, nvfp4_selector_metric_mode::POLICY);

    nvfp4_selector_metric_rank out;
    out.pass = main_score.pass;
    out.score = main_score.score;

    if (holdout_dm != nullptr && holdout_dm->ok) {
        const bool have_holdout_base = baseline_holdout != nullptr && baseline_holdout->ok;
        const nvfp4_selector_metric_score holdout_score =
            nvfp4_selector_score_metrics(
                *holdout_dm,
                have_holdout_base ? baseline_holdout : nullptr,
                rank_cfg,
                nvfp4_selector_metric_mode::POLICY);
        if (have_holdout_base) {
            out.pass = out.pass && holdout_score.pass;
        }
        const double w = std::clamp(rank_cfg.holdout_weight, 0.0, 0.95);
        out.score = (1.0 - w) * out.score + w * holdout_score.score;
    }

    return out;
}

static int nvfp4_selector_compare_one(
    const nvfp4_selector_derived_metrics & a,
    const nvfp4_selector_derived_metrics & b,
    const selector_rank_config & rank_cfg) {
    if (!a.ok || !b.ok) {
        if (a.ok != b.ok) {
            return a.ok ? -1 : 1;
        }
        return 0;
    }

    const int same_top_cmp = nvfp4_selector_compare_higher_noise(
        a.same_top, a.same_top_unc,
        b.same_top, b.same_top_unc,
        rank_cfg.same_top_sigma,
        rank_cfg.same_top_abs_floor);

    int cmp = nvfp4_selector_compare_lower_noise(
        a.mean_kld, a.mean_kld_unc,
        b.mean_kld, b.mean_kld_unc,
        rank_cfg.kld_sigma,
        rank_cfg.mean_kld_abs_floor);
    if (cmp != 0) {
        return cmp;
    }

    if (same_top_cmp != 0) {
        return same_top_cmp;
    }

    cmp = nvfp4_selector_compare_lower_margin(a.kld_p95, b.kld_p95, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin);
    if (cmp != 0) {
        return cmp;
    }

    cmp = nvfp4_selector_compare_lower_margin(a.kld_p99, b.kld_p99, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin);
    if (cmp != 0) {
        return cmp;
    }

    cmp = nvfp4_selector_compare_lower_margin(a.kld_p999, b.kld_p999, rank_cfg.p999_abs_margin, rank_cfg.p999_rel_margin);
    if (cmp != 0) {
        return cmp;
    }

    cmp = nvfp4_selector_compare_lower_margin(a.kld_tail_mean, b.kld_tail_mean, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin);
    if (cmp != 0) {
        return cmp;
    }

    cmp = nvfp4_selector_compare_lower_noise(
        a.rms_dp, a.rms_dp_unc,
        b.rms_dp, b.rms_dp_unc,
        rank_cfg.rms_dp_sigma,
        rank_cfg.rms_dp_abs_floor);
    if (cmp != 0) {
        return cmp;
    }

    cmp = nvfp4_selector_compare_lower_margin(a.top_flip_weight, b.top_flip_weight, rank_cfg.top_flip_weight_abs_floor, 0.0);
    if (cmp != 0) {
        return cmp;
    }

    cmp = nvfp4_selector_compare_lower_margin(a.top_prob_rmse, b.top_prob_rmse, rank_cfg.top_prob_rmse_abs_floor, 0.0);
    if (cmp != 0) {
        return cmp;
    }

    cmp = nvfp4_selector_compare_lower_margin(a.entropy_rmse, b.entropy_rmse, rank_cfg.entropy_rmse_abs_floor, 0.0);
    if (cmp != 0) {
        return cmp;
    }

    cmp = nvfp4_selector_compare_lower_margin(a.max_kld, b.max_kld, rank_cfg.max_kld_abs_margin, rank_cfg.max_kld_rel_margin);
    if (cmp != 0) {
        return cmp;
    }

    return nvfp4_selector_compare_lower_noise(
        a.ln_ratio, a.ln_ratio_unc,
        b.ln_ratio, b.ln_ratio_unc,
        rank_cfg.ppl_sigma,
        rank_cfg.ln_ratio_abs_floor);
}

static double nvfp4_selector_measured_score_one(
        const nvfp4_selector_derived_metrics & dm,
        const selector_rank_config & rank_cfg) {
    return nvfp4_selector_score_metrics(
        dm, nullptr, rank_cfg, nvfp4_selector_metric_mode::POLICY).score;
}

static double nvfp4_selector_measured_score(
    const nvfp4_selector_derived_metrics & main_dm,
    const nvfp4_selector_derived_metrics * holdout_dm,
    const selector_rank_config & rank_cfg) {
    const double main_score = nvfp4_selector_measured_score_one(main_dm, rank_cfg);
    if (holdout_dm == nullptr || !holdout_dm->ok) {
        return main_score;
    }
    const double holdout_score = nvfp4_selector_measured_score_one(*holdout_dm, rank_cfg);
    const double w = std::clamp(rank_cfg.holdout_weight, 0.0, 0.95);
    return (1.0 - w) * main_score + w * holdout_score;
}

static int nvfp4_selector_compare_measured_score(double a, double b) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        if (std::isfinite(a) != std::isfinite(b)) {
            return std::isfinite(a) ? -1 : 1;
        }
        return 0;
    }

    const double margin = std::max(1e-5, 0.001 * std::max(std::fabs(a), std::fabs(b)));
    if (a + margin < b) {
        return -1;
    }
    if (b + margin < a) {
        return 1;
    }
    return 0;
}

struct nvfp4_selector_best_objectives {
    bool ok = false;
    double ln_ratio = 0.0;
    double mean_kld = 0.0;
    double kld_p95 = 0.0;
    double kld_p99 = 0.0;
    double kld_p999 = 0.0;
    double kld_tail_mean = 0.0;
    double max_kld = 0.0;
    double rms_dp = 0.0;
    double same_top = 0.0;
    double entropy_rmse = 0.0;
    double top_prob_rmse = 0.0;
    double top_flip_weight = 0.0;
};

static nvfp4_selector_best_objectives nvfp4_selector_best_objectives_for(
        const nvfp4_selector_policy & policy,
        double holdout_weight) {
    nvfp4_selector_best_objectives obj;
    if (!policy.measured.ok) {
        return obj;
    }

    obj.ok = true;
    obj.ln_ratio = policy.measured.ln_ratio;
    obj.mean_kld = policy.measured.mean_kld;
    obj.kld_p95 = policy.measured.kld_p95;
    obj.kld_p99 = policy.measured.kld_p99;
    obj.kld_p999 = policy.measured.kld_p999;
    obj.kld_tail_mean = policy.measured.kld_tail_mean;
    obj.max_kld = policy.measured.max_kld;
    obj.rms_dp = policy.measured.rms_dp;
    obj.same_top = policy.measured.same_top;
    obj.entropy_rmse = policy.measured.entropy_rmse;
    obj.top_prob_rmse = policy.measured.top_prob_rmse;
    obj.top_flip_weight = policy.measured.top_flip_weight;

    if (policy.has_holdout && policy.measured_holdout.ok) {
        const double w = std::clamp(holdout_weight, 0.0, 0.95);
        auto mix_lower = [w](double main_v, double holdout_v) {
            return (1.0 - w) * main_v + w * holdout_v;
        };
        obj.ln_ratio = mix_lower(obj.ln_ratio, policy.measured_holdout.ln_ratio);
        obj.mean_kld = mix_lower(obj.mean_kld, policy.measured_holdout.mean_kld);
        obj.kld_p95 = mix_lower(obj.kld_p95, policy.measured_holdout.kld_p95);
        obj.kld_p99 = mix_lower(obj.kld_p99, policy.measured_holdout.kld_p99);
        obj.kld_p999 = mix_lower(obj.kld_p999, policy.measured_holdout.kld_p999);
        obj.kld_tail_mean = mix_lower(obj.kld_tail_mean, policy.measured_holdout.kld_tail_mean);
        obj.max_kld = mix_lower(obj.max_kld, policy.measured_holdout.max_kld);
        obj.rms_dp = mix_lower(obj.rms_dp, policy.measured_holdout.rms_dp);
        obj.same_top = mix_lower(obj.same_top, policy.measured_holdout.same_top);
        obj.entropy_rmse = mix_lower(obj.entropy_rmse, policy.measured_holdout.entropy_rmse);
        obj.top_prob_rmse = mix_lower(obj.top_prob_rmse, policy.measured_holdout.top_prob_rmse);
        obj.top_flip_weight = mix_lower(obj.top_flip_weight, policy.measured_holdout.top_flip_weight);
    }

    return obj;
}

static bool nvfp4_selector_best_lower_no_worse(double a, double b, double abs_margin, double rel_margin) {
    const double margin = std::max(std::max(0.0, abs_margin), std::max(0.0, rel_margin) * std::max(std::fabs(a), std::fabs(b)));
    return a <= b + margin;
}

static bool nvfp4_selector_best_lower_better(double a, double b, double abs_margin, double rel_margin) {
    const double margin = std::max(std::max(0.0, abs_margin), std::max(0.0, rel_margin) * std::max(std::fabs(a), std::fabs(b)));
    return a + margin < b;
}

static bool nvfp4_selector_best_higher_no_worse(double a, double b, double abs_margin, double rel_margin) {
    const double margin = std::max(std::max(0.0, abs_margin), std::max(0.0, rel_margin) * std::max(std::fabs(a), std::fabs(b)));
    return a + margin >= b;
}

static bool nvfp4_selector_best_higher_better(double a, double b, double abs_margin, double rel_margin) {
    const double margin = std::max(std::max(0.0, abs_margin), std::max(0.0, rel_margin) * std::max(std::fabs(a), std::fabs(b)));
    return a > b + margin;
}

static bool nvfp4_selector_best_dominates(
        const nvfp4_selector_policy & a,
        const nvfp4_selector_policy & b,
        const selector_rank_config & rank_cfg) {
    const auto ao = nvfp4_selector_best_objectives_for(a, rank_cfg.holdout_weight);
    const auto bo = nvfp4_selector_best_objectives_for(b, rank_cfg.holdout_weight);
    if (!ao.ok || !bo.ok) {
        return false;
    }

    bool strictly_better = false;
    auto lower = [&](double av, double bv, double abs_margin, double rel_margin) {
        if (!nvfp4_selector_best_lower_no_worse(av, bv, abs_margin, rel_margin)) {
            return false;
        }
        strictly_better = strictly_better || nvfp4_selector_best_lower_better(av, bv, abs_margin, rel_margin);
        return true;
    };
    auto higher = [&](double av, double bv, double abs_margin, double rel_margin) {
        if (!nvfp4_selector_best_higher_no_worse(av, bv, abs_margin, rel_margin)) {
            return false;
        }
        strictly_better = strictly_better || nvfp4_selector_best_higher_better(av, bv, abs_margin, rel_margin);
        return true;
    };

    return
        lower(ao.ln_ratio, bo.ln_ratio, rank_cfg.ln_ratio_abs_floor, 0.0) &&
        lower(ao.mean_kld, bo.mean_kld, rank_cfg.mean_kld_abs_floor, 0.0) &&
        lower(ao.kld_p95, bo.kld_p95, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin) &&
        lower(ao.kld_p99, bo.kld_p99, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin) &&
        lower(ao.kld_p999, bo.kld_p999, rank_cfg.p999_abs_margin, rank_cfg.p999_rel_margin) &&
        lower(ao.kld_tail_mean, bo.kld_tail_mean, rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin) &&
        lower(ao.max_kld, bo.max_kld, rank_cfg.max_kld_abs_margin, rank_cfg.max_kld_rel_margin) &&
        lower(ao.rms_dp, bo.rms_dp, rank_cfg.rms_dp_abs_floor, 0.0) &&
        lower(ao.entropy_rmse, bo.entropy_rmse, rank_cfg.entropy_rmse_abs_floor, 0.0) &&
        lower(ao.top_prob_rmse, bo.top_prob_rmse, rank_cfg.top_prob_rmse_abs_floor, 0.0) &&
        lower(ao.top_flip_weight, bo.top_flip_weight, rank_cfg.top_flip_weight_abs_floor, 0.0) &&
        higher(ao.same_top, bo.same_top, rank_cfg.same_top_abs_floor, 0.0) &&
        strictly_better;
}

static std::vector<size_t> nvfp4_selector_best_set(
        const std::vector<nvfp4_selector_policy> & policies,
        const std::vector<size_t> & candidates,
        const selector_rank_config & rank_cfg) {
    std::vector<size_t> best_set;
    best_set.reserve(candidates.size());

    for (size_t idx : candidates) {
        if (idx >= policies.size() || !std::isfinite(policies[idx].measured_score) || !policies[idx].measured.ok) {
            continue;
        }
        bool dominated = false;
        for (size_t other : candidates) {
            if (other == idx || other >= policies.size()) {
                continue;
            }
            if (nvfp4_selector_best_dominates(policies[other], policies[idx], rank_cfg)) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            best_set.push_back(idx);
        }
    }

    return best_set;
}

struct nvfp4_selector_best_set_stats {
    bool ok = false;
    double min_ln_ratio = std::numeric_limits<double>::infinity();
    double max_ln_ratio = -std::numeric_limits<double>::infinity();
    double min_mean_kld = std::numeric_limits<double>::infinity();
    double max_mean_kld = -std::numeric_limits<double>::infinity();
    double min_kld_p95 = std::numeric_limits<double>::infinity();
    double max_kld_p95 = -std::numeric_limits<double>::infinity();
    double min_kld_p99 = std::numeric_limits<double>::infinity();
    double max_kld_p99 = -std::numeric_limits<double>::infinity();
    double min_kld_p999 = std::numeric_limits<double>::infinity();
    double max_kld_p999 = -std::numeric_limits<double>::infinity();
    double min_kld_tail_mean = std::numeric_limits<double>::infinity();
    double max_kld_tail_mean = -std::numeric_limits<double>::infinity();
    double min_max_kld = std::numeric_limits<double>::infinity();
    double max_max_kld = -std::numeric_limits<double>::infinity();
    double min_rms_dp = std::numeric_limits<double>::infinity();
    double max_rms_dp = -std::numeric_limits<double>::infinity();
    double min_same_top = std::numeric_limits<double>::infinity();
    double max_same_top = -std::numeric_limits<double>::infinity();
    double min_entropy_rmse = std::numeric_limits<double>::infinity();
    double max_entropy_rmse = -std::numeric_limits<double>::infinity();
    double min_top_prob_rmse = std::numeric_limits<double>::infinity();
    double max_top_prob_rmse = -std::numeric_limits<double>::infinity();
    double min_top_flip_weight = std::numeric_limits<double>::infinity();
    double max_top_flip_weight = -std::numeric_limits<double>::infinity();
};

static void nvfp4_selector_best_set_stats_add(
        nvfp4_selector_best_set_stats & stats,
        const nvfp4_selector_best_objectives & obj) {
    if (!obj.ok ||
            !std::isfinite(obj.ln_ratio) ||
            !std::isfinite(obj.mean_kld) ||
            !std::isfinite(obj.kld_p95) ||
            !std::isfinite(obj.kld_p99) ||
            !std::isfinite(obj.kld_p999) ||
            !std::isfinite(obj.kld_tail_mean) ||
            !std::isfinite(obj.max_kld) ||
            !std::isfinite(obj.rms_dp) ||
            !std::isfinite(obj.same_top) ||
            !std::isfinite(obj.entropy_rmse) ||
            !std::isfinite(obj.top_prob_rmse) ||
            !std::isfinite(obj.top_flip_weight)) {
        return;
    }
    stats.ok = true;
    stats.min_ln_ratio = std::min(stats.min_ln_ratio, obj.ln_ratio);
    stats.max_ln_ratio = std::max(stats.max_ln_ratio, obj.ln_ratio);
    stats.min_mean_kld = std::min(stats.min_mean_kld, obj.mean_kld);
    stats.max_mean_kld = std::max(stats.max_mean_kld, obj.mean_kld);
    stats.min_kld_p95 = std::min(stats.min_kld_p95, obj.kld_p95);
    stats.max_kld_p95 = std::max(stats.max_kld_p95, obj.kld_p95);
    stats.min_kld_p99 = std::min(stats.min_kld_p99, obj.kld_p99);
    stats.max_kld_p99 = std::max(stats.max_kld_p99, obj.kld_p99);
    stats.min_kld_p999 = std::min(stats.min_kld_p999, obj.kld_p999);
    stats.max_kld_p999 = std::max(stats.max_kld_p999, obj.kld_p999);
    stats.min_kld_tail_mean = std::min(stats.min_kld_tail_mean, obj.kld_tail_mean);
    stats.max_kld_tail_mean = std::max(stats.max_kld_tail_mean, obj.kld_tail_mean);
    stats.min_max_kld = std::min(stats.min_max_kld, obj.max_kld);
    stats.max_max_kld = std::max(stats.max_max_kld, obj.max_kld);
    stats.min_rms_dp = std::min(stats.min_rms_dp, obj.rms_dp);
    stats.max_rms_dp = std::max(stats.max_rms_dp, obj.rms_dp);
    stats.min_same_top = std::min(stats.min_same_top, obj.same_top);
    stats.max_same_top = std::max(stats.max_same_top, obj.same_top);
    stats.min_entropy_rmse = std::min(stats.min_entropy_rmse, obj.entropy_rmse);
    stats.max_entropy_rmse = std::max(stats.max_entropy_rmse, obj.entropy_rmse);
    stats.min_top_prob_rmse = std::min(stats.min_top_prob_rmse, obj.top_prob_rmse);
    stats.max_top_prob_rmse = std::max(stats.max_top_prob_rmse, obj.top_prob_rmse);
    stats.min_top_flip_weight = std::min(stats.min_top_flip_weight, obj.top_flip_weight);
    stats.max_top_flip_weight = std::max(stats.max_top_flip_weight, obj.top_flip_weight);
}

static nvfp4_selector_best_set_stats nvfp4_selector_best_set_stats_for(
        const std::vector<nvfp4_selector_policy> & policies,
        const std::vector<size_t> & candidates,
        const selector_rank_config & rank_cfg) {
    nvfp4_selector_best_set_stats stats;
    for (size_t idx : candidates) {
        if (idx >= policies.size()) {
            continue;
        }
        nvfp4_selector_best_set_stats_add(
            stats,
            nvfp4_selector_best_objectives_for(policies[idx], rank_cfg.holdout_weight));
    }
    return stats;
}

static double nvfp4_selector_range_floor(double best, double worst, double abs_floor, double rel_floor) {
    return std::max({
        std::fabs(worst - best),
        std::max(0.0, abs_floor),
        std::max(0.0, rel_floor) * std::max(std::fabs(best), 1e-12),
        1e-12,
    });
}

static double nvfp4_selector_norm_lower(double value, double best, double worst, double abs_floor, double rel_floor) {
    if (!std::isfinite(value) || !std::isfinite(best) || !std::isfinite(worst)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max(0.0, value - best) / nvfp4_selector_range_floor(best, worst, abs_floor, rel_floor);
}

static double nvfp4_selector_norm_higher(double value, double best, double worst, double abs_floor, double rel_floor) {
    if (!std::isfinite(value) || !std::isfinite(best) || !std::isfinite(worst)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max(0.0, best - value) / nvfp4_selector_range_floor(best, worst, abs_floor, rel_floor);
}

static double nvfp4_selector_best_set_utility(
        const nvfp4_selector_policy & policy,
        const nvfp4_selector_best_set_stats & stats,
        const selector_rank_config & rank_cfg) {
    if (!stats.ok || !policy.measured.ok) {
        return std::numeric_limits<double>::infinity();
    }
    const auto obj = nvfp4_selector_best_objectives_for(policy, rank_cfg.holdout_weight);
    if (!obj.ok) {
        return std::numeric_limits<double>::infinity();
    }

    const double ppl = nvfp4_selector_norm_lower(
        obj.ln_ratio, stats.min_ln_ratio, stats.max_ln_ratio,
        rank_cfg.ln_ratio_abs_floor, 0.0);
    const double mean = nvfp4_selector_norm_lower(
        obj.mean_kld, stats.min_mean_kld, stats.max_mean_kld,
        rank_cfg.mean_kld_abs_floor, 0.0);
    const double p95 = nvfp4_selector_norm_lower(
        obj.kld_p95, stats.min_kld_p95, stats.max_kld_p95,
        rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin);
    const double p99 = nvfp4_selector_norm_lower(
        obj.kld_p99, stats.min_kld_p99, stats.max_kld_p99,
        rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin);
    const double p999 = nvfp4_selector_norm_lower(
        obj.kld_p999, stats.min_kld_p999, stats.max_kld_p999,
        rank_cfg.p999_abs_margin, rank_cfg.p999_rel_margin);
    const double tail_mean = nvfp4_selector_norm_lower(
        obj.kld_tail_mean, stats.min_kld_tail_mean, stats.max_kld_tail_mean,
        rank_cfg.p99_abs_margin, rank_cfg.p99_rel_margin);
    const double max_kld = nvfp4_selector_norm_lower(
        obj.max_kld, stats.min_max_kld, stats.max_max_kld,
        rank_cfg.max_kld_abs_margin, rank_cfg.max_kld_rel_margin);
    const double rms = nvfp4_selector_norm_lower(
        obj.rms_dp, stats.min_rms_dp, stats.max_rms_dp,
        rank_cfg.rms_dp_abs_floor, 0.0);
    const double same_top = nvfp4_selector_norm_higher(
        obj.same_top, stats.max_same_top, stats.min_same_top,
        rank_cfg.same_top_abs_floor, 0.0);
    const double entropy = nvfp4_selector_norm_lower(
        obj.entropy_rmse, stats.min_entropy_rmse, stats.max_entropy_rmse,
        rank_cfg.entropy_rmse_abs_floor, 0.0);
    const double top_prob = nvfp4_selector_norm_lower(
        obj.top_prob_rmse, stats.min_top_prob_rmse, stats.max_top_prob_rmse,
        rank_cfg.top_prob_rmse_abs_floor, 0.0);
    const double top_flip = nvfp4_selector_norm_lower(
        obj.top_flip_weight, stats.min_top_flip_weight, stats.max_top_flip_weight,
        rank_cfg.top_flip_weight_abs_floor, 0.0);

    const double weighted_sum =
        3.25 * mean +
        0.85 * p95 +
        2.10 * p99 +
        1.45 * p999 +
        1.75 * tail_mean +
        1.25 * rms +
        2.20 * same_top +
        2.70 * top_flip +
        1.35 * top_prob +
        0.75 * entropy +
        0.30 * max_kld +
        0.06 * ppl;

    const double worst_kld_axis = std::max({
        3.25 * mean,
        0.85 * p95,
        2.10 * p99,
        1.45 * p999,
        1.75 * tail_mean,
        1.25 * rms,
        2.20 * same_top,
        2.70 * top_flip,
    });

    // Distinguish "one bad token but good average" from a broader tail failure.
    // A max-only spike is a small penalty; p99/p999 degradation relative to mean
    // indicates many tokens moved and should dominate the final utility.
    const double tail_shape_penalty =
        0.30 * std::max(0.0, p95 - mean) +
        0.60 * std::max(0.0, p99 - mean) +
        0.45 * std::max(0.0, p999 - mean) +
        0.55 * std::max(0.0, tail_mean - mean) +
        0.08 * std::max(0.0, max_kld - p999);
    const double behavior_shape_penalty =
        0.95 * std::max(0.0, top_flip - same_top) +
        0.20 * std::max(0.0, entropy - mean);

    return 0.62 * weighted_sum + 0.38 * worst_kld_axis + tail_shape_penalty + behavior_shape_penalty;
}

static int nvfp4_selector_compare_policy_measured(
    const nvfp4_selector_policy & a,
    const nvfp4_selector_policy & b,
    const nvfp4_selector_derived_metrics & baseline_eval,
    const nvfp4_selector_derived_metrics * baseline_holdout_eval,
    const selector_rank_config & rank_cfg) {
    const nvfp4_selector_metric_rank a_rank =
        nvfp4_selector_rank_policy_metrics(
            a.measured,
            a.has_holdout ? &a.measured_holdout : nullptr,
            baseline_eval,
            baseline_holdout_eval,
            rank_cfg);
    const nvfp4_selector_metric_rank b_rank =
        nvfp4_selector_rank_policy_metrics(
            b.measured,
            b.has_holdout ? &b.measured_holdout : nullptr,
            baseline_eval,
            baseline_holdout_eval,
            rank_cfg);
    const bool a_pass = a_rank.pass;
    const bool b_pass = b_rank.pass;

    if (a_pass != b_pass) {
        return a_pass ? -1 : 1;
    }

    const int cmp_main = nvfp4_selector_compare_one(a.measured, b.measured, rank_cfg);
    if (cmp_main != 0) {
        if (baseline_holdout_eval != nullptr && a.has_holdout && b.has_holdout) {
            const int cmp_holdout = nvfp4_selector_compare_one(a.measured_holdout, b.measured_holdout, rank_cfg);
            if (cmp_holdout != 0 && cmp_holdout != cmp_main) {
                return cmp_holdout;
            }
        }
        return cmp_main;
    }

    if (baseline_holdout_eval != nullptr && a.has_holdout && b.has_holdout) {
        const int cmp_holdout = nvfp4_selector_compare_one(a.measured_holdout, b.measured_holdout, rank_cfg);
        if (cmp_holdout != 0) {
            return cmp_holdout;
        }
    }

    const int score_cmp = nvfp4_selector_compare_measured_score(a.measured_score, b.measured_score);
    if (score_cmp != 0) {
        return score_cmp;
    }

    if (a.proxy_score != b.proxy_score) {
        return a.proxy_score < b.proxy_score ? -1 : 1;
    }
    if (a.name != b.name) {
        return a.name < b.name ? -1 : 1;
    }
    return 0;
}

static int32_t nvfp4_selector_parse_layer(const std::string & name) {
    if (name.rfind("blk.", 0) != 0) {
        return -1;
    }
    size_t pos = 4;
    int32_t layer = 0;
    bool have_digit = false;
    while (pos < name.size() && std::isdigit((unsigned char) name[pos])) {
        have_digit = true;
        layer = layer * 10 + (name[pos] - '0');
        ++pos;
    }
    return have_digit ? layer : -1;
}

static nvfp4_selector_tensor_class nvfp4_selector_classify_tensor(const std::string & name) {
    if (name == "token_embd.weight" || name == "output.weight" || name == "token_embd_norm.weight" || name == "output_norm.weight") {
        return nvfp4_selector_tensor_class::EMBEDDING_OUTPUT;
    }
    if (name.find(".ffn_gate_inp") != std::string::npos || name.find(".altup_router") != std::string::npos || name.find(".router") != std::string::npos) {
        return nvfp4_selector_tensor_class::ROUTER;
    }
    if (name.find(".attn_gate.weight") != std::string::npos) {
        return nvfp4_selector_tensor_class::ROUTER;
    }
    if (name.find(".attn_output.weight") != std::string::npos || name.find(".wo.weight") != std::string::npos) {
        return nvfp4_selector_tensor_class::ATTN_OUT;
    }
    if (name.find(".attn_q.weight") != std::string::npos ||
        name.find(".attn_k.weight") != std::string::npos ||
        name.find(".attn_v.weight") != std::string::npos ||
        name.find(".attn_qkv.weight") != std::string::npos) {
        return nvfp4_selector_tensor_class::ATTN_QKV;
    }
    if (name.find(".ffn_up_exps.weight") != std::string::npos ||
        name.find(".ffn_gate.weight") != std::string::npos ||
        name.find(".ffn_up.weight") != std::string::npos) {
        return nvfp4_selector_tensor_class::EXPERT_UP_GATE;
    }
    if (name.find(".ffn_down_exps.weight") != std::string::npos) {
        return nvfp4_selector_tensor_class::EXPERT_DOWN;
    }
    if (name.find(".ffn_down.weight") != std::string::npos) {
        return nvfp4_selector_tensor_class::DENSE_MLP;
    }
    if (name.find(".ssm_out.weight") != std::string::npos ||
        name.find(".ssm_in.weight") != std::string::npos ||
        name.find(".ssm_alpha.weight") != std::string::npos ||
        name.find(".ssm_beta.weight") != std::string::npos ||
        name.find(".ssm_conv1d.weight") != std::string::npos ||
        name.find(".ssm_") != std::string::npos) {
        return nvfp4_selector_tensor_class::SSM;
    }
    return nvfp4_selector_tensor_class::OTHER;
}

static const char * nvfp4_selector_tensor_class_name(nvfp4_selector_tensor_class cls) {
    switch (cls) {
        case nvfp4_selector_tensor_class::EMBEDDING_OUTPUT: return "embedding_output";
        case nvfp4_selector_tensor_class::ATTN_QKV:         return "attn_qkv";
        case nvfp4_selector_tensor_class::ATTN_OUT:         return "attn_out";
        case nvfp4_selector_tensor_class::ROUTER:           return "router";
        case nvfp4_selector_tensor_class::EXPERT_UP_GATE:   return "expert_up_gate";
        case nvfp4_selector_tensor_class::EXPERT_DOWN:      return "expert_down";
        case nvfp4_selector_tensor_class::DENSE_MLP:        return "dense_mlp";
        case nvfp4_selector_tensor_class::SSM:              return "ssm";
        case nvfp4_selector_tensor_class::OTHER:            return "other";
    }
    return "other";
}

static int nvfp4_selector_layer_bucket(int32_t layer, int32_t n_layer) {
    if (layer < 0 || n_layer <= 0) {
        return -1;
    }
    if (n_layer <= 3) {
        return std::clamp(layer, 0, n_layer - 1);
    }
    if (layer < n_layer / 3) {
        return 0;
    }
    if (layer < (2 * n_layer) / 3) {
        return 1;
    }
    return 2;
}

static int32_t nvfp4_selector_detect_n_layer(const std::vector<std::pair<std::string, ggml_tensor *>> & tmap) {
    int32_t max_layer = -1;
    for (const auto & kv : tmap) {
        max_layer = std::max(max_layer, nvfp4_selector_parse_layer(kv.first));
    }
    return max_layer + 1;
}

static std::string nvfp4_selector_regex_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() * 2 + 2);
    out.push_back('^');
    for (char ch : s) {
        switch (ch) {
            case '\\':
            case '^':
            case '$':
            case '.':
            case '|':
            case '?':
            case '*':
            case '+':
            case '(':
            case ')':
            case '[':
            case ']':
            case '{':
            case '}':
                out.push_back('\\');
                break;
            default:
                break;
        }
        out.push_back(ch);
    }
    out.push_back('$');
    return out;
}

static size_t quantize_tensor_nbytes_as_type(const ggml_tensor * tensor, ggml_type type) {
    if (tensor == nullptr || type == GGML_TYPE_COUNT) {
        return 0;
    }
    const int64_t ne0 = std::max<int64_t>(1, tensor->ne[0]);
    const int64_t rows = std::max<int64_t>(1, tensor->ne[1]) * std::max<int64_t>(1, tensor->ne[2]) * std::max<int64_t>(1, tensor->ne[3]);
    return ggml_row_size(type, ne0) * (size_t) rows;
}

static float nvfp4_selector_slice_absmax(const ggml_tensor * tensor, int64_t slice_index) {
    if (tensor == nullptr || tensor->data == nullptr || slice_index < 0 || slice_index >= std::max<int64_t>(1, tensor->ne[2])) {
        return 0.0f;
    }

    const int64_t n_per_row = tensor->ne[0];
    const int64_t nrows = tensor->ne[1];
    const size_t row_size = ggml_row_size(tensor->type, n_per_row);
    const uint8_t * src_slice = (const uint8_t *) tensor->data + (size_t) slice_index * (size_t) nrows * row_size;

    float amax = 0.0f;
    std::vector<float> tmp;

    for (int64_t row = 0; row < nrows; ++row) {
        const uint8_t * row_ptr = src_slice + (size_t) row * row_size;
        if (tensor->type == GGML_TYPE_F32) {
            const float * x = (const float *) row_ptr;
            for (int64_t i = 0; i < n_per_row; ++i) {
                const float a = std::fabs(x[i]);
                if (a > amax && std::isfinite(a)) {
                    amax = a;
                }
            }
        } else if (tensor->type == GGML_TYPE_BF16) {
            const ggml_bf16_t * x = (const ggml_bf16_t *) row_ptr;
            for (int64_t i = 0; i < n_per_row; ++i) {
                const float a = std::fabs(ggml_bf16_to_fp32(x[i]));
                if (a > amax && std::isfinite(a)) {
                    amax = a;
                }
            }
        } else if (tensor->type == GGML_TYPE_F16) {
            const ggml_fp16_t * x = (const ggml_fp16_t *) row_ptr;
            for (int64_t i = 0; i < n_per_row; ++i) {
                const float a = std::fabs(ggml_fp16_to_fp32(x[i]));
                if (a > amax && std::isfinite(a)) {
                    amax = a;
                }
            }
        } else {
            const auto * traits = ggml_get_type_traits(tensor->type);
            if (traits == nullptr || traits->to_float == nullptr) {
                return 0.0f;
            }
            tmp.resize((size_t) n_per_row);
            traits->to_float(row_ptr, tmp.data(), n_per_row);
            for (float v : tmp) {
                const float a = std::fabs(v);
                if (a > amax && std::isfinite(a)) {
                    amax = a;
                }
            }
        }
    }

    return amax;
}

static std::vector<float> nvfp4_selector_quant_tensor_scales(const ggml_tensor * tensor, float correction_denom) {
    std::vector<float> scales;
    if (tensor == nullptr || !(correction_denom > 0.0f) || !std::isfinite(correction_denom)) {
        return scales;
    }

    const int64_t n_slices = std::max<int64_t>(1, tensor->ne[2]);
    scales.resize((size_t) n_slices, 1.0f);
    for (int64_t is = 0; is < n_slices; ++is) {
        const float amax = nvfp4_selector_slice_absmax(tensor, is);
        if (amax > 0.0f && std::isfinite(amax)) {
            const float scale = amax / correction_denom;
            if (scale > 0.0f && std::isfinite(scale)) {
                scales[(size_t) is] = scale;
            }
        }
    }
    return scales;
}

static bool quantize_tensor_copy_out(const ggml_tensor * tensor, void * dst, size_t nbytes) {
    if (tensor == nullptr || dst == nullptr) {
        return false;
    }
    if (tensor->buffer != nullptr) {
        ggml_backend_tensor_get(const_cast<ggml_tensor *>(tensor), dst, 0, nbytes);
        return true;
    }
    if (tensor->data != nullptr) {
        if (ggml_cuda_tensor_get_host(tensor, dst, nbytes, nullptr)) {
            return true;
        }
        memcpy(dst, tensor->data, nbytes);
        return true;
    }
    return false;
}

static bool quantize_tensor_copy_in(ggml_tensor * tensor, const void * src, size_t nbytes) {
    if (tensor == nullptr || src == nullptr) {
        return false;
    }
    if (ggml_cuda_tensor_set_host(tensor, src, nbytes, nullptr)) {
        return true;
    }
    if (tensor->buffer != nullptr) {
        if (ggml_backend_buffer_is_host(tensor->buffer)) {
            return false;
        }
        const char * buffer_name = ggml_backend_buffer_name(tensor->buffer);
        if (buffer_name != nullptr && std::strstr(buffer_name, "CUDA") != nullptr) {
            return false;
        }
        ggml_backend_tensor_set(tensor, const_cast<void *>(src), 0, nbytes);
        return true;
    }
    if (tensor->data != nullptr) {
        if (ggml_cuda_tensor_set_host(tensor, src, nbytes, nullptr)) {
            return true;
        }
    }
    return false;
}

static bool quantize_tensor_host_buffer(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer);
}

static bool quantize_binding_ensure_target_bytes(nvfp4_selector_binding & binding) {
    if (binding.target == nullptr || binding.target_nbytes == 0) {
        return false;
    }
    if (!binding.original_target_device.valid_for(binding.target_nbytes)) {
        (void) binding.original_target_device.capture(binding.target, binding.target_nbytes);
    }
    if (binding.original_target_bytes.size() != binding.target_nbytes) {
        binding.original_target_bytes.resize(binding.target_nbytes);
        if (!quantize_tensor_copy_out(binding.target, binding.original_target_bytes.data(), binding.target_nbytes)) {
            return false;
        }
    }
    if (binding.working_target_bytes.size() != binding.target_nbytes) {
        binding.working_target_bytes.resize(binding.target_nbytes);
    }
    return true;
}

static bool quantize_binding_ensure_scale_bytes(nvfp4_selector_binding & binding) {
    if (binding.target_scale_nbytes == 0) {
        return false;
    }
    if (binding.target_scale == nullptr) {
        if (binding.target == nullptr || binding.target->type != GGML_TYPE_MXFP6_E2M3 ||
                binding.target_nbytes < MXFP6_HEADER_OFFSET ||
                binding.target_scale_nbytes != sizeof(float)) {
            return false;
        }
        if (!quantize_binding_ensure_target_bytes(binding)) {
            return false;
        }
        if (binding.original_scale_bytes.size() != binding.target_scale_nbytes) {
            binding.original_scale_bytes.resize(binding.target_scale_nbytes);
            const auto * header = (const tensor_mxfp6 *) binding.original_target_bytes.data();
            const float weight_scale = header->weight_scale > 0.0f && std::isfinite(header->weight_scale) ? header->weight_scale : 1.0f;
            memcpy(binding.original_scale_bytes.data(), &weight_scale, sizeof(float));
        }
        if (binding.working_scale_bytes.size() != binding.target_scale_nbytes) {
            binding.working_scale_bytes.resize(binding.target_scale_nbytes);
        }
        return true;
    }
    if (binding.original_scale_bytes.size() != binding.target_scale_nbytes) {
        binding.original_scale_bytes.resize(binding.target_scale_nbytes);
        if (!quantize_tensor_copy_out(binding.target_scale, binding.original_scale_bytes.data(), binding.target_scale_nbytes)) {
            return false;
        }
    }
    if (!binding.original_scale_device.valid_for(binding.target_scale_nbytes)) {
        (void) binding.original_scale_device.capture(binding.target_scale, binding.target_scale_nbytes);
    }
    if (binding.working_scale_bytes.size() != binding.target_scale_nbytes) {
        binding.working_scale_bytes.resize(binding.target_scale_nbytes);
    }
    return true;
}

static bool quantize_binding_ensure_input_scale_bytes(nvfp4_selector_binding & binding) {
    if (binding.target_input_scale == nullptr || binding.target_input_scale_nbytes == 0) {
        return false;
    }
    if (binding.original_input_scale_bytes.size() != binding.target_input_scale_nbytes) {
        binding.original_input_scale_bytes.resize(binding.target_input_scale_nbytes);
        if (!quantize_tensor_copy_out(
                binding.target_input_scale,
                binding.original_input_scale_bytes.data(),
                binding.target_input_scale_nbytes)) {
            return false;
        }
    }
    if (!binding.original_input_scale_device.valid_for(binding.target_input_scale_nbytes)) {
        (void) binding.original_input_scale_device.capture(binding.target_input_scale, binding.target_input_scale_nbytes);
    }
    if (binding.working_input_scale_bytes.size() != binding.target_input_scale_nbytes) {
        binding.working_input_scale_bytes.resize(binding.target_input_scale_nbytes);
    }
    return true;
}

static bool nvfp4_selector_prepare_nvfp4_runtime_scales(
        nvfp4_selector_binding & binding,
        int32_t input_scale_policy,
        float * header_weight_scale,
        float * header_input_scale) {
    const int64_t n_slices = binding.source != nullptr ? std::max<int64_t>(1, binding.source->ne[2]) : 1;
    const int64_t n_per_row = binding.source != nullptr ? binding.source->ne[0] : 0;

    std::vector<float> weight_scales((size_t) n_slices, 1.0f);
    for (int64_t is = 0; is < n_slices; ++is) {
        if ((size_t) is < binding.quant_tensor_scales.size() &&
                std::isfinite(binding.quant_tensor_scales[(size_t) is]) &&
                binding.quant_tensor_scales[(size_t) is] > 0.0f) {
            weight_scales[(size_t) is] = binding.quant_tensor_scales[(size_t) is];
        }
    }

    std::vector<float> input_scales((size_t) n_slices, 1.0f);
    for (int64_t is = 0; is < n_slices; ++is) {
        const float * imatrix_slice =
            binding.imatrix_row != nullptr && n_per_row > 0 ? binding.imatrix_row + is * n_per_row : nullptr;
        input_scales[(size_t) is] =
            llama_nvfp4_input_scale_from_imatrix(imatrix_slice, n_per_row, input_scale_policy);
    }

    if (binding.target_scale != nullptr) {
        if (!quantize_binding_ensure_scale_bytes(binding) ||
                binding.working_scale_bytes.size() != binding.target_scale_nbytes ||
                binding.target_scale_nbytes < weight_scales.size() * sizeof(float)) {
            return false;
        }
        memcpy(binding.working_scale_bytes.data(), weight_scales.data(), weight_scales.size() * sizeof(float));
    }
    if (binding.target_input_scale != nullptr) {
        if (!quantize_binding_ensure_input_scale_bytes(binding) ||
                binding.working_input_scale_bytes.size() != binding.target_input_scale_nbytes ||
                binding.target_input_scale_nbytes < input_scales.size() * sizeof(float)) {
            return false;
        }
        memcpy(binding.working_input_scale_bytes.data(), input_scales.data(), input_scales.size() * sizeof(float));
    }

    if (header_weight_scale != nullptr) {
        *header_weight_scale = weight_scales.empty() ? 1.0f : weight_scales[0];
    }
    if (header_input_scale != nullptr) {
        *header_input_scale = input_scales.empty() ? 1.0f : input_scales[0];
    }
    return true;
}

struct nvfp4_selector_restore_counters {
    int64_t device = 0;
    int64_t host = 0;
    int64_t failed = 0;
};

static bool quantize_restore_from_snapshot_or_host(
        ggml_tensor * tensor,
        const nvfp4_selector_device_snapshot & snapshot,
        const std::vector<uint8_t> & host_bytes,
        size_t nbytes,
        nvfp4_selector_restore_counters * counters) {
    if (tensor == nullptr || nbytes == 0) {
        if (counters != nullptr) {
            ++counters->failed;
        }
        return false;
    }
    if (!snapshot.valid_for(nbytes) && host_bytes.size() != nbytes) {
        return false;
    }
    if (snapshot.restore(tensor, nbytes)) {
        if (counters != nullptr) {
            ++counters->device;
        }
        return true;
    }
    if (host_bytes.size() == nbytes && quantize_tensor_copy_in(tensor, host_bytes.data(), nbytes)) {
        if (counters != nullptr) {
            ++counters->host;
        }
        return true;
    }
    if (counters != nullptr) {
        ++counters->failed;
    }
    return false;
}

static bool quantize_restore_from_host(
        ggml_tensor * tensor,
        const std::vector<uint8_t> & host_bytes,
        size_t nbytes,
        nvfp4_selector_restore_counters * counters) {
    if (tensor == nullptr || nbytes == 0 || host_bytes.size() != nbytes ||
            !quantize_tensor_copy_in(tensor, host_bytes.data(), nbytes)) {
        if (counters != nullptr) {
            ++counters->failed;
        }
        return false;
    }
    if (counters != nullptr) {
        ++counters->host;
    }
    return true;
}

static std::vector<std::string> nvfp4_selector_build_stress_tensors(
    const std::vector<std::pair<std::string, ggml_tensor *>> & tmap,
    int32_t n_layer) {
    std::vector<std::string> out;
    if (tmap.empty()) {
        return out;
    }

    struct tensor_pick {
        std::string name;
        nvfp4_selector_tensor_class cls = nvfp4_selector_tensor_class::OTHER;
        int bucket = -1;
        int32_t layer = -1;
        size_t nbytes = 0;
    };

    std::vector<tensor_pick> picks;
    picks.reserve(tmap.size());
    for (const auto & kv : tmap) {
        if (kv.second == nullptr || kv.second->type != GGML_TYPE_NVFP4) {
            continue;
        }
        tensor_pick pick;
        pick.name = kv.first;
        pick.cls = nvfp4_selector_classify_tensor(kv.first);
        pick.layer = nvfp4_selector_parse_layer(kv.first);
        pick.bucket = nvfp4_selector_layer_bucket(pick.layer, n_layer);
        pick.nbytes = ggml_nbytes(kv.second);
        picks.push_back(std::move(pick));
    }

    auto append_unique = [&](const std::string & name) {
        if (std::find(out.begin(), out.end(), name) == out.end()) {
            out.push_back(name);
        }
    };

    const int per_bucket = (int) std::max<int64_t>(1, SELECTOR_CLASS_BUCKET_LIMIT);
    const int special_limit = (int) std::max<int64_t>(1, SELECTOR_SPECIAL_LIMIT);
    const int exception_limit = (int) std::max<int64_t>(0, SELECTOR_EXCEPTION_LIMIT);

    const std::vector<std::string> special_names = {
        "token_embd.weight",
        "output.weight",
        "token_embd_norm.weight",
        "output_norm.weight",
    };
    for (const std::string & special : special_names) {
        if ((int) out.size() >= special_limit) {
            break;
        }
        for (const auto & pick : picks) {
            if (pick.name == special) {
                append_unique(pick.name);
                break;
            }
        }
    }

    std::stable_sort(picks.begin(), picks.end(), [](const tensor_pick & a, const tensor_pick & b) {
        if (a.cls != b.cls) return a.cls < b.cls;
        if (a.bucket != b.bucket) return a.bucket < b.bucket;
        if (a.nbytes != b.nbytes) return a.nbytes > b.nbytes;
        if (a.layer != b.layer) return a.layer < b.layer;
        return a.name < b.name;
    });

    const std::vector<nvfp4_selector_tensor_class> class_order = {
        nvfp4_selector_tensor_class::EMBEDDING_OUTPUT,
        nvfp4_selector_tensor_class::ROUTER,
        nvfp4_selector_tensor_class::ATTN_QKV,
        nvfp4_selector_tensor_class::ATTN_OUT,
        nvfp4_selector_tensor_class::EXPERT_UP_GATE,
        nvfp4_selector_tensor_class::EXPERT_DOWN,
        nvfp4_selector_tensor_class::DENSE_MLP,
        nvfp4_selector_tensor_class::SSM,
    };
    for (nvfp4_selector_tensor_class cls : class_order) {
        for (int bucket = -1; bucket < 3; ++bucket) {
            int added = 0;
            for (const auto & pick : picks) {
                if (pick.cls != cls || pick.bucket != bucket) {
                    continue;
                }
                append_unique(pick.name);
                if (++added >= per_bucket) {
                    break;
                }
            }
        }
    }

    if (exception_limit > 0) {
        std::vector<tensor_pick> by_size = picks;
        std::stable_sort(by_size.begin(), by_size.end(), [](const tensor_pick & a, const tensor_pick & b) {
            if (a.nbytes != b.nbytes) return a.nbytes > b.nbytes;
            return a.name < b.name;
        });
        int added = 0;
        for (const auto & pick : by_size) {
            if (pick.cls == nvfp4_selector_tensor_class::OTHER) {
                continue;
            }
            const size_t before = out.size();
            append_unique(pick.name);
            if (out.size() != before && ++added >= exception_limit) {
                break;
            }
        }
    }

    return out;
}

static double nvfp4_selector_class_hotness(nvfp4_selector_tensor_class cls) {
    switch (cls) {
        case nvfp4_selector_tensor_class::ATTN_QKV:        return 1.00;
        case nvfp4_selector_tensor_class::ATTN_OUT:        return 0.95;
        case nvfp4_selector_tensor_class::EXPERT_UP_GATE:  return 0.90;
        case nvfp4_selector_tensor_class::EXPERT_DOWN:     return 0.90;
        case nvfp4_selector_tensor_class::SSM:             return 1.00;
        case nvfp4_selector_tensor_class::EMBEDDING_OUTPUT:return 0.80;
        case nvfp4_selector_tensor_class::ROUTER:          return 0.70;
        case nvfp4_selector_tensor_class::DENSE_MLP:       return 0.75;
        case nvfp4_selector_tensor_class::OTHER:           return 0.60;
    }
    return 0.75;
}

static bool nvfp4_selector_bf16_allowed(nvfp4_selector_tensor_class cls) {
    switch (cls) {
        case nvfp4_selector_tensor_class::ROUTER:
        case nvfp4_selector_tensor_class::SSM:
        case nvfp4_selector_tensor_class::EMBEDDING_OUTPUT:
        case nvfp4_selector_tensor_class::ATTN_OUT:
        case nvfp4_selector_tensor_class::ATTN_QKV:
            return true;
        default:
            return false;
    }
}

static bool nvfp4_selector_expert_class(nvfp4_selector_tensor_class cls) {
    return
        cls == nvfp4_selector_tensor_class::EXPERT_UP_GATE ||
        cls == nvfp4_selector_tensor_class::EXPERT_DOWN;
}

static double nvfp4_selector_rescue_roi(double proxy_score, size_t delta_bytes, double speed_penalty, double type_gain, double speed_weight) {
    if (delta_bytes == 0 || !std::isfinite(proxy_score) || proxy_score <= 0.0) {
        return 0.0;
    }

    const double delta_mb = std::max(1e-6, (double) delta_bytes / (1024.0 * 1024.0));
    const double denom = delta_mb * (1.0 + std::max(0.0, speed_weight) * std::max(0.0, speed_penalty));
    return (proxy_score * std::max(0.0, type_gain)) / denom;
}

static int64_t nvfp4_selector_autotune_sample_blocks(int64_t nb_total, int64_t override_cap = 0) {
    if (nb_total <= 0) {
        return 0;
    }
    const int64_t env_cap = quantize_control_i64("LLAMA_NVFP4_AUTOTUNE_MAX_BLOCKS", 0);
    const int64_t cap_hint = override_cap > 0 ? override_cap : env_cap;
    if (cap_hint > 0) {
        return std::min(nb_total, cap_hint);
    }
    const int64_t cap =
        nb_total >= 262144 ? 4096 :
        nb_total >= 65536  ? 3072 :
        nb_total >= 16384  ? 2048 :
        nb_total >= 4096   ? 1024 :
        nb_total >= 1024   ? 512  :
        nb_total >= 256    ? 256  : 64;
    return std::min(nb_total, cap);
}

static bool quantize_tensor_slice_to_f32(
    const ggml_tensor * src_tensor,
    int64_t slice_index,
    float tensor_scale,
    std::vector<float> & out) {
    if (src_tensor == nullptr || slice_index < 0 || slice_index >= src_tensor->ne[2]) {
        return false;
    }

    const int64_t n_per_row = src_tensor->ne[0];
    const int64_t nrows = src_tensor->ne[1];
    const size_t n = (size_t) n_per_row * (size_t) nrows;
    out.resize(n);

    const size_t src_row_size = ggml_row_size(src_tensor->type, n_per_row);
    const auto * src_traits = ggml_get_type_traits(src_tensor->type);
    if (src_tensor->type != GGML_TYPE_F32 && src_tensor->type != GGML_TYPE_F16 && src_tensor->type != GGML_TYPE_BF16 &&
        (!src_traits || src_traits->to_float == nullptr)) {
        return false;
    }

    for (int64_t row = 0; row < nrows; ++row) {
        const uint8_t * src_ptr = (const uint8_t *) src_tensor->data + ((size_t) slice_index * (size_t) nrows + (size_t) row) * src_row_size;
        float * dst_ptr = out.data() + (size_t) row * (size_t) n_per_row;
        if (src_tensor->type == GGML_TYPE_F32) {
            memcpy(dst_ptr, src_ptr, (size_t) n_per_row * sizeof(float));
        } else if (src_tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src_ptr, dst_ptr, n_per_row);
        } else if (src_tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) src_ptr, dst_ptr, n_per_row);
        } else {
            src_traits->to_float(src_ptr, dst_ptr, n_per_row);
        }
    }

    if (std::isfinite(tensor_scale) && tensor_scale > 0.0f && std::fabs(tensor_scale - 1.0f) > 1e-12f) {
        for (float & v : out) {
            v *= tensor_scale;
        }
    }
    return true;
}

struct nvfp4_selector_source_slice_view {
    const void * raw = nullptr;
    bool raw_bf16 = false;
    float x_scale = 1.0f;
    std::vector<float> owned_f32;
};

static bool nvfp4_selector_prepare_sample_from_f32(
    const float * src_f32,
    const float * imatrix_row,
    int64_t nrows,
    int64_t n_per_row,
    std::vector<float> & sample_x,
    std::vector<float> & sample_qw,
    int64_t sample_blocks_override = 0,
    int64_t sample_phase = 0) {
    const int64_t nb_total = (nrows * n_per_row) / NVFP4_SELECTOR_BLOCK_SIZE;
    const int64_t sample_nb = nvfp4_selector_autotune_sample_blocks(nb_total, sample_blocks_override);
    if (sample_nb <= 0) {
        return false;
    }
    sample_x.resize((size_t) sample_nb * (size_t) NVFP4_SELECTOR_BLOCK_SIZE);
    sample_qw.clear();

    const int64_t row_blocks = n_per_row / NVFP4_SELECTOR_BLOCK_SIZE;
    const int64_t slice_elems = nrows * n_per_row;
    const bool build_qw = imatrix_row != nullptr && row_blocks > 0;
    if (build_qw) {
        sample_qw.resize(sample_x.size());
    }

    for (int64_t ib = 0; ib < sample_nb; ++ib) {
        const int64_t src_block = llama_nvfp4_sample_block_index(ib, sample_nb, nb_total, row_blocks, sample_phase);
        const int64_t src_off = src_block * NVFP4_SELECTOR_BLOCK_SIZE;
        if (src_off < 0 || src_off + NVFP4_SELECTOR_BLOCK_SIZE > slice_elems) {
            return false;
        }
        const int64_t dst_off = ib * NVFP4_SELECTOR_BLOCK_SIZE;
        memcpy(sample_x.data() + dst_off, src_f32 + src_off, (size_t) NVFP4_SELECTOR_BLOCK_SIZE * sizeof(float));
        if (build_qw) {
            const int64_t block_in_row = src_block % row_blocks;
            memcpy(sample_qw.data() + dst_off, imatrix_row + block_in_row * NVFP4_SELECTOR_BLOCK_SIZE, (size_t) NVFP4_SELECTOR_BLOCK_SIZE * sizeof(float));
        }
    }

    return true;
}

static bool nvfp4_selector_prepare_sample(
    const nvfp4_selector_binding & binding,
    int64_t slice_index,
    std::vector<float> & sample_x,
    std::vector<float> & sample_qw,
    nvfp4_selector_source_slice_view & source_view,
    int64_t sample_blocks_override = 0,
    int64_t sample_phase = 0) {
    const ggml_tensor * src_tensor = binding.source;
    if (src_tensor == nullptr || slice_index < 0 || slice_index >= src_tensor->ne[2]) {
        return false;
    }

    const int64_t n_per_row = src_tensor->ne[0];
    const int64_t nrows = src_tensor->ne[1];
    const float * imatrix_row = binding.imatrix_row ? (binding.imatrix_row + slice_index * n_per_row) : nullptr;
    const float quant_tensor_scale =
        slice_index >= 0 && (size_t) slice_index < binding.quant_tensor_scales.size() &&
        std::isfinite(binding.quant_tensor_scales[(size_t) slice_index]) &&
        binding.quant_tensor_scales[(size_t) slice_index] > 0.0f
        ? binding.quant_tensor_scales[(size_t) slice_index]
        : 1.0f;
    const bool tensor_scale_identity =
        !(std::isfinite(binding.source_tensor_scale) &&
          binding.source_tensor_scale > 0.0f &&
          std::fabs(binding.source_tensor_scale - 1.0f) > 1e-12f);

    const size_t src_row_size = ggml_row_size(src_tensor->type, n_per_row);
    const uint8_t * src_slice = (const uint8_t *) src_tensor->data + (size_t) slice_index * (size_t) nrows * src_row_size;
    const int64_t slice_elems = nrows * n_per_row;

    source_view = {};

    if (tensor_scale_identity && src_tensor->type == GGML_TYPE_F32) {
        source_view.raw = src_slice;
        source_view.raw_bf16 = false;
        source_view.x_scale = quant_tensor_scale;
        return nvfp4_selector_prepare_sample_from_f32((const float *) src_slice, imatrix_row, nrows, n_per_row, sample_x, sample_qw, sample_blocks_override, sample_phase);
    }

    const int64_t nb_total = (nrows * n_per_row) / NVFP4_SELECTOR_BLOCK_SIZE;
    const int64_t sample_nb = nvfp4_selector_autotune_sample_blocks(nb_total, sample_blocks_override);
    if (sample_nb <= 0) {
        return false;
    }
    sample_x.resize((size_t) sample_nb * (size_t) NVFP4_SELECTOR_BLOCK_SIZE);
    sample_qw.clear();

    const int64_t row_blocks = n_per_row / NVFP4_SELECTOR_BLOCK_SIZE;
    const bool build_qw = imatrix_row != nullptr && row_blocks > 0;
    if (build_qw) {
        sample_qw.resize(sample_x.size());
    }

    if (tensor_scale_identity && src_tensor->type == GGML_TYPE_BF16) {
        source_view.raw = src_slice;
        source_view.raw_bf16 = true;
        source_view.x_scale = quant_tensor_scale;
        const auto * src_bf16 = (const ggml_bf16_t *) src_slice;
        for (int64_t ib = 0; ib < sample_nb; ++ib) {
            const int64_t src_block = llama_nvfp4_sample_block_index(ib, sample_nb, nb_total, row_blocks, sample_phase);
            const int64_t src_off = src_block * NVFP4_SELECTOR_BLOCK_SIZE;
            if (src_off < 0 || src_off + NVFP4_SELECTOR_BLOCK_SIZE > slice_elems) {
                return false;
            }
            const int64_t dst_off = ib * NVFP4_SELECTOR_BLOCK_SIZE;
            ggml_bf16_to_fp32_row(src_bf16 + src_off, sample_x.data() + dst_off, NVFP4_SELECTOR_BLOCK_SIZE);
            if (build_qw) {
                const int64_t block_in_row = src_block % row_blocks;
                memcpy(sample_qw.data() + dst_off, imatrix_row + block_in_row * NVFP4_SELECTOR_BLOCK_SIZE, (size_t) NVFP4_SELECTOR_BLOCK_SIZE * sizeof(float));
            }
        }
        return true;
    }

    if (tensor_scale_identity && src_tensor->type == GGML_TYPE_F16) {
        const auto * src_f16 = (const ggml_fp16_t *) src_slice;
        for (int64_t ib = 0; ib < sample_nb; ++ib) {
            const int64_t src_block = llama_nvfp4_sample_block_index(ib, sample_nb, nb_total, row_blocks, sample_phase);
            const int64_t src_off = src_block * NVFP4_SELECTOR_BLOCK_SIZE;
            if (src_off < 0 || src_off + NVFP4_SELECTOR_BLOCK_SIZE > slice_elems) {
                return false;
            }
            const int64_t dst_off = ib * NVFP4_SELECTOR_BLOCK_SIZE;
            ggml_fp16_to_fp32_row(src_f16 + src_off, sample_x.data() + dst_off, NVFP4_SELECTOR_BLOCK_SIZE);
            if (build_qw) {
                const int64_t block_in_row = src_block % row_blocks;
                memcpy(sample_qw.data() + dst_off, imatrix_row + block_in_row * NVFP4_SELECTOR_BLOCK_SIZE, (size_t) NVFP4_SELECTOR_BLOCK_SIZE * sizeof(float));
            }
        }
        source_view.owned_f32.resize((size_t) nrows * (size_t) n_per_row);
        ggml_fp16_to_fp32_row(src_f16, source_view.owned_f32.data(), (int64_t) source_view.owned_f32.size());
        source_view.raw = source_view.owned_f32.data();
        source_view.raw_bf16 = false;
        source_view.x_scale = quant_tensor_scale;
        return true;
    }

    if (!quantize_tensor_slice_to_f32(src_tensor, slice_index, binding.source_tensor_scale, source_view.owned_f32)) {
        return false;
    }
    source_view.raw = source_view.owned_f32.data();
    source_view.raw_bf16 = false;
    source_view.x_scale = quant_tensor_scale;
    return nvfp4_selector_prepare_sample_from_f32(source_view.owned_f32.data(), imatrix_row, nrows, n_per_row, sample_x, sample_qw, sample_blocks_override, sample_phase);
}

static bool nvfp4_selector_device_sample_supported(const ggml_tensor * src_tensor, float source_tensor_scale) {
    if (src_tensor == nullptr || src_tensor->data == nullptr) {
        return false;
    }
    if (std::isfinite(source_tensor_scale) && source_tensor_scale > 0.0f &&
            std::fabs(source_tensor_scale - 1.0f) > 1e-12f) {
        return false;
    }
    return
        src_tensor->type == GGML_TYPE_F32 ||
        src_tensor->type == GGML_TYPE_F16 ||
        src_tensor->type == GGML_TYPE_BF16;
}

static bool nvfp4_selector_get_device_sample(
    const nvfp4_selector_binding & binding,
    int64_t slice_index,
    int64_t sample_nb,
    int64_t sample_phase,
    void * stream_key,
    nvfp4_selector_device_sample_view & out) {
    out = {};
    const ggml_tensor * src_tensor = binding.source;
    if (sample_nb <= 0 || !binding.device_samples ||
            !nvfp4_selector_device_sample_supported(src_tensor, binding.source_tensor_scale) ||
            slice_index < 0 || slice_index >= src_tensor->ne[2]) {
        return false;
    }

    const int64_t n_per_row = src_tensor->ne[0];
    const int64_t nrows = src_tensor->ne[1];
    if (n_per_row <= 0 || nrows <= 0 || (n_per_row % NVFP4_SELECTOR_BLOCK_SIZE) != 0) {
        return false;
    }
    const int64_t nb_total = (nrows * n_per_row) / NVFP4_SELECTOR_BLOCK_SIZE;
    if (sample_nb > nb_total || sample_nb * 32 < nb_total) {
        return false;
    }

    const size_t src_row_size = ggml_row_size(src_tensor->type, n_per_row);
    const uint8_t * src_slice = (const uint8_t *) src_tensor->data + (size_t) slice_index * (size_t) nrows * src_row_size;
    const float * imatrix_row = binding.imatrix_row ? (binding.imatrix_row + slice_index * n_per_row) : nullptr;
    const float quant_tensor_scale =
        slice_index >= 0 && (size_t) slice_index < binding.quant_tensor_scales.size() &&
        std::isfinite(binding.quant_tensor_scales[(size_t) slice_index]) &&
        binding.quant_tensor_scales[(size_t) slice_index] > 0.0f
        ? binding.quant_tensor_scales[(size_t) slice_index]
        : 1.0f;
    const float tune_x_mul =
        std::isfinite(quant_tensor_scale) && quant_tensor_scale > 0.0f &&
        std::fabs(quant_tensor_scale - 1.0f) > 1e-12f
        ? 1.0f / quant_tensor_scale
        : 1.0f;

    std::lock_guard<std::mutex> lock(binding.device_samples->mutex);
    for (const auto & entry : binding.device_samples->entries) {
        if (entry && entry->matches(slice_index, sample_nb, sample_phase, (int32_t) src_tensor->type, src_slice, imatrix_row, tune_x_mul)) {
            out.x_device = entry->x_device;
            out.tune_x_device = entry->tune_x_device;
            out.qw_device = entry->qw_device;
            out.n_device = entry->n_device;
            out.x_scale = quant_tensor_scale;
            return out.x_device != nullptr && out.tune_x_device != nullptr && out.n_device == sample_nb * NVFP4_SELECTOR_BLOCK_SIZE;
        }
    }

    auto entry = std::make_unique<nvfp4_selector_device_sample_entry>();
    entry->slice_index = slice_index;
    entry->sample_nb = sample_nb;
    entry->sample_phase = sample_phase;
    entry->source_type = (int32_t) src_tensor->type;
    entry->source_ptr = src_slice;
    entry->imatrix_ptr = imatrix_row;
    entry->tune_x_mul = tune_x_mul;

    if (!nvfp4_sample_cache_cuda_create(
            src_slice,
            (int32_t) src_tensor->type,
            nrows,
            n_per_row,
            imatrix_row,
            sample_nb,
            sample_phase,
            tune_x_mul,
            &entry->cache,
            &entry->x_device,
            &entry->tune_x_device,
            &entry->qw_device,
            &entry->n_device,
            stream_key)) {
        return false;
    }

    out.x_device = entry->x_device;
    out.tune_x_device = entry->tune_x_device;
    out.qw_device = entry->qw_device;
    out.n_device = entry->n_device;
    out.x_scale = quant_tensor_scale;
    const bool ok = out.x_device != nullptr && out.tune_x_device != nullptr && out.n_device == sample_nb * NVFP4_SELECTOR_BLOCK_SIZE;
    if (ok) {
        binding.device_samples->entries.push_back(std::move(entry));
    }
    return ok;
}

static bool nvfp4_selector_quantize_binding(
    const nvfp4_selector_binding & binding,
    const nvfp4_cuda_runtime_cfg & cfg,
    int nthread,
    std::vector<uint8_t> & out_bytes,
    double & sum_sq,
    double & sum_abs,
    double & max_abs,
    int64_t & count,
    int64_t sample_blocks_override = 0,
    int64_t sample_phase = 0,
    bool direct_runtime_patch = false) {
    using qtype = ggml_quantize_type_traits<GGML_TYPE_NVFP4>;

    auto prepare = [&](const quantize_binding_shape_t & shape) -> bool {
        direct_runtime_patch = direct_runtime_patch && !shape.sample_only;
        if (quantize_control_i64("LLAMA_NVFP4_TRACE", 0) != 0) {
            fprintf(stderr,
                "%s: tensor=%s type=%s rows=%" PRId64 " cols=%" PRId64 " slices=%" PRId64 " sample_blocks=%" PRId64 " cfg={choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f}\n",
                __func__,
                binding.name.c_str(),
                ggml_type_name(binding.source->type),
                shape.nrows,
                shape.n_per_row,
                shape.n_slices,
                sample_blocks_override,
                cfg.choose46_mode,
                cfg.refit_iters,
                cfg.use_compand_sat,
                (double) cfg.cap_m6,
                (double) cfg.cap_m4);
        }
        if (direct_runtime_patch) {
            out_bytes.clear();
        } else {
            out_bytes.resize((size_t) shape.n_slices * shape.out_slice_bytes);
        }
        return true;
    };

    auto run_slice = [&](int64_t i03, void * stream_key, const quantize_binding_shape_t & shape, quantize_binding_slice_accum_t & acc) {
        const int64_t n_per_row = shape.n_per_row;
        const int64_t nrows = shape.nrows;
        const bool sample_only = shape.sample_only;
        const int64_t sample_nb_for_bytes = shape.sample_nb_for_bytes;
        const size_t out_slice_bytes = shape.out_slice_bytes;
        if (sample_only) {
            nvfp4_selector_device_sample_view device_sample;
            if (nvfp4_selector_get_device_sample(binding, i03, sample_nb_for_bytes, sample_phase, stream_key, device_sample)) {
                nvfp4_cuda_tune_result tune = {
                    NVFP4_A0,
                    NVFP4_B0,
                    1.0f,
                    cfg,
                    1,
                };
                if (nvfp4_autotune_cuda_cfg(
                        device_sample.tune_x_device,
                        device_sample.qw_device,
                        device_sample.n_device,
                        &cfg,
                        &tune,
                        stream_key)) {
                    nvfp4_cuda_eval_result eval = {};
                    quantize_cuda_eval_params_t eval_params;
                    eval_params.x = device_sample.x_device;
                    eval_params.out = out_bytes.data() + (size_t) i03 * out_slice_bytes;
                    eval_params.nrow = 1;
                    eval_params.n_per_row = device_sample.n_device;
                    eval_params.qw = device_sample.qw_device;
                    eval_params.x_scale = device_sample.x_scale;
                    eval_params.a = tune.a;
                    eval_params.b = tune.b;
                    eval_params.nvfp4_cfg = &cfg;
                    eval_params.eval = &eval;
                    eval_params.stream_key = stream_key;
                    if (qtype::quantize_eval(eval_params)) {
                        acc.sum_sq += eval.sum_sq;
                        acc.sum_abs += eval.sum_abs;
                        acc.max_abs = std::max(acc.max_abs, eval.max_abs);
                        acc.count += eval.count;
                        return;
                    }
                }
            }
        }

        nvfp4_selector_source_slice_view source_view;
        std::vector<float> sample_x;
        std::vector<float> sample_qw;
        if (!nvfp4_selector_prepare_sample(binding, i03, sample_x, sample_qw, source_view, sample_blocks_override, sample_phase)) {
            fprintf(stderr,
                "%s: failed preparing selector sample for %s slice=%" PRId64 " source_type=%s rows=%" PRId64 " cols=%" PRId64 " sample_blocks=%" PRId64 " phase=%" PRId64 "\n",
                __func__,
                binding.name.c_str(),
                i03,
                ggml_type_name(binding.source->type),
                nrows,
                n_per_row,
                sample_blocks_override,
                sample_phase);
            acc.ok = false;
            return;
        }
        const float * imatrix_slice = binding.imatrix_row ? (binding.imatrix_row + i03 * n_per_row) : nullptr;

        nvfp4_cuda_tune_result tune = {
            NVFP4_A0,
            NVFP4_B0,
            1.0f,
            cfg,
            1,
        };
        const float sample_inv_scale =
            std::isfinite(source_view.x_scale) && source_view.x_scale > 0.0f &&
            std::fabs(source_view.x_scale - 1.0f) > 1e-12f
            ? 1.0f / source_view.x_scale
            : 1.0f;
        std::vector<float> tune_x;
        const float * tune_x_ptr = sample_x.data();
        if (sample_inv_scale != 1.0f) {
            tune_x.resize(sample_x.size());
            for (size_t i = 0; i < sample_x.size(); ++i) {
                tune_x[i] = sample_x[i] * sample_inv_scale;
            }
            tune_x_ptr = tune_x.data();
        }
        if (!nvfp4_autotune_cuda_cfg(tune_x_ptr, sample_qw.empty() ? nullptr : sample_qw.data(), (int64_t) sample_x.size(), &cfg, &tune, stream_key)) {
            fprintf(stderr,
                "%s: CUDA autotune failed for %s slice=%" PRId64 " sample_elems=%zu sample_qw=%s choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f\n",
                __func__,
                binding.name.c_str(),
                i03,
                sample_x.size(),
                sample_qw.empty() ? "no" : "yes",
                cfg.choose46_mode,
                cfg.refit_iters,
                cfg.use_compand_sat,
                (double) cfg.cap_m6,
                (double) cfg.cap_m4);
            acc.ok = false;
            return;
        }

        const void * eval_x = source_view.raw;
        bool eval_bf16 = source_view.raw_bf16;
        float eval_x_scale = source_view.x_scale;
        int64_t eval_nrow = nrows;
        int64_t eval_n_per_row = n_per_row;
        const float * eval_qw = imatrix_slice;

        if (sample_only) {
            eval_x = sample_x.data();
            eval_bf16 = false;
            eval_x_scale = source_view.x_scale;
            eval_nrow = 1;
            eval_n_per_row = (int64_t) sample_x.size();
            eval_qw = sample_qw.empty() ? nullptr : sample_qw.data();
        }

        nvfp4_cuda_eval_result eval = {};
        quantize_cuda_eval_params_t eval_params;
        eval_params.x = eval_x;
        eval_params.x_bf16 = eval_bf16;
        eval_params.out = direct_runtime_patch ? nullptr : out_bytes.data() + (size_t) i03 * out_slice_bytes;
        eval_params.target = direct_runtime_patch ? binding.target : nullptr;
        eval_params.slice_index = i03;
        eval_params.nrow = eval_nrow;
        eval_params.n_per_row = eval_n_per_row;
        eval_params.qw = eval_qw;
        eval_params.x_scale = eval_x_scale;
        eval_params.a = tune.a;
        eval_params.b = tune.b;
        eval_params.nvfp4_cfg = &cfg;
        eval_params.eval = &eval;
        eval_params.stream_key = stream_key;
        const bool quant_ok = qtype::quantize_eval(eval_params);
        if (!quant_ok) {
            fprintf(stderr,
                "%s: CUDA quant/eval failed for %s slice=%" PRId64 " eval_rows=%" PRId64 " eval_cols=%" PRId64 " eval_bf16=%d eval_qw=%s x_scale=%g a=%g b=%g choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f\n",
                __func__,
                binding.name.c_str(),
                i03,
                eval_nrow,
                eval_n_per_row,
                eval_bf16 ? 1 : 0,
                eval_qw ? "yes" : "no",
                (double) eval_x_scale,
                (double) tune.a,
                (double) tune.b,
                cfg.choose46_mode,
                cfg.refit_iters,
                cfg.use_compand_sat,
                (double) cfg.cap_m6,
                (double) cfg.cap_m4);
            acc.ok = false;
            return;
        }
        acc.sum_sq += eval.sum_sq;
        acc.sum_abs += eval.sum_abs;
        acc.max_abs = std::max(acc.max_abs, eval.max_abs);
        acc.count += eval.count;
    };

    return quantize_binding_quantize<qtype::quant_type>(
        binding,
        nthread,
        sample_blocks_override,
        sample_phase,
        reinterpret_cast<uintptr_t>(&out_bytes),
        __func__,
        prepare,
        run_slice,
        sum_sq,
        sum_abs,
        max_abs,
        count);
}

static bool mxfp6_selector_quantize_binding(
    nvfp4_selector_binding & binding,
    float scale_mul,
    int nthread,
    std::vector<uint8_t> & out_bytes,
    std::vector<uint8_t> & out_scale_bytes,
    double & sum_sq,
    double & sum_abs,
    double & max_abs,
    int64_t & count,
    bool direct_runtime_patch = false,
    bool * direct_applied = nullptr) {
    if (direct_applied != nullptr) {
        *direct_applied = false;
    }
    using qtype = ggml_quantize_type_traits<GGML_TYPE_MXFP6_E2M3>;
    if (!(scale_mul > 0.0f) || !std::isfinite(scale_mul)) {
        return false;
    }

    bool scale_in_header = false;
    bool direct_patch = false;
    size_t out_header_bytes = 0;
    size_t out_slice_bytes = 0;
    const float * base_scales = nullptr;
    float * tuned_scales = nullptr;
    float input_scale = 1.0f;

    auto prepare = [&](const quantize_binding_shape_t & shape) -> bool {
        if (!quantize_binding_ensure_scale_bytes(binding)) {
            fprintf(stderr, "%s: missing MXFP6_E2M3 scale tensor for %s\n", __func__, binding.name.c_str());
            return false;
        }

        scale_in_header = binding.target_scale == nullptr;
        if (scale_in_header && shape.n_slices != 1) {
            fprintf(stderr, "%s: embedded MXFP6_E2M3 scale tuning only supports scalar header scale for %s (slices=%" PRId64 ")\n",
                    __func__, binding.name.c_str(), shape.n_slices);
            return false;
        }

        out_slice_bytes = shape.out_slice_bytes;
        out_header_bytes = scale_in_header ? MXFP6_HEADER_OFFSET : 0;
        direct_patch = direct_runtime_patch && binding.target != nullptr;
        if (direct_patch) {
            out_bytes.clear();
        } else {
            out_bytes.resize(out_header_bytes + (size_t) shape.n_slices * out_slice_bytes);
        }
        out_scale_bytes.resize((size_t) shape.n_slices * sizeof(float));
        base_scales = (const float *) binding.original_scale_bytes.data();
        tuned_scales = (float *) out_scale_bytes.data();
        input_scale = 1.0f;
        if (binding.original_target_bytes.size() >= MXFP6_HEADER_OFFSET) {
            const auto * header = (const tensor_mxfp6 *) binding.original_target_bytes.data();
            input_scale = header->input_scale > 0.0f && std::isfinite(header->input_scale) ? header->input_scale : 1.0f;
        }
        return true;
    };

    auto run_slice = [&](int64_t i03, void * stream_key, const quantize_binding_shape_t & shape, quantize_binding_slice_accum_t & acc) {
        const int64_t n_per_row = shape.n_per_row;
        const int64_t nrows = shape.nrows;
        std::vector<float> src_f32;
        if (!quantize_tensor_slice_to_f32(binding.source, i03, binding.source_tensor_scale, src_f32)) {
            fprintf(stderr, "%s: failed dequantizing source slice for %s slice=%" PRId64 "\n", __func__, binding.name.c_str(), i03);
            acc.ok = false;
            return;
        }
        if (src_f32.size() != (size_t) nrows * (size_t) n_per_row) {
            acc.ok = false;
            return;
        }

        float tensor_scale = base_scales[i03] * scale_mul;
        if (!(tensor_scale > 0.0f) || !std::isfinite(tensor_scale)) {
            tensor_scale = 1.0f;
        }
        tuned_scales[i03] = tensor_scale;

        const float * imatrix_slice = binding.imatrix_row ? (binding.imatrix_row + i03 * n_per_row) : nullptr;

        nvfp4_cuda_eval_result eval{};
        quantize_cuda_eval_params_t eval_params;
        eval_params.x = src_f32.data();
        eval_params.out = direct_patch ? nullptr : out_bytes.data() + out_header_bytes + (size_t) i03 * out_slice_bytes;
        eval_params.target = direct_patch ? binding.target : nullptr;
        eval_params.slice_index = i03;
        eval_params.nrow = nrows;
        eval_params.n_per_row = n_per_row;
        eval_params.qw = imatrix_slice;
        eval_params.weight_scale = tensor_scale;
        eval_params.input_scale = input_scale;
        eval_params.eval = &eval;
        eval_params.stream_key = stream_key;
        const bool ok = qtype::quantize_eval(eval_params);
        if (!ok) {
            fprintf(stderr, "%s: CUDA MXFP6_E2M3 quantize/eval failed for %s slice=%" PRId64 "\n",
                    __func__, binding.name.c_str(), i03);
            acc.ok = false;
            return;
        }
        acc.sum_sq += eval.sum_sq;
        acc.sum_abs += eval.sum_abs;
        acc.max_abs = std::max(acc.max_abs, eval.max_abs);
        acc.count += eval.count;
    };

    if (!quantize_binding_quantize<qtype::quant_type>(
            binding,
            nthread,
            0,
            0,
            0,
            __func__,
            prepare,
            run_slice,
            sum_sq,
            sum_abs,
            max_abs,
            count)) {
        return false;
    }

    if (!direct_patch && scale_in_header && count > 0) {
        auto * header = (tensor_mxfp6 *) out_bytes.data();
        header->weight_scale  = tuned_scales[0];
        header->input_scale   = input_scale;
        header->weight_scales = nullptr;
        header->input_scales  = nullptr;
    }

    if (direct_patch && count > 0 && direct_applied != nullptr) {
        *direct_applied = true;
    }
    return count > 0;
}

struct nvfp4_selector_proxy_metrics {
    bool ok = false;
    double rmse = std::numeric_limits<double>::infinity();
    double abs_mean = std::numeric_limits<double>::infinity();
    double max_abs = std::numeric_limits<double>::infinity();
    double score = std::numeric_limits<double>::infinity();
};

static nvfp4_selector_proxy_metrics nvfp4_selector_proxy_score(
        double sum_sq,
        double sum_abs,
        double max_abs,
        int64_t count) {
    nvfp4_selector_proxy_metrics out;
    if (count <= 0) {
        return out;
    }
    const double inv_count = 1.0 / (double) count;
    out.rmse = std::sqrt(std::max(0.0, sum_sq * inv_count));
    out.abs_mean = sum_abs * inv_count;
    out.max_abs = max_abs;
    out.score = out.rmse + 4.0 * out.abs_mean + 0.2 * out.max_abs;
    out.ok =
        std::isfinite(out.rmse) &&
        std::isfinite(out.abs_mean) &&
        std::isfinite(out.max_abs) &&
        std::isfinite(out.score);
    return out;
}

static nvfp4_selector_proxy_metrics nvfp4_selector_policy_proxy_metrics(
        const nvfp4_selector_policy & policy,
        bool prefer_survey) {
    nvfp4_selector_proxy_metrics out;
    if (prefer_survey && policy.has_survey) {
        out.ok = std::isfinite(policy.survey_proxy_score);
        out.rmse = policy.survey_proxy_rmse;
        out.abs_mean = policy.survey_proxy_abs_mean;
        out.max_abs = policy.survey_proxy_max_abs;
        out.score = policy.survey_proxy_score;
        return out;
    }
    out.ok = std::isfinite(policy.proxy_score);
    out.rmse = policy.proxy_rmse;
    out.abs_mean = policy.proxy_abs_mean;
    out.max_abs = policy.proxy_max_abs;
    out.score = policy.proxy_score;
    return out;
}

static bool nvfp4_selector_metric_close(double a, double b, double abs_tol, double rel_tol);

static int nvfp4_selector_policy_proxy_tie_rank(const nvfp4_selector_policy & policy) {
    int rank = 0;
    if (policy.name.find("recipe_") == 0) {
        rank -= 300;
    }
    if (policy.name == "baseline_auto") {
        rank -= 250;
    } else if (policy.name.find("baseline") != std::string::npos) {
        rank -= 150;
    }
    if (policy.cfg.use_compand_sat == 0) {
        rank += 1000;
    }
    rank += 2 * std::abs(policy.cfg.refit_iters - 8);
    rank += (int) std::llround(std::fabs((double) policy.cfg.cap_m6 - 448.0) / 16.0);
    rank += (int) std::llround(std::fabs((double) policy.cfg.cap_m4 - 256.0) / 16.0);
    return rank;
}

static bool nvfp4_selector_policy_proxy_less(const nvfp4_selector_policy & a, const nvfp4_selector_policy & b) {
    if (a.has_survey != b.has_survey) return a.has_survey > b.has_survey;
    const bool prefer_survey = a.has_survey && b.has_survey;
    const nvfp4_selector_proxy_metrics am = nvfp4_selector_policy_proxy_metrics(a, prefer_survey);
    const nvfp4_selector_proxy_metrics bm = nvfp4_selector_policy_proxy_metrics(b, prefer_survey);
    if (!nvfp4_selector_metric_close(am.score, bm.score, 1e-8, 1e-7)) return am.score < bm.score;
    const int ar = nvfp4_selector_policy_proxy_tie_rank(a);
    const int br = nvfp4_selector_policy_proxy_tie_rank(b);
    if (ar != br) return ar < br;
    if (am.score != bm.score) return am.score < bm.score;
    return a.name < b.name;
}

static bool nvfp4_selector_metric_close(double a, double b, double abs_tol, double rel_tol) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return a == b;
    }
    const double tol = std::max(abs_tol, rel_tol * std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= tol;
}

static bool nvfp4_selector_policy_is_recipe(const std::string & name) {
    return name == "recipe_current" || name.rfind("recipe_", 0) == 0;
}

static bool nvfp4_selector_proxy_equivalent(
        const nvfp4_selector_policy & a,
        const nvfp4_selector_policy & b,
        bool prefer_survey) {
    if (nvfp4_selector_policy_is_recipe(a.name) != nvfp4_selector_policy_is_recipe(b.name)) {
        return false;
    }
    const bool require_cfg_match = quantize_control_i64("LLAMA_NVFP4_SELECTOR_DEDUP_REQUIRE_CFG", 0) != 0;
    if (require_cfg_match && (
            a.cfg.choose46_mode != b.cfg.choose46_mode ||
            a.cfg.refit_iters != b.cfg.refit_iters ||
            a.cfg.use_compand_sat != b.cfg.use_compand_sat ||
            a.cfg.reserved_i32 != b.cfg.reserved_i32 ||
            a.cfg.cap_m6 != b.cfg.cap_m6 ||
            a.cfg.cap_m4 != b.cfg.cap_m4)) {
        return false;
    }
    const bool use_survey = prefer_survey && a.has_survey && b.has_survey;
    const nvfp4_selector_proxy_metrics am = nvfp4_selector_policy_proxy_metrics(a, use_survey);
    const nvfp4_selector_proxy_metrics bm = nvfp4_selector_policy_proxy_metrics(b, use_survey);

    return
        nvfp4_selector_metric_close(am.score,    bm.score,    1e-8, 1e-7) &&
        nvfp4_selector_metric_close(am.rmse,     bm.rmse,     1e-8, 1e-7) &&
        nvfp4_selector_metric_close(am.abs_mean, bm.abs_mean, 1e-8, 1e-7) &&
        nvfp4_selector_metric_close(am.max_abs,  bm.max_abs,  1e-7, 1e-7);
}

static bool nvfp4_selector_cfg_equal(const nvfp4_cuda_runtime_cfg & a, const nvfp4_cuda_runtime_cfg & b) {
    return
        a.choose46_mode == b.choose46_mode &&
        a.refit_iters == b.refit_iters &&
        a.use_compand_sat == b.use_compand_sat &&
        a.reserved_i32 == b.reserved_i32 &&
        a.cap_m6 == b.cap_m6 &&
        a.cap_m4 == b.cap_m4;
}

static std::vector<nvfp4_selector_policy> nvfp4_selector_default_policies(
        const nvfp4_cuda_runtime_cfg * recipe_cfg = nullptr,
        const std::string & recipe_policy_name = {}) {
    std::vector<nvfp4_selector_policy> out;
    auto push = [&](const std::string & name, nvfp4_cuda_runtime_cfg cfg) {
        for (const auto & existing : out) {
            if (nvfp4_selector_cfg_equal(existing.cfg, cfg)) {
                return;
            }
        }
        nvfp4_selector_policy p;
        p.name = name;
        p.cfg = cfg;
        out.push_back(std::move(p));
    };
    if (recipe_cfg != nullptr) {
        push(recipe_policy_name.empty() ? "recipe_current" : "recipe_" + recipe_policy_name, *recipe_cfg);
    }
    for (const llama_nvfp4_named_preset & preset : llama_nvfp4_preset_catalog()) {
        push(preset.name, preset.cfg);
    }
    return out;
}

static bool nvfp4_selector_policy_is_awq_tail(const std::string & name) {
    return name.find("asym_tail") != std::string::npos ||
           name.find("awq_tail") != std::string::npos;
}

static std::unordered_set<std::string> selector_policy_set_from_control(const char * key) {
    std::unordered_set<std::string> out;
    const std::string raw = quantize_control_string(key);
    if (raw.empty()) {
        return out;
    }

    std::string normalized(raw);
    for (char & ch : normalized) {
        if (ch == ';' || ch == '\n' || ch == '\r' || ch == '\t') {
            ch = ',';
        }
    }

    std::stringstream ss(normalized);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim_copy(token);
        if (!token.empty()) {
            out.insert(std::move(token));
        }
    }
    return out;
}

static void nvfp4_selector_push_policy_unique(std::vector<nvfp4_selector_policy> & out, std::string name, const nvfp4_cuda_runtime_cfg & cfg) {
    for (const auto & p : out) {
        if (nvfp4_selector_cfg_equal(p.cfg, cfg)) {
            return;
        }
    }
    nvfp4_selector_policy p;
    p.name = std::move(name);
    p.cfg = cfg;
    out.push_back(std::move(p));
}

static std::vector<std::pair<float, float>> nvfp4_selector_cap_pairs_for_class(
    nvfp4_selector_tensor_class cls,
    const nvfp4_cuda_runtime_cfg & base_cfg) {
    std::vector<std::pair<float, float>> out;
    auto add = [&](float c6, float c4) {
        c6 = std::clamp(c6, 256.0f, 544.0f);
        c4 = std::clamp(c4, 160.0f, 352.0f);
        if (c4 > c6) {
            c4 = c6;
        }
        const std::pair<float, float> p { c6, c4 };
        if (std::find(out.begin(), out.end(), p) == out.end()) {
            out.push_back(p);
        }
    };

    add(base_cfg.cap_m6, base_cfg.cap_m4);

    switch (cls) {
        case nvfp4_selector_tensor_class::ATTN_QKV:
        case nvfp4_selector_tensor_class::ATTN_OUT:
        case nvfp4_selector_tensor_class::SSM:
            add(320.0f, 224.0f);
            add(352.0f, 224.0f);
            add(416.0f, 192.0f);
            add(384.0f, 256.0f);
            add(448.0f, 224.0f);
            add(416.0f, 256.0f);
            add(448.0f, 320.0f);
            add(480.0f, 320.0f);
            add(384.0f, 384.0f);
            break;
        case nvfp4_selector_tensor_class::EXPERT_UP_GATE:
        case nvfp4_selector_tensor_class::EXPERT_DOWN:
            add(352.0f, 224.0f);
            add(384.0f, 224.0f);
            add(384.0f, 256.0f);
            add(416.0f, 256.0f);
            add(448.0f, 256.0f);
            add(480.0f, 256.0f);
            add(512.0f, 288.0f);
            add(448.0f, 320.0f);
            add(480.0f, 320.0f);
            add(512.0f, 320.0f);
            add(512.0f, 352.0f);
            add(544.0f, 320.0f);
            add(544.0f, 352.0f);
            break;
        default:
            add(320.0f, 224.0f);
            add(352.0f, 224.0f);
            add(384.0f, 256.0f);
            add(448.0f, 224.0f);
            add(448.0f, 256.0f);
            add(480.0f, 256.0f);
            add(448.0f, 320.0f);
            break;
    }

    return out;
}

static std::vector<int> nvfp4_selector_refit_candidates_for_class(
    nvfp4_selector_tensor_class cls,
    int base_refit) {
    std::vector<int> out;
    auto add = [&](int refit) {
        refit = std::max(2, std::min(32, refit));
        if (std::find(out.begin(), out.end(), refit) == out.end()) {
            out.push_back(refit);
        }
    };

    add(base_refit);
    add(4);
    add(8);
    add(12);
    add(16);
    add(24);
    if (cls == nvfp4_selector_tensor_class::ATTN_QKV ||
        cls == nvfp4_selector_tensor_class::ATTN_OUT ||
        cls == nvfp4_selector_tensor_class::SSM ||
        cls == nvfp4_selector_tensor_class::EXPERT_UP_GATE ||
        cls == nvfp4_selector_tensor_class::EXPERT_DOWN) {
        add(32);
    }
    return out;
}

static std::vector<nvfp4_selector_policy> nvfp4_selector_refine_policies(const std::vector<nvfp4_selector_policy> & ranked) {
    const int refine_top = (int) std::max<int64_t>(1, quantize_control_i64("LLAMA_NVFP4_SELECTOR_REFINE_TOP", 2));
    const int refine_budget = (int) std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_REFINE_BUDGET", 10));
    if (refine_budget <= 0 || ranked.empty()) {
        return {};
    }

    std::vector<nvfp4_selector_policy> out;
    for (int i = 0; i < refine_top && i < (int) ranked.size() && (int) out.size() < refine_budget; ++i) {
        const auto & seed = ranked[(size_t) i];
        auto add = [&](const std::string & suffix, nvfp4_cuda_runtime_cfg cfg) {
            if ((int) out.size() >= refine_budget) {
                return;
            }
            cfg.refit_iters = std::max(2, std::min(24, cfg.refit_iters));
            cfg.cap_m6 = std::clamp(cfg.cap_m6, 256.0f, 512.0f);
            cfg.cap_m4 = std::clamp(cfg.cap_m4, 160.0f, 320.0f);
            nvfp4_selector_push_policy_unique(out, seed.name + suffix, cfg);
        };

        nvfp4_cuda_runtime_cfg cfg = seed.cfg;
        add("_recover_m6", { cfg.choose46_mode, cfg.refit_iters, cfg.use_compand_sat, 0, cfg.cap_m6 - 32.0f, cfg.cap_m4 });
        add("_gentler_m4", { cfg.choose46_mode, cfg.refit_iters, cfg.use_compand_sat, 0, cfg.cap_m6, cfg.cap_m4 - 32.0f });
        add("_refit4",     { cfg.choose46_mode, 4,               cfg.use_compand_sat, 0, cfg.cap_m6, cfg.cap_m4 });
        add("_refit16",    { cfg.choose46_mode, 16,              cfg.use_compand_sat, 0, cfg.cap_m6, cfg.cap_m4 });
        add("_compand_flip", { cfg.choose46_mode, cfg.refit_iters, cfg.use_compand_sat ? 0 : 1, 0, cfg.cap_m6, cfg.cap_m4 });
    }

    return out;
}

static int64_t nvfp4_selector_rescue_sample_blocks(int64_t nb_total, nvfp4_selector_tensor_class cls) {
    const int64_t base = nvfp4_selector_autotune_sample_blocks(nb_total);
    const int64_t mult = SELECTOR_RESCUE_NVFP4_SAMPLE_MULT;
    const int64_t max_blocks = quantize_control_i64("LLAMA_NVFP4_SELECTOR_RESCUE_NVFP4_MAX_BLOCKS", 65536);
    int64_t min_blocks = SELECTOR_RESCUE_NVFP4_MIN_BLOCKS;
    switch (cls) {
        case nvfp4_selector_tensor_class::ATTN_QKV:
        case nvfp4_selector_tensor_class::ATTN_OUT:
        case nvfp4_selector_tensor_class::SSM:
            min_blocks = std::max<int64_t>(min_blocks, 8192);
            break;
        case nvfp4_selector_tensor_class::EXPERT_UP_GATE:
        case nvfp4_selector_tensor_class::EXPERT_DOWN:
            min_blocks = std::max<int64_t>(min_blocks, 16384);
            break;
        default:
            break;
    }
    const int64_t target = std::max<int64_t>(base * std::max<int64_t>(1, mult), min_blocks);
    return std::min<int64_t>(nb_total, std::max<int64_t>(1, std::min<int64_t>(target, max_blocks)));
}

static int64_t nvfp4_selector_rescue_refine_blocks(int64_t nb_total, nvfp4_selector_tensor_class cls, int64_t coarse_blocks) {
    const int64_t mult = SELECTOR_RESCUE_NVFP4_REFINE_MULT;
    const int64_t max_blocks = quantize_control_i64("LLAMA_NVFP4_SELECTOR_RESCUE_NVFP4_REFINE_MAX_BLOCKS", 131072);
    int64_t min_blocks = SELECTOR_RESCUE_NVFP4_REFINE_MIN_BLOCKS;
    switch (cls) {
        case nvfp4_selector_tensor_class::ATTN_QKV:
        case nvfp4_selector_tensor_class::ATTN_OUT:
        case nvfp4_selector_tensor_class::SSM:
            min_blocks = std::max<int64_t>(min_blocks, 16384);
            break;
        case nvfp4_selector_tensor_class::EXPERT_UP_GATE:
        case nvfp4_selector_tensor_class::EXPERT_DOWN:
            min_blocks = std::max<int64_t>(min_blocks, 32768);
            break;
        default:
            break;
    }
    const int64_t target = std::max<int64_t>(coarse_blocks * std::max<int64_t>(1, mult), min_blocks);
    return std::min<int64_t>(nb_total, std::max<int64_t>(1, std::min<int64_t>(target, max_blocks)));
}

static int64_t nvfp4_selector_rescue_guard_blocks(int64_t nb_total, nvfp4_selector_tensor_class cls, int64_t refine_blocks) {
    const int64_t mult = SELECTOR_RESCUE_NVFP4_GUARD_MULT;
    const int64_t max_blocks = quantize_control_i64("LLAMA_NVFP4_SELECTOR_RESCUE_NVFP4_GUARD_MAX_BLOCKS", 262144);
    int64_t min_blocks = SELECTOR_RESCUE_NVFP4_GUARD_MIN_BLOCKS;
    switch (cls) {
        case nvfp4_selector_tensor_class::ATTN_QKV:
        case nvfp4_selector_tensor_class::ATTN_OUT:
        case nvfp4_selector_tensor_class::SSM:
            min_blocks = std::max<int64_t>(min_blocks, 32768);
            break;
        case nvfp4_selector_tensor_class::EXPERT_UP_GATE:
        case nvfp4_selector_tensor_class::EXPERT_DOWN:
            min_blocks = std::max<int64_t>(min_blocks, 65536);
            break;
        default:
            break;
    }
    const int64_t target = std::max<int64_t>(refine_blocks * std::max<int64_t>(1, mult), min_blocks);
    return std::min<int64_t>(nb_total, std::max<int64_t>(1, std::min<int64_t>(target, max_blocks)));
}

static std::vector<nvfp4_selector_policy> nvfp4_selector_rescue_policies(
    const nvfp4_cuda_runtime_cfg & base_cfg,
    nvfp4_selector_tensor_class cls) {
    std::vector<nvfp4_selector_policy> out;
    nvfp4_selector_push_policy_unique(out, "rescue_base", base_cfg);

    const int rescue_budget = (int) std::max<int64_t>(8, SELECTOR_RESCUE_NVFP4_POLICY_BUDGET);
    auto add = [&](const std::string & name, nvfp4_cuda_runtime_cfg cfg) {
        if ((int) out.size() >= rescue_budget) {
            return;
        }
        cfg.refit_iters = std::max(2, std::min(32, cfg.refit_iters));
        cfg.cap_m6 = std::clamp(cfg.cap_m6, 256.0f, 544.0f);
        cfg.cap_m4 = std::clamp(cfg.cap_m4, 160.0f, 352.0f);
        nvfp4_selector_push_policy_unique(out, name, cfg);
    };

    const auto cap_pairs = nvfp4_selector_cap_pairs_for_class(cls, base_cfg);
    const auto refits = nvfp4_selector_refit_candidates_for_class(cls, base_cfg.refit_iters);

    for (int refit : refits) {
        add("rescue_refit", { base_cfg.choose46_mode, refit, base_cfg.use_compand_sat, 0, base_cfg.cap_m6, base_cfg.cap_m4 });
    }
    add("rescue_compand_flip", { base_cfg.choose46_mode, std::max(base_cfg.refit_iters, 8), base_cfg.use_compand_sat ? 0 : 1, 0, base_cfg.cap_m6, base_cfg.cap_m4 });

    for (const auto & caps : cap_pairs) {
        add("rescue_adaptive", { NVFP4_CUDA_CHOOSE46_ADAPTIVE, std::max(base_cfg.refit_iters, 8), 1, 0, caps.first, caps.second });
    }
    for (const auto & caps : cap_pairs) {
        add("rescue_adaptive_nocomp", { NVFP4_CUDA_CHOOSE46_ADAPTIVE, std::max(base_cfg.refit_iters, 8), 0, 0, caps.first, caps.second });
    }

    if (base_cfg.choose46_mode == NVFP4_CUDA_CHOOSE46_ADAPTIVE) {
        add("rescue_adaptive_tighter_m6", { NVFP4_CUDA_CHOOSE46_ADAPTIVE, std::max(base_cfg.refit_iters, 10), base_cfg.use_compand_sat, 0, std::max(320.0f, base_cfg.cap_m6 - 32.0f), std::min(256.0f, base_cfg.cap_m4) });
        add("rescue_adaptive_tighter_m4", { NVFP4_CUDA_CHOOSE46_ADAPTIVE, std::max(base_cfg.refit_iters, 10), base_cfg.use_compand_sat, 0, std::max(320.0f, base_cfg.cap_m6 - 64.0f), std::max(224.0f, base_cfg.cap_m4) });
    } else {
        add("rescue_adaptive", { NVFP4_CUDA_CHOOSE46_ADAPTIVE, std::max(base_cfg.refit_iters, 10), base_cfg.use_compand_sat, 0, base_cfg.cap_m6, base_cfg.cap_m4 });
    }

    for (const auto & p : nvfp4_selector_default_policies()) {
        if ((int) out.size() >= rescue_budget) {
            break;
        }
        nvfp4_selector_push_policy_unique(out, "rescue_" + p.name, p.cfg);
    }

    auto refined = nvfp4_selector_refine_policies(out);
    for (const auto & p : refined) {
        if ((int) out.size() >= rescue_budget) {
            break;
        }
        nvfp4_selector_push_policy_unique(out, "rescue_" + p.name, p.cfg);
    }

    if ((int) out.size() > rescue_budget) {
        out.resize((size_t) rescue_budget);
    }
    return out;
}

static bool nvfp4_selector_find_tensor_rescue_cfg(
    const nvfp4_selector_binding & binding,
    const nvfp4_cuda_runtime_cfg & base_cfg,
    int nthread,
    nvfp4_cuda_runtime_cfg & out_cfg,
    std::string & out_policy_name,
    int64_t & out_sample_blocks,
    double & out_base_score,
    double & out_best_score) {
    const int64_t nb_total = (binding.source->ne[0] * binding.source->ne[1]) / NVFP4_SELECTOR_BLOCK_SIZE;
    const int64_t sample_blocks = nvfp4_selector_rescue_sample_blocks(nb_total, binding.cls);
    const int64_t refine_blocks = nvfp4_selector_rescue_refine_blocks(nb_total, binding.cls, sample_blocks);
    const int64_t guard_blocks  = nvfp4_selector_rescue_guard_blocks(nb_total, binding.cls, refine_blocks);
    const int phase_count =
        (binding.cls == nvfp4_selector_tensor_class::EXPERT_UP_GATE || binding.cls == nvfp4_selector_tensor_class::EXPERT_DOWN) ? 6 :
        (binding.cls == nvfp4_selector_tensor_class::ATTN_QKV || binding.cls == nvfp4_selector_tensor_class::ATTN_OUT || binding.cls == nvfp4_selector_tensor_class::SSM) ? 4 :
        3;
    const double phase_tail_weight = SELECTOR_RESCUE_NVFP4_PHASE_TAIL_WEIGHT;
    const int64_t phase_stride = nvfp4_selector_sample_phase_stride(nb_total, binding.cls);
    auto policies = nvfp4_selector_rescue_policies(base_cfg, binding.cls);
    if (policies.empty()) {
        return false;
    }

    out_base_score = std::numeric_limits<double>::infinity();
    out_best_score = std::numeric_limits<double>::infinity();
    out_cfg = base_cfg;
    out_policy_name = "rescue_base";
    out_sample_blocks = sample_blocks;

    struct scored_policy {
        size_t index = 0;
        double score = std::numeric_limits<double>::infinity();
    };

    auto score_policy = [&](const nvfp4_selector_policy & policy, int64_t blocks, std::vector<uint8_t> & tmp_bytes, double & score_out) -> bool {
        double phase_sum = 0.0;
        double phase_worst = 0.0;
        int phases_done = 0;
        for (int phase_idx = 0; phase_idx < phase_count; ++phase_idx) {
            double sum_sq = 0.0;
            double sum_abs = 0.0;
            double max_abs = 0.0;
            int64_t count = 0;
            const int64_t sample_phase =
                phase_idx == 0 ? 0 :
                (phase_stride * phase_idx + std::max<int32_t>(0, binding.layer + 1)) % std::max<int64_t>(1, nb_total);
            if (!nvfp4_selector_quantize_binding(binding, policy.cfg, nthread, tmp_bytes, sum_sq, sum_abs, max_abs, count, blocks, sample_phase) || count <= 0) {
                return false;
            }
            const nvfp4_selector_proxy_metrics phase_metrics =
                nvfp4_selector_proxy_score(sum_sq, sum_abs, max_abs, count);
            if (!phase_metrics.ok) {
                return false;
            }
            const double phase_score = phase_metrics.score;
            phase_sum += phase_score;
            phase_worst = std::max(phase_worst, phase_score);
            ++phases_done;
        }
        if (phases_done <= 0) {
            return false;
        }
        score_out = (phase_sum / (double) phases_done) + phase_tail_weight * phase_worst;
        return std::isfinite(score_out);
    };

    const int refine_top = (int) std::max<int64_t>(1, SELECTOR_RESCUE_NVFP4_REFINE_TOP);
    const int guard_top  = (int) std::max<int64_t>(1, SELECTOR_RESCUE_NVFP4_GUARD_TOP);

    std::vector<uint8_t> tmp_bytes;
    std::vector<scored_policy> coarse_ranked;
    coarse_ranked.reserve(policies.size());
    for (size_t i = 0; i < policies.size(); ++i) {
        double score = std::numeric_limits<double>::infinity();
        if (!score_policy(policies[i], sample_blocks, tmp_bytes, score)) {
            continue;
        }
        coarse_ranked.push_back({ i, score });
        if (nvfp4_selector_cfg_equal(policies[i].cfg, base_cfg)) {
            out_base_score = score;
        }
        if (score < out_best_score) {
            out_best_score = score;
            out_cfg = policies[i].cfg;
            out_policy_name = policies[i].name;
        }
    }

    if (coarse_ranked.empty()) {
        return false;
    }

    std::sort(coarse_ranked.begin(), coarse_ranked.end(), [](const scored_policy & a, const scored_policy & b) {
        if (a.score != b.score) return a.score < b.score;
        return a.index < b.index;
    });

    if (refine_blocks > sample_blocks) {
        std::vector<scored_policy> refine_ranked;
        refine_ranked.reserve((size_t) refine_top + 1);
        auto maybe_push_refine = [&](size_t idx) {
            if (std::find_if(refine_ranked.begin(), refine_ranked.end(), [&](const scored_policy & sp) { return sp.index == idx; }) != refine_ranked.end()) {
                return;
            }
            double score = std::numeric_limits<double>::infinity();
            if (!score_policy(policies[idx], refine_blocks, tmp_bytes, score)) {
                return;
            }
            refine_ranked.push_back({ idx, score });
        };

        for (int i = 0; i < refine_top && i < (int) coarse_ranked.size(); ++i) {
            maybe_push_refine(coarse_ranked[(size_t) i].index);
        }
        for (size_t i = 0; i < policies.size(); ++i) {
            if (nvfp4_selector_cfg_equal(policies[i].cfg, base_cfg)) {
                maybe_push_refine(i);
                break;
            }
        }

        if (!refine_ranked.empty()) {
            std::sort(refine_ranked.begin(), refine_ranked.end(), [](const scored_policy & a, const scored_policy & b) {
                if (a.score != b.score) return a.score < b.score;
                return a.index < b.index;
            });
            coarse_ranked.swap(refine_ranked);
            out_best_score = coarse_ranked.front().score;
            out_cfg = policies[coarse_ranked.front().index].cfg;
            out_policy_name = policies[coarse_ranked.front().index].name;
            for (const auto & ranked : coarse_ranked) {
                if (nvfp4_selector_cfg_equal(policies[ranked.index].cfg, base_cfg)) {
                    out_base_score = ranked.score;
                    break;
                }
            }
        }
    }

    if (guard_blocks > refine_blocks) {
        std::vector<scored_policy> guard_ranked;
        guard_ranked.reserve((size_t) guard_top + 1);
        auto maybe_push_guard = [&](size_t idx) {
            if (std::find_if(guard_ranked.begin(), guard_ranked.end(), [&](const scored_policy & sp) { return sp.index == idx; }) != guard_ranked.end()) {
                return;
            }
            double score = std::numeric_limits<double>::infinity();
            if (!score_policy(policies[idx], guard_blocks, tmp_bytes, score)) {
                return;
            }
            guard_ranked.push_back({ idx, score });
        };

        for (int i = 0; i < guard_top && i < (int) coarse_ranked.size(); ++i) {
            maybe_push_guard(coarse_ranked[(size_t) i].index);
        }
        for (size_t i = 0; i < policies.size(); ++i) {
            if (nvfp4_selector_cfg_equal(policies[i].cfg, base_cfg)) {
                maybe_push_guard(i);
                break;
            }
        }

        if (!guard_ranked.empty()) {
            std::sort(guard_ranked.begin(), guard_ranked.end(), [](const scored_policy & a, const scored_policy & b) {
                if (a.score != b.score) return a.score < b.score;
                return a.index < b.index;
            });
            out_best_score = guard_ranked.front().score;
            out_cfg = policies[guard_ranked.front().index].cfg;
            out_policy_name = policies[guard_ranked.front().index].name;
            for (const auto & ranked : guard_ranked) {
                if (nvfp4_selector_cfg_equal(policies[ranked.index].cfg, base_cfg)) {
                    out_base_score = ranked.score;
                    break;
                }
            }
        }
    }

    if (!std::isfinite(out_base_score)) {
        out_base_score = out_best_score;
    }
    out_sample_blocks = guard_blocks > refine_blocks ? guard_blocks : (refine_blocks > sample_blocks ? refine_blocks : sample_blocks);
    return std::isfinite(out_best_score);
}

static bool nvfp4_selector_choose_policy(
    const std::string & source_model_path,
    const std::string & checkpoint_model_path,
    const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
    int nthread,
    const nvfp4_selector_kld_subset & kld,
    const nvfp4_selector_kld_subset * kld_holdout,
    const selector_rank_config & rank_cfg,
    const nvfp4_cuda_runtime_cfg * recipe_cfg,
    const std::string & recipe_policy_name,
    int32_t nvfp4_input_scale_policy,
    int32_t selector_eval_batch_override,
    int32_t nvfp4_autotune_threads,
    nvfp4_cuda_runtime_cfg & out_cfg,
    std::string & out_name,
        bool * out_kept_seed,
	    std::vector<tensor_type_option> * out_tensor_overrides) {
	    common_init();
    if (out_kept_seed != nullptr) {
        *out_kept_seed = false;
    }
    nvfp4_selector_progress_heartbeat full_quant_eta("full_quant_eta");
    int64_t full_quant_eta_done = 0;
    int64_t full_quant_eta_total = 0;
    int64_t stageb_policy_total = 0;
    int64_t stageb_policy_done = 0;
    double stageb_policy_done_seconds = 0.0;
    bool stageb_policy_active = false;
    std::chrono::steady_clock::time_point stageb_policy_start;
    auto stageb_eta_hint = [&]() -> std::string {
        if (full_quant_eta_total > 0 && full_quant_eta_done >= full_quant_eta_total) {
            return "done";
        }
        if (stageb_policy_total <= 0) {
            return {};
        }
        if (stageb_policy_done <= 0 || !(stageb_policy_done_seconds > 0.0)) {
            return "TBD...";
        }
        const double avg_policy_s = stageb_policy_done_seconds / (double) stageb_policy_done;
        int64_t remaining_policies = std::max<int64_t>(0, stageb_policy_total - stageb_policy_done);
        double remaining_s = 0.0;
        if (stageb_policy_active && remaining_policies > 0) {
            const double active_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - stageb_policy_start).count();
            remaining_s += std::max(0.0, avg_policy_s - active_s);
            --remaining_policies;
        }
        remaining_s += (double) remaining_policies * avg_policy_s;
        return nvfp4_selector_format_duration(remaining_s);
    };
    auto update_full_quant_eta = [&](const char * phase, bool print_now = false) {
        char detail[256];
        snprintf(detail, sizeof(detail),
            "phase=%s scope=whole_artifact_estimate basis=selector_stageb_policy_units_plus_final_materialization",
            phase);
        full_quant_eta.eta_hint(stageb_eta_hint());
        full_quant_eta.update(full_quant_eta_done, full_quant_eta_total, detail, print_now);
    };
    auto finish_stageb_policy_unit = [&](const char * phase, bool print_now = false) {
        if (stageb_policy_active) {
            stageb_policy_done_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - stageb_policy_start).count();
            ++stageb_policy_done;
            stageb_policy_active = false;
        }
        full_quant_eta_done = std::min<int64_t>(full_quant_eta_total, 2 + stageb_policy_done);
        update_full_quant_eta(phase, print_now);
    };
    update_full_quant_eta("selector-setup", true);

	    const int eval_top = (int) std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_EVAL_TOP", 6));
    const bool want_measured_eval_requested = eval_top > 0;
    const bool want_measured_eval = want_measured_eval_requested && quantize_control_i64("LLAMA_NVFP4_SELECTOR_ENABLE_EVAL", 0) != 0;
    const bool require_runtime_cache = quantize_control_i64("LLAMA_NVFP4_SELECTOR_REQUIRE_RUNTIME_CACHE", 0) != 0;

    std::vector<std::string> src_splits;
    llama_model_loader src_ml(
        /*metadata=*/ nullptr,
        /*set_tensor_data=*/ nullptr,
        /*set_tensor_data_ud=*/ nullptr,
        source_model_path,
        src_splits,
        /*file=*/ nullptr,
        /*use_mmap=*/ true,
        /*use_direct_io=*/ false,
        /*check_tensors=*/ false,
        /*no_alloc=*/ true,
        /*param_overrides_p=*/ nullptr,
        /*param_tensor_buft_overrides_p=*/ nullptr);
    src_ml.init_mappings(false);

    std::vector<std::string> seed_splits;
	    llama_model_loader seed_ml(
        /*metadata=*/ nullptr,
        /*set_tensor_data=*/ nullptr,
        /*set_tensor_data_ud=*/ nullptr,
	        checkpoint_model_path,
        seed_splits,
        /*file=*/ nullptr,
        /*use_mmap=*/ true,
        /*use_direct_io=*/ false,
        /*check_tensors=*/ false,
        /*no_alloc=*/ true,
        /*param_overrides_p=*/ nullptr,
        /*param_tensor_buft_overrides_p=*/ nullptr);
    seed_ml.init_mappings(false);

    std::vector<std::pair<std::string, ggml_tensor *>> seed_tmap;
    seed_tmap.reserve(seed_ml.weights_map.size());
    for (const auto & kv : seed_ml.weights_map) {
        if (kv.second.tensor == nullptr) {
            continue;
        }
        seed_tmap.emplace_back(kv.first, kv.second.tensor);
    }
    if (seed_tmap.empty()) {
        fprintf(stderr, "%s: failed to load candidate search checkpoint GGUF %s\n", __func__, checkpoint_model_path.c_str());
        return false;
    }

    const int32_t n_layer = nvfp4_selector_detect_n_layer(seed_tmap);
    std::vector<std::string> tensor_names = nvfp4_selector_build_stress_tensors(seed_tmap, n_layer);
    const int64_t max_tensors = std::max<int64_t>(1, quantize_control_i64("LLAMA_NVFP4_SELECTOR_MAX_TENSORS", 24));
    if ((int64_t) tensor_names.size() > max_tensors) {
        tensor_names.resize((size_t) max_tensors);
    }

    std::unordered_set<std::string> stress_name_set(tensor_names.begin(), tensor_names.end());
    std::vector<nvfp4_selector_binding> all_bindings;
    all_bindings.reserve(seed_tmap.size());
    std::vector<nvfp4_selector_binding> mxfp6_bindings;
    mxfp6_bindings.reserve(seed_tmap.size());
    std::vector<size_t> stress_binding_indices;
    const float selector_correction_denom =
        (float) quantize_control_f64("LLAMA_NVFP4_SELECTOR_CORRECTION_DENOM", NVFP4_SELECTOR_CORRECTION_DENOM_DEFAULT);
    for (const auto & kv : seed_tmap) {
        const std::string & tname = kv.first;
        ggml_tensor * target_tensor = kv.second;
        if (target_tensor == nullptr || (target_tensor->type != GGML_TYPE_NVFP4 && target_tensor->type != GGML_TYPE_MXFP6_E2M3)) {
            continue;
        }
        const bool is_mxfp6_target = target_tensor->type == GGML_TYPE_MXFP6_E2M3;
        const auto * src_weight = src_ml.get_weight(tname.c_str());
        ggml_tensor * src_tensor = src_weight ? src_weight->tensor : nullptr;
        if (src_tensor == nullptr) {
            continue;
        }
        src_ml.load_data_for(src_tensor);
        const float * imatrix_row = nullptr;
        auto it = imatrix_data.find(tname);
        if (it != imatrix_data.end() && it->second.size() >= (size_t) src_tensor->ne[0] * (size_t) std::max<int64_t>(1, src_tensor->ne[2])) {
            imatrix_row = it->second.data();
        }

        nvfp4_selector_binding b;
        b.name = tname;
        b.cls = nvfp4_selector_classify_tensor(tname);
        if (b.cls == nvfp4_selector_tensor_class::EMBEDDING_OUTPUT) {
            continue;
        }
        b.layer = nvfp4_selector_parse_layer(tname);
        b.bucket = nvfp4_selector_layer_bucket(b.layer, n_layer);
        b.target = target_tensor;
        if (is_mxfp6_target) {
            const std::string scale_name = llama_nvfp4_scale_tensor_name(tname);
            auto scale_it = seed_ml.weights_map.find(scale_name);
            if (scale_it != seed_ml.weights_map.end() && scale_it->second.tensor != nullptr &&
                    scale_it->second.tensor->type == GGML_TYPE_F32 &&
                    ggml_nelements(scale_it->second.tensor) >= std::max<int64_t>(1, target_tensor->ne[2])) {
                seed_ml.load_data_for(scale_it->second.tensor);
                b.target_scale = scale_it->second.tensor;
                b.target_scale_nbytes = (size_t) std::max<int64_t>(1, target_tensor->ne[2]) * sizeof(float);
            } else if (std::max<int64_t>(1, target_tensor->ne[2]) == 1 && ggml_nbytes(target_tensor) >= MXFP6_HEADER_OFFSET) {
                b.target_scale = nullptr;
                b.target_scale_nbytes = sizeof(float);
            }
        } else {
            const int64_t scale_len = std::max<int64_t>(1, src_tensor->ne[2]);
            const std::string scale_name = llama_nvfp4_scale_tensor_name(tname);
            auto scale_it = seed_ml.weights_map.find(scale_name);
            if (scale_it != seed_ml.weights_map.end() && scale_it->second.tensor != nullptr &&
                    scale_it->second.tensor->type == GGML_TYPE_F32 &&
                    ggml_nelements(scale_it->second.tensor) >= scale_len) {
                seed_ml.load_data_for(scale_it->second.tensor);
                b.target_scale = scale_it->second.tensor;
                b.target_scale_nbytes = (size_t) scale_len * sizeof(float);
            }
            const std::string input_scale_name = llama_nvfp4_input_scale_tensor_name(tname);
            auto input_scale_it = seed_ml.weights_map.find(input_scale_name);
            if (input_scale_it != seed_ml.weights_map.end() && input_scale_it->second.tensor != nullptr &&
                    input_scale_it->second.tensor->type == GGML_TYPE_F32 &&
                    ggml_nelements(input_scale_it->second.tensor) >= scale_len) {
                seed_ml.load_data_for(input_scale_it->second.tensor);
                b.target_input_scale = input_scale_it->second.tensor;
                b.target_input_scale_nbytes = (size_t) scale_len * sizeof(float);
            }
        }
        b.source = src_tensor;
        b.device_samples = std::make_shared<nvfp4_selector_device_sample_bank>();
        b.imatrix_row = imatrix_row;
        b.source_tensor_scale = src_tensor->type == GGML_TYPE_NVFP4 ? gguf_get_nvfp4_tensor_scale(src_ml.metadata, tname.c_str()) : 1.0f;
        if (src_tensor->type != GGML_TYPE_NVFP4) {
            b.quant_tensor_scales = nvfp4_selector_quant_tensor_scales(src_tensor, selector_correction_denom);
        }
        b.source_nbytes = ggml_nbytes(src_tensor);
        b.target_nbytes = ggml_nbytes(target_tensor);
        if (is_mxfp6_target) {
            if (b.target_scale_nbytes > 0) {
                mxfp6_bindings.push_back(std::move(b));
            }
        } else {
            all_bindings.push_back(std::move(b));
            if (stress_name_set.find(tname) != stress_name_set.end()) {
                stress_binding_indices.push_back(all_bindings.size() - 1);
            }
        }
    }

    nvfp4_selector_restore_counters runtime_restore_counters;
    auto restore_all = [&]() {
        if (!want_measured_eval) {
            return;
        }
        for (auto & b : all_bindings) {
            if (!quantize_tensor_host_buffer(b.target) &&
                    b.original_target_bytes.size() == b.target_nbytes) {
                quantize_restore_from_host(
                    b.target, b.original_target_bytes, b.target_nbytes, &runtime_restore_counters);
            }
            if (b.target_scale != nullptr && !quantize_tensor_host_buffer(b.target_scale) &&
                    b.original_scale_bytes.size() == b.target_scale_nbytes) {
                quantize_restore_from_snapshot_or_host(
                    b.target_scale, b.original_scale_device, b.original_scale_bytes, b.target_scale_nbytes,
                    &runtime_restore_counters);
            }
            if (b.target_input_scale != nullptr && !quantize_tensor_host_buffer(b.target_input_scale) &&
                    b.original_input_scale_bytes.size() == b.target_input_scale_nbytes) {
                quantize_restore_from_snapshot_or_host(
                    b.target_input_scale, b.original_input_scale_device, b.original_input_scale_bytes,
                    b.target_input_scale_nbytes, &runtime_restore_counters);
            }
        }
        for (auto & b : mxfp6_bindings) {
            if (!quantize_tensor_host_buffer(b.target)) {
                quantize_restore_from_snapshot_or_host(
                    b.target, b.original_target_device, b.original_target_bytes, b.target_nbytes, &runtime_restore_counters);
            }
            if (b.target_scale != nullptr && !quantize_tensor_host_buffer(b.target_scale)) {
                quantize_restore_from_snapshot_or_host(
                    b.target_scale, b.original_scale_device, b.original_scale_bytes, b.target_scale_nbytes,
                    &runtime_restore_counters);
            }
        }
    };

    auto covers_all_nvfp4_bindings = [&](const std::vector<size_t> & binding_indices) {
        if (binding_indices.size() != all_bindings.size()) {
            return false;
        }
        for (size_t i = 0; i < binding_indices.size(); ++i) {
            if (binding_indices[i] != i) {
                return false;
            }
        }
        return true;
    };

    if (all_bindings.empty() && mxfp6_bindings.empty()) {
        fprintf(stderr, "%s: selector found no NVFP4/MXFP6_E2M3 candidate tensors in %s\n", __func__, checkpoint_model_path.c_str());
        return false;
    }
    if (stress_binding_indices.empty() && !all_bindings.empty()) {
        for (size_t i = 0; i < all_bindings.size() && (int64_t) i < max_tensors; ++i) {
            stress_binding_indices.push_back(i);
        }
    }
    std::vector<size_t> all_binding_indices;
    all_binding_indices.reserve(all_bindings.size());
    for (size_t i = 0; i < all_bindings.size(); ++i) {
        all_binding_indices.push_back(i);
    }

    fprintf(stderr,
	        "%s: selector loaded %zu NVFP4 candidate tensors from checkpoint=%s (stress=%zu, survey=%zu, eval=%s, eval_rank=best)\n",
	        __func__,
	        all_bindings.size(),
	        checkpoint_model_path.c_str(),
        stress_binding_indices.size(),
        all_binding_indices.size(),
        want_measured_eval ? "enabled" : "disabled");
    if (!mxfp6_bindings.empty()) {
	        fprintf(stderr,
	            "%s: selector loaded %zu MXFP6_E2M3 scale-refine candidate tensors from checkpoint=%s\n",
	            __func__, mxfp6_bindings.size(), checkpoint_model_path.c_str());
    }
    if (want_measured_eval_requested && !want_measured_eval) {
        fprintf(stderr,
            "%s: selector measured stage is unavailable because the runtime patch cache is not active\n",
            __func__);
    }

    auto score_proxy_policy = [&](nvfp4_selector_policy & policy, const std::vector<size_t> & binding_indices) -> bool {
        restore_all();
        double sum_sq = 0.0;
        double sum_abs = 0.0;
        double max_abs = 0.0;
        int64_t count = 0;
        const int64_t stagea_default_sample_blocks = want_measured_eval ? 8192 : 0;
        const int64_t stagea_sample_blocks = std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_STAGEA_SAMPLE_BLOCKS", stagea_default_sample_blocks));
        policy.proxy_rejected = false;
        if (binding_indices.empty()) {
            policy.proxy_rmse = 0.0;
            policy.proxy_abs_mean = 0.0;
            policy.proxy_max_abs = 0.0;
            policy.proxy_score = 0.0;
            policy.first_tensor_rmse = 0.0;
            policy.first_tensor_abs_mean = 0.0;
            policy.first_tensor_max_abs = 0.0;
            fprintf(stderr,
                "selector proxy-rank phase=stage-a policy=%s objective=weight_reconstruction_sampled "
                "rank=lower_is_better score=0.000000 rmse=0.000000 mean_abs=0.000000 max_abs=0.000000 "
                "tensors=0 sample_blocks=full measured_kld_ppl=no skipped=no_nvfp4_tensors\n",
                policy.name.c_str());
            return true;
        }
        struct binding_score {
            bool ok = true;
            double tensor_sq = 0.0;
            double tensor_abs = 0.0;
            double tensor_max = 0.0;
            int64_t tensor_n = 0;
        };
        const int policy_threads = (int) std::max<int64_t>(1, std::min<int64_t>(
            (int64_t) binding_indices.size(),
            quantize_control_i64("LLAMA_NVFP4_SELECTOR_POLICY_THREADS", std::max(1, nthread))));
        const int binding_nthread = std::max(1, nthread / std::max(1, policy_threads));
        std::vector<binding_score> binding_scores(binding_indices.size());

        if (policy_threads <= 1 || binding_indices.size() <= 1) {
            std::vector<uint8_t> tmp_bytes;
            for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
                auto & b = all_bindings[binding_indices[ib]];
                std::vector<uint8_t> & quant_bytes = want_measured_eval ? b.working_target_bytes : tmp_bytes;
                if (!nvfp4_selector_quantize_binding(b, policy.cfg, binding_nthread, quant_bytes,
                        binding_scores[ib].tensor_sq, binding_scores[ib].tensor_abs, binding_scores[ib].tensor_max, binding_scores[ib].tensor_n,
                        stagea_sample_blocks)) {
                    fprintf(stderr, "%s: selector failed patching tensor %s for policy %s\n", __func__, b.name.c_str(), policy.name.c_str());
                    return false;
                }
            }
        } else {
            std::atomic<size_t> next_binding{0};
            std::vector<std::thread> workers_local;
            workers_local.reserve((size_t) policy_threads);
            for (int ti = 0; ti < policy_threads; ++ti) {
                workers_local.emplace_back([&, ti]() {
                    std::vector<uint8_t> tmp_bytes_local;
                    while (true) {
                        const size_t ib = next_binding.fetch_add(1, std::memory_order_relaxed);
                        if (ib >= binding_indices.size()) {
                            break;
                        }
                        auto & b = all_bindings[binding_indices[ib]];
                        std::vector<uint8_t> & quant_bytes = want_measured_eval ? b.working_target_bytes : tmp_bytes_local;
                        if (!nvfp4_selector_quantize_binding(b, policy.cfg, binding_nthread, quant_bytes,
                                binding_scores[ib].tensor_sq, binding_scores[ib].tensor_abs, binding_scores[ib].tensor_max, binding_scores[ib].tensor_n,
                                stagea_sample_blocks)) {
                            binding_scores[ib].ok = false;
                        }
                    }
                });
            }
            for (auto & worker : workers_local) {
                worker.join();
            }
            for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
                if (!binding_scores[ib].ok) {
                    auto & b = all_bindings[binding_indices[ib]];
                    fprintf(stderr, "%s: selector failed patching tensor %s for policy %s\n", __func__, b.name.c_str(), policy.name.c_str());
                    return false;
                }
            }
        }

        for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
            auto & b = all_bindings[binding_indices[ib]];
            const auto & bs = binding_scores[ib];
            if (want_measured_eval && stagea_sample_blocks <= 0 &&
                    !quantize_tensor_copy_in(b.target, b.working_target_bytes.data(), b.target_nbytes)) {
                fprintf(stderr, "%s: selector failed writing policy bytes for %s\n", __func__, b.name.c_str());
                return false;
            }
            sum_sq += bs.tensor_sq;
            sum_abs += bs.tensor_abs;
            max_abs = std::max(max_abs, bs.tensor_max);
            count += bs.tensor_n;
            if (ib == 0 && bs.tensor_n > 0) {
                policy.first_tensor_rmse = std::sqrt(bs.tensor_sq / (double) bs.tensor_n);
                policy.first_tensor_abs_mean = bs.tensor_abs / (double) bs.tensor_n;
                policy.first_tensor_max_abs = bs.tensor_max;
            }
        }

        const nvfp4_selector_proxy_metrics proxy_metrics =
            nvfp4_selector_proxy_score(sum_sq, sum_abs, max_abs, count);
        if (proxy_metrics.ok) {
            policy.proxy_rmse = proxy_metrics.rmse;
            policy.proxy_abs_mean = proxy_metrics.abs_mean;
            policy.proxy_max_abs = proxy_metrics.max_abs;
            policy.proxy_score = proxy_metrics.score;
        }

        fprintf(stderr,
            "selector proxy-rank phase=stage-a policy=%s objective=weight_reconstruction_sampled rank=lower_is_better "
            "score=%.6f rmse=%.6f mean_abs=%.6f max_abs=%.6f tensors=%zu sample_blocks=full measured_kld_ppl=no\n",
            policy.name.c_str(),
            policy.proxy_score,
            policy.proxy_rmse,
            policy.proxy_abs_mean,
            policy.proxy_max_abs,
            binding_indices.size());
        return true;
    };

    auto score_proxy_policy_extended = [&](nvfp4_selector_policy & policy,
                                          const std::vector<size_t> & binding_indices,
                                          int64_t sample_blocks_override,
                                          bool store_survey,
                                          const char * label) -> bool {
        restore_all();
        if (binding_indices.empty()) {
            if (store_survey) {
                policy.has_survey = true;
                policy.survey_proxy_rmse = 0.0;
                policy.survey_proxy_abs_mean = 0.0;
                policy.survey_proxy_max_abs = 0.0;
                policy.survey_proxy_score = 0.0;
            } else {
                policy.proxy_rmse = 0.0;
                policy.proxy_abs_mean = 0.0;
                policy.proxy_max_abs = 0.0;
                policy.proxy_score = 0.0;
            }
            fprintf(stderr,
                "selector proxy-rank phase=%s policy=%s objective=weight_reconstruction_sampled rank=lower_is_better "
                "score=0.000000 rmse=0.000000 mean_abs=0.000000 max_abs=0.000000 "
                "tensors=0 sample_blocks_per_tensor=%" PRId64 " measured_kld_ppl=no skipped=no_nvfp4_tensors\n",
                label,
                policy.name.c_str(),
                sample_blocks_override);
            return true;
        }
        double sum_sq = 0.0;
        double sum_abs = 0.0;
        double max_abs = 0.0;
        int64_t count = 0;
        struct binding_score {
            bool ok = true;
            double tensor_sq = 0.0;
            double tensor_abs = 0.0;
            double tensor_max = 0.0;
            int64_t tensor_n = 0;
        };
        const int policy_threads = (int) std::max<int64_t>(1, std::min<int64_t>(
            (int64_t) binding_indices.size(),
            quantize_control_i64("LLAMA_NVFP4_SELECTOR_POLICY_THREADS", std::max(1, nthread))));
        const int binding_nthread = std::max(1, nthread / std::max(1, policy_threads));
        std::vector<binding_score> binding_scores(binding_indices.size());

        if (policy_threads <= 1 || binding_indices.size() <= 1) {
            std::vector<uint8_t> tmp_bytes;
            for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
                auto & b = all_bindings[binding_indices[ib]];
                std::vector<uint8_t> & quant_bytes = want_measured_eval ? b.working_target_bytes : tmp_bytes;
                if (!nvfp4_selector_quantize_binding(b, policy.cfg, binding_nthread, quant_bytes,
                        binding_scores[ib].tensor_sq, binding_scores[ib].tensor_abs, binding_scores[ib].tensor_max, binding_scores[ib].tensor_n,
                        sample_blocks_override)) {
                    fprintf(stderr, "%s: selector failed patching tensor %s for policy %s\n", __func__, b.name.c_str(), policy.name.c_str());
                    return false;
                }
            }
        } else {
            std::atomic<size_t> next_binding{0};
            std::vector<std::thread> workers_local;
            workers_local.reserve((size_t) policy_threads);
            for (int ti = 0; ti < policy_threads; ++ti) {
                workers_local.emplace_back([&, ti]() {
                    std::vector<uint8_t> tmp_bytes_local;
                    while (true) {
                        const size_t ib = next_binding.fetch_add(1, std::memory_order_relaxed);
                        if (ib >= binding_indices.size()) {
                            break;
                        }
                        auto & b = all_bindings[binding_indices[ib]];
                        std::vector<uint8_t> & quant_bytes = want_measured_eval ? b.working_target_bytes : tmp_bytes_local;
                        if (!nvfp4_selector_quantize_binding(b, policy.cfg, binding_nthread, quant_bytes,
                                binding_scores[ib].tensor_sq, binding_scores[ib].tensor_abs, binding_scores[ib].tensor_max, binding_scores[ib].tensor_n,
                                sample_blocks_override)) {
                            binding_scores[ib].ok = false;
                        }
                    }
                });
            }
            for (auto & worker : workers_local) {
                worker.join();
            }
            for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
                if (!binding_scores[ib].ok) {
                    auto & b = all_bindings[binding_indices[ib]];
                    fprintf(stderr, "%s: selector failed patching tensor %s for policy %s\n", __func__, b.name.c_str(), policy.name.c_str());
                    return false;
                }
            }
        }

        for (size_t ib = 0; ib < binding_indices.size(); ++ib) {
            auto & b = all_bindings[binding_indices[ib]];
            const auto & bs = binding_scores[ib];
            if (want_measured_eval && sample_blocks_override <= 0 &&
                    !quantize_tensor_copy_in(b.target, b.working_target_bytes.data(), b.target_nbytes)) {
                fprintf(stderr, "%s: selector failed writing policy bytes for %s\n", __func__, b.name.c_str());
                return false;
            }
            sum_sq += bs.tensor_sq;
            sum_abs += bs.tensor_abs;
            max_abs = std::max(max_abs, bs.tensor_max);
            count += bs.tensor_n;
        }

        const nvfp4_selector_proxy_metrics proxy_metrics =
            nvfp4_selector_proxy_score(sum_sq, sum_abs, max_abs, count);
        if (proxy_metrics.ok) {
            if (store_survey) {
                policy.has_survey = true;
                policy.survey_proxy_rmse = proxy_metrics.rmse;
                policy.survey_proxy_abs_mean = proxy_metrics.abs_mean;
                policy.survey_proxy_max_abs = proxy_metrics.max_abs;
                policy.survey_proxy_score = proxy_metrics.score;
            } else {
                policy.proxy_rmse = proxy_metrics.rmse;
                policy.proxy_abs_mean = proxy_metrics.abs_mean;
                policy.proxy_max_abs = proxy_metrics.max_abs;
                policy.proxy_score = proxy_metrics.score;
            }
            fprintf(stderr,
                "selector proxy-rank phase=%s policy=%s objective=weight_reconstruction_sampled rank=lower_is_better "
                "score=%.6f rmse=%.6f mean_abs=%.6f max_abs=%.6f tensors=%zu sample_blocks_per_tensor=%" PRId64 " measured_kld_ppl=no\n",
                label,
                policy.name.c_str(),
                proxy_metrics.score,
                proxy_metrics.rmse,
                proxy_metrics.abs_mean,
                proxy_metrics.max_abs,
                binding_indices.size(),
                sample_blocks_override);
        }
        return true;
    };

    auto reject_policy = [&](nvfp4_selector_policy & policy, const char * phase) {
        restore_all();
        policy.proxy_score = std::numeric_limits<double>::infinity();
        policy.proxy_rmse = std::numeric_limits<double>::infinity();
        policy.proxy_abs_mean = std::numeric_limits<double>::infinity();
        policy.proxy_max_abs = std::numeric_limits<double>::infinity();
        policy.measured_score = std::numeric_limits<double>::infinity();
        policy.measured_pass = false;
        policy.proxy_rejected = true;
        fprintf(stderr, "%s: selector rejected policy=%s during %s after quantize/patch failure\n",
            __func__, policy.name.c_str(), phase);
    };

    bool skip_remaining_tuning = false;
    auto note_skip_remaining = [&](const char * phase) {
        if (skip_remaining_tuning) {
            return true;
        }
        if (!nvfp4_selector_skip_requested(phase)) {
            return false;
        }
        skip_remaining_tuning = true;
        fprintf(stderr,
            "%s: selector skip requested after %s; remaining optional tuning will be skipped\n",
            __func__,
            phase != nullptr && phase[0] != '\0' ? phase : "current phase");
        return true;
    };

    auto policies = nvfp4_selector_default_policies(recipe_cfg, recipe_policy_name);
    const bool has_expert_bindings = std::any_of(all_bindings.begin(), all_bindings.end(), [](const nvfp4_selector_binding & b) {
        return b.name.find(".ffn_gate_exps.weight") != std::string::npos ||
               b.name.find(".ffn_up_exps.weight") != std::string::npos ||
               b.name.find(".ffn_down_exps.weight") != std::string::npos;
    });
    const bool include_moe_policies = quantize_control_i64(
        "LLAMA_NVFP4_SELECTOR_INCLUDE_MOE_POLICIES", has_expert_bindings ? 1 : 0) != 0;
    if (!include_moe_policies) {
        const size_t before = policies.size();
        policies.erase(std::remove_if(policies.begin(), policies.end(), [](const nvfp4_selector_policy & policy) {
            return !nvfp4_selector_policy_is_recipe(policy.name) &&
                   (policy.name.find("_moe") != std::string::npos ||
                    policy.name.find("moe_") != std::string::npos ||
                    policy.name.find("awq_tail_moe") != std::string::npos);
        }), policies.end());
        if (before != policies.size()) {
            fprintf(stderr,
                "%s: selector pruned %zu MoE-specialized policy candidate(s) for dense/non-expert model\n",
                __func__,
                before - policies.size());
        }
    }
    const std::unordered_set<std::string> skip_policy_names =
        selector_policy_set_from_control("LLAMA_NVFP4_SELECTOR_SKIP_POLICIES");
    auto selector_policy_skipped = [&](const nvfp4_selector_policy & policy) {
        return policy.name != "seed_keep" &&
               skip_policy_names.find(policy.name) != skip_policy_names.end();
    };
    if (!skip_policy_names.empty()) {
        fprintf(stderr, "%s: selector will skip %zu named policy candidate(s) during survey/eval\n",
            __func__, skip_policy_names.size());
    }
    const int stagea_max_policies = (int) std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_STAGEA_MAX_POLICIES", 0));
    if (stagea_max_policies > 0 && (int) policies.size() > stagea_max_policies) {
        const int awq_keep = std::min<int>(
            stagea_max_policies,
            (int) std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_AWQ_TOP", 0)));
        if (awq_keep > 0) {
            std::vector<nvfp4_selector_policy> limited;
            limited.reserve((size_t) stagea_max_policies);
            auto push_policy = [&](const nvfp4_selector_policy & policy) {
                for (const auto & existing : limited) {
                    if (existing.name == policy.name) {
                        return false;
                    }
                }
                if ((int) limited.size() >= stagea_max_policies) {
                    return false;
                }
                limited.push_back(policy);
                return true;
            };

            const int primary_keep = std::max(0, stagea_max_policies - awq_keep);
            for (const auto & policy : policies) {
                if ((int) limited.size() >= primary_keep) {
                    break;
                }
                if (!nvfp4_selector_policy_is_awq_tail(policy.name)) {
                    push_policy(policy);
                }
            }
            size_t awq_added = 0;
            for (const auto & policy : policies) {
                if ((int) awq_added >= awq_keep) {
                    break;
                }
                if (nvfp4_selector_policy_is_awq_tail(policy.name)) {
                    if (push_policy(policy)) {
                        ++awq_added;
                    }
                }
            }
            for (const auto & policy : policies) {
                if ((int) limited.size() >= stagea_max_policies) {
                    break;
                }
                push_policy(policy);
            }
            fprintf(stderr,
                "%s: selector capped stage-a policies to %zu/%zu while preserving %zu asym/AWQ-tail candidate(s)\n",
                __func__,
                limited.size(),
                policies.size(),
                awq_added);
            policies.swap(limited);
        } else {
            policies.resize((size_t) stagea_max_policies);
        }
    }
    for (size_t policy_idx = 0; policy_idx < policies.size(); ++policy_idx) {
        auto & policy = policies[policy_idx];
        fprintf(stderr,
            "selector stage-a start [%zu/%zu] policy=%s tensors=%zu\n",
            policy_idx + 1,
            policies.size(),
            policy.name.c_str(),
            stress_binding_indices.size());
        if (!score_proxy_policy(policy, stress_binding_indices)) {
            reject_policy(policy, "stage-a");
            continue;
        }
    }
    note_skip_remaining("stage-a");

    std::sort(policies.begin(), policies.end(), nvfp4_selector_policy_proxy_less);

    if (!skip_remaining_tuning) {
        auto refined = nvfp4_selector_refine_policies(policies);
        for (size_t policy_idx = 0; policy_idx < refined.size(); ++policy_idx) {
            auto & policy = refined[policy_idx];
            fprintf(stderr,
                "selector refine start [%zu/%zu] policy=%s tensors=%zu\n",
                policy_idx + 1,
                refined.size(),
                policy.name.c_str(),
                stress_binding_indices.size());
            if (!score_proxy_policy(policy, stress_binding_indices)) {
                reject_policy(policy, "refine");
                continue;
            }
            policies.push_back(std::move(policy));
            if (note_skip_remaining("refine")) {
                break;
            }
        }
    }

    auto baseline_it = std::find_if(policies.begin(), policies.end(), [](const nvfp4_selector_policy & policy) {
        return policy.name == "baseline_auto";
    });
    if (baseline_it == policies.end()) {
        baseline_it = policies.begin();
    }
    if (baseline_it != policies.end()) {
        const double soft_factor = SELECTOR_FIRST_TENSOR_SOFT_FACTOR;
        const double hard_factor = SELECTOR_FIRST_TENSOR_HARD_FACTOR;
        const double soft_penalty = SELECTOR_FIRST_TENSOR_PENALTY;
        const double base_rmse = std::max(1e-9, baseline_it->first_tensor_rmse);
        const double base_max = std::max(1e-9, baseline_it->first_tensor_max_abs);

        for (auto & policy : policies) {
            if (policy.name == baseline_it->name) {
                continue;
            }

            const double rmse_ratio = policy.first_tensor_rmse / base_rmse;
            const double max_ratio = policy.first_tensor_max_abs / base_max;
            const double worst_ratio = std::max(rmse_ratio, max_ratio);
            if (!std::isfinite(worst_ratio)) {
                continue;
            }

            if (worst_ratio > hard_factor) {
                policy.proxy_score = std::numeric_limits<double>::infinity();
                policy.proxy_rejected = true;
                fprintf(stderr,
                    "selector stage-a reject policy=%s first_tensor ratio=%.3f hard=%.3f\n",
                    policy.name.c_str(), worst_ratio, hard_factor);
            } else if (worst_ratio > soft_factor) {
                const double penalty = soft_penalty * (worst_ratio - soft_factor);
                policy.proxy_score += penalty;
                fprintf(stderr,
                    "selector stage-a penalize policy=%s first_tensor ratio=%.3f soft=%.3f penalty=%.3f\n",
                    policy.name.c_str(), worst_ratio, soft_factor, penalty);
            }
        }
    }

    std::sort(policies.begin(), policies.end(), nvfp4_selector_policy_proxy_less);

    const int survey_top = (int) std::max<int64_t>(1, quantize_control_i64("LLAMA_NVFP4_SELECTOR_SURVEY_TOP", 6));
    const int64_t survey_sample_blocks = std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_SURVEY_SAMPLE_BLOCKS", 1024));
    const bool dedup_survey = NVFP4_SELECTOR_DEDUP_SURVEY_DEFAULT;
    std::vector<size_t> survey_policy_indices;
    survey_policy_indices.reserve((size_t) survey_top + 1);
    size_t survey_dedup_skipped = 0;
    for (size_t i = 0; i < policies.size() && survey_policy_indices.size() < (size_t) survey_top; ++i) {
        if (policies[i].proxy_rejected || !std::isfinite(policies[i].proxy_score)) {
            continue;
        }
        if (selector_policy_skipped(policies[i])) {
            continue;
        }
        if (dedup_survey) {
            bool duplicate = false;
            for (size_t selected : survey_policy_indices) {
                if (nvfp4_selector_proxy_equivalent(policies[i], policies[selected], false)) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                ++survey_dedup_skipped;
                continue;
            }
        }
        survey_policy_indices.push_back(i);
    }
    for (size_t i = 0; i < policies.size(); ++i) {
        if (policies[i].name == "baseline_auto") {
            if (!selector_policy_skipped(policies[i]) &&
                    std::find(survey_policy_indices.begin(), survey_policy_indices.end(), i) == survey_policy_indices.end()) {
                survey_policy_indices.push_back(i);
            }
            break;
        }
    }
    if (dedup_survey && survey_dedup_skipped > 0) {
        fprintf(stderr,
            "%s: selector skipped %zu proxy-equivalent policies before full-tensor survey\n",
            __func__,
            survey_dedup_skipped);
    }
    if (!skip_remaining_tuning) {
        for (size_t survey_pos = 0; survey_pos < survey_policy_indices.size(); ++survey_pos) {
            const size_t policy_idx = survey_policy_indices[survey_pos];
            fprintf(stderr,
                "selector survey start [%zu/%zu] policy=%s tensors=%zu sample_blocks=%" PRId64 "\n",
                survey_pos + 1,
                survey_policy_indices.size(),
                policies[policy_idx].name.c_str(),
                all_binding_indices.size(),
                survey_sample_blocks);
            if (!score_proxy_policy_extended(policies[policy_idx], all_binding_indices, survey_sample_blocks, true, "stage-a-survey")) {
                reject_policy(policies[policy_idx], "survey");
                continue;
            }
            if (note_skip_remaining("survey")) {
                break;
            }
        }
    }

    std::sort(policies.begin(), policies.end(), nvfp4_selector_policy_proxy_less);

    fprintf(stderr,
        "%s: selector survey complete; shortlisted %zu policies for full-model proxy ranking, top policy=%s\n",
        __func__,
        survey_policy_indices.size(),
        policies.empty() ? "<none>" : policies.front().name.c_str());

    const bool dedup_eval = NVFP4_SELECTOR_DEDUP_EVAL_DEFAULT;
    std::vector<size_t> eval_policy_indices;
    eval_policy_indices.reserve((size_t) std::max<int64_t>(1, eval_top) + 1);
    size_t eval_dedup_skipped = 0;
    for (size_t i = 0; i < policies.size() && eval_policy_indices.size() < (size_t) std::max<int64_t>(1, eval_top); ++i) {
        if (policies[i].proxy_rejected || !std::isfinite(policies[i].proxy_score)) {
            continue;
        }
        if (selector_policy_skipped(policies[i])) {
            continue;
        }
        if (dedup_eval) {
            bool duplicate = false;
            for (size_t selected : eval_policy_indices) {
                if (nvfp4_selector_proxy_equivalent(policies[i], policies[selected], true)) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                ++eval_dedup_skipped;
                continue;
            }
        }
        eval_policy_indices.push_back(i);
    }
    for (size_t i = 0; i < policies.size(); ++i) {
        if (policies[i].name == "baseline_auto") {
            if (!selector_policy_skipped(policies[i]) &&
                    std::find(eval_policy_indices.begin(), eval_policy_indices.end(), i) == eval_policy_indices.end()) {
                eval_policy_indices.push_back(i);
            }
            break;
        }
    }
    if (dedup_eval && eval_dedup_skipped > 0) {
        fprintf(stderr,
            "%s: selector skipped %zu proxy-equivalent policies before full PPL/KLD eval\n",
            __func__,
            eval_dedup_skipped);
    }
    note_skip_remaining("stage-b start");
    const bool run_stageb_eval = want_measured_eval && !skip_remaining_tuning;
    if (run_stageb_eval) {
        full_quant_eta_total = (int64_t) eval_policy_indices.size() + 2;
        full_quant_eta_done = 0;
        stageb_policy_total = (int64_t) eval_policy_indices.size();
        stageb_policy_done = 0;
        stageb_policy_done_seconds = 0.0;
        stageb_policy_active = false;
        update_full_quant_eta("selector-stage-b-runtime-load", true);
    } else {
        full_quant_eta_total = 1;
        full_quant_eta_done = 0;
        stageb_policy_total = 0;
        stageb_policy_done = 0;
        stageb_policy_done_seconds = 0.0;
        stageb_policy_active = false;
        update_full_quant_eta(skip_remaining_tuning ? "selector-skipping-to-final-materialization" : "selector-proxy-only", true);
    }
    const nvfp4_selector_kld_subset kld_budget = nvfp4_selector_make_kld_budget_subset(kld, (int32_t) quantize_control_i64("LLAMA_NVFP4_SELECTOR_EVAL_CHUNKS", 4));
    const std::unique_ptr<nvfp4_selector_kld_subset> holdout_budget =
        (kld_holdout != nullptr)
        ? std::make_unique<nvfp4_selector_kld_subset>(nvfp4_selector_make_kld_budget_subset(*kld_holdout, (int32_t) quantize_control_i64("LLAMA_NVFP4_SELECTOR_EVAL_CHUNKS", 4)))
        : nullptr;
    int selector_n_seq = (int) std::max<int64_t>(1, quantize_control_i64("LLAMA_NVFP4_SELECTOR_N_SEQ", 4));
    const nvfp4_selector_logits_budget logits_budget = nvfp4_selector_default_logits_budget();
    const double max_logits_gib = std::max(NVFP4_SELECTOR_LOGITS_MIN_GIB, logits_budget.max_gib);
    const double logits_gib_per_seq =
        (double) kld.n_ctx * (double) kld.n_vocab * (double) sizeof(float) / (1024.0 * 1024.0 * 1024.0);
    if (logits_gib_per_seq > 0.0 && std::isfinite(logits_gib_per_seq)) {
        const int max_n_seq_by_logits = std::max(1, (int) std::floor(max_logits_gib / logits_gib_per_seq));
        if (selector_n_seq > max_n_seq_by_logits) {
            fprintf(stderr,
                "%s: selector eval n_seq capped %d -> %d to keep logits workspace under %.2f GiB "
                "(host_available=%.2f GiB reserve=%.2f GiB ctx=%d vocab=%d, %.2f GiB/seq)\n",
                __func__,
                selector_n_seq,
                max_n_seq_by_logits,
                max_logits_gib,
                logits_budget.available_gib,
                logits_budget.reserve_gib,
                kld.n_ctx,
                kld.n_vocab,
                logits_gib_per_seq);
            selector_n_seq = max_n_seq_by_logits;
        }
    }
    common_params params;
    params.model.path = checkpoint_model_path;
    params.n_ctx = kld.n_ctx * selector_n_seq;
    params.n_batch = selector_eval_batch_override > 0
        ? std::min<int32_t>((int32_t) params.n_ctx, selector_eval_batch_override)
        : params.n_ctx;
    params.n_ubatch = params.n_batch;
    params.n_parallel = selector_n_seq;
    params.n_gpu_layers = (int32_t) quantize_control_i64("LLAMA_NVFP4_SELECTOR_N_GPU_LAYERS", 9999);
    params.cpuparams.n_threads = std::max(1, nthread);
    params.cpuparams_batch.n_threads = std::max(1, nthread);
    // Selector full PPL/KLD evaluation patches tensor buffers in a calibration-only context with
    // explicit n_ctx/n_batch/n_parallel. Let it load exactly that context instead of
    // running the general fit-to-VRAM probe, which can fail on freshly written local
    // NVFP4/MXFP6 GGUFs before the real selector load is attempted.
    params.fit_params = false;
    // Keep mmap enabled for selector full PPL/KLD evaluation. The non-mmap loader path
    // is fragile with derived/native NVFP4 GGUF tensors, while runtime tensor
    // patching still works through ggml_backend_tensor_set on loaded buffers.
    params.use_mmap = true;
    // Do not pre-fault the whole checkpoint mapping in one loader thread.
    // Stage-b only needs a calibration context; letting pages fault on demand
    // avoids a long CPU-only MAP_POPULATE phase before CUDA work can begin.
    params.use_mmap_prefetch = false;
    params.warmup = false;
    params.verbosity = 0;
    params.graph_reuse_disable = true;
    std::unique_ptr<nvfp4_selector_progress_heartbeat> load_heartbeat;
    if (run_stageb_eval) {
        load_heartbeat = std::make_unique<nvfp4_selector_progress_heartbeat>(
            "selector stage-b checkpoint load",
            10000);
    }
    params.load_progress_callback = [](float progress, void * user_data) -> bool {
        auto * heartbeat = static_cast<nvfp4_selector_progress_heartbeat *>(user_data);
        if (heartbeat != nullptr) {
            const int64_t done = (int64_t) std::llround(
                10000.0 * std::max(0.0f, std::min(1.0f, progress)));
            heartbeat->update(done, 10000, "loading runtime checkpoint", progress >= 1.0f);
        }
        return true;
    };
    params.load_progress_callback_user_data = load_heartbeat.get();

    common_init_result_ptr init_res;
    llama_context * lctx = nullptr;
    if (run_stageb_eval) {
#if defined(GGML_USE_CUDA)
        if (ggml_backend_cuda_get_device_count() > 0) {
            size_t cuda_free = 0;
            size_t cuda_total = 0;
            ggml_backend_cuda_get_device_memory(0, &cuda_free, &cuda_total);
            const bool reset_before_eval = true;
            if (reset_before_eval && cuda_total > 0 && cuda_free < cuda_total / 2) {
                fprintf(stderr,
                    "%s: selector stage-b resetting CUDA device 0 before checkpoint load to release autotune/proxy workspaces (free=%.2f GiB total=%.2f GiB)\n",
                    __func__,
                    (double) cuda_free / (1024.0 * 1024.0 * 1024.0),
                    (double) cuda_total / (1024.0 * 1024.0 * 1024.0));
                if (nvfp4_selector_reset_cuda_device(0)) {
                    ggml_backend_cuda_get_device_memory(0, &cuda_free, &cuda_total);
                    fprintf(stderr,
                        "%s: selector stage-b CUDA memory after reset free=%.2f GiB total=%.2f GiB\n",
                        __func__,
                        (double) cuda_free / (1024.0 * 1024.0 * 1024.0),
                        (double) cuda_total / (1024.0 * 1024.0 * 1024.0));
                }
            }
            fprintf(stderr,
                "%s: selector stage-b CUDA memory before checkpoint load free=%.2f GiB total=%.2f GiB n_gpu_layers=%d n_seq=%d n_ctx=%d n_batch=%d n_ubatch=%d\n",
                __func__,
                (double) cuda_free / (1024.0 * 1024.0 * 1024.0),
                (double) cuda_total / (1024.0 * 1024.0 * 1024.0),
                params.n_gpu_layers,
                selector_n_seq,
                (int) params.n_ctx,
                (int) params.n_batch,
                (int) params.n_ubatch);
        }
#endif
        // The selector mutates already-loaded tensor buffers between KLD probes,
        // so this calibration context must not reuse graphs across evaluations.
        if (load_heartbeat) {
            load_heartbeat->detail("loading runtime checkpoint", true);
        }
        init_res = common_init_from_params(params);
        if (load_heartbeat) {
            load_heartbeat->finish(init_res ? "runtime checkpoint loaded" : "runtime checkpoint load failed");
        }
        lctx = init_res ? init_res->context() : nullptr;
    }
    const bool have_measured_eval = run_stageb_eval && lctx != nullptr;
    if (run_stageb_eval && !have_measured_eval) {
        fprintf(stderr, "%s: selector stage-b unavailable for checkpoint %s; continuing with proxy-only selection and measured tensor policy map\n", __func__, checkpoint_model_path.c_str());
        if (require_runtime_cache) {
            fprintf(stderr, "%s: selector runtime cache is required; aborting selector\n", __func__);
            return false;
        }
    }
    if (have_measured_eval) {
        nvfp4_selector_progress_heartbeat runtime_bind_heartbeat(
            "selector stage-b runtime bind",
            (int64_t) all_bindings.size() + (int64_t) mxfp6_bindings.size());
        runtime_bind_heartbeat.detail("binding runtime tensors", true);
        std::unordered_map<std::string, ggml_tensor *> runtime_tensors;
        for (const auto & kv : llama_internal_get_tensor_map(init_res->model())) {
            if (kv.second != nullptr) {
                runtime_tensors.emplace(kv.first, kv.second);
            }
        }
        int64_t runtime_bind_done = 0;
        for (auto & b : all_bindings) {
            auto it = runtime_tensors.find(b.name);
            if (it == runtime_tensors.end() || it->second == nullptr || it->second->type != GGML_TYPE_NVFP4) {
                fprintf(stderr, "%s: selector stage-b missing runtime NVFP4 tensor %s; continuing with proxy-only selection\n",
                    __func__, b.name.c_str());
                if (require_runtime_cache) {
                    return false;
                }
                lctx = nullptr;
                break;
            }
            const size_t runtime_nbytes = ggml_nbytes(it->second);
            if (runtime_nbytes != b.target_nbytes) {
                fprintf(stderr, "%s: selector stage-b runtime tensor size mismatch for %s (%zu != %zu); continuing with proxy-only selection\n",
                    __func__, b.name.c_str(), runtime_nbytes, b.target_nbytes);
                if (require_runtime_cache) {
                    return false;
                }
                lctx = nullptr;
                break;
            }
            b.target = it->second;
            const int64_t scale_len = std::max<int64_t>(1, b.source ? b.source->ne[2] : 1);
            const std::string scale_name = llama_nvfp4_scale_tensor_name(b.name);
            auto sit = runtime_tensors.find(scale_name);
            if (sit != runtime_tensors.end() && sit->second != nullptr && sit->second->type == GGML_TYPE_F32 &&
                    ggml_nelements(sit->second) >= scale_len) {
                b.target_scale = sit->second;
                b.target_scale_nbytes = (size_t) scale_len * sizeof(float);
            }
            const std::string input_scale_name = llama_nvfp4_input_scale_tensor_name(b.name);
            auto iit = runtime_tensors.find(input_scale_name);
            if (iit != runtime_tensors.end() && iit->second != nullptr && iit->second->type == GGML_TYPE_F32 &&
                    ggml_nelements(iit->second) >= scale_len) {
                b.target_input_scale = iit->second;
                b.target_input_scale_nbytes = (size_t) scale_len * sizeof(float);
            }
            runtime_bind_heartbeat.update(++runtime_bind_done, "binding runtime tensors");
        }
        for (auto & b : mxfp6_bindings) {
            auto it = runtime_tensors.find(b.name);
            if (it == runtime_tensors.end() || it->second == nullptr || it->second->type != GGML_TYPE_MXFP6_E2M3) {
                fprintf(stderr, "%s: selector stage-b missing runtime MXFP6_E2M3 tensor %s; continuing without MXFP6_E2M3 scale measured tuning\n",
                    __func__, b.name.c_str());
                b.target = nullptr;
                continue;
            }
            const size_t runtime_nbytes = ggml_nbytes(it->second);
            if (runtime_nbytes != b.target_nbytes) {
                fprintf(stderr, "%s: selector stage-b runtime MXFP6_E2M3 tensor size mismatch for %s (%zu != %zu); continuing without that tensor\n",
                    __func__, b.name.c_str(), runtime_nbytes, b.target_nbytes);
                b.target = nullptr;
                continue;
            }
            b.target = it->second;

            const std::string scale_name = llama_nvfp4_scale_tensor_name(b.name);
            auto sit = runtime_tensors.find(scale_name);
            if (sit == runtime_tensors.end() || sit->second == nullptr || sit->second->type != GGML_TYPE_F32) {
                if (b.target_scale_nbytes == sizeof(float) && runtime_nbytes >= MXFP6_HEADER_OFFSET) {
                    b.target_scale = nullptr;
                    continue;
                }
                fprintf(stderr, "%s: selector stage-b missing runtime MXFP6_E2M3 scale tensor %s; continuing without %s\n",
                    __func__, scale_name.c_str(), b.name.c_str());
                b.target = nullptr;
                b.target_scale = nullptr;
                continue;
            }
            b.target_scale = sit->second;
            b.target_scale_nbytes = (size_t) std::max<int64_t>(1, b.source->ne[2]) * sizeof(float);
            runtime_bind_heartbeat.update(++runtime_bind_done, "binding runtime tensors");
        }
        runtime_bind_heartbeat.finish("runtime tensors bound");
    }
    const bool have_runtime_eval = run_stageb_eval && lctx != nullptr;

    nvfp4_selector_derived_metrics baseline_eval;
    nvfp4_selector_derived_metrics baseline_holdout_eval;
    bool has_holdout_eval = false;
    if (have_runtime_eval) {
        nvfp4_selector_progress_heartbeat runtime_cache_heartbeat(
            "selector stage-b runtime cache",
            (int64_t) all_bindings.size() + (int64_t) mxfp6_bindings.size());
        runtime_cache_heartbeat.detail("caching original runtime tensors", true);
        int64_t runtime_cache_done = 0;
        for (auto & b : all_bindings) {
            if (!quantize_binding_ensure_target_bytes(b)) {
                fprintf(stderr, "%s: selector failed to cache original runtime tensor bytes for %s\n", __func__, b.name.c_str());
                restore_all();
                return false;
            }
            if (b.target_scale != nullptr && !quantize_binding_ensure_scale_bytes(b)) {
                fprintf(stderr, "%s: selector failed to cache original runtime NVFP4 scale bytes for %s\n", __func__, b.name.c_str());
                restore_all();
                return false;
            }
            if (b.target_input_scale != nullptr && !quantize_binding_ensure_input_scale_bytes(b)) {
                fprintf(stderr, "%s: selector failed to cache original runtime NVFP4 input-scale bytes for %s\n", __func__, b.name.c_str());
                restore_all();
                return false;
            }
            runtime_cache_heartbeat.update(++runtime_cache_done, "caching original runtime tensors");
        }
        for (auto & b : mxfp6_bindings) {
            if (b.target == nullptr) {
                continue;
            }
            if (!quantize_binding_ensure_target_bytes(b) || !quantize_binding_ensure_scale_bytes(b)) {
                fprintf(stderr, "%s: selector failed to cache original runtime MXFP6_E2M3 bytes for %s\n", __func__, b.name.c_str());
                restore_all();
                return false;
            }
            runtime_cache_heartbeat.update(++runtime_cache_done, "caching original runtime tensors");
        }
        runtime_cache_heartbeat.finish("original runtime tensors cached");
        int64_t kld_metric_threads = nvfp4_selector_kld_threads_override();
        if (kld_metric_threads <= 0) {
            kld_metric_threads = std::max<unsigned int>(1, std::thread::hardware_concurrency());
        }
        fprintf(stderr,
            "selector stage-b runtime cache active checkpoint=%s tensors=%zu mx6_tensors=%zu ctx=%d n_seq=%d n_batch=%d n_ubatch=%d kld_threads=%" PRId64 "; candidates patch loaded tensor buffers without GGUF reload\n",
            checkpoint_model_path.c_str(),
            all_bindings.size(),
            mxfp6_bindings.size(),
            kld.n_ctx,
            selector_n_seq,
            params.n_batch,
            params.n_ubatch,
            kld_metric_threads);
        full_quant_eta_done = 1;
        update_full_quant_eta("selector-stage-b-baseline-kld", true);
        nvfp4_selector_kld_metrics baseline_km;
        if (!nvfp4_selector_eval_kld_subset(lctx, kld_budget, params.n_batch, baseline_km, true)) {
            restore_all();
            return false;
        }
        baseline_eval = nvfp4_selector_derive_metrics(baseline_km);

        if (holdout_budget && holdout_budget->n_chunk > 0) {
            nvfp4_selector_kld_metrics baseline_holdout_km;
            if (!nvfp4_selector_eval_kld_subset(lctx, *holdout_budget, params.n_batch, baseline_holdout_km, true)) {
                restore_all();
                return false;
            }
            baseline_holdout_eval = nvfp4_selector_derive_metrics(baseline_holdout_km);
            has_holdout_eval = baseline_holdout_eval.ok;
        }

        const std::string baseline_main_summary =
            nvfp4_selector_format_metrics("search", baseline_eval);
        fprintf(stderr,
            "selector stage-b baseline %s\n",
            baseline_main_summary.c_str());
        if (has_holdout_eval) {
            const std::string baseline_holdout_summary =
                nvfp4_selector_format_metrics("validation", baseline_holdout_eval);
            fprintf(stderr,
                "selector stage-b baseline %s\n",
                baseline_holdout_summary.c_str());
        }
        full_quant_eta_done = 2;
        update_full_quant_eta("selector-stage-b-policy-eval", true);

        for (size_t i = 0; i < eval_policy_indices.size(); ++i) {
            auto & policy = policies[eval_policy_indices[i]];
            stageb_policy_active = true;
            stageb_policy_start = std::chrono::steady_clock::now();
            full_quant_eta_done = std::min<int64_t>(full_quant_eta_total, 2 + stageb_policy_done);
            update_full_quant_eta("selector-stage-b-policy-running", true);
            fprintf(stderr,
                "selector stage-b start [%zu/%zu] policy=%s tensors=%zu search_chunks=%d validation_chunks=%d\n",
                i + 1,
                eval_policy_indices.size(),
                policy.name.c_str(),
                policy.has_survey ? all_binding_indices.size() : stress_binding_indices.size(),
                kld_budget.n_chunk,
                holdout_budget ? holdout_budget->n_chunk : 0);
            const std::vector<size_t> & eval_binding_indices = policy.has_survey ? all_binding_indices : stress_binding_indices;
            if (!covers_all_nvfp4_bindings(eval_binding_indices)) {
                restore_all();
            }
            bool policy_patch_ok = true;
            const auto patch_t0 = std::chrono::steady_clock::now();
            for (size_t idx : eval_binding_indices) {
                auto & b = all_bindings[idx];
                if (!quantize_binding_ensure_target_bytes(b)) {
                    restore_all();
                    return false;
                }
            }
            const auto patch_ensure_t1 = std::chrono::steady_clock::now();
            struct stageb_patch_result {
                bool ok = true;
                bool direct_applied = false;
                double tensor_sq = 0.0;
                double tensor_abs = 0.0;
                double tensor_max = 0.0;
                int64_t tensor_n = 0;
            };
            // Direct stage-B patches keep the selector fast by updating the
            // loaded CUDA tensor in place. The CUDA helper must resolve the
            // actual active storage (including split-buffer data_device) and
            // write the runtime layout, not assume tensor->data is what matmul
            // will read.
            const bool stageb_direct_patch = true;
            std::vector<stageb_patch_result> patch_results(eval_binding_indices.size());
            const int requested_stageb_patch_threads = (int) std::max<int64_t>(1, std::min<int64_t>(
                (int64_t) eval_binding_indices.size(),
                quantize_control_i64("LLAMA_NVFP4_SELECTOR_POLICY_THREADS",
                    std::max<int>(1, nthread))));
            const int stageb_autotune_threads = nvfp4_autotune_threads > 0 ?
                std::min<int>(4, std::max<int>(1, nvfp4_autotune_threads)) : 4;
            const int stageb_patch_threads = std::max<int>(1, std::min<int>(
                requested_stageb_patch_threads,
                std::max<int>(1, nthread / stageb_autotune_threads)));
            const int stageb_binding_nthread = std::max(1, nthread / std::max(1, stageb_patch_threads));
            if (stageb_patch_threads > 1) {
                fprintf(stderr,
                    "selector stage-b patch policy=%s threads=%d tensor_threads=%d autotune_threads=%d tensors=%zu\n",
                    policy.name.c_str(),
                    stageb_patch_threads,
                    stageb_binding_nthread,
                    stageb_autotune_threads,
                    eval_binding_indices.size());
            }
            const size_t no_failed_patch = std::numeric_limits<size_t>::max();
            auto clear_stageb_retry_caches = [&]() {
                nvfp4_clear_cuda_stream_cache();
                for (auto & b : all_bindings) {
                    if (b.device_samples) {
                        std::lock_guard<std::mutex> lock(b.device_samples->mutex);
                        b.device_samples->entries.clear();
                    }
                }
            };
            auto patch_policy_once = [&](int attempt, size_t & failed_pos) -> bool {
                failed_pos = no_failed_patch;
                for (auto & r : patch_results) {
                    r = stageb_patch_result{};
                }
                if (stageb_patch_threads <= 1 || eval_binding_indices.size() <= 1) {
                    for (size_t pos = 0; pos < eval_binding_indices.size(); ++pos) {
                        auto & b = all_bindings[eval_binding_indices[pos]];
                        auto & r = patch_results[pos];
                        if (stageb_direct_patch &&
                                nvfp4_selector_quantize_binding(b, policy.cfg, stageb_binding_nthread, b.working_target_bytes,
                                    r.tensor_sq, r.tensor_abs, r.tensor_max, r.tensor_n, 0, 0, true)) {
                            r.direct_applied = true;
                            continue;
                        }
                        if (!nvfp4_selector_quantize_binding(b, policy.cfg, stageb_binding_nthread, b.working_target_bytes,
                                r.tensor_sq, r.tensor_abs, r.tensor_max, r.tensor_n, 0, 0, false)) {
                            r.ok = false;
                            failed_pos = pos;
                            return false;
                        }
                    }
                    return true;
                } else {
                    std::atomic<size_t> next_patch { 0 };
                    std::atomic<size_t> failed_patch { no_failed_patch };
                    std::atomic<bool> patch_failed { false };
                    std::vector<std::thread> workers;
                    workers.reserve((size_t) stageb_patch_threads);
                    for (int ti = 0; ti < stageb_patch_threads; ++ti) {
                        workers.emplace_back([&]() {
                            while (true) {
                                if (patch_failed.load(std::memory_order_acquire)) {
                                    break;
                                }
                                const size_t pos = next_patch.fetch_add(1, std::memory_order_relaxed);
                                if (pos >= eval_binding_indices.size()) {
                                    break;
                                }
                                auto & b = all_bindings[eval_binding_indices[pos]];
                                auto & r = patch_results[pos];
                                if (stageb_direct_patch &&
                                        nvfp4_selector_quantize_binding(b, policy.cfg, stageb_binding_nthread, b.working_target_bytes,
                                            r.tensor_sq, r.tensor_abs, r.tensor_max, r.tensor_n, 0, 0, true)) {
                                    r.direct_applied = true;
                                    continue;
                                }
                                if (!nvfp4_selector_quantize_binding(b, policy.cfg, stageb_binding_nthread, b.working_target_bytes,
                                        r.tensor_sq, r.tensor_abs, r.tensor_max, r.tensor_n, 0, 0, false)) {
                                    r.ok = false;
                                    size_t expected = no_failed_patch;
                                    (void) failed_patch.compare_exchange_strong(
                                        expected, pos, std::memory_order_acq_rel, std::memory_order_acquire);
                                    patch_failed.store(true, std::memory_order_release);
                                }
                            }
                        });
                    }
                    for (auto & worker : workers) {
                        worker.join();
                    }
                    failed_pos = failed_patch.load(std::memory_order_acquire);
                    if (failed_pos != no_failed_patch) {
                        auto & b = all_bindings[eval_binding_indices[failed_pos]];
                        fprintf(stderr,
                            "%s: selector stage-b patch attempt=%d failed policy=%s tensor=%s; stopped remaining workers\n",
                            __func__, attempt, policy.name.c_str(), b.name.c_str());
                        return false;
                    }
                    return true;
                }
            };
            size_t failed_patch_pos = no_failed_patch;
            bool stageb_patch_quant_ok = patch_policy_once(1, failed_patch_pos);
            if (!stageb_patch_quant_ok) {
                const char * failed_name = failed_patch_pos != no_failed_patch
                    ? all_bindings[eval_binding_indices[failed_patch_pos]].name.c_str()
                    : "<unknown>";
                fprintf(stderr,
                    "%s: selector stage-b CUDA patch recovery policy=%s failed_tensor=%s; "
                    "clearing CUDA caches and retrying once with threads=%d\n",
                    __func__, policy.name.c_str(), failed_name, stageb_patch_threads);
                clear_stageb_retry_caches();
                restore_all();
                clear_stageb_retry_caches();
                failed_patch_pos = no_failed_patch;
                stageb_patch_quant_ok = patch_policy_once(2, failed_patch_pos);
                if (!stageb_patch_quant_ok) {
                    const char * retry_failed_name = failed_patch_pos != no_failed_patch
                        ? all_bindings[eval_binding_indices[failed_patch_pos]].name.c_str()
                        : "<unknown>";
                    fprintf(stderr,
                        "%s: selector stage-b CUDA patch recovery failed policy=%s tensor=%s after retry\n",
                        __func__, policy.name.c_str(), retry_failed_name);
                }
            }
            const auto patch_quant_t1 = std::chrono::steady_clock::now();
            size_t direct_patch_count = 0;
            size_t direct_fallback_count = 0;
            if (!stageb_patch_quant_ok) {
                policy_patch_ok = false;
            }
            for (size_t pos = 0; policy_patch_ok && pos < eval_binding_indices.size(); ++pos) {
                auto & b = all_bindings[eval_binding_indices[pos]];
                const auto & r = patch_results[pos];
                if (!r.ok) {
                    fprintf(stderr, "%s: selector rejected policy=%s during stage-b tensor=%s after quantize failure\n",
                        __func__, policy.name.c_str(), b.name.c_str());
                    policy_patch_ok = false;
                    break;
                }
                direct_patch_count += r.direct_applied ? 1 : 0;
                float header_weight_scale = 1.0f;
                float header_input_scale = 1.0f;
                if (!nvfp4_selector_prepare_nvfp4_runtime_scales(
                        b, nvfp4_input_scale_policy, &header_weight_scale, &header_input_scale)) {
                    restore_all();
                    return false;
                }
                if (b.target_scale != nullptr &&
                        !quantize_tensor_copy_in(b.target_scale, b.working_scale_bytes.data(), b.target_scale_nbytes)) {
                    restore_all();
                    return false;
                }
                if (b.target_input_scale != nullptr &&
                        !quantize_tensor_copy_in(
                            b.target_input_scale,
                            b.working_input_scale_bytes.data(),
                            b.target_input_scale_nbytes)) {
                    restore_all();
                    return false;
                }
                if (r.direct_applied &&
                        !ggml_cuda_nvfp4_tensor_set_header_scales(b.target, header_weight_scale, header_input_scale, nullptr)) {
                    restore_all();
                    return false;
                }
                if (!r.direct_applied && !quantize_tensor_copy_in(b.target, b.working_target_bytes.data(), b.target_nbytes)) {
                    restore_all();
                    return false;
                }
            }
            if (policy_patch_ok && direct_patch_count > 0) {
                bool direct_verified = false;
                bool direct_mismatch = false;
                std::string direct_mismatch_name;
                std::vector<uint8_t> verify_bytes;
                std::vector<uint8_t> reference_bytes;
                for (size_t pos = 0; pos < eval_binding_indices.size(); ++pos) {
                    auto & b = all_bindings[eval_binding_indices[pos]];
                    const auto & r = patch_results[pos];
                    if (!r.direct_applied) {
                        continue;
                    }
                    double ref_sq = 0.0;
                    double ref_abs = 0.0;
                    double ref_max = 0.0;
                    int64_t ref_n = 0;
                    if (!nvfp4_selector_quantize_binding(
                            b, policy.cfg, stageb_binding_nthread, reference_bytes,
                            ref_sq, ref_abs, ref_max, ref_n, 0, 0, false) || ref_n <= 0) {
                        direct_mismatch = true;
                        direct_mismatch_name = b.name;
                        direct_verified = true;
                        break;
                    }
                    verify_bytes.resize(b.target_nbytes);
                    if (!quantize_tensor_copy_out(b.target, verify_bytes.data(), b.target_nbytes) ||
                            reference_bytes.size() != b.target_nbytes ||
                            std::memcmp(verify_bytes.data(), reference_bytes.data(), b.target_nbytes) != 0) {
                        direct_mismatch = true;
                        direct_mismatch_name = b.name;
                    }
                    direct_verified = true;
                    break;
                }
                if (direct_verified && direct_mismatch) {
                    fprintf(stderr,
                        "%s: selector direct runtime patch readback mismatch policy=%s tensor=%s; falling back to backend tensor setter for this policy\n",
                        __func__, policy.name.c_str(), direct_mismatch_name.c_str());
                    for (size_t pos = 0; pos < eval_binding_indices.size(); ++pos) {
                        auto & b = all_bindings[eval_binding_indices[pos]];
                        const auto & r = patch_results[pos];
                        if (!r.direct_applied) {
                            continue;
                        }
                        double ref_sq = 0.0;
                        double ref_abs = 0.0;
                        double ref_max = 0.0;
                        int64_t ref_n = 0;
                        if (!nvfp4_selector_quantize_binding(
                                b, policy.cfg, stageb_binding_nthread, b.working_target_bytes,
                                ref_sq, ref_abs, ref_max, ref_n, 0, 0, false) || ref_n <= 0) {
                            restore_all();
                            return false;
                        }
                        if (!quantize_tensor_copy_in(b.target, b.working_target_bytes.data(), b.target_nbytes)) {
                            restore_all();
                            return false;
                        }
                        ++direct_fallback_count;
                    }
                }
            }
            const auto patch_apply_t1 = std::chrono::steady_clock::now();
            const double patch_ensure_s = std::chrono::duration<double>(patch_ensure_t1 - patch_t0).count();
            const double patch_quant_s = std::chrono::duration<double>(patch_quant_t1 - patch_ensure_t1).count();
            const double patch_apply_s = std::chrono::duration<double>(patch_apply_t1 - patch_quant_t1).count();
            fprintf(stderr,
                "selector stage-b patch complete policy=%s tensors=%zu direct=%zu direct_fallback=%zu threads=%d ensure=%.3fs encode=%.3fs apply=%.3fs total=%.3fs\n",
                policy.name.c_str(),
                eval_binding_indices.size(),
                direct_patch_count,
                direct_fallback_count,
                stageb_patch_threads,
                patch_ensure_s,
                patch_quant_s,
                patch_apply_s,
                patch_ensure_s + patch_quant_s + patch_apply_s);
            update_full_quant_eta("selector-stage-b-policy-kld", false);
            if (!policy_patch_ok) {
                restore_all();
                policy.proxy_rejected = true;
                policy.measured_pass = false;
                policy.measured_score = std::numeric_limits<double>::infinity();
                finish_stageb_policy_unit("selector-stage-b-policy-rejected", true);
                continue;
            }

            nvfp4_selector_kld_metrics km;
            if (!nvfp4_selector_eval_kld_subset(lctx, kld_budget, params.n_batch, km, true)) {
                restore_all();
                return false;
            }
            policy.measured = nvfp4_selector_derive_metrics(km);
            if (holdout_budget && has_holdout_eval) {
                nvfp4_selector_kld_metrics km_holdout;
                if (!nvfp4_selector_eval_kld_subset(lctx, *holdout_budget, params.n_batch, km_holdout, true)) {
                    restore_all();
                    return false;
                }
                policy.measured_holdout = nvfp4_selector_derive_metrics(km_holdout);
                policy.has_holdout = policy.measured_holdout.ok;
            }
            const nvfp4_selector_metric_rank policy_rank =
                nvfp4_selector_rank_policy_metrics(
                    policy.measured,
                    policy.has_holdout ? &policy.measured_holdout : nullptr,
                    baseline_eval,
                    has_holdout_eval ? &baseline_holdout_eval : nullptr,
                    rank_cfg);
            policy.measured_pass = policy_rank.pass;
            policy.measured_score = policy_rank.score;
            const std::string policy_main_summary =
                nvfp4_selector_format_metrics("search", policy.measured);
            fprintf(stderr,
                "selector stage-b policy=%s measured_score=%.6f pass=%s %s\n",
                policy.name.c_str(),
                policy.measured_score,
                policy.measured_pass ? "yes" : "no",
                policy_main_summary.c_str());
            if (policy.has_holdout) {
                const std::string policy_holdout_summary =
                    nvfp4_selector_format_metrics("validation", policy.measured_holdout);
                fprintf(stderr,
                    "selector stage-b policy=%s %s\n",
                    policy.name.c_str(),
                    policy_holdout_summary.c_str());
            }
            finish_stageb_policy_unit("selector-stage-b-policy-complete", true);
            if (note_skip_remaining("stage-b")) {
                break;
            }
        }
    } else {
        for (auto & policy : policies) {
            policy.measured_pass = !policy.proxy_rejected;
            policy.measured_score = policy.proxy_score;
        }
    }

    std::optional<nvfp4_selector_policy> seed_keep_policy;
    if (have_runtime_eval && baseline_eval.ok) {
        nvfp4_selector_policy seed;
        seed.name = "seed_keep";
        seed.cfg = baseline_it != policies.end() ? baseline_it->cfg : seed.cfg;
        seed.proxy_score = 0.0;
        seed.proxy_rmse = 0.0;
        seed.proxy_abs_mean = 0.0;
        seed.proxy_max_abs = 0.0;
        seed.measured = baseline_eval;
        seed.measured_holdout = baseline_holdout_eval;
        seed.has_holdout = has_holdout_eval;
        const nvfp4_selector_metric_rank seed_rank =
            nvfp4_selector_rank_policy_metrics(
                seed.measured,
                seed.has_holdout ? &seed.measured_holdout : nullptr,
                baseline_eval,
                seed.has_holdout ? &baseline_holdout_eval : nullptr,
                rank_cfg);
        seed.measured_pass = seed_rank.pass;
        seed.measured_score = seed_rank.score;
        seed_keep_policy = seed;
        const std::string seed_summary =
            nvfp4_selector_format_metrics("search", seed.measured);
        fprintf(stderr,
            "selector stage-b seed_keep measured_score=%.6f %s\n",
            seed.measured_score,
            seed_summary.c_str());
    }

    auto best_it = policies.begin();
    if (have_measured_eval) {
        if (seed_keep_policy.has_value()) {
            policies.push_back(*seed_keep_policy);
        }

        std::vector<size_t> eval_candidates;
        eval_candidates.reserve(eval_policy_indices.size() + (seed_keep_policy.has_value() ? 1 : 0));
        for (size_t idx : eval_policy_indices) {
            eval_candidates.push_back(idx);
        }
        if (seed_keep_policy.has_value()) {
            eval_candidates.push_back(policies.size() - 1);
        }

        std::vector<size_t> gated_candidates;
        gated_candidates.reserve(eval_candidates.size());
        for (size_t idx : eval_candidates) {
            const auto & policy = policies[idx];
            if (!std::isfinite(policy.measured_score) || !policy.measured.ok) {
                continue;
            }
            if (policy.measured_pass) {
                gated_candidates.push_back(idx);
            }
        }

        const std::vector<size_t> & best_input = gated_candidates.empty() ? eval_candidates : gated_candidates;
        std::vector<size_t> best_set = nvfp4_selector_best_set(policies, best_input, rank_cfg);
        if (best_set.empty() && !gated_candidates.empty()) {
            best_set = nvfp4_selector_best_set(policies, eval_candidates, rank_cfg);
        }

        const nvfp4_selector_best_set_stats best_stats =
            nvfp4_selector_best_set_stats_for(policies, best_set, rank_cfg);
        bool have_best = false;
        size_t best_idx = 0;
        double best_utility = std::numeric_limits<double>::infinity();
        for (size_t idx : best_set) {
            const double utility = nvfp4_selector_best_set_utility(
                policies[idx],
                best_stats,
                rank_cfg);
            bool better = !have_best || utility < best_utility;
            if (!better && have_best && std::fabs(utility - best_utility) <= 1e-9) {
                better = nvfp4_selector_compare_policy_measured(
                        policies[idx],
                        policies[best_idx],
                        baseline_eval,
                        has_holdout_eval ? &baseline_holdout_eval : nullptr,
                        rank_cfg) < 0;
            }
            if (better) {
                best_idx = idx;
                best_utility = utility;
                have_best = true;
            }
        }
        if (have_best) {
            best_it = policies.begin() + (std::ptrdiff_t) best_idx;
            fprintf(stderr,
                "%s: selector best candidates=%zu gated=%zu non_dominated=%zu chosen=%s measured_score=%.6f utility=%.6f\n",
                __func__,
                eval_candidates.size(),
                gated_candidates.size(),
                best_set.size(),
                best_it->name.c_str(),
                best_it->measured_score,
                best_utility);
            for (size_t rank = 0; rank < best_set.size(); ++rank) {
                const auto & p = policies[best_set[rank]];
                const auto obj = nvfp4_selector_best_objectives_for(p, rank_cfg.holdout_weight);
                const double utility = nvfp4_selector_best_set_utility(p, best_stats, rank_cfg);
                fprintf(stderr,
                    "%s: selector Best[%zu] policy=%s pass=%s score=%.6f utility=%.6f ln=%.6f kld=%.6f p95=%.6f p99=%.6f p999=%.6f tail99=%.6f max=%.6f rms=%.6f top=%.4f flip_w=%.6f top_p=%.6f entropy=%.6f\n",
                    __func__,
                    rank + 1,
                    p.name.c_str(),
                    p.measured_pass ? "yes" : "no",
                    p.measured_score,
                    utility,
                    obj.ln_ratio,
                    obj.mean_kld,
                    obj.kld_p95,
                    obj.kld_p99,
                    obj.kld_p999,
                    obj.kld_tail_mean,
                    obj.max_kld,
                    obj.rms_dp,
                    obj.same_top,
                    obj.top_flip_weight,
                    obj.top_prob_rmse,
                    obj.entropy_rmse);
            }
        } else {
            best_it = policies.begin();
        }
    } else {
        best_it = std::min_element(policies.begin(), policies.end(), [](const auto & a, const auto & b) {
            if (a.proxy_score != b.proxy_score) return a.proxy_score < b.proxy_score;
            return a.name < b.name;
        });
    }

    const std::string sensitivity_report_file = quantize_control_string("LLAMA_NVFP4_SELECTOR_SENSITIVITY_REPORT_FILE");
    if (!skip_remaining_tuning &&
            have_runtime_eval &&
            !sensitivity_report_file.empty() &&
            best_it != policies.end()) {
        const std::string tensor_filter =
            quantize_control_string("LLAMA_NVFP4_SELECTOR_SENSITIVITY_TENSOR");
        const int32_t layer_filter =
            (int32_t) quantize_control_i64("LLAMA_NVFP4_SELECTOR_SENSITIVITY_LAYER", -999999);
        const int sensitivity_top = (int) std::max<int64_t>(
            0,
            quantize_control_i64("LLAMA_NVFP4_SELECTOR_SENSITIVITY_TOP",
                (!tensor_filter.empty() || layer_filter != -999999) ? 0 : 16));
        const int64_t sensitivity_sample_blocks = std::max<int64_t>(
            0,
            quantize_control_i64("LLAMA_NVFP4_SELECTOR_SENSITIVITY_SAMPLE_BLOCKS", 8192));

        std::ofstream sensitivity_report(sensitivity_report_file, std::ios::trunc);
        if (!sensitivity_report) {
            fprintf(stderr, "%s: failed to open selector sensitivity report %s\n", __func__, sensitivity_report_file.c_str());
        } else {
            fprintf(stderr,
                "%s: selector writing exact one-tensor sensitivity report %s top=%d layer_filter=%d tensor_filter=%s\n",
                __func__,
                sensitivity_report_file.c_str(),
                sensitivity_top,
                layer_filter,
                tensor_filter.empty() ? "<none>" : tensor_filter.c_str());

            std::vector<std::vector<uint8_t>> best_target_bytes(all_bindings.size());
            bool best_patch_ok = true;
            restore_all();
            for (size_t i = 0; i < all_bindings.size(); ++i) {
                auto & b = all_bindings[i];
                double sq = 0.0;
                double ab = 0.0;
                double mx = 0.0;
                int64_t n = 0;
                if (!nvfp4_selector_quantize_binding(b, best_it->cfg, nthread, best_target_bytes[i], sq, ab, mx, n, 0) ||
                        !quantize_tensor_copy_in(b.target, best_target_bytes[i].data(), b.target_nbytes)) {
                    fprintf(stderr, "%s: selector sensitivity failed to patch best policy tensor=%s\n", __func__, b.name.c_str());
                    best_patch_ok = false;
                    break;
                }
            }

            nvfp4_selector_derived_metrics sensitivity_base = best_it->measured;
            if (best_patch_ok) {
                nvfp4_selector_kld_metrics base_km;
                if (!nvfp4_selector_eval_kld_subset(lctx, kld_budget, params.n_batch, base_km, true)) {
                    restore_all();
                    return false;
                }
                sensitivity_base = nvfp4_selector_derive_metrics(base_km);
            }

            sensitivity_report
                << "{\"schema\":\"blackwell-selector-sensitivity-v1\", \"event\":\"baseline\", \"policy\":\""
                << nvfp4_selector_json_escape(best_it->name)
                << "\", \"chunks\": " << kld_budget.n_chunk
                << ", \"ctx\": " << kld_budget.n_ctx
                << ", \"score_tokens\": " << kld_budget.n_score;
            nvfp4_selector_write_metric_json_fields(sensitivity_report, sensitivity_base, "");
            sensitivity_report << "}\n";

            std::vector<size_t> sensitivity_indices;
            sensitivity_indices.reserve(all_bindings.size());
            for (size_t i = 0; i < all_bindings.size(); ++i) {
                const auto & b = all_bindings[i];
                if (!tensor_filter.empty() && b.name.find(tensor_filter) == std::string::npos) {
                    continue;
                }
                if (layer_filter != -999999 && b.layer != layer_filter) {
                    continue;
                }
                sensitivity_indices.push_back(i);
            }
            std::sort(sensitivity_indices.begin(), sensitivity_indices.end(), [&](size_t ai, size_t bi) {
                const auto & a = all_bindings[ai];
                const auto & b = all_bindings[bi];
                const double as = nvfp4_selector_class_hotness(a.cls) * (double) std::max<size_t>(1, a.source_nbytes);
                const double bs = nvfp4_selector_class_hotness(b.cls) * (double) std::max<size_t>(1, b.source_nbytes);
                if (as != bs) return as > bs;
                if (a.layer != b.layer) return a.layer < b.layer;
                return a.name < b.name;
            });
            if (sensitivity_top > 0 && sensitivity_indices.size() > (size_t) sensitivity_top) {
                sensitivity_indices.resize((size_t) sensitivity_top);
            }

            int sensitivity_written = 0;
            if (best_patch_ok) {
                struct sensitivity_alt {
                    size_t index = 0;
                    nvfp4_cuda_runtime_cfg cfg = {};
                    std::string policy_name;
                    int64_t sample_blocks = 0;
                    double base_proxy_score = std::numeric_limits<double>::infinity();
                    double alt_proxy_score = std::numeric_limits<double>::infinity();
                };

                auto find_sensitivity_alt = [&](size_t idx, sensitivity_alt & out_alt) -> bool {
                    auto & b = all_bindings[idx];
                    out_alt.index = idx;
                    if (!nvfp4_selector_find_tensor_rescue_cfg(
                            b,
                            best_it->cfg,
                            nthread,
                            out_alt.cfg,
                            out_alt.policy_name,
                            out_alt.sample_blocks,
                            out_alt.base_proxy_score,
                            out_alt.alt_proxy_score) ||
                            nvfp4_selector_cfg_equal(out_alt.cfg, best_it->cfg)) {
                        return false;
                    }
                    return true;
                };

                if (layer_filter != -999999 && !sensitivity_indices.empty()) {
                    std::vector<sensitivity_alt> layer_alts;
                    layer_alts.reserve(sensitivity_indices.size());
                    for (const size_t idx : sensitivity_indices) {
                        sensitivity_alt alt;
                        if (find_sensitivity_alt(idx, alt)) {
                            layer_alts.push_back(std::move(alt));
                        }
                    }

                    if (!layer_alts.empty()) {
                        bool layer_patch_ok = true;
                        std::vector<std::string> tensor_names;
                        tensor_names.reserve(layer_alts.size());
                        double proxy_gain_sum = 0.0;
                        int64_t alt_sample_blocks_max = 0;

                        for (const auto & alt : layer_alts) {
                            auto & b = all_bindings[alt.index];
                            std::vector<uint8_t> alt_bytes;
                            double sq = 0.0;
                            double ab = 0.0;
                            double mx = 0.0;
                            int64_t n = 0;
                            if (!quantize_tensor_copy_in(b.target, best_target_bytes[alt.index].data(), b.target_nbytes) ||
                                    !nvfp4_selector_quantize_binding(
                                        b,
                                        alt.cfg,
                                        nthread,
                                        alt_bytes,
                                        sq,
                                        ab,
                                        mx,
                                        n,
                                        sensitivity_sample_blocks) ||
                                    !quantize_tensor_copy_in(b.target, alt_bytes.data(), b.target_nbytes)) {
                                fprintf(stderr, "%s: selector layer sensitivity failed tensor=%s alt=%s\n",
                                    __func__, b.name.c_str(), alt.policy_name.c_str());
                                layer_patch_ok = false;
                                break;
                            }
                            tensor_names.push_back(b.name + ":" + alt.policy_name);
                            if (std::isfinite(alt.base_proxy_score) && std::isfinite(alt.alt_proxy_score)) {
                                proxy_gain_sum += alt.base_proxy_score - alt.alt_proxy_score;
                            }
                            alt_sample_blocks_max = std::max<int64_t>(alt_sample_blocks_max, alt.sample_blocks);
                        }

                        if (layer_patch_ok) {
                            nvfp4_selector_kld_metrics km;
                            if (!nvfp4_selector_eval_kld_subset(lctx, kld_budget, params.n_batch, km, true)) {
                                restore_all();
                                return false;
                            }
                            const nvfp4_selector_derived_metrics dm = nvfp4_selector_derive_metrics(km);
                            const double utility = dm.ok
                                ? nvfp4_selector_measured_score(dm, nullptr, rank_cfg)
                                : std::numeric_limits<double>::infinity();
                            const double base_utility = nvfp4_selector_measured_score(sensitivity_base, nullptr, rank_cfg);
                            const double delta_utility = utility - base_utility;

                            sensitivity_report
                                << "{\"schema\":\"blackwell-selector-sensitivity-v1\", \"event\":\"layer_delta\""
                                << ", \"layer\": " << layer_filter
                                << ", \"tensor_count\": " << layer_alts.size()
                                << ", \"base_policy\":\"" << nvfp4_selector_json_escape(best_it->name) << "\""
                                << ", \"alt_policy\":\"rescue_adaptive_layer\""
                                << ", \"alt_sample_blocks\": " << alt_sample_blocks_max
                                << ", \"measured_mode\":\"exact_layer\""
                                << ", \"tensors\":\"";
                            for (size_t ti = 0; ti < tensor_names.size(); ++ti) {
                                if (ti > 0) {
                                    sensitivity_report << ",";
                                }
                                sensitivity_report << nvfp4_selector_json_escape(tensor_names[ti]);
                            }
                            sensitivity_report
                                << "\"";
                            nvfp4_selector_write_json_number_field(sensitivity_report, "proxy_gain_sum", proxy_gain_sum);
                            nvfp4_selector_write_json_number_field(sensitivity_report, "utility", utility);
                            nvfp4_selector_write_json_number_field(sensitivity_report, "delta_utility", delta_utility);
                            nvfp4_selector_write_metric_json_fields(sensitivity_report, dm, "");
                            nvfp4_selector_write_delta_metric_json_fields(sensitivity_report, dm, sensitivity_base);
                            sensitivity_report << "}\n";

                            fprintf(stderr,
                                "%s: selector layer sensitivity layer=%d tensors=%zu delta_kld=%.6f delta_p99=%.6f delta_top_flip_w=%.6f delta_ln=%.6f\n",
                                __func__,
                                layer_filter,
                                layer_alts.size(),
                                dm.mean_kld - sensitivity_base.mean_kld,
                                dm.kld_p99 - sensitivity_base.kld_p99,
                                dm.top_flip_weight - sensitivity_base.top_flip_weight,
                                dm.ln_ratio - sensitivity_base.ln_ratio);
                        }

                        for (const auto & alt : layer_alts) {
                            auto & b = all_bindings[alt.index];
                            if (!quantize_tensor_copy_in(b.target, best_target_bytes[alt.index].data(), b.target_nbytes)) {
                                restore_all();
                                return false;
                            }
                        }
                    }
                }

                for (size_t rank = 0; rank < sensitivity_indices.size(); ++rank) {
                    const size_t idx = sensitivity_indices[rank];
                    auto & b = all_bindings[idx];
                    sensitivity_alt alt;
                    if (!find_sensitivity_alt(idx, alt)) {
                        continue;
                    }

                    std::vector<uint8_t> alt_bytes;
                    double sq = 0.0;
                    double ab = 0.0;
                    double mx = 0.0;
                    int64_t n = 0;
                    if (!quantize_tensor_copy_in(b.target, best_target_bytes[idx].data(), b.target_nbytes) ||
                            !nvfp4_selector_quantize_binding(
                                b,
                                alt.cfg,
                                nthread,
                                alt_bytes,
                                sq,
                                ab,
                                mx,
                                n,
                                sensitivity_sample_blocks) ||
                            !quantize_tensor_copy_in(b.target, alt_bytes.data(), b.target_nbytes)) {
                        fprintf(stderr, "%s: selector sensitivity failed tensor=%s alt=%s\n",
                            __func__, b.name.c_str(), alt.policy_name.c_str());
                        continue;
                    }

                    nvfp4_selector_kld_metrics km;
                    if (!nvfp4_selector_eval_kld_subset(lctx, kld_budget, params.n_batch, km, true)) {
                        restore_all();
                        return false;
                    }
                    const nvfp4_selector_derived_metrics dm = nvfp4_selector_derive_metrics(km);
                    const double utility = dm.ok
                        ? nvfp4_selector_measured_score(dm, nullptr, rank_cfg)
                        : std::numeric_limits<double>::infinity();
                    const double base_utility = nvfp4_selector_measured_score(sensitivity_base, nullptr, rank_cfg);
                    const double delta_utility = utility - base_utility;

                    sensitivity_report
                        << "{\"schema\":\"blackwell-selector-sensitivity-v1\", \"event\":\"tensor_delta\", \"rank\": "
                        << (rank + 1)
                        << ", \"tensor\":\"" << nvfp4_selector_json_escape(b.name) << "\""
                        << ", \"class\":\"" << nvfp4_selector_tensor_class_name(b.cls) << "\""
                        << ", \"layer\": " << b.layer
                        << ", \"bucket\": " << b.bucket
                        << ", \"base_policy\":\"" << nvfp4_selector_json_escape(best_it->name) << "\""
                        << ", \"alt_policy\":\"" << nvfp4_selector_json_escape(alt.policy_name) << "\""
                        << ", \"alt_sample_blocks\": " << alt.sample_blocks
                        << ", \"measured_mode\":\"exact_single_tensor\"";
                    nvfp4_selector_write_json_number_field(sensitivity_report, "base_proxy_score", alt.base_proxy_score);
                    nvfp4_selector_write_json_number_field(sensitivity_report, "alt_proxy_score", alt.alt_proxy_score);
                    nvfp4_selector_write_json_number_field(sensitivity_report, "proxy_gain", alt.base_proxy_score - alt.alt_proxy_score);
                    nvfp4_selector_write_json_number_field(sensitivity_report, "utility", utility);
                    nvfp4_selector_write_json_number_field(sensitivity_report, "delta_utility", delta_utility);
                    nvfp4_selector_write_metric_json_fields(sensitivity_report, dm, "");
                    nvfp4_selector_write_delta_metric_json_fields(sensitivity_report, dm, sensitivity_base);
                    sensitivity_report << "}\n";
                    ++sensitivity_written;

                    fprintf(stderr,
                        "%s: selector sensitivity tensor=%s alt=%s delta_kld=%.6f delta_p99=%.6f delta_top_flip_w=%.6f delta_ln=%.6f\n",
                        __func__,
                        b.name.c_str(),
                        alt.policy_name.c_str(),
                        dm.mean_kld - sensitivity_base.mean_kld,
                        dm.kld_p99 - sensitivity_base.kld_p99,
                        dm.top_flip_weight - sensitivity_base.top_flip_weight,
                        dm.ln_ratio - sensitivity_base.ln_ratio);

                    if (!quantize_tensor_copy_in(b.target, best_target_bytes[idx].data(), b.target_nbytes)) {
                        restore_all();
                        return false;
                    }
                }
            }
            fprintf(stderr,
                "%s: wrote selector sensitivity report %s (%d tensor deltas)\n",
                __func__,
                sensitivity_report_file.c_str(),
                sensitivity_written);
            restore_all();
        }
    }

    out_cfg = best_it->cfg;
    out_name = best_it->name.empty() ? "unnamed" : best_it->name;
    if (out_kept_seed != nullptr) {
        *out_kept_seed = out_name == "seed_keep";
    }

    const int rescue_apply_top = (!skip_remaining_tuning && out_tensor_overrides) ? (int) std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_RESCUE_TOP", 0)) : 0;
    const int rescue_report_top_default = rescue_apply_top > 0 ? 12 : 0;
    const int rescue_report_top = (int) std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_RESCUE_REPORT_TOP", rescue_report_top_default));
    const size_t rescue_budget_bytes = (size_t) std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_RESCUE_BUDGET_MB", 0)) * 1024ull * 1024ull;
    const size_t rescue_bf16_budget_bytes = (size_t) std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_RESCUE_BF16_BUDGET_MB", 0)) * 1024ull * 1024ull;
    const int rescue_class_limit = (int) std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_RESCUE_CLASS_LIMIT", 0));
    const int rescue_nvfp4_top = (int) std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_RESCUE_NVFP4_TOP", std::max<int>(32, rescue_apply_top * 8)));
    const int64_t rescue_sens_sample_blocks = std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_RESCUE_SENS_SAMPLE_BLOCKS", 8192));
    const double rescue_nvfp4_min_gain = SELECTOR_RESCUE_NVFP4_MIN_GAIN;
    const double rescue_nvfp4_min_rel_gain = SELECTOR_RESCUE_NVFP4_MIN_REL_GAIN;
    const double rescue_nvfp4_prefer_gain = SELECTOR_RESCUE_NVFP4_PREFER_GAIN;
    const double rescue_nvfp4_prefer_rel_gain = SELECTOR_RESCUE_NVFP4_PREFER_REL_GAIN;
    const ggml_type rescue_q8_type = quantize_control_type("LLAMA_NVFP4_SELECTOR_RESCUE_Q8_TYPE", GGML_TYPE_Q8_0);
    const double rescue_speed_weight = SELECTOR_RESCUE_SPEED_WEIGHT;
    const double rescue_q8_speed = SELECTOR_RESCUE_Q8_SPEED_PENALTY;
    const double rescue_bf16_speed = SELECTOR_RESCUE_BF16_SPEED_PENALTY;
    const double rescue_q8_gain = SELECTOR_RESCUE_Q8_GAIN;
    const double rescue_bf16_gain = SELECTOR_RESCUE_BF16_GAIN;
    const double rescue_bf16_margin = SELECTOR_RESCUE_BF16_MARGIN;
    const std::string rescue_report_file = quantize_control_string("LLAMA_NVFP4_SELECTOR_RESCUE_REPORT_FILE");
    const std::string rescue_tensor_types_file = quantize_control_string("LLAMA_NVFP4_SELECTOR_RESCUE_TENSOR_TYPES_FILE");
    if (out_tensor_overrides) {
        out_tensor_overrides->clear();
    }
    if (rescue_report_top > 0 || rescue_apply_top > 0) {
        std::vector<nvfp4_selector_tensor_sensitivity> sens;
        sens.reserve(all_bindings.size());
        for (size_t bind_i = 0; bind_i < all_bindings.size(); ++bind_i) {
            auto & b = all_bindings[bind_i];
            fprintf(stderr,
                "selector sensitivity [%zu/%zu] tensor=%s cls=%s bucket=%d layer=%d\n",
                bind_i + 1,
                all_bindings.size(),
                b.name.c_str(),
                nvfp4_selector_tensor_class_name(b.cls),
                b.bucket,
                b.layer);
            double tensor_sq = 0.0;
            double tensor_abs = 0.0;
            double tensor_max = 0.0;
            int64_t tensor_n = 0;
            std::vector<uint8_t> sens_bytes;
            if (!nvfp4_selector_quantize_binding(
                    b, best_it->cfg, nthread, sens_bytes,
                    tensor_sq, tensor_abs, tensor_max, tensor_n,
                    rescue_sens_sample_blocks) || tensor_n <= 0) {
                continue;
            }
            nvfp4_selector_tensor_sensitivity s;
            s.name = b.name;
            s.cls = b.cls;
            s.bucket = b.bucket;
            s.layer = b.layer;
            s.binding_index = &b - all_bindings.data();
            s.target_nbytes = b.target_nbytes;
            s.q8_type = rescue_q8_type;
            s.q8_nbytes = quantize_tensor_nbytes_as_type(b.source, rescue_q8_type);
            s.bf16_nbytes = quantize_tensor_nbytes_as_type(b.source, GGML_TYPE_BF16);
            const nvfp4_selector_proxy_metrics proxy_metrics =
                nvfp4_selector_proxy_score(tensor_sq, tensor_abs, tensor_max, tensor_n);
            if (!proxy_metrics.ok) {
                continue;
            }
            s.proxy_rmse = proxy_metrics.rmse;
            s.proxy_abs_mean = proxy_metrics.abs_mean;
            s.proxy_max_abs = proxy_metrics.max_abs;
            s.proxy_score = proxy_metrics.score;
            s.q8_delta_bytes = s.q8_nbytes > s.target_nbytes ? s.q8_nbytes - s.target_nbytes : 0;
            s.bf16_delta_bytes = s.bf16_nbytes > s.target_nbytes ? s.bf16_nbytes - s.target_nbytes : 0;
            const double hotness = nvfp4_selector_class_hotness(b.cls);
            s.q8_speed_penalty = hotness * rescue_q8_speed;
            s.bf16_speed_penalty = hotness * rescue_bf16_speed;
            s.q8_roi = nvfp4_selector_rescue_roi(s.proxy_score, s.q8_delta_bytes, s.q8_speed_penalty, rescue_q8_gain, rescue_speed_weight);
            s.bf16_roi = nvfp4_selector_rescue_roi(s.proxy_score, s.bf16_delta_bytes, s.bf16_speed_penalty, rescue_bf16_gain, rescue_speed_weight);
            sens.push_back(std::move(s));
        }

        std::sort(sens.begin(), sens.end(), [](const auto & a, const auto & b) {
            if (a.proxy_score != b.proxy_score) return a.proxy_score > b.proxy_score;
            if (a.proxy_max_abs != b.proxy_max_abs) return a.proxy_max_abs > b.proxy_max_abs;
            if (a.target_nbytes != b.target_nbytes) return a.target_nbytes > b.target_nbytes;
            return a.name < b.name;
        });

        const int rescue_scan_n = std::min<int>(rescue_nvfp4_top, (int) sens.size());
        for (int i = 0; i < rescue_scan_n; ++i) {
            auto & s = sens[(size_t) i];
            auto & b = all_bindings[s.binding_index];
            fprintf(stderr,
                "selector rescue scan [%d/%d] tensor=%s cls=%s bucket=%d layer=%d base_score=%.6f\n",
                i + 1,
                rescue_scan_n,
                s.name.c_str(),
                nvfp4_selector_tensor_class_name(s.cls),
                s.bucket,
                s.layer,
                s.proxy_score);
            nvfp4_cuda_runtime_cfg alt_cfg = best_it->cfg;
            std::string alt_policy_name;
            int64_t alt_sample_blocks = 0;
            double base_score = std::numeric_limits<double>::infinity();
            double best_score = std::numeric_limits<double>::infinity();
            if (!nvfp4_selector_find_tensor_rescue_cfg(
                    b,
                    best_it->cfg,
                    nthread,
                    alt_cfg,
                    alt_policy_name,
                    alt_sample_blocks,
                    base_score,
                    best_score)) {
                continue;
            }

            s.alt_nvfp4_proxy_score = best_score;
            s.alt_nvfp4_gain = std::max(0.0, base_score - best_score);
            s.alt_nvfp4_gain_rel = s.alt_nvfp4_gain / std::max(1e-9, base_score);
            if ((s.alt_nvfp4_gain >= rescue_nvfp4_min_gain || s.alt_nvfp4_gain_rel >= rescue_nvfp4_min_rel_gain) &&
                !nvfp4_selector_cfg_equal(alt_cfg, best_it->cfg)) {
                s.has_alt_nvfp4_cfg = true;
                s.alt_nvfp4_cfg = alt_cfg;
                s.alt_nvfp4_sample_blocks = alt_sample_blocks;
                s.alt_nvfp4_policy_name = alt_policy_name;
                fprintf(stderr,
                    "selector rescue alt tensor=%s policy=%s gain=%.6f rel=%.4f sample_blocks=%" PRId64 "\n",
                    s.name.c_str(),
                    s.alt_nvfp4_policy_name.c_str(),
                    s.alt_nvfp4_gain,
                    s.alt_nvfp4_gain_rel,
                    s.alt_nvfp4_sample_blocks);
            }
        }

        for (auto & s : sens) {
            s.suggested_type = s.q8_type;
            s.suggested_priority = std::max(s.q8_roi, s.bf16_roi);
            if (nvfp4_selector_bf16_allowed(s.cls) && s.bf16_roi > s.q8_roi * rescue_bf16_margin) {
                s.suggested_type = GGML_TYPE_BF16;
                s.suggested_priority = s.bf16_roi;
            }
            if (s.has_alt_nvfp4_cfg) {
                const bool expert_nvfp4_first = nvfp4_selector_expert_class(s.cls);
                const bool prefer_alt_nvfp4 =
                    expert_nvfp4_first ||
                    s.alt_nvfp4_gain >= rescue_nvfp4_prefer_gain ||
                    s.alt_nvfp4_gain_rel >= rescue_nvfp4_prefer_rel_gain;
                if (prefer_alt_nvfp4 || s.suggested_priority <= 0.0) {
                    const double nvfp4_priority =
                        s.alt_nvfp4_gain +
                        (expert_nvfp4_first ? 0.60 : 0.25) * s.proxy_score;
                    s.suggested_type = GGML_TYPE_NVFP4;
                    s.suggested_priority = std::max(nvfp4_priority, expert_nvfp4_first ? s.q8_roi * 1.05 : 0.0);
                }
            }
        }

        std::sort(sens.begin(), sens.end(), [](const auto & a, const auto & b) {
            if (a.suggested_priority != b.suggested_priority) return a.suggested_priority > b.suggested_priority;
            if (a.proxy_score != b.proxy_score) return a.proxy_score > b.proxy_score;
            const size_t a_delta = a.suggested_type == GGML_TYPE_BF16 ? a.bf16_delta_bytes : a.q8_delta_bytes;
            const size_t b_delta = b.suggested_type == GGML_TYPE_BF16 ? b.bf16_delta_bytes : b.q8_delta_bytes;
            if (a_delta != b_delta) return a_delta < b_delta;
            return a.name < b.name;
        });

        if (!rescue_report_file.empty()) {
            std::ofstream report(rescue_report_file, std::ios::trunc);
            if (!report) {
                fprintf(stderr, "%s: failed to open selector rescue report file %s\n", __func__, rescue_report_file.c_str());
            } else {
                report << "rank,name,class,bucket,layer,proxy_score,proxy_rmse,proxy_abs_mean,proxy_max_abs,target_nbytes,alt_nvfp4_policy,alt_nvfp4_score,alt_nvfp4_gain,alt_nvfp4_gain_rel,alt_nvfp4_sample_blocks,alt_nvfp4_choose46,alt_nvfp4_refit,alt_nvfp4_compand,alt_nvfp4_cap6,alt_nvfp4_cap4,q8_type,q8_nbytes,q8_delta_bytes,q8_roi,q8_speed_penalty,bf16_nbytes,bf16_delta_bytes,bf16_roi,bf16_speed_penalty,suggested_type,suggested_priority\n";
                for (size_t i = 0; i < sens.size(); ++i) {
                    const auto & s = sens[i];
                    report
                        << (i + 1) << ','
                        << '"' << s.name << '"' << ','
                        << nvfp4_selector_tensor_class_name(s.cls) << ','
                        << s.bucket << ','
                        << s.layer << ','
                        << s.proxy_score << ','
                        << s.proxy_rmse << ','
                        << s.proxy_abs_mean << ','
                        << s.proxy_max_abs << ','
                        << s.target_nbytes << ','
                        << '"' << s.alt_nvfp4_policy_name << '"' << ','
                        << s.alt_nvfp4_proxy_score << ','
                        << s.alt_nvfp4_gain << ','
                        << s.alt_nvfp4_gain_rel << ','
                        << s.alt_nvfp4_sample_blocks << ','
                        << nvfp4_choose46_mode_name(s.alt_nvfp4_cfg.choose46_mode) << ','
                        << s.alt_nvfp4_cfg.refit_iters << ','
                        << s.alt_nvfp4_cfg.use_compand_sat << ','
                        << s.alt_nvfp4_cfg.cap_m6 << ','
                        << s.alt_nvfp4_cfg.cap_m4 << ','
                        << ggml_type_name(s.q8_type) << ','
                        << s.q8_nbytes << ','
                        << s.q8_delta_bytes << ','
                        << s.q8_roi << ','
                        << s.q8_speed_penalty << ','
                        << s.bf16_nbytes << ','
                        << s.bf16_delta_bytes << ','
                        << s.bf16_roi << ','
                        << s.bf16_speed_penalty << ','
                        << format_rescue_suggested_type(s) << ','
                        << s.suggested_priority
                        << '\n';
                }
                fprintf(stderr, "%s: wrote selector rescue report %s (%zu tensors)\n", __func__, rescue_report_file.c_str(), sens.size());
            }
        }

        for (int i = 0; i < rescue_report_top && i < (int) sens.size(); ++i) {
            const auto & s = sens[(size_t) i];
            fprintf(stderr,
                "selector rescue rank=%d tensor=%s cls=%s bucket=%d layer=%d score=%.6f rmse=%.6f abs=%.6f max=%.6f alt_nvfp4_gain=%.6f alt_nvfp4_rel=%.4f q8_roi=%.6f bf16_roi=%.6f suggest=%s\n",
                i + 1, s.name.c_str(), nvfp4_selector_tensor_class_name(s.cls), s.bucket, s.layer,
                s.proxy_score, s.proxy_rmse, s.proxy_abs_mean, s.proxy_max_abs,
                s.alt_nvfp4_gain, s.alt_nvfp4_gain_rel,
                s.q8_roi, s.bf16_roi,
                    format_rescue_suggested_type(s).c_str());
        }

        if (out_tensor_overrides != nullptr && rescue_apply_top > 0) {
            size_t budget_used = 0;
            size_t bf16_budget_used = 0;
            int applied = 0;
            std::unordered_map<int, int> class_used;
            for (const auto & s : sens) {
                if (applied >= rescue_apply_top) {
                    break;
                }
                if (rescue_class_limit > 0 && class_used[(int) s.cls] >= rescue_class_limit) {
                    continue;
                }
                if (s.suggested_type == GGML_TYPE_NVFP4 && s.has_alt_nvfp4_cfg) {
                    tensor_type_option opt;
                    opt.name = nvfp4_selector_regex_escape(s.name);
                    opt.type = GGML_TYPE_NVFP4;
                    opt.has_nvfp4_cfg = true;
                    opt.nvfp4_cfg = s.alt_nvfp4_cfg;
                    opt.nvfp4_sample_blocks = s.alt_nvfp4_sample_blocks;
                    opt.nvfp4_policy_name = s.alt_nvfp4_policy_name;
                    out_tensor_overrides->push_back(std::move(opt));
                    class_used[(int) s.cls]++;
                    ++applied;
                    fprintf(stderr,
                        "selector rescue apply tensor=%s type=%s policy=%s sample_blocks=%" PRId64 "\n",
                        s.name.c_str(),
                        ggml_type_name(s.suggested_type),
                        s.alt_nvfp4_policy_name.c_str(),
                        s.alt_nvfp4_sample_blocks);
                    continue;
                }
                if (s.suggested_type == GGML_TYPE_NVFP4) {
                    continue;
                }
                const size_t rescue_nbytes = s.suggested_type == GGML_TYPE_BF16 ? s.bf16_nbytes : s.q8_nbytes;
                if (rescue_nbytes <= s.target_nbytes) {
                    continue;
                }
                const size_t delta = rescue_nbytes - s.target_nbytes;
                if (rescue_budget_bytes > 0 && budget_used + delta > rescue_budget_bytes) {
                    continue;
                }
                if (s.suggested_type == GGML_TYPE_BF16 &&
                    rescue_bf16_budget_bytes > 0 &&
                    bf16_budget_used + delta > rescue_bf16_budget_bytes) {
                    continue;
                }
                tensor_type_option opt;
                opt.name = nvfp4_selector_regex_escape(s.name);
                opt.type = s.suggested_type;
                out_tensor_overrides->push_back(std::move(opt));
                budget_used += delta;
                if (s.suggested_type == GGML_TYPE_BF16) {
                    bf16_budget_used += delta;
                }
                class_used[(int) s.cls]++;
                ++applied;
                fprintf(stderr,
                    "selector rescue apply tensor=%s type=%s delta_bytes=%zu budget_used=%zu bf16_budget_used=%zu\n",
                    s.name.c_str(),
                    ggml_type_name(s.suggested_type),
                    delta,
                    budget_used,
                    bf16_budget_used);
            }

            if (!rescue_tensor_types_file.empty()) {
                std::ofstream patch_types(rescue_tensor_types_file, std::ios::trunc);
                if (!patch_types) {
                    fprintf(stderr, "%s: failed to open selector rescue tensor-type file %s\n", __func__, rescue_tensor_types_file.c_str());
                } else {
                    for (const auto & opt : *out_tensor_overrides) {
                        patch_types << opt.name << '=' << format_tensor_type_value(opt) << '\n';
                    }
                    fprintf(stderr, "%s: wrote selector rescue tensor-type file %s (%zu overrides)\n",
                        __func__, rescue_tensor_types_file.c_str(), out_tensor_overrides->size());
                }
            }
        }
    }

    const int mxfp6_e2m3_scale_top_default =
        rescue_apply_top > 0 ? rescue_apply_top : (mxfp6_bindings.empty() ? 0 : 96);
    const int mxfp6_e2m3_scale_top = out_tensor_overrides && have_runtime_eval
        ? (int) std::max<int64_t>(0, quantize_control_i64("LLAMA_MXFP6_SELECTOR_SCALE_TOP", mxfp6_e2m3_scale_top_default))
        : 0;
    if (mxfp6_e2m3_scale_top > 0 && !mxfp6_bindings.empty()) {
        struct mxfp6_e2m3_scale_sens {
            size_t binding_index = 0;
            double proxy_score = 0.0;
            double rmse = 0.0;
            double abs_mean = 0.0;
            double max_abs = 0.0;
        };
        std::vector<mxfp6_e2m3_scale_sens> mx6_sens;
        mx6_sens.reserve(mxfp6_bindings.size());
        const bool scan_all_mxfp6 = mxfp6_e2m3_scale_top >= (int) mxfp6_bindings.size();
        if (scan_all_mxfp6) {
            auto class_priority = [](nvfp4_selector_tensor_class cls) {
                switch (cls) {
                    case nvfp4_selector_tensor_class::ATTN_QKV:       return 90.0;
                    case nvfp4_selector_tensor_class::ATTN_OUT:       return 88.0;
                    case nvfp4_selector_tensor_class::ROUTER:         return 76.0;
                    case nvfp4_selector_tensor_class::SSM:            return 72.0;
                    case nvfp4_selector_tensor_class::DENSE_MLP:      return 64.0;
                    case nvfp4_selector_tensor_class::EXPERT_DOWN:    return 62.0;
                    case nvfp4_selector_tensor_class::EXPERT_UP_GATE: return 60.0;
                    case nvfp4_selector_tensor_class::EMBEDDING_OUTPUT:return 40.0;
                    case nvfp4_selector_tensor_class::OTHER:          return 32.0;
                }
                return 32.0;
            };
            for (size_t i = 0; i < mxfp6_bindings.size(); ++i) {
                auto & b = mxfp6_bindings[i];
                if (b.target == nullptr || b.target_scale_nbytes == 0) {
                    continue;
                }
                mxfp6_e2m3_scale_sens s;
                s.binding_index = i;
                const double layer = b.layer >= 0 ? (double) b.layer : 9999.0;
                s.proxy_score = class_priority(b.cls) * 1000000.0 - layer * 1000.0 - (double) i;
                mx6_sens.push_back(s);
            }
        } else {
            for (size_t i = 0; i < mxfp6_bindings.size(); ++i) {
                auto & b = mxfp6_bindings[i];
                if (b.target == nullptr || b.target_scale_nbytes == 0) {
                    continue;
                }
                if (!quantize_binding_ensure_target_bytes(b) || !quantize_binding_ensure_scale_bytes(b)) {
                    continue;
                }
                double sq = 0.0;
                double ab = 0.0;
                double mx = 0.0;
                int64_t n = 0;
                if (!mxfp6_selector_quantize_binding(b, 1.0f, nthread, b.working_target_bytes, b.working_scale_bytes, sq, ab, mx, n) || n <= 0) {
                    continue;
                }
                mxfp6_e2m3_scale_sens s;
                s.binding_index = i;
                const nvfp4_selector_proxy_metrics proxy_metrics =
                    nvfp4_selector_proxy_score(sq, ab, mx, n);
                if (!proxy_metrics.ok) {
                    continue;
                }
                s.rmse = proxy_metrics.rmse;
                s.abs_mean = proxy_metrics.abs_mean;
                s.max_abs = proxy_metrics.max_abs;
                s.proxy_score = proxy_metrics.score;
                mx6_sens.push_back(s);
            }
        }
        std::sort(mx6_sens.begin(), mx6_sens.end(), [](const auto & a, const auto & b) {
            if (a.proxy_score != b.proxy_score) return a.proxy_score > b.proxy_score;
            return a.binding_index < b.binding_index;
        });

        const std::vector<float> scale_candidates = quantize_mxfp6_scale_candidates();
        const int scan_n = std::min<int>(mxfp6_e2m3_scale_top, (int) mx6_sens.size());
        update_full_quant_eta("selector-mxfp6-scale-refine", true);
        fprintf(stderr,
            "%s: MXFP6_E2M3 KLD-first scale refine scanning %d/%zu tensors with %zu scale candidates\n",
            __func__, scan_n, mx6_sens.size(), scale_candidates.size());

        struct mxfp6_e2m3_scale_accept {
            size_t binding_index = 0;
            std::vector<uint8_t> target_bytes;
            std::vector<uint8_t> scale_bytes;
            nvfp4_selector_device_snapshot target_device;
            nvfp4_selector_device_snapshot scale_device;
        };
        struct mxfp6_e2m3_scale_pending {
            size_t binding_index = 0;
            float scale_mul = 1.0f;
            double base_score = 0.0;
            double independent_score = 0.0;
            double replay_score = std::numeric_limits<double>::infinity();
            nvfp4_selector_derived_metrics base_metrics;
            nvfp4_selector_derived_metrics independent_metrics;
            nvfp4_selector_derived_metrics replay_metrics;
            bool active = true;
            int replay_count = 0;
        };

        std::vector<mxfp6_e2m3_scale_accept> accepted_scales;
        std::vector<mxfp6_e2m3_scale_pending> pending_scales;
        nvfp4_selector_derived_metrics current_scale_metrics = baseline_eval;
        int64_t mx6_runtime_direct_patches = 0;
        int64_t mx6_runtime_fallback_patches = 0;
        int64_t mx6_host_materializations = 0;
        int64_t mx6_accepted_snapshot_captures = 0;
        int64_t mx6_accepted_snapshot_failures = 0;
        nvfp4_selector_restore_counters mx6_accepted_restore_counters;
        const int pool_per_tensor = (int) std::max<int64_t>(1, MXFP6_SELECTOR_SCALE_POOL_PER_TENSOR);
        const int pool_top = (int) std::max<int64_t>(1, MXFP6_SELECTOR_SCALE_POOL_TOP);
        const int pool_rescore_top = (int) std::max<int64_t>(1, MXFP6_SELECTOR_SCALE_POOL_RESCORE_TOP);
        const int pool_apply_top = (int) std::max<int64_t>(0, MXFP6_SELECTOR_SCALE_POOL_APPLY_TOP);
        const double pool_score_slack = std::max(0.0, MXFP6_SELECTOR_SCALE_POOL_SCORE_SLACK);

        auto restore_accepted_scales = [&]() -> bool {
            restore_all();
            for (const auto & accepted : accepted_scales) {
                auto & ab = mxfp6_bindings[accepted.binding_index];
                if (!quantize_tensor_host_buffer(ab.target) &&
                        !quantize_restore_from_snapshot_or_host(
                        ab.target, accepted.target_device, accepted.target_bytes, ab.target_nbytes,
                        &mx6_accepted_restore_counters)) {
                    return false;
                }
                if (ab.target_scale != nullptr && !quantize_tensor_host_buffer(ab.target_scale) &&
                        !quantize_restore_from_snapshot_or_host(
                            ab.target_scale, accepted.scale_device, accepted.scale_bytes, ab.target_scale_nbytes,
                            &mx6_accepted_restore_counters)) {
                    return false;
                }
            }
            return true;
        };
        auto copy_mxfp6_scale_tensor = [&](nvfp4_selector_binding & b) -> bool {
            return b.target_scale == nullptr ||
                quantize_tensor_copy_in(b.target_scale, b.working_scale_bytes.data(), b.target_scale_nbytes);
        };
        auto quantize_mxfp6_runtime_patch = [&](
                nvfp4_selector_binding & b,
                float mul,
                bool materialize_host_bytes,
                double & sq,
                double & ab,
                double & mx,
                int64_t & n,
                bool & fatal_copy) -> bool {
            fatal_copy = false;
            bool direct_applied = false;
            if (mxfp6_selector_quantize_binding(
                    b, mul, nthread, b.working_target_bytes, b.working_scale_bytes, sq, ab, mx, n,
                    true, &direct_applied) && n > 0 && direct_applied) {
                if (!copy_mxfp6_scale_tensor(b)) {
                    return false;
                }
                ++mx6_runtime_direct_patches;
                if (!materialize_host_bytes) {
                    return true;
                }
                ++mx6_host_materializations;
                double host_sq = 0.0;
                double host_ab = 0.0;
                double host_mx = 0.0;
                int64_t host_n = 0;
                if (!mxfp6_selector_quantize_binding(
                        b, mul, nthread, b.working_target_bytes, b.working_scale_bytes,
                        host_sq, host_ab, host_mx, host_n) || host_n <= 0) {
                    return false;
                }
                return true;
            }
            if (!mxfp6_selector_quantize_binding(
                    b, mul, nthread, b.working_target_bytes, b.working_scale_bytes, sq, ab, mx, n) || n <= 0) {
                return false;
            }
            if (!quantize_tensor_copy_in(b.target, b.working_target_bytes.data(), b.target_nbytes)) {
                return false;
            }
            if (!copy_mxfp6_scale_tensor(b)) {
                return false;
            }
            ++mx6_runtime_fallback_patches;
            return true;
        };

        for (int si = 0; si < scan_n; ++si) {
            auto & b = mxfp6_bindings[mx6_sens[(size_t) si].binding_index];
            std::vector<mxfp6_e2m3_scale_pending> tensor_pending;
            nvfp4_selector_derived_metrics base_metrics = baseline_eval;
            if (!base_metrics.ok) {
                restore_all();
                nvfp4_selector_kld_metrics base_km;
                if (!nvfp4_selector_eval_kld_subset(lctx, kld_budget, params.n_batch, base_km, true)) {
                    restore_all();
                    return false;
                }
                base_metrics = nvfp4_selector_derive_metrics(base_km);
            }
            const nvfp4_selector_metric_score base_metric_score =
                nvfp4_selector_score_metrics(base_metrics, &base_metrics, rank_cfg, nvfp4_selector_metric_mode::MXFP6_SCALE);
            double base_score = base_metric_score.score;

            fprintf(stderr,
                "MXFP6_E2M3 scale refine tensor=%s rank=%d proxy=%.6f base_score=%.6f\n",
                b.name.c_str(), si + 1, mx6_sens[(size_t) si].proxy_score, base_score);

            for (float mul : scale_candidates) {
                if (std::fabs(mul - 1.0f) < 1e-6f) {
                    continue;
                }
                restore_all();
                double sq = 0.0;
                double ab = 0.0;
                double mx = 0.0;
                int64_t n = 0;
                bool fatal_copy = false;
                if (!quantize_mxfp6_runtime_patch(b, mul, false, sq, ab, mx, n, fatal_copy)) {
                    if (!fatal_copy) {
                        continue;
                    }
                    restore_all();
                    return false;
                }

                nvfp4_selector_kld_metrics km;
                if (!nvfp4_selector_eval_kld_subset(lctx, kld_budget, params.n_batch, km, true)) {
                    restore_all();
                    return false;
                }
                const nvfp4_selector_derived_metrics dm = nvfp4_selector_derive_metrics(km);
                const nvfp4_selector_metric_score metric_score =
                    nvfp4_selector_score_metrics(dm, &base_metrics, rank_cfg, nvfp4_selector_metric_mode::MXFP6_SCALE);
                const double score = metric_score.score;
                const bool pass = metric_score.pass;
                fprintf(stderr,
                    "MXFP6_E2M3 scale refine tensor=%s mul=%.8g pass=%d score=%.6f ln=%.6f kld=%.6f p99=%.6f p999=%.6f max=%.6f rms=%.6f top=%.4f\n",
                    b.name.c_str(), (double) mul, pass ? 1 : 0, score,
                    dm.ln_ratio, dm.mean_kld, dm.kld_p99, dm.kld_p999, dm.max_kld, dm.rms_dp, dm.same_top);
                if (pass && std::isfinite(score) && score < base_score + pool_score_slack) {
                    mxfp6_e2m3_scale_pending pending;
                    pending.binding_index = mx6_sens[(size_t) si].binding_index;
                    pending.scale_mul = mul;
                    pending.base_score = base_score;
                    pending.independent_score = score;
                    pending.base_metrics = base_metrics;
                    pending.independent_metrics = dm;
                    tensor_pending.push_back(std::move(pending));
                }
            }

            if (!restore_accepted_scales()) {
                return false;
            }

            std::sort(tensor_pending.begin(), tensor_pending.end(), [](const auto & a, const auto & b) {
                if (a.independent_score != b.independent_score) return a.independent_score < b.independent_score;
                if (a.independent_metrics.max_kld != b.independent_metrics.max_kld) return a.independent_metrics.max_kld < b.independent_metrics.max_kld;
                return a.scale_mul < b.scale_mul;
            });
            if ((int) tensor_pending.size() > pool_per_tensor) {
                tensor_pending.resize((size_t) pool_per_tensor);
            }
            for (auto & pending : tensor_pending) {
                auto & pb = mxfp6_bindings[pending.binding_index];
                fprintf(stderr,
                    "MXFP6_E2M3 scale refine candidate tensor=%s scale_mul=%.8g score %.6f -> %.6f kld %.6f -> %.6f p99 %.6f -> %.6f max %.6f -> %.6f\n",
                    pb.name.c_str(), (double) pending.scale_mul,
                    pending.base_score, pending.independent_score,
                    pending.base_metrics.mean_kld, pending.independent_metrics.mean_kld,
                    pending.base_metrics.kld_p99, pending.independent_metrics.kld_p99,
                    pending.base_metrics.max_kld, pending.independent_metrics.max_kld);
                pending_scales.push_back(std::move(pending));
            }
        }

        std::sort(pending_scales.begin(), pending_scales.end(), [](const auto & a, const auto & b) {
            if (a.independent_score != b.independent_score) return a.independent_score < b.independent_score;
            if (a.independent_metrics.max_kld != b.independent_metrics.max_kld) return a.independent_metrics.max_kld < b.independent_metrics.max_kld;
            return a.binding_index < b.binding_index;
        });
        if ((int) pending_scales.size() > pool_top) {
            pending_scales.resize((size_t) pool_top);
        }

            fprintf(stderr,
            "%s: MXFP6_E2M3 KLD-first scale refine pooled replay candidates=%zu per_tensor=%d rescore_top=%d apply_top=%d score_slack=%.6f\n",
            __func__, pending_scales.size(), pool_per_tensor, pool_rescore_top, pool_apply_top, pool_score_slack);

        std::vector<uint8_t> binding_accepted(mxfp6_bindings.size(), 0);
        int applied_pool = 0;
        for (int replay_round = 1; !pending_scales.empty(); ++replay_round) {
            std::sort(pending_scales.begin(), pending_scales.end(), [](const auto & a, const auto & b) {
                const double as = std::isfinite(a.replay_score) ? a.replay_score : a.independent_score;
                const double bs = std::isfinite(b.replay_score) ? b.replay_score : b.independent_score;
                if (as != bs) return as < bs;
                if (a.independent_score != b.independent_score) return a.independent_score < b.independent_score;
                if (a.replay_count != b.replay_count) return a.replay_count < b.replay_count;
                return a.binding_index < b.binding_index;
            });

            const nvfp4_selector_metric_score base_metric_score =
                nvfp4_selector_score_metrics(current_scale_metrics, &current_scale_metrics, rank_cfg, nvfp4_selector_metric_mode::MXFP6_SCALE);
            const double base_score = base_metric_score.score;
            int best_index = -1;
            double best_score = base_score;
            nvfp4_selector_derived_metrics best_metrics;
            std::vector<uint8_t> best_target_bytes;
            std::vector<uint8_t> best_scale_bytes;
            int rescored = 0;

            for (size_t pi = 0; pi < pending_scales.size() && rescored < pool_rescore_top; ++pi) {
                auto & pending = pending_scales[pi];
                if (!pending.active || pending.binding_index >= binding_accepted.size() || binding_accepted[pending.binding_index]) {
                    pending.active = false;
                    continue;
                }
                auto & b = mxfp6_bindings[pending.binding_index];
                if (!restore_accepted_scales()) {
                    return false;
                }
                double sq = 0.0;
                double ab = 0.0;
                double mx = 0.0;
                int64_t n = 0;
                bool fatal_copy = false;
                if (!quantize_mxfp6_runtime_patch(b, pending.scale_mul, false, sq, ab, mx, n, fatal_copy)) {
                    if (fatal_copy) {
                        restore_all();
                        return false;
                    }
                    pending.active = false;
                    continue;
                }

                nvfp4_selector_kld_metrics km;
                if (!nvfp4_selector_eval_kld_subset(lctx, kld_budget, params.n_batch, km, true)) {
                    restore_all();
                    return false;
                }
                const nvfp4_selector_derived_metrics dm = nvfp4_selector_derive_metrics(km);
                const nvfp4_selector_metric_score metric_score =
                    nvfp4_selector_score_metrics(dm, &current_scale_metrics, rank_cfg, nvfp4_selector_metric_mode::MXFP6_SCALE);
                const double score = metric_score.score;
                const bool pass = metric_score.pass;
                ++pending.replay_count;
                pending.replay_score = score;
                pending.replay_metrics = dm;
                ++rescored;
                fprintf(stderr,
                    "MXFP6_E2M3 scale refine pool round=%d tensor=%s mul=%.8g pass=%d score %.6f -> %.6f independent=%.6f ln=%.6f kld=%.6f p99=%.6f p999=%.6f max=%.6f rms=%.6f top=%.4f\n",
                    replay_round, b.name.c_str(), (double) pending.scale_mul, pass ? 1 : 0,
                    base_score, score, pending.independent_score,
                    dm.ln_ratio, dm.mean_kld, dm.kld_p99, dm.kld_p999, dm.max_kld, dm.rms_dp, dm.same_top);
                if (!pass || !std::isfinite(score) || score >= base_score) {
                    pending.active = false;
                    continue;
                }
                if (score < best_score) {
                    bool materialize_fatal = false;
                    double mat_sq = 0.0;
                    double mat_ab = 0.0;
                    double mat_mx = 0.0;
                    int64_t mat_n = 0;
                    if (!quantize_mxfp6_runtime_patch(
                            b, pending.scale_mul, true,
                            mat_sq, mat_ab, mat_mx, mat_n, materialize_fatal)) {
                        if (materialize_fatal) {
                            restore_all();
                            return false;
                        }
                        pending.active = false;
                        continue;
                    }
                    best_index = (int) pi;
                    best_score = score;
                    best_metrics = dm;
                    best_target_bytes = b.working_target_bytes;
                    best_scale_bytes = b.working_scale_bytes;
                }
            }

            if (best_index < 0) {
                fprintf(stderr,
                    "MXFP6_E2M3 scale refine pool stop round=%d rescored=%d base_score=%.6f active=%zu\n",
                    replay_round, rescored, base_score,
                    std::count_if(pending_scales.begin(), pending_scales.end(), [](const auto & p) { return p.active; }));
                break;
            }

            auto & accepted_pending = pending_scales[(size_t) best_index];
            auto & b = mxfp6_bindings[accepted_pending.binding_index];
            if (!restore_accepted_scales()) {
                return false;
            }
            if (!quantize_tensor_copy_in(b.target, best_target_bytes.data(), b.target_nbytes)) {
                restore_all();
                return false;
            }
            if (b.target_scale != nullptr &&
                    !quantize_tensor_copy_in(b.target_scale, best_scale_bytes.data(), b.target_scale_nbytes)) {
                restore_all();
                return false;
            }

            tensor_type_option opt;
            opt.name = nvfp4_selector_regex_escape(b.name);
            opt.type = GGML_TYPE_MXFP6_E2M3;
            opt.has_mxfp6_scale_mul = true;
            opt.mxfp6_e2m3_scale_mul = accepted_pending.scale_mul;
            opt.mxfp6_policy_name = "kld_scale";
            out_tensor_overrides->push_back(std::move(opt));
            fprintf(stderr,
                "MXFP6_E2M3 scale refine apply round=%d tensor=%s scale_mul=%.8g score %.6f -> %.6f kld %.6f -> %.6f p99 %.6f -> %.6f p999 %.6f -> %.6f max %.6f -> %.6f\n",
                replay_round, b.name.c_str(), (double) accepted_pending.scale_mul,
                base_score, best_score,
                current_scale_metrics.mean_kld, best_metrics.mean_kld,
                current_scale_metrics.kld_p99, best_metrics.kld_p99,
                current_scale_metrics.kld_p999, best_metrics.kld_p999,
                current_scale_metrics.max_kld, best_metrics.max_kld);
            mxfp6_e2m3_scale_accept accepted;
            accepted.binding_index = accepted_pending.binding_index;
            accepted.target_bytes = std::move(best_target_bytes);
            accepted.scale_bytes = std::move(best_scale_bytes);
            if (accepted.target_device.capture(b.target, b.target_nbytes)) {
                ++mx6_accepted_snapshot_captures;
            } else {
                ++mx6_accepted_snapshot_failures;
            }
            if (b.target_scale != nullptr) {
                if (accepted.scale_device.capture(b.target_scale, b.target_scale_nbytes)) {
                    ++mx6_accepted_snapshot_captures;
                } else {
                    ++mx6_accepted_snapshot_failures;
                }
            }
            accepted_scales.push_back(std::move(accepted));
            binding_accepted[accepted_pending.binding_index] = 1;
            current_scale_metrics = best_metrics;
            ++applied_pool;
            pending_scales.erase(
                std::remove_if(pending_scales.begin(), pending_scales.end(), [&](const auto & p) {
                    return !p.active || p.binding_index == accepted.binding_index;
                }),
                pending_scales.end());
            if (pool_apply_top > 0 && applied_pool >= pool_apply_top) {
                fprintf(stderr, "MXFP6_E2M3 scale refine pool reached apply_top=%d\n", pool_apply_top);
                break;
            }
        }
        fprintf(stderr,
            "%s: MXFP6_E2M3 scale refine runtime patches direct=%" PRId64 " fallback=%" PRId64 " materialize=%" PRId64
            " accepted_snapshots=%" PRId64 " accepted_snapshot_fail=%" PRId64
            " accepted_restore_device=%" PRId64 " accepted_restore_host=%" PRId64 " accepted_restore_fail=%" PRId64 "\n",
            __func__,
            mx6_runtime_direct_patches,
            mx6_runtime_fallback_patches,
            mx6_host_materializations,
            mx6_accepted_snapshot_captures,
            mx6_accepted_snapshot_failures,
            mx6_accepted_restore_counters.device,
            mx6_accepted_restore_counters.host,
            mx6_accepted_restore_counters.failed);
    }

    restore_all();
    if (runtime_restore_counters.device || runtime_restore_counters.host || runtime_restore_counters.failed) {
        fprintf(stderr,
            "%s: selector runtime original restores device=%" PRId64 " host=%" PRId64 " fail=%" PRId64 "\n",
            __func__,
            runtime_restore_counters.device,
            runtime_restore_counters.host,
            runtime_restore_counters.failed);
    }
    return true;
}

static bool parse_mixed_format_policy(const char * value, int32_t & out) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    std::string v = trim_copy(value);
    std::replace(v.begin(), v.end(), '-', '_');
    if (striequals(v.c_str(), "off") || striequals(v.c_str(), "none")) {
        out = LLAMA_NV4MX6_POLICY_OFF;
        return true;
    }
    if (striequals(v.c_str(), "auto")) {
        out = LLAMA_NV4MX6_POLICY_AUTO;
        return true;
    }
    if (striequals(v.c_str(), "nvfp4_quality_boost") ||
            striequals(v.c_str(), "nvfp4_primary") ||
            striequals(v.c_str(), "quality_boost") ||
            striequals(v.c_str(), "promote_mxfp6") ||
            striequals(v.c_str(), "nv4_promote_mx6")) {
        out = LLAMA_NV4MX6_POLICY_NV4_PROMOTE_MX6;
        return true;
    }
    if (striequals(v.c_str(), "mxfp6_primary") ||
            striequals(v.c_str(), "mxfp6_quality") ||
            striequals(v.c_str(), "quality_first") ||
            striequals(v.c_str(), "demote_nvfp4") ||
            striequals(v.c_str(), "mx6_demote_nv4")) {
        out = LLAMA_NV4MX6_POLICY_MX6_DEMOTE_NV4;
        return true;
    }
    if (striequals(v.c_str(), "mxfp6_slot") || striequals(v.c_str(), "mx6_slot")) {
        out = LLAMA_NV4MX6_POLICY_MX6_SLOT;
        return true;
    }
    if (striequals(v.c_str(), "bf16_mx6") || striequals(v.c_str(), "bf16_to_mxfp6")) {
        out = LLAMA_NV4MX6_POLICY_BF16_MX6;
        return true;
    }
    if (striequals(v.c_str(), "bf16_mx6_sse") || striequals(v.c_str(), "bf16_to_mxfp6_sse")) {
        out = LLAMA_NV4MX6_POLICY_BF16_MX6_SSE;
        return true;
    }
    return false;
}

int llama_quantize(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc < 3) {
        usage(argv[0]);
    }

    quantize_control_clear();
    llama_model_quantize_params params = llama_model_quantize_default_params();

    int arg_idx = 1;
    std::string imatrix_file;
    std::string patch_base_model;
    std::vector<std::string> included_weights, excluded_weights;
    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<tensor_type_option> tensor_type_opts;
    std::vector<int> prune_layers;
    std::vector<std::pair<std::string, std::string>> selector_controls;
    std::string assignment_jsonl;
    int32_t selector_eval_batch_override = 0;
    int32_t cli_nvfp4_autotune_threads = 0;
    int32_t selector_kld_threads_override = 0;
    bool cli_nvfp4_fast_quantize = false;
    bool selector_skip_remaining = false;
    std::string selector_skip_file;
    std::string selector_skip_policies_cli;
    bool cli_nvfp4_cfg_valid = false;
    nvfp4_cuda_runtime_cfg cli_nvfp4_cfg = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    std::string cli_nvfp4_policy_name;
    const auto add_selector_controls = [&](const char * key, const char * value) {
        selector_controls.emplace_back(key, value);
    };
    auto append_selector_skip_policy = [&](const char * value) {
        if (value == nullptr || value[0] == '\0') {
            return;
        }
        if (!selector_skip_policies_cli.empty()) {
            selector_skip_policies_cli.push_back(',');
        }
        selector_skip_policies_cli += value;
    };

    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        if (strcmp(argv[arg_idx], "--leave-output-tensor") == 0) {
            params.quantize_output_tensor = false;
        } else if (strcmp(argv[arg_idx], "--output-tensor-type") == 0) {
            if (arg_idx < argc-1) {
                params.output_tensor_type = parse_ggml_type(argv[++arg_idx]);
                if (params.output_tensor_type == GGML_TYPE_COUNT) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--token-embedding-type") == 0) {
            if (arg_idx < argc-1) {
                params.token_embedding_type = parse_ggml_type(argv[++arg_idx]);
                if (params.token_embedding_type == GGML_TYPE_COUNT) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--tensor-type") == 0) {
            if (arg_idx == argc-1 || !parse_tensor_type(argv[++arg_idx], tensor_type_opts)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-cfg") == 0) {
            if (arg_idx < argc-1) {
                tensor_type_option opt;
                if (!parse_tensor_type_nvfp4_cfg(argv[++arg_idx], opt)) {
                    usage(argv[0]);
                }
                cli_nvfp4_cfg = opt.nvfp4_cfg;
                cli_nvfp4_policy_name = opt.nvfp4_policy_name.empty() ? "cli" : opt.nvfp4_policy_name;
                cli_nvfp4_cfg_valid = true;
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-preset") == 0) {
            if (arg_idx < argc-1) {
                if (!parse_nvfp4_preset(argv[++arg_idx], cli_nvfp4_cfg, &cli_nvfp4_policy_name)) {
                    usage(argv[0]);
                }
                cli_nvfp4_cfg_valid = true;
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-correction-denom") == 0) {
            if (arg_idx < argc-1) {
                const char * value = argv[++arg_idx];
                if (!parse_nvfp4_scale_denom(value, params.nvfp4_correction_denom)) {
                    usage(argv[0]);
                }
                add_selector_controls("LLAMA_NVFP4_SELECTOR_CORRECTION_DENOM", value);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-input-scale-policy") == 0) {
            if (arg_idx < argc-1) {
                if (!parse_nvfp4_input_scale_policy(argv[++arg_idx], params.nvfp4_input_scale_policy)) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-autotune-max-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_AUTOTUNE_MAX_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-autotune-threads") == 0) {
            if (arg_idx < argc-1) {
                cli_nvfp4_autotune_threads = std::max<int32_t>(1, (int32_t) std::strtol(argv[++arg_idx], nullptr, 10));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-fast-quantize") == 0) {
            cli_nvfp4_fast_quantize = true;
        } else if (strcmp(argv[arg_idx], "--tensor-type-file") == 0) {
            if (arg_idx == argc-1 || !parse_tensor_type_file(argv[++arg_idx], tensor_type_opts)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--prune-layers") == 0) {
            if (arg_idx == argc-1 || !parse_layer_prune(argv[++arg_idx], prune_layers)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--override-kv") == 0) {
            if (arg_idx == argc-1 || !string_parse_kv_override(argv[++arg_idx], kv_overrides)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--patch-base") == 0) {
            if (arg_idx < argc-1) {
                patch_base_model = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mxfp6_e2m3-tensor-scale") == 0) {
            if (arg_idx < argc-1) {
                if (!parse_on_off_value(argv[++arg_idx], params.mxfp6_tensor_scale)) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mxfp6_e2m3-min-savings") == 0) {
            if (arg_idx < argc-1) {
                params.mxfp6_min_savings_bytes = std::max<int64_t>(0, std::stoll(argv[++arg_idx]));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mxfp6_e2m3-input-scale-denom") == 0) {
            if (arg_idx < argc-1) {
                params.mxfp6_input_scale_denom = std::max(0.0f, std::strtof(argv[++arg_idx], nullptr));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mxfp6_e2m3-input-scale-quantile") == 0) {
            if (arg_idx < argc-1) {
                params.mxfp6_input_scale_quantile = std::clamp(std::strtof(argv[++arg_idx], nullptr), 0.0f, 1.0f);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mxfp6_e2m3-tensor-scale-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                params.mxfp6_tensor_scale_sample_blocks = std::max<int64_t>(0, std::stoll(argv[++arg_idx]));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mxfp6_e2m3-tensor-scale-steps") == 0) {
            if (arg_idx < argc-1) {
                params.mxfp6_tensor_scale_steps = (int32_t) std::max<int64_t>(0, std::stoll(argv[++arg_idx]));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mxfp6_e2m3-selector-scale-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_MXFP6_SELECTOR_SCALE_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mxfp6_e2m3-selector-scale-candidates") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_MXFP6_SELECTOR_SCALE_CANDIDATES", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mixed-format-policy") == 0 ||
                   strcmp(argv[arg_idx], "--nv4mx6-policy") == 0) {
            if (arg_idx < argc-1) {
                if (!parse_mixed_format_policy(argv[++arg_idx], params.nv4mx6_policy)) {
                    usage(argv[0]);
                }
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mixed-mx6-penalty") == 0 ||
                   strcmp(argv[arg_idx], "--nv4mx6-mx6-penalty") == 0) {
            if (arg_idx < argc-1) {
                params.nv4mx6_mx6_penalty = std::max(0.0f, std::strtof(argv[++arg_idx], nullptr));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mixed-bf16-mx6-threshold") == 0 ||
                   strcmp(argv[arg_idx], "--nv4mx6-bf16-mx6-threshold") == 0) {
            if (arg_idx < argc-1) {
                params.nv4mx6_bf16_mx6_max_sse_ratio = std::max(0.0f, std::strtof(argv[++arg_idx], nullptr));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mixed-sample-blocks") == 0 ||
                   strcmp(argv[arg_idx], "--nv4mx6-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                params.mixed_format_sample_blocks = std::max<int64_t>(0, std::stoll(argv[++arg_idx]));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mixed-sample-cap") == 0 ||
                   strcmp(argv[arg_idx], "--nv4mx6-sample-cap") == 0) {
            if (arg_idx < argc-1) {
                params.mixed_format_sample_cap = std::max<int64_t>(0, std::stoll(argv[++arg_idx]));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mixed-imatrix-weight-blend") == 0 ||
                   strcmp(argv[arg_idx], "--nv4mx6-qw-blend") == 0) {
            if (arg_idx < argc-1) {
                params.mixed_format_imatrix_blend = std::strtof(argv[++arg_idx], nullptr);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mixed-imatrix-weight-power") == 0 ||
                   strcmp(argv[arg_idx], "--nv4mx6-qw-power") == 0) {
            if (arg_idx < argc-1) {
                params.mixed_format_imatrix_power = std::strtof(argv[++arg_idx], nullptr);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mixed-imatrix-weight-min") == 0 ||
                   strcmp(argv[arg_idx], "--nv4mx6-qw-min") == 0) {
            if (arg_idx < argc-1) {
                params.mixed_format_imatrix_min = std::strtof(argv[++arg_idx], nullptr);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--mixed-imatrix-weight-max") == 0 ||
                   strcmp(argv[arg_idx], "--nv4mx6-qw-max") == 0) {
            if (arg_idx < argc-1) {
                params.mixed_format_imatrix_max = std::strtof(argv[++arg_idx], nullptr);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-kld") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_KLD_FILE", argv[++arg_idx]);
                add_selector_controls("LLAMA_NVFP4_SELECTOR_ENABLE_EVAL", "1");
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-checkpoint-model") == 0 ||
                   strcmp(argv[arg_idx], "--nvfp4-selector-seed-model") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_CHECKPOINT_MODEL", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-cache-dir") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_CACHE_DIR", argv[++arg_idx]);
                add_selector_controls("LLAMA_NVFP4_SELECTOR_KEEP_CHECKPOINT", "1");
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-keep-checkpoint") == 0) {
            add_selector_controls("LLAMA_NVFP4_SELECTOR_KEEP_CHECKPOINT", "1");
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-require-runtime-cache") == 0) {
            add_selector_controls("LLAMA_NVFP4_SELECTOR_REQUIRE_RUNTIME_CACHE", "1");
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-skip-file") == 0) {
            if (arg_idx < argc-1) {
                selector_skip_file = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-skip-remaining") == 0) {
            selector_skip_remaining = true;
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-chunks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_N_CHUNKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-chunk-start") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_CHUNK_START", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-holdout-chunks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_HOLDOUT_CHUNKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-holdout-start") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_HOLDOUT_START", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-stagea-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_STAGEA_SAMPLE_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-stagea-max-policies") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_STAGEA_MAX_POLICIES", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-awq-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_AWQ_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-skip-policy") == 0) {
            if (arg_idx < argc-1) {
                append_selector_skip_policy(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-skip-policies") == 0) {
            if (arg_idx < argc-1) {
                append_selector_skip_policy(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-refine-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_REFINE_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-refine-budget") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_REFINE_BUDGET", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-survey-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_SURVEY_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-survey-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_SURVEY_SAMPLE_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-max-tensors") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_MAX_TENSORS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-trace") == 0) {
            add_selector_controls("LLAMA_NVFP4_TRACE", "1");
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-policy-threads") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_POLICY_THREADS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-threads") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_THREADS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-kld-threads") == 0) {
            if (arg_idx < argc-1) {
                selector_kld_threads_override = std::max<int32_t>(1, (int32_t) std::strtol(argv[++arg_idx], nullptr, 10));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-auto-rescue") == 0) {
            add_selector_controls("LLAMA_NVFP4_AUTO_RESCUE", "1");
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-only") == 0) {
            add_selector_controls("LLAMA_NVFP4_SELECTOR_ONLY", "1");
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-eval-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_EVAL_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-eval-chunks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_EVAL_CHUNKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-n-seq") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_N_SEQ", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-eval-batch") == 0) {
            if (arg_idx < argc-1) {
                selector_eval_batch_override = std::max<int32_t>(1, (int32_t) std::strtol(argv[++arg_idx], nullptr, 10));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-n-gpu-layers") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_N_GPU_LAYERS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-sensitivity-report") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_SENSITIVITY_REPORT_FILE", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-sensitivity-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_SENSITIVITY_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-sensitivity-layer") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_SENSITIVITY_LAYER", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-sensitivity-tensor") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_SENSITIVITY_TENSOR", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-sensitivity-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_SENSITIVITY_SAMPLE_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-type") == 0) {
            if (arg_idx < argc-1) {
                const ggml_type rescue_type = parse_ggml_type(argv[++arg_idx]);
                if (rescue_type == GGML_TYPE_COUNT) {
                    usage(argv[0]);
                }
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_Q8_TYPE", ggml_type_name(rescue_type));
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-report-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_REPORT_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-budget-mb") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_BUDGET_MB", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-bf16-budget-mb") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_BF16_BUDGET_MB", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-class-limit") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_CLASS_LIMIT", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-nvfp4-top") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_NVFP4_TOP", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-sample-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_SENS_SAMPLE_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-coarse-max-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_NVFP4_MAX_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-refine-max-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_NVFP4_REFINE_MAX_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-guard-max-blocks") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_NVFP4_GUARD_MAX_BLOCKS", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-report") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_REPORT_FILE", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rescue-tensor-types") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RESCUE_TENSOR_TYPES_FILE", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-kld-penalty") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RANK_KLD_PENALTY", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-p99-penalty") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RANK_P99_PENALTY", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-p999-penalty") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RANK_P999_PENALTY", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-max-kld-penalty") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RANK_MAX_KLD_PENALTY", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rank-kld-threshold") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RANK_KLD_THRESHOLD", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rank-kld-hard-gate") == 0) {
            add_selector_controls("LLAMA_NVFP4_SELECTOR_RANK_KLD_HARD_GATE", "1");
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rank-p99-threshold") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RANK_P99_THRESHOLD", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rank-p99-hard-gate") == 0) {
            add_selector_controls("LLAMA_NVFP4_SELECTOR_RANK_P99_HARD_GATE", "1");
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rank-p999-threshold") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RANK_P999_THRESHOLD", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rank-p999-hard-gate") == 0) {
            add_selector_controls("LLAMA_NVFP4_SELECTOR_RANK_P999_HARD_GATE", "1");
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rank-max-kld-threshold") == 0) {
            if (arg_idx < argc-1) {
                add_selector_controls("LLAMA_NVFP4_SELECTOR_RANK_MAX_KLD_THRESHOLD", argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--nvfp4-selector-rank-max-kld-hard-gate") == 0) {
            add_selector_controls("LLAMA_NVFP4_SELECTOR_RANK_MAX_KLD_HARD_GATE", "1");
        } else if (strcmp(argv[arg_idx], "--assignment-jsonl") == 0) {
            if (arg_idx < argc-1) {
                assignment_jsonl = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--dry-run") == 0) {
            params.dry_run = true;
        } else if (strcmp(argv[arg_idx], "--allow-requantize") == 0) {
            params.allow_requantize = true;
        } else if (strcmp(argv[arg_idx], "--pure") == 0) {
            params.pure = true;
        } else if (strcmp(argv[arg_idx], "--imatrix") == 0) {
            if (arg_idx < argc-1) {
                imatrix_file = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--include-weights") == 0) {
            if (arg_idx < argc-1) {
                included_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--exclude-weights") == 0) {
            if (arg_idx < argc-1) {
                excluded_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--keep-split") == 0) {
            params.keep_split = true;
        } else {
            usage(argv[0]);
        }
    }

    if (argc - arg_idx < 2) {
        printf("%s: bad arguments\n", argv[0]);
        usage(argv[0]);
    }
    if (!included_weights.empty() && !excluded_weights.empty()) {
        usage(argv[0]);
    }

    auto selector_controls_has = [&](const char * key) {
        return std::any_of(selector_controls.begin(), selector_controls.end(), [&](const auto & entry) {
            return entry.first == key;
        });
    };
    auto add_selector_controls_default = [&](const char * key, const char * value) {
        if (!selector_controls_has(key)) {
            selector_controls.emplace_back(key, value);
        }
    };
    if (cli_nvfp4_fast_quantize) {
        add_selector_controls_default("LLAMA_NVFP4_AUTOTUNE_MAX_BLOCKS", "4096");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_STAGEA_SAMPLE_BLOCKS", "512");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_STAGEA_MAX_POLICIES", "8");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_AWQ_TOP", "3");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_REFINE_TOP", "4");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_REFINE_BUDGET", "16");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_SURVEY_TOP", "8");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_SURVEY_SAMPLE_BLOCKS", "512");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_HOLDOUT_CHUNKS", "0");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_EVAL_TOP", "2");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_EVAL_CHUNKS", "2");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_N_SEQ", "1");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_RESCUE_TOP", "0");
        add_selector_controls_default("LLAMA_NVFP4_SELECTOR_RESCUE_REPORT_TOP", "0");
        fprintf(stderr, "llama_quantize: fast quantize minimal autotuning enabled\n");
    }
    if (!selector_skip_policies_cli.empty()) {
        add_selector_controls("LLAMA_NVFP4_SELECTOR_SKIP_POLICIES", selector_skip_policies_cli.c_str());
    }
    nvfp4_selector_set_skip_state(selector_skip_file, selector_skip_remaining);
    std::vector<std::string> imatrix_datasets;
    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    std::vector<llama_model_imatrix_data> imatrix_entries;
    std::vector<int32_t> prune_layers_param;
    int m_last_call = prepare_imatrix(imatrix_file, imatrix_datasets, included_weights, excluded_weights, imatrix_data);

    std::vector<llama_model_imatrix_data> i_data;
    std::vector<llama_model_tensor_override> t_override;
    if (!imatrix_data.empty()) {
        imatrix_entries.reserve(imatrix_data.size() + 1);
        for (const auto & entry : imatrix_data) {
            imatrix_entries.push_back({
                entry.first.c_str(),
                entry.second.data(),
                entry.second.size(),
            });
        }
        imatrix_entries.push_back({ nullptr, nullptr, 0 });
        params.imatrix = imatrix_entries.data();
        {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_FILE);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            strncpy(kvo.val_str, imatrix_file.c_str(), 127);
            kvo.val_str[127] = '\0';
            kv_overrides.emplace_back(std::move(kvo));
        }
        if (!imatrix_datasets.empty()) {
            llama_model_kv_override kvo;
            // TODO: list multiple datasets when there are more than one
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_DATASET);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            strncpy(kvo.val_str, imatrix_datasets[0].c_str(), 127);
            kvo.val_str[127] = '\0';
            kv_overrides.emplace_back(std::move(kvo));
        }
        {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            kvo.val_i64 = imatrix_data.size();
            kv_overrides.emplace_back(std::move(kvo));
        }
        if (m_last_call > 0) {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            kvo.val_i64 = m_last_call;
            kv_overrides.emplace_back(std::move(kvo));
        }
    }
    if (!kv_overrides.empty()) {
        kv_overrides.emplace_back();
        kv_overrides.back().key[0] = 0;
        params.kv_overrides = kv_overrides.data();
    }
    if (!tensor_type_opts.empty()) {
        t_override.reserve(tensor_type_opts.size() + 1);
        for (const auto & tt : tensor_type_opts) {
            t_override.push_back({tt.name.c_str(), tt.type});
        }
        t_override.push_back({nullptr, GGML_TYPE_COUNT});  // array terminator
        params.tt_overrides = t_override.data();
        params.tensor_types = &tensor_type_opts;
    }
    if (!prune_layers.empty()) {
        prune_layers_param.assign(prune_layers.begin(), prune_layers.end());
        prune_layers_param.push_back(-1);
        params.prune_layers = prune_layers_param.data();
    }
    if (!patch_base_model.empty()) {
        params.patch_base_model = patch_base_model.c_str();
    }
    if (!assignment_jsonl.empty()) {
        params.assignment_jsonl = assignment_jsonl.c_str();
    }
    llama_backend_init();

    // parse command line arguments
    const std::string fname_inp = argv[arg_idx];
    arg_idx++;
    std::string fname_out;

    std::string ftype_str;
    std::string suffix = ".gguf";
    if (try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
        // argv[arg_idx] is the ftype directly: <input> <ftype>
        if (!params.dry_run) {
            std::string fpath;
            const size_t pos = fname_inp.find_last_of("/\\");
            if (pos != std::string::npos) {
                fpath = fname_inp.substr(0, pos + 1);
            }

            // export as [inp path]/ggml-model-[ftype]. Only add extension if there is no splitting
            fname_out = fpath + "ggml-model-" + ftype_str;
            if (!params.keep_split) {
                fname_out += suffix;
            }
        }
        arg_idx++;
        if (ftype_str == "COPY") {
            params.only_copy = true;
        }
    } else {
        // argv[arg_idx] is not a valid ftype, so treat it as output path: <input> <output> <ftype>
        fname_out = argv[arg_idx];
        if (params.keep_split && fname_out.find(suffix) != std::string::npos) {
            fname_out = fname_out.substr(0, fname_out.length() - suffix.length());
        }
        arg_idx++;

        if (argc <= arg_idx) {
            fprintf(stderr, "%s: missing ftype\n", __func__);
            return 1;
        }
        if (!try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
            fprintf(stderr, "%s: invalid ftype '%s'\n", __func__, argv[arg_idx]);
            return 1;
        }
        if (ftype_str == "COPY") {
           params.only_copy = true;
        }
        arg_idx++;
    }

    const bool ftype_mixed_nvfp4_mxfp6 = ftype_is_nvfp4_mxfp6_alias(ftype_str);
    if (ftype_mixed_nvfp4_mxfp6) {
        params.ftype = LLAMA_FTYPE_MOSTLY_NVFP4;
        if (params.nv4mx6_policy == LLAMA_NV4MX6_POLICY_OFF) {
            params.nv4mx6_policy = LLAMA_NV4MX6_POLICY_AUTO;
        }
    }

    // parse nthreads
    if (argc > arg_idx) {
        try {
            params.nthread = std::stoi(argv[arg_idx]);
        }
        catch (const std::exception & e) {
            fprintf(stderr, "%s: invalid nthread '%s' (%s)\n", __func__, argv[arg_idx], e.what());
            return 1;
        }
    }

    if (params.ftype == LLAMA_FTYPE_MOSTLY_NVFP4 && params.nv4mx6_policy == LLAMA_NV4MX6_POLICY_OFF) {
        if (params.token_embedding_type >= GGML_TYPE_COUNT) {
            params.token_embedding_type = GGML_TYPE_NVFP4;
        }
        if (params.output_tensor_type >= GGML_TYPE_COUNT) {
            params.output_tensor_type = GGML_TYPE_Q6_K;
        }
    }

    if (params.ftype == LLAMA_FTYPE_MOSTLY_MXFP6_E2M3 || ftype_mixed_nvfp4_mxfp6) {
        if (params.token_embedding_type >= GGML_TYPE_COUNT) {
            params.token_embedding_type = GGML_TYPE_MXFP6_E2M3;
        }
        if (params.output_tensor_type >= GGML_TYPE_COUNT) {
            params.output_tensor_type = GGML_TYPE_MXFP6_E2M3;
        }
    }

    const bool has_selector_kld =
        selector_controls_has("LLAMA_NVFP4_SELECTOR_KLD_FILE");
    if (params.ftype == LLAMA_FTYPE_MOSTLY_MXFP6_E2M3 && has_selector_kld) {
        // Pure MXFP6 has no NVFP4 policy to choose. Keep the measured-eval selector
        // alive, but spend the KLD budget on MXFP6 tensor-scale refinement.
        if (!selector_controls_has("LLAMA_NVFP4_SELECTOR_EVAL_TOP")) {
            selector_controls.emplace_back("LLAMA_NVFP4_SELECTOR_EVAL_TOP", "1");
        }
        if (!selector_controls_has("LLAMA_NVFP4_SELECTOR_STAGEA_MAX_POLICIES")) {
            selector_controls.emplace_back("LLAMA_NVFP4_SELECTOR_STAGEA_MAX_POLICIES", "1");
        }
        if (!selector_controls_has("LLAMA_NVFP4_SELECTOR_SURVEY_TOP")) {
            selector_controls.emplace_back("LLAMA_NVFP4_SELECTOR_SURVEY_TOP", "1");
        }
    }

    std::vector<std::unique_ptr<scoped_quantize_control>> selector_controls_scope;
    selector_controls_scope.reserve(selector_controls.size());
    for (const auto & entry : selector_controls) {
        selector_controls_scope.emplace_back(std::make_unique<scoped_quantize_control>(entry.first.c_str(), entry.second.c_str()));
    }

    params.nvfp4_autotune_max_blocks =
        std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_AUTOTUNE_MAX_BLOCKS", params.nvfp4_autotune_max_blocks));
    nvfp4_set_autotune_threads(cli_nvfp4_autotune_threads);
    nvfp4_selector_set_kld_threads_override(selector_kld_threads_override);

    if (!params.dry_run) {
        if (std::error_code ec; std::filesystem::equivalent(fname_inp, fname_out, ec)) {
            fprintf(stderr, "%s: error: input and output files are the same: '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }
    }

    const std::vector<tensor_type_option> tensor_type_opts_cli = tensor_type_opts;
    const std::string selector_kld_path = quantize_control_string("LLAMA_NVFP4_SELECTOR_KLD_FILE");
    const bool selector_bootstrap = quantize_control_has("LLAMA_NVFP4_SELECTOR_BOOTSTRAP");
    const bool selector_auto_rescue_running = quantize_control_has("LLAMA_NVFP4_AUTO_RESCUE_RUNNING");
    const bool selector_only = quantize_control_i64("LLAMA_NVFP4_SELECTOR_ONLY", 0) != 0;
    bool selector_cfg_valid = false;
    nvfp4_cuda_runtime_cfg selector_cfg = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    std::string selector_policy_name;
    std::string selector_checkpoint_path;
    bool selector_checkpoint_created = false;
    bool selector_inputs_ready = false;
    nvfp4_selector_kld_subset selector_kld;
    std::unique_ptr<nvfp4_selector_kld_subset> selector_kld_holdout;
    selector_rank_config selector_rank_cfg;

    const bool selector_enabled =
        !selector_bootstrap &&
        !selector_auto_rescue_running &&
        !selector_kld_path.empty() &&
        (params.ftype == LLAMA_FTYPE_MOSTLY_NVFP4 || params.ftype == LLAMA_FTYPE_MOSTLY_MXFP6_E2M3) &&
        !params.dry_run &&
        !params.keep_split &&
        !params.only_copy;
    const bool selector_auto_rescue =
        selector_enabled &&
        !selector_only &&
        quantize_control_i64("LLAMA_NVFP4_AUTO_RESCUE", 0) != 0;

    llama_print_build_info();

    if (params.dry_run) {
        fprintf(stderr, "%s: calculating quantization size for '%s' as %s", __func__, fname_inp.c_str(), ftype_str.c_str());
    } else {
        fprintf(stderr, "%s: quantizing '%s' to '%s' as %s", __func__, fname_inp.c_str(), fname_out.c_str(), ftype_str.c_str());
    }
    if (!patch_base_model.empty()) {
        fprintf(stderr, " (patch-base=%s)", patch_base_model.c_str());
    }

    if (params.nthread > 0) {
        fprintf(stderr, " using %d threads", params.nthread);
    }
    fprintf(stderr, "\n");

	    auto run_selector_pass = [&](const std::string & seed_path,
	                                 nvfp4_cuda_runtime_cfg & out_cfg,
	                                 std::string & out_policy_name,
                                     bool * out_kept_seed,
	                                 std::vector<tensor_type_option> * out_overrides) -> bool {
        if (!selector_inputs_ready || seed_path.empty()) {
            return false;
        }
        if (out_overrides != nullptr) {
            out_overrides->clear();
        }
        return nvfp4_selector_choose_policy(
            fname_inp,
            seed_path,
            imatrix_data,
            std::max(1, params.nthread),
            selector_kld,
            selector_kld_holdout.get(),
            selector_rank_cfg,
            cli_nvfp4_cfg_valid ? &cli_nvfp4_cfg : nullptr,
            cli_nvfp4_policy_name,
            params.nvfp4_input_scale_policy,
	            selector_eval_batch_override,
                cli_nvfp4_autotune_threads,
	            out_cfg,
	            out_policy_name,
                out_kept_seed,
	            out_overrides);
	    };

    if (selector_enabled) {
        const int32_t selector_chunk_start = (int32_t) std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_CHUNK_START", 0));
        const int32_t selector_n_chunks = (int32_t) std::max<int64_t>(1, quantize_control_i64("LLAMA_NVFP4_SELECTOR_N_CHUNKS", 4));
        const int32_t selector_holdout_chunks = (int32_t) std::max<int64_t>(0, quantize_control_i64("LLAMA_NVFP4_SELECTOR_HOLDOUT_CHUNKS", 2));
        const int32_t selector_holdout_start = (int32_t) quantize_control_i64("LLAMA_NVFP4_SELECTOR_HOLDOUT_START", selector_chunk_start + selector_n_chunks);
        selector_rank_cfg.kld_threshold = quantize_control_f64("LLAMA_NVFP4_SELECTOR_RANK_KLD_THRESHOLD", -1.0);
        selector_rank_cfg.p99_threshold = quantize_control_f64("LLAMA_NVFP4_SELECTOR_RANK_P99_THRESHOLD", -1.0);
        selector_rank_cfg.p999_threshold = quantize_control_f64("LLAMA_NVFP4_SELECTOR_RANK_P999_THRESHOLD", -1.0);
        selector_rank_cfg.max_kld_threshold = quantize_control_f64("LLAMA_NVFP4_SELECTOR_RANK_MAX_KLD_THRESHOLD", -1.0);
        selector_rank_cfg.kld_penalty = quantize_control_f64("LLAMA_NVFP4_SELECTOR_RANK_KLD_PENALTY", 0.0);
        selector_rank_cfg.p99_penalty = quantize_control_f64("LLAMA_NVFP4_SELECTOR_RANK_P99_PENALTY", 0.0);
        selector_rank_cfg.p999_penalty = quantize_control_f64("LLAMA_NVFP4_SELECTOR_RANK_P999_PENALTY", 0.0);
        selector_rank_cfg.max_kld_penalty = quantize_control_f64("LLAMA_NVFP4_SELECTOR_RANK_MAX_KLD_PENALTY", 0.0);
        selector_rank_cfg.holdout_weight = SELECTOR_HOLDOUT_WEIGHT;
        selector_rank_cfg.kld_hard_gate = quantize_control_i64("LLAMA_NVFP4_SELECTOR_RANK_KLD_HARD_GATE", 0) != 0;
        selector_rank_cfg.p99_hard_gate = quantize_control_i64("LLAMA_NVFP4_SELECTOR_RANK_P99_HARD_GATE", 0) != 0;
        selector_rank_cfg.p999_hard_gate = quantize_control_i64("LLAMA_NVFP4_SELECTOR_RANK_P999_HARD_GATE", 0) != 0;
        selector_rank_cfg.max_kld_hard_gate = quantize_control_i64("LLAMA_NVFP4_SELECTOR_RANK_MAX_KLD_HARD_GATE", 0) != 0;
        selector_rank_cfg.ppl_sigma = SELECTOR_PPL_SIGMA;
        selector_rank_cfg.kld_sigma = SELECTOR_KLD_SIGMA;
        selector_rank_cfg.rms_dp_sigma = SELECTOR_RMS_DP_SIGMA;
        selector_rank_cfg.same_top_sigma = SELECTOR_SAME_TOP_SIGMA;
        selector_rank_cfg.ln_ratio_abs_floor = SELECTOR_LN_RATIO_ABS_FLOOR;
        selector_rank_cfg.mean_kld_abs_floor = SELECTOR_MEAN_KLD_ABS_FLOOR;
        selector_rank_cfg.rms_dp_abs_floor = SELECTOR_RMS_DP_ABS_FLOOR;
        selector_rank_cfg.same_top_abs_floor = SELECTOR_SAME_TOP_ABS_FLOOR;
        selector_rank_cfg.entropy_rmse_abs_floor = SELECTOR_ENTROPY_RMSE_ABS_FLOOR;
        selector_rank_cfg.top_prob_rmse_abs_floor = SELECTOR_TOP_PROB_RMSE_ABS_FLOOR;
        selector_rank_cfg.top_flip_weight_abs_floor = SELECTOR_TOP_FLIP_WEIGHT_ABS_FLOOR;
        selector_rank_cfg.p99_abs_margin = SELECTOR_P99_ABS_MARGIN;
        selector_rank_cfg.p99_rel_margin = SELECTOR_P99_REL_MARGIN;
        selector_rank_cfg.p999_abs_margin = SELECTOR_P999_ABS_MARGIN;
        selector_rank_cfg.p999_rel_margin = SELECTOR_P999_REL_MARGIN;
        selector_rank_cfg.max_kld_abs_margin = SELECTOR_MAX_KLD_ABS_MARGIN;
        selector_rank_cfg.max_kld_rel_margin = SELECTOR_MAX_KLD_REL_MARGIN;
        const bool selector_mxfp6_possible =
            params.ftype == LLAMA_FTYPE_MOSTLY_MXFP6_E2M3 ||
            params.nv4mx6_policy != LLAMA_NV4MX6_POLICY_OFF ||
            std::any_of(tensor_type_opts.begin(), tensor_type_opts.end(), [](const tensor_type_option & opt) {
                return opt.type == GGML_TYPE_MXFP6_E2M3;
            });
        if (selector_mxfp6_possible) {
            fprintf(stderr,
                "%s: selector rank penalties: mean=%.6g p99=%.6g p999=%.6g max=%.6g thresholds={mean=%.6g p99=%.6g p999=%.6g max=%.6g} mxfp6_tol={ppl_abs=%.6g ppl_rel=%.6g mean_kld_abs=%.6g mean_kld_rel=%.6g}\n",
                __func__,
                selector_rank_cfg.kld_penalty,
                selector_rank_cfg.p99_penalty,
                selector_rank_cfg.p999_penalty,
                selector_rank_cfg.max_kld_penalty,
                selector_rank_cfg.kld_threshold,
                selector_rank_cfg.p99_threshold,
                selector_rank_cfg.p999_threshold,
                selector_rank_cfg.max_kld_threshold,
                selector_rank_cfg.mxfp6.ppl_abs_tol,
                selector_rank_cfg.mxfp6.ppl_rel_tol,
                selector_rank_cfg.mxfp6.mean_kld_abs_tol,
                selector_rank_cfg.mxfp6.mean_kld_rel_tol);
        } else {
            fprintf(stderr,
                "%s: selector rank penalties: mean=%.6g p99=%.6g p999=%.6g max=%.6g thresholds={mean=%.6g p99=%.6g p999=%.6g max=%.6g}\n",
                __func__,
                selector_rank_cfg.kld_penalty,
                selector_rank_cfg.p99_penalty,
                selector_rank_cfg.p999_penalty,
                selector_rank_cfg.max_kld_penalty,
                selector_rank_cfg.kld_threshold,
                selector_rank_cfg.p99_threshold,
                selector_rank_cfg.p999_threshold,
                selector_rank_cfg.max_kld_threshold);
        }
        if (!nvfp4_selector_load_kld_subset(selector_kld_path.c_str(), selector_chunk_start, selector_n_chunks, selector_kld)) {
            fprintf(stderr, "%s: selector disabled because KLD subset could not be loaded from %s\n", __func__, selector_kld_path.c_str());
        } else {
            selector_inputs_ready = true;
            if (selector_holdout_chunks > 0) {
                selector_kld_holdout = std::make_unique<nvfp4_selector_kld_subset>();
                if (!nvfp4_selector_load_kld_subset(selector_kld_path.c_str(), selector_holdout_start, selector_holdout_chunks, *selector_kld_holdout)) {
                    fprintf(stderr, "%s: selector holdout subset could not be loaded from %s (start=%d chunks=%d); continuing without holdout\n",
                        __func__, selector_kld_path.c_str(), selector_holdout_start, selector_holdout_chunks);
                    selector_kld_holdout.reset();
                }
            }

            std::string selector_checkpoint_override = quantize_control_string("LLAMA_NVFP4_SELECTOR_CHECKPOINT_MODEL");
            const bool selector_keep_checkpoint = quantize_control_i64("LLAMA_NVFP4_SELECTOR_KEEP_CHECKPOINT", 0) != 0;
            const std::string selector_cache_dir = quantize_control_string("LLAMA_NVFP4_SELECTOR_CACHE_DIR");
            if (!selector_checkpoint_override.empty()) {
                selector_checkpoint_path = selector_checkpoint_override;
                fprintf(stderr, "%s: selector using existing candidate search checkpoint %s\n", __func__, selector_checkpoint_path.c_str());
            } else {
                if (!selector_cache_dir.empty()) {
                    std::string stem = std::filesystem::path(fname_inp).stem().string();
                    for (char & ch : stem) {
                        if (!std::isalnum((unsigned char) ch) && ch != '-' && ch != '_') {
                            ch = '_';
                        }
                    }
                    if (stem.empty()) {
                        stem = "model";
                    }
                    uint64_t h = 1469598103934665603ull;
                    const std::string hash_input = fname_inp + "|" + ftype_str;
                    for (unsigned char ch : hash_input) {
                        h ^= (uint64_t) ch;
                        h *= 1099511628211ull;
                    }
                    std::ostringstream hss;
                    hss << std::hex << h;
                    selector_checkpoint_path = (std::filesystem::path(selector_cache_dir) /
                        (stem + "-" + ftype_str + "-" + hss.str() + ".bwq-checkpoint.gguf")).string();
                    std::error_code ec;
                    std::filesystem::create_directories(selector_cache_dir, ec);
                    if (ec) {
                        fprintf(stderr, "%s: failed to create selector checkpoint cache dir %s: %s\n",
                            __func__, selector_cache_dir.c_str(), ec.message().c_str());
                    }
                } else {
                    selector_checkpoint_path = fname_out + ".bwq-checkpoint.gguf";
                }

                bool reuse_cached_checkpoint = false;
                if (selector_keep_checkpoint && !selector_checkpoint_path.empty()) {
                    std::error_code ec;
                    reuse_cached_checkpoint =
                        std::filesystem::exists(selector_checkpoint_path, ec) &&
                        !ec &&
                        std::filesystem::file_size(selector_checkpoint_path, ec) > 0 &&
                        !ec;
                    if (reuse_cached_checkpoint) {
                        fprintf(stderr, "%s: selector reusing cached candidate search checkpoint %s\n",
                            __func__, selector_checkpoint_path.c_str());
                    }
                }

                if (!reuse_cached_checkpoint) {
                    selector_checkpoint_created = true;
                    fprintf(stderr, "%s: candidate search checkpoint quantization -> %s\n", __func__, selector_checkpoint_path.c_str());
                    {
                        scoped_quantize_control bootstrap("LLAMA_NVFP4_SELECTOR_BOOTSTRAP", "1");
                        nvfp4_clear_runtime_cfg();
                        if (llama_model_quantize(fname_inp.c_str(), selector_checkpoint_path.c_str(), &params)) {
                            fprintf(stderr, "%s: candidate search checkpoint build failed, continuing without selector\n", __func__);
                            selector_checkpoint_path.clear();
                            selector_checkpoint_created = false;
                        }
                    }
                }
            }

	            std::vector<tensor_type_option> selector_overrides;
                bool selector_chose_seed_keep = false;
	            if (run_selector_pass(selector_checkpoint_path, selector_cfg, selector_policy_name, &selector_chose_seed_keep, &selector_overrides)) {
	                nvfp4_set_runtime_cfg(&selector_cfg);
	                selector_cfg_valid = true;
	                fprintf(stderr,
                    "%s: selector chose policy=%s cfg={choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f}\n",
                    __func__,
                    selector_policy_name.c_str(),
                    selector_cfg.choose46_mode,
                    selector_cfg.refit_iters,
                    selector_cfg.use_compand_sat,
                    (double) selector_cfg.cap_m6,
                    (double) selector_cfg.cap_m4);
                if (!selector_overrides.empty()) {
                    tensor_type_opts = tensor_type_opts_cli;
                    tensor_type_opts.insert(tensor_type_opts.end(), selector_overrides.begin(), selector_overrides.end());
	                    params.tensor_types = &tensor_type_opts;
	                    fprintf(stderr, "%s: selector added %zu exact tensor override(s)\n", __func__, tensor_type_opts.size());
	                }
                    if (selector_chose_seed_keep && selector_overrides.empty() && !patch_base_model.empty()) {
                        std::error_code ec;
                        const bool seed_is_patch_base =
                            selector_checkpoint_path == patch_base_model ||
                            std::filesystem::equivalent(selector_checkpoint_path, patch_base_model, ec);
                        if (seed_is_patch_base) {
                            fprintf(stderr,
                                "%s: selector chose seed_keep; patch-base already contains the measured winner, leaving output unwritten: %s\n",
                                __func__,
                                patch_base_model.c_str());
                            if (selector_checkpoint_created && !selector_checkpoint_path.empty() &&
                                    quantize_control_i64("LLAMA_NVFP4_SELECTOR_KEEP_CHECKPOINT", 0) == 0) {
                                std::error_code remove_ec;
                                std::filesystem::remove(selector_checkpoint_path, remove_ec);
                            }
                            nvfp4_clear_runtime_cfg();
                            llama_backend_free();
                            return 0;
                        }
                    }
	            } else if (!selector_checkpoint_path.empty()) {
                fprintf(stderr, "%s: selector failed, continuing with the built-in CUDA policy search\n", __func__);
                if (quantize_control_i64("LLAMA_NVFP4_SELECTOR_REQUIRE_RUNTIME_CACHE", 0) != 0) {
                    fprintf(stderr, "%s: selector runtime cache is required; aborting quantization\n", __func__);
                    if (selector_checkpoint_created && !selector_checkpoint_path.empty() &&
                            quantize_control_i64("LLAMA_NVFP4_SELECTOR_KEEP_CHECKPOINT", 0) == 0) {
                        std::error_code ec;
                        std::filesystem::remove(selector_checkpoint_path, ec);
                    }
                    llama_backend_free();
                    return 1;
                }
            }
        }
    }

    if (selector_only) {
        if (!selector_enabled) {
            fprintf(stderr, "%s: selector-only requested but selector is not enabled for this invocation\n", __func__);
            llama_backend_free();
            return 1;
        }
        if (!selector_cfg_valid) {
            fprintf(stderr, "%s: selector-only requested but no selector policy was chosen\n", __func__);
            if (selector_checkpoint_created && !selector_checkpoint_path.empty() &&
                    quantize_control_i64("LLAMA_NVFP4_SELECTOR_KEEP_CHECKPOINT", 0) == 0) {
                std::error_code ec;
                std::filesystem::remove(selector_checkpoint_path, ec);
            }
            llama_backend_free();
            return 1;
        }
        if (selector_checkpoint_created && !selector_checkpoint_path.empty() &&
                quantize_control_i64("LLAMA_NVFP4_SELECTOR_KEEP_CHECKPOINT", 0) == 0) {
            std::error_code ec;
            std::filesystem::remove(selector_checkpoint_path, ec);
        }
        nvfp4_clear_runtime_cfg();
        llama_backend_free();
        return 0;
    }

    if (!selector_cfg_valid && cli_nvfp4_cfg_valid) {
        nvfp4_set_runtime_cfg(&cli_nvfp4_cfg);
        selector_cfg_valid = true;
        fprintf(stderr,
            "%s: using CLI NVFP4 policy=%s cfg={choose46=%d refit=%d compand=%d cap6=%.1f cap4=%.1f}\n",
            __func__,
            cli_nvfp4_policy_name.empty() ? "cli" : cli_nvfp4_policy_name.c_str(),
            cli_nvfp4_cfg.choose46_mode,
            cli_nvfp4_cfg.refit_iters,
            cli_nvfp4_cfg.use_compand_sat,
            (double) cli_nvfp4_cfg.cap_m6,
            (double) cli_nvfp4_cfg.cap_m4);
    }

    const int64_t t_main_start_us = llama_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = llama_time_us();

        if (llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), &params)) {
            if (selector_cfg_valid) {
                nvfp4_clear_runtime_cfg();
            }
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = llama_time_us() - t_start_us;
    }

    if (selector_auto_rescue && selector_inputs_ready) {
        nvfp4_cuda_runtime_cfg rescue_cfg = selector_cfg;
        std::string rescue_policy_name;
        std::vector<tensor_type_option> rescue_overrides;

        nvfp4_clear_runtime_cfg();
        llama_backend_free();
        llama_backend_init();

        fprintf(stderr, "%s: selector rescue pass inspecting completed model %s\n", __func__, fname_out.c_str());
        if (run_selector_pass(fname_out, rescue_cfg, rescue_policy_name, nullptr, &rescue_overrides)) {
            if (!rescue_overrides.empty()) {
                const std::string rescue_out = fname_out + ".rescue.tmp.gguf";
                std::vector<tensor_type_option> rescue_tensor_types = tensor_type_opts_cli;
                rescue_tensor_types.insert(rescue_tensor_types.end(), rescue_overrides.begin(), rescue_overrides.end());

                llama_model_quantize_params rescue_params = params;
                rescue_params.tensor_types = &rescue_tensor_types;
                rescue_params.patch_base_model = fname_out.c_str();

                fprintf(stderr,
                    "%s: selector rescue chose policy=%s and %zu exact override(s); patching via %s\n",
                    __func__,
                    rescue_policy_name.c_str(),
                    rescue_overrides.size(),
                    fname_out.c_str());

                {
                    std::error_code ec;
                    std::filesystem::remove(rescue_out, ec);
                }

                bool rescue_ok = false;
                {
                    scoped_quantize_control rescue_running("LLAMA_NVFP4_AUTO_RESCUE_RUNNING", "1");
                    nvfp4_clear_runtime_cfg();
                    rescue_ok = llama_model_quantize(fname_inp.c_str(), rescue_out.c_str(), &rescue_params) == 0;
                }

                if (rescue_ok) {
                    std::error_code ec;
                    std::filesystem::rename(rescue_out, fname_out, ec);
                    if (ec) {
                        fprintf(stderr, "%s: selector rescue produced %s but failed to replace %s: %s\n",
                            __func__, rescue_out.c_str(), fname_out.c_str(), ec.message().c_str());
                    } else {
                        fprintf(stderr, "%s: selector rescue patched final output in-place\n", __func__);
                    }
                } else {
                    std::error_code ec;
                    std::filesystem::remove(rescue_out, ec);
                    fprintf(stderr, "%s: selector rescue patching failed; keeping original output\n", __func__);
                }

            } else {
                fprintf(stderr, "%s: selector rescue found no exact tensor overrides to patch\n", __func__);
            }
        } else {
            fprintf(stderr, "%s: selector rescue analysis failed; keeping original output\n", __func__);
        }

        if (selector_cfg_valid) {
            nvfp4_set_runtime_cfg(&selector_cfg);
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = llama_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
    }

    if (selector_cfg_valid) {
        nvfp4_clear_runtime_cfg();
    }
    if (selector_checkpoint_created && !selector_checkpoint_path.empty() &&
            quantize_control_i64("LLAMA_NVFP4_SELECTOR_KEEP_CHECKPOINT", 0) == 0) {
        std::error_code ec;
        std::filesystem::remove(selector_checkpoint_path, ec);
    } else if (selector_checkpoint_created && !selector_checkpoint_path.empty()) {
        fprintf(stderr, "%s: selector kept candidate search checkpoint %s\n", __func__, selector_checkpoint_path.c_str());
    }

    llama_backend_free();

    return 0;
}
