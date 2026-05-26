#pragma once

#include "ggml-cuda.h"
#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct nvfp4_selector_kld_subset {
    struct mmap_file {
        int fd = -1;
        size_t size = 0;
        const uint8_t * data = nullptr;

        ~mmap_file();
    };

    int32_t n_ctx = 0;
    int32_t n_vocab = 0;
    int32_t n_chunk_total = 0;
    int32_t chunk_start = 0;
    int32_t n_chunk = 0;
    int32_t first = 0;
    int32_t n_score = 0;
    int32_t nv = 0;
    std::string source_path;
    uintmax_t source_size = 0;
    int64_t source_mtime = 0;
    std::vector<llama_token> tokens;
    std::vector<uint16_t> log_probs_u16;
    std::shared_ptr<mmap_file> log_probs_mapping;
    const uint16_t * log_probs_u16_mapped = nullptr;
};

struct nvfp4_selector_kld_metrics {
    double sum_nll = 0.0;
    double sum_nll2 = 0.0;
    double sum_nll_base = 0.0;
    double sum_nll_base2 = 0.0;
    double sum_nll_nll_base = 0.0;
    double sum_kld = 0.0;
    double sum_kld2 = 0.0;
    double max_kld = 0.0;
    double sum_p_diff2 = 0.0;
    double sum_p_diff4 = 0.0;
    double sum_entropy_diff2 = 0.0;
    double sum_top_prob_diff2 = 0.0;
    double sum_top_flip_weight = 0.0;
    int64_t same_top = 0;
    int64_t count = 0;
    bool collect_kld_values = false;
    std::vector<double> kld_values;
};

struct nvfp4_selector_derived_metrics {
    bool ok = false;
    double ppl_q = 0.0;
    double ppl_base = 0.0;
    double ln_ratio = 0.0;
    double ln_ratio_unc = 0.0;
    double mean_kld = 0.0;
    double mean_kld_unc = 0.0;
    double kld_p95 = 0.0;
    double kld_p99 = 0.0;
    double kld_p999 = 0.0;
    double kld_tail_mean = 0.0;
    double max_kld = 0.0;
    double rms_dp = 0.0;
    double rms_dp_unc = 0.0;
    double same_top = 0.0;
    double same_top_unc = 0.0;
    double entropy_rmse = 0.0;
    double top_prob_rmse = 0.0;
    double top_flip_weight = 0.0;
};

const uint16_t * nvfp4_selector_kld_log_probs_data(const nvfp4_selector_kld_subset & kld);
bool nvfp4_selector_load_kld_subset(const std::string & path, int32_t chunk_start, int32_t n_chunks, nvfp4_selector_kld_subset & out);
nvfp4_selector_kld_subset nvfp4_selector_make_kld_budget_subset(const nvfp4_selector_kld_subset & src, int32_t max_chunks);

void nvfp4_selector_eval_one_token(
    int n_vocab,
    const float * logits,
    const uint16_t * base_logp_u16,
    llama_token tok,
    nvfp4_selector_kld_metrics & m);

void nvfp4_selector_merge_kld_metrics(nvfp4_selector_kld_metrics & dst, nvfp4_selector_kld_metrics && src);
void nvfp4_selector_merge_cuda_kld_metrics(
    nvfp4_selector_kld_metrics & dst,
    const nvfp4_cuda_kld_result & src,
    std::vector<double> && kld_values);

std::pair<double, double> nvfp4_selector_mean_and_uncertainty(double sum, double sum2, int64_t count);
nvfp4_selector_derived_metrics nvfp4_selector_derive_metrics(const nvfp4_selector_kld_metrics & km);
