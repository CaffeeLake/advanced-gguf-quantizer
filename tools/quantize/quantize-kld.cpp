#include "quantize-kld.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <system_error>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

selector_kld_subset::mmap_file::~mmap_file() {
#if defined(__unix__) || defined(__APPLE__)
    if (data != nullptr && size > 0) {
        munmap(const_cast<uint8_t *>(data), size);
    }
    if (fd >= 0) {
        close(fd);
    }
#endif
}

const uint16_t * selector_kld_log_probs_data(const selector_kld_subset & kld) {
    return kld.log_probs_u16_mapped != nullptr ? kld.log_probs_u16_mapped : kld.log_probs_u16.data();
}

static std::shared_ptr<selector_kld_subset::mmap_file> selector_try_mmap_kld_file(const std::string & path) {
#if defined(__unix__) || defined(__APPLE__)
    std::error_code ec;
    const uintmax_t file_size_u = std::filesystem::file_size(path, ec);
    if (ec || file_size_u == 0 || file_size_u > (uintmax_t) std::numeric_limits<size_t>::max()) {
        fprintf(stderr, "%s: could not stat selector kld file for mmap, falling back to read: %s\n",
            __func__, path.c_str());
        return nullptr;
    }

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "%s: could not open selector kld file for mmap, falling back to read: %s\n",
            __func__, path.c_str());
        return nullptr;
    }

    const size_t file_size = (size_t) file_size_u;
    void * mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        fprintf(stderr, "%s: mmap failed for selector kld file, falling back to read: %s\n",
            __func__, path.c_str());
        return nullptr;
    }

    auto out = std::make_shared<selector_kld_subset::mmap_file>();
    out->fd = fd;
    out->size = file_size;
    out->data = (const uint8_t *) mapped;
    return out;
#else
    (void) path;
    return nullptr;
#endif
}

bool selector_load_kld_subset(const std::string & path, selector_kld_subset & out) {
    out.source_path = path;
    std::error_code file_ec;
    out.source_size = std::filesystem::file_size(path, file_ec);
    if (file_ec) {
        out.source_size = 0;
    }
    const auto write_time = std::filesystem::last_write_time(path, file_ec);
    out.source_mtime = file_ec ? 0 : write_time.time_since_epoch().count();

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "%s: failed to open selector kld file: %s\n", __func__, path.c_str());
        return false;
    }

    char magic[8];
    in.read(magic, 8);
    if (!in || memcmp(magic, "_logits_", 8) != 0) {
        fprintf(stderr, "%s: invalid selector kld magic in %s\n", __func__, path.c_str());
        return false;
    }

    in.read((char *) &out.n_ctx, sizeof(out.n_ctx));
    in.read((char *) &out.n_vocab, sizeof(out.n_vocab));
    in.read((char *) &out.n_chunk_total, sizeof(out.n_chunk_total));
    if (!in || out.n_ctx <= 0 || out.n_vocab <= 0 || out.n_chunk_total <= 0) {
        fprintf(stderr, "%s: invalid selector kld header\n", __func__);
        return false;
    }

    out.n_chunk = out.n_chunk_total;
    out.first = out.n_ctx / 2;
    out.n_score = out.n_ctx - 1 - out.first;
    out.nv = 2 * ((out.n_vocab + 1) / 2) + 4;

    const std::streamoff header_sz = 8 + 3 * (std::streamoff) sizeof(int32_t);
    const std::streamoff tokens_all_sz = (std::streamoff) out.n_chunk_total * out.n_ctx * (std::streamoff) sizeof(llama_token);
    const std::streamoff chunk_logp_sz = (std::streamoff) out.n_score * out.nv * (std::streamoff) sizeof(uint16_t);

    out.tokens.resize((size_t) out.n_chunk * (size_t) out.n_ctx);

    in.seekg(header_sz, std::ios::beg);
    in.read((char *) out.tokens.data(), (std::streamsize) out.tokens.size() * (std::streamsize) sizeof(llama_token));
    if (!in) {
        fprintf(stderr, "%s: failed reading selector tokens\n", __func__);
        return false;
    }

    in.seekg(header_sz + tokens_all_sz, std::ios::beg);
    const std::streamoff logp_offset = header_sz + tokens_all_sz;
    const std::streamoff logp_bytes = (std::streamoff) out.n_chunk * chunk_logp_sz;
    out.log_probs_mapping = selector_try_mmap_kld_file(path);
    if (out.log_probs_mapping != nullptr &&
            logp_offset >= 0 &&
            logp_bytes >= 0 &&
            (uintmax_t) logp_offset + (uintmax_t) logp_bytes <= (uintmax_t) out.log_probs_mapping->size) {
        out.log_probs_u16.clear();
        out.log_probs_u16.shrink_to_fit();
        out.log_probs_u16_mapped = (const uint16_t *) (out.log_probs_mapping->data + (size_t) logp_offset);
        fprintf(stderr, "%s: selector mmap full kld chunks=%d ctx=%d vocab=%d virtual=%.2f GiB\n",
            __func__, out.n_chunk, out.n_ctx, out.n_vocab,
            (double) out.log_probs_mapping->size / (1024.0 * 1024.0 * 1024.0));
    } else {
        if (out.log_probs_mapping != nullptr) {
            fprintf(stderr, "%s: selector kld mmap range check failed, falling back to read\n", __func__);
            out.log_probs_mapping.reset();
        }
        out.log_probs_u16.resize((size_t) out.n_chunk * (size_t) out.n_score * (size_t) out.nv);
        in.seekg(logp_offset, std::ios::beg);
        in.read((char *) out.log_probs_u16.data(), (std::streamsize) out.log_probs_u16.size() * (std::streamsize) sizeof(uint16_t));
        if (!in) {
            fprintf(stderr, "%s: failed reading selector log-probs\n", __func__);
            return false;
        }
    }

    return true;
}

void selector_eval_one_token(
    int n_vocab,
    const float * logits,
    const uint16_t * base_logp_u16,
    llama_token tok,
    selector_kld_metrics & m) {
    float max_l = logits[0];
    int i_max = 0;

    float scale = 0.0f;
    float min_log_prob = 0.0f;
    memcpy(&scale, base_logp_u16 + 0, sizeof(float));
    memcpy(&min_log_prob, base_logp_u16 + 2, sizeof(float));
    if (!(scale >= 0.0f) || !std::isfinite(scale)) {
        scale = 0.0f;
    }
    if (!std::isfinite(min_log_prob)) {
        min_log_prob = -16.0f;
    }

    const uint16_t * idx = base_logp_u16 + 4;
    const double nll_base = -(scale * idx[tok] + min_log_prob);

    int i_max_base = 0;
    float p_log_base_max = scale * idx[0] + min_log_prob;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_l) {
            max_l = logits[i];
            i_max = i;
        }
        const float p_log_base = scale * idx[i] + min_log_prob;
        if (p_log_base > p_log_base_max) {
            p_log_base_max = p_log_base;
            i_max_base = i;
        }
    }

    double sum_exp = 0.0;
    double sum_exp_shifted_logit = 0.0;
    double kld = 0.0;
    double p_base_sum = 0.0;
    double base_entropy = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        const double exp_shifted = expf(logits[i] - max_l);
        sum_exp += exp_shifted;
        sum_exp_shifted_logit += exp_shifted * (double) (logits[i] - max_l);
        const float p_log_base = scale * idx[i] + min_log_prob;
        if (p_log_base > -16.0f) {
            const double p_base = expf(p_log_base);
            p_base_sum += p_base;
            base_entropy -= p_base * p_log_base;
            kld += p_base * (p_log_base - logits[i] + max_l);
        }
    }
    const double log_sum_exp = log(sum_exp);
    kld += p_base_sum * log_sum_exp;
    const double nll = max_l + log_sum_exp - logits[tok];

    const double p_q = exp(-nll);
    const double p_base_tok = exp(-nll_base);
    const double p_diff = p_q - p_base_tok;
    const double p_diff2 = p_diff * p_diff;
    const double base_norm = std::max(p_base_sum, 1e-300);
    const double q_entropy = log_sum_exp - sum_exp_shifted_logit / std::max(sum_exp, 1e-300);
    const double base_entropy_norm = base_entropy / base_norm + std::log(base_norm);
    const double entropy_diff = q_entropy - base_entropy_norm;
    const double q_prob_base_top = exp((double) logits[i_max_base] - (double) max_l - log_sum_exp);
    const double base_top_prob = exp((double) p_log_base_max) / base_norm;
    const double base_prob_q_top = exp((double) (scale * idx[i_max] + min_log_prob)) / base_norm;
    const double top_prob_diff = q_prob_base_top - base_top_prob;
    const double top_flip_weight = i_max == i_max_base ? 0.0 : std::max(0.0, base_top_prob - base_prob_q_top);

    m.sum_nll += nll;
    m.sum_nll2 += nll * nll;
    m.sum_nll_base += nll_base;
    m.sum_nll_base2 += nll_base * nll_base;
    m.sum_nll_nll_base += nll * nll_base;
    m.sum_kld += kld;
    m.sum_kld2 += kld * kld;
    m.max_kld = std::max(m.max_kld, kld);
    if (m.collect_kld_values) {
        m.kld_values.push_back(kld);
    }
    m.sum_p_diff2 += p_diff2;
    m.sum_p_diff4 += p_diff2 * p_diff2;
    m.sum_entropy_diff2 += entropy_diff * entropy_diff;
    m.sum_top_prob_diff2 += top_prob_diff * top_prob_diff;
    m.sum_top_flip_weight += top_flip_weight;
    m.same_top += (i_max == i_max_base);
    m.count++;
}

void selector_merge_kld_metrics(selector_kld_metrics & dst, selector_kld_metrics && src) {
    dst.sum_nll += src.sum_nll;
    dst.sum_nll2 += src.sum_nll2;
    dst.sum_nll_base += src.sum_nll_base;
    dst.sum_nll_base2 += src.sum_nll_base2;
    dst.sum_nll_nll_base += src.sum_nll_nll_base;
    dst.sum_kld += src.sum_kld;
    dst.sum_kld2 += src.sum_kld2;
    dst.max_kld = std::max(dst.max_kld, src.max_kld);
    dst.sum_p_diff2 += src.sum_p_diff2;
    dst.sum_p_diff4 += src.sum_p_diff4;
    dst.sum_entropy_diff2 += src.sum_entropy_diff2;
    dst.sum_top_prob_diff2 += src.sum_top_prob_diff2;
    dst.sum_top_flip_weight += src.sum_top_flip_weight;
    dst.same_top += src.same_top;
    dst.count += src.count;
    if (dst.collect_kld_values && !src.kld_values.empty()) {
        dst.kld_values.insert(dst.kld_values.end(),
            std::make_move_iterator(src.kld_values.begin()),
            std::make_move_iterator(src.kld_values.end()));
    }
}

void selector_merge_cuda_kld_metrics(
        selector_kld_metrics & dst,
        const nvfp4_cuda_kld_result & src,
        std::vector<double> && kld_values) {
    dst.sum_nll += src.sum_nll;
    dst.sum_nll2 += src.sum_nll2;
    dst.sum_nll_base += src.sum_nll_base;
    dst.sum_nll_base2 += src.sum_nll_base2;
    dst.sum_nll_nll_base += src.sum_nll_nll_base;
    dst.sum_kld += src.sum_kld;
    dst.sum_kld2 += src.sum_kld2;
    dst.max_kld = std::max(dst.max_kld, src.max_kld);
    dst.sum_p_diff2 += src.sum_p_diff2;
    dst.sum_p_diff4 += src.sum_p_diff4;
    dst.sum_entropy_diff2 += src.sum_entropy_diff2;
    dst.sum_top_prob_diff2 += src.sum_top_prob_diff2;
    dst.sum_top_flip_weight += src.sum_top_flip_weight;
    dst.same_top += src.same_top;
    dst.count += src.count;
    if (dst.collect_kld_values && !kld_values.empty()) {
        dst.kld_values.insert(dst.kld_values.end(),
            std::make_move_iterator(kld_values.begin()),
            std::make_move_iterator(kld_values.end()));
    }
}

std::pair<double, double> selector_mean_and_uncertainty(double sum, double sum2, int64_t count) {
    if (count < 1) {
        return { 0.0, 0.0 };
    }
    const double mean = sum / (double) count;
    double unc = sum2 / (double) count - mean * mean;
    unc = unc > 0.0 && count > 10 ? std::sqrt(unc / (double) (count - 1)) : 0.0;
    return { mean, unc };
}

static double selector_covariance(double suma, double sumb, double sumab, int64_t count) {
    if (count < 10) {
        return 0.0;
    }
    double cov = sumab / (double) count - (suma / (double) count) * (sumb / (double) count);
    cov /= (double) (count - 1);
    return cov;
}

static double selector_percentile_sorted(const std::vector<double> & values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    if (fraction <= 0.0) {
        return values.front();
    }
    if (fraction >= 1.0) {
        return values.back();
    }
    double p = fraction * (double) (values.size() - 1);
    const size_t ip = (size_t) p;
    p -= (double) ip;
    return (1.0 - p) * values[ip] + p * values[std::min(ip + 1, values.size() - 1)];
}

static double selector_tail_mean_sorted(const std::vector<double> & values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    size_t begin = (size_t) std::floor(std::clamp(fraction, 0.0, 1.0) * (double) (values.size() - 1));
    begin = std::min(begin, values.size() - 1);
    double sum = 0.0;
    for (size_t i = begin; i < values.size(); ++i) {
        sum += values[i];
    }
    return sum / (double) (values.size() - begin);
}

selector_derived_metrics selector_derive_metrics(const selector_kld_metrics & km) {
    selector_derived_metrics d;
    if (km.count <= 0) {
        return d;
    }
    d.ok = true;
    const auto log_ppl = selector_mean_and_uncertainty(km.sum_nll, km.sum_nll2, km.count);
    const auto log_ppl_base = selector_mean_and_uncertainty(km.sum_nll_base, km.sum_nll_base2, km.count);
    const double log_ppl_cov = selector_covariance(km.sum_nll, km.sum_nll_base, km.sum_nll_nll_base, km.count);
    d.ppl_q = exp(log_ppl.first);
    d.ppl_base = exp(log_ppl_base.first);
    d.ln_ratio = log_ppl.first - log_ppl_base.first;
    d.ln_ratio_unc = std::sqrt(std::max(0.0,
        log_ppl.second * log_ppl.second +
        log_ppl_base.second * log_ppl_base.second -
        2.0 * log_ppl_cov));
    const auto kld = selector_mean_and_uncertainty(km.sum_kld, km.sum_kld2, km.count);
    d.mean_kld = kld.first;
    d.mean_kld_unc = kld.second;
    if (!km.kld_values.empty()) {
        std::vector<double> sorted_kld = km.kld_values;
        std::sort(sorted_kld.begin(), sorted_kld.end());
        d.kld_p95 = selector_percentile_sorted(sorted_kld, 0.95);
        d.kld_p99 = selector_percentile_sorted(sorted_kld, 0.99);
        d.kld_p999 = selector_percentile_sorted(sorted_kld, 0.999);
        d.kld_tail_mean = selector_tail_mean_sorted(sorted_kld, 0.99);
    } else {
        d.kld_p95 = km.max_kld;
        d.kld_p99 = km.max_kld;
        d.kld_p999 = km.max_kld;
        d.kld_tail_mean = km.max_kld;
    }
    d.max_kld = km.max_kld;
    const auto p_diff_mse = selector_mean_and_uncertainty(km.sum_p_diff2, km.sum_p_diff4, km.count);
    d.rms_dp = std::sqrt(std::max(0.0, p_diff_mse.first));
    d.rms_dp_unc = d.rms_dp > 0.0 ? (0.5 / d.rms_dp) * p_diff_mse.second : 0.0;
    d.same_top = (double) km.same_top / km.count;
    d.same_top_unc = km.count > 1 ? std::sqrt(std::max(0.0, d.same_top * (1.0 - d.same_top) / (double) (km.count - 1))) : 0.0;
    d.entropy_rmse = std::sqrt(std::max(0.0, km.sum_entropy_diff2 / (double) km.count));
    d.top_prob_rmse = std::sqrt(std::max(0.0, km.sum_top_prob_diff2 / (double) km.count));
    d.top_flip_weight = km.sum_top_flip_weight / (double) km.count;
    d.ok =
        std::isfinite(d.ppl_q) &&
        std::isfinite(d.ppl_base) &&
        std::isfinite(d.ln_ratio) &&
        std::isfinite(d.mean_kld) &&
        std::isfinite(d.kld_p95) &&
        std::isfinite(d.kld_p99) &&
        std::isfinite(d.kld_p999) &&
        std::isfinite(d.kld_tail_mean) &&
        std::isfinite(d.max_kld) &&
        std::isfinite(d.rms_dp) &&
        std::isfinite(d.same_top) &&
        std::isfinite(d.entropy_rmse) &&
        std::isfinite(d.top_prob_rmse) &&
        std::isfinite(d.top_flip_weight);
    return d;
}
