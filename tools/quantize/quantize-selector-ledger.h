#pragma once

#include "nlohmann/json.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

class selector_ledger {
public:
    static constexpr const char * schema = "advanced-gguf-selector-ledger-v1";
    static constexpr int schema_version = 1;

    selector_ledger() = default;
    explicit selector_ledger(std::string path);

    struct local_metric {
        double sum_sq = 0.0;
        double sum_abs = 0.0;
        double max_abs = 0.0;
        double proxy_score = 0.0;
        int64_t count = 0;
        bool proxy_ok = false;
        bool rsf_changed = false;
        int64_t rsf_slices = 0;
        int64_t rsf_scale_mul_count = 0;
        double rsf_scale_mul_mean = 0.0;
    };

    void configure(std::string path);
    bool enabled() const;
    const std::string & path() const;

    bool load_local_metrics();
    bool find_local_metric(const nlohmann::ordered_json & key, local_metric & out) const;

    bool append_context(const char * event, nlohmann::ordered_json payload = nlohmann::ordered_json::object()) const;
    bool append_tensor_local_metric(nlohmann::ordered_json payload) const;
    bool append_exact_eval(nlohmann::ordered_json payload) const;
    bool append_row(const char * row_type, nlohmann::ordered_json payload) const;

private:
    static bool metric_from_json(const nlohmann::ordered_json & row, local_metric & out);
    static std::string key_string(const nlohmann::ordered_json & key);

    std::string path_;
    mutable bool local_metrics_loaded_ = false;
    mutable std::unordered_map<std::string, local_metric> local_metrics_;
    mutable std::mutex mutex_;
};
