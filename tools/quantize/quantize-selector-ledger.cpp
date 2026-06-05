#include "quantize-selector-ledger.h"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <utility>

selector_ledger::selector_ledger(std::string path) {
    configure(std::move(path));
}

void selector_ledger::configure(std::string path) {
    std::lock_guard<std::mutex> lock(mutex_);
    path_ = std::move(path);
    local_metrics_loaded_ = false;
    local_metrics_.clear();
}

bool selector_ledger::enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !path_.empty();
}

const std::string & selector_ledger::path() const {
    return path_;
}

bool selector_ledger::metric_from_json(const nlohmann::ordered_json & row, local_metric & out) {
    if (!row.is_object()) {
        return false;
    }
    out.sum_sq = row.value("sum_sq", 0.0);
    out.sum_abs = row.value("sum_abs", 0.0);
    out.max_abs = row.value("max_abs", 0.0);
    out.proxy_score = row.value("proxy_score", 0.0);
    out.count = row.value("count", (int64_t) 0);
    out.proxy_ok = row.value("proxy_ok", false);
    if (const auto rsf_it = row.find("rsf"); rsf_it != row.end() && rsf_it->is_object()) {
        out.rsf_changed = rsf_it->value("changed", false);
        out.rsf_slices = rsf_it->value("slices", (int64_t) 0);
        out.rsf_scale_mul_count = rsf_it->value("scale_mul_count", (int64_t) 0);
        out.rsf_scale_mul_mean = rsf_it->value("scale_mul_mean", 0.0);
    }
    return out.count > 0;
}

std::string selector_ledger::key_string(const nlohmann::ordered_json & key) {
    return key.dump();
}

bool selector_ledger::load_local_metrics() {
    std::lock_guard<std::mutex> lock(mutex_);
    local_metrics_.clear();
    local_metrics_loaded_ = true;
    if (path_.empty()) {
        return true;
    }

    std::ifstream in(path_);
    if (!in) {
        return true;
    }

    std::string line;
    size_t loaded = 0;
    size_t ignored = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        try {
            const auto row = nlohmann::ordered_json::parse(line);
            if (!row.is_object() ||
                    row.value("schema", std::string()) != schema ||
                    row.value("row_type", std::string()) != "tensor.local_metric") {
                ++ignored;
                continue;
            }
            const auto key_it = row.find("key");
            if (key_it == row.end()) {
                ++ignored;
                continue;
            }
            local_metric metric;
            if (!metric_from_json(row, metric)) {
                ++ignored;
                continue;
            }
            local_metrics_[key_string(*key_it)] = metric;
            ++loaded;
        } catch (const std::exception &) {
            ++ignored;
        }
    }
    std::fprintf(stderr,
        "%s: selector ledger loaded local_metrics=%zu ignored=%zu path=%s\n",
        __func__, loaded, ignored, path_.c_str());
    return true;
}

bool selector_ledger::find_local_metric(const nlohmann::ordered_json & key, local_metric & out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!local_metrics_loaded_) {
        return false;
    }
    const auto it = local_metrics_.find(key_string(key));
    if (it == local_metrics_.end()) {
        return false;
    }
    out = it->second;
    return true;
}

bool selector_ledger::append_context(const char * event, nlohmann::ordered_json payload) const {
    if (!payload.is_object()) {
        nlohmann::ordered_json wrapped = nlohmann::ordered_json::object();
        wrapped["value"] = std::move(payload);
        payload = std::move(wrapped);
    }
    if (event != nullptr && event[0] != '\0') {
        payload["event"] = event;
    }
    return append_row("context", std::move(payload));
}

bool selector_ledger::append_tensor_local_metric(nlohmann::ordered_json payload) const {
    return append_row("tensor.local_metric", std::move(payload));
}

bool selector_ledger::append_exact_eval(nlohmann::ordered_json payload) const {
    return append_row("exact_eval", std::move(payload));
}

bool selector_ledger::append_row(const char * row_type, nlohmann::ordered_json payload) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (path_.empty()) {
        return true;
    }
    if (!payload.is_object()) {
        nlohmann::ordered_json wrapped = nlohmann::ordered_json::object();
        wrapped["value"] = std::move(payload);
        payload = std::move(wrapped);
    }

    std::filesystem::path out_path(path_);
    if (out_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
        if (ec) {
            std::fprintf(stderr,
                "%s: selector ledger directory unavailable %s: %s\n",
                __func__,
                out_path.parent_path().string().c_str(),
                ec.message().c_str());
            return false;
        }
    }

    nlohmann::ordered_json record = nlohmann::ordered_json::object();
    record["schema"] = schema;
    record["schema_version"] = schema_version;
    record["row_type"] = row_type != nullptr && row_type[0] != '\0' ? row_type : "unknown";
    for (auto it = payload.begin(); it != payload.end(); ++it) {
        record[it.key()] = *it;
    }

    std::ofstream out(path_, std::ios::app);
    if (!out) {
        std::fprintf(stderr, "%s: selector ledger append failed path=%s\n", __func__, path_.c_str());
        return false;
    }
    out << record.dump() << '\n';
    return (bool) out;
}
