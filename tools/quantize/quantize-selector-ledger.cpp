#include "quantize-selector-ledger.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <utility>

nvfp4_selector_ledger::nvfp4_selector_ledger(std::string path) {
    configure(std::move(path));
}

void nvfp4_selector_ledger::configure(std::string path) {
    std::lock_guard<std::mutex> lock(mutex_);
    path_ = std::move(path);
}

bool nvfp4_selector_ledger::enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !path_.empty();
}

const std::string & nvfp4_selector_ledger::path() const {
    return path_;
}

bool nvfp4_selector_ledger::append_context(const char * event, nlohmann::ordered_json payload) const {
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

bool nvfp4_selector_ledger::append_tensor_local_metric(nlohmann::ordered_json payload) const {
    return append_row("tensor.local_metric", std::move(payload));
}

bool nvfp4_selector_ledger::append_exact_eval(nlohmann::ordered_json payload) const {
    return append_row("exact_eval", std::move(payload));
}

bool nvfp4_selector_ledger::append_row(const char * row_type, nlohmann::ordered_json payload) const {
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
