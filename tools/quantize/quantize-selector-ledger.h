#pragma once

#include "nlohmann/json.hpp"

#include <mutex>
#include <string>

class nvfp4_selector_ledger {
public:
    static constexpr const char * schema = "advanced-gguf-selector-ledger-v1";
    static constexpr int schema_version = 1;

    nvfp4_selector_ledger() = default;
    explicit nvfp4_selector_ledger(std::string path);

    void configure(std::string path);
    bool enabled() const;
    const std::string & path() const;

    bool append_context(const char * event, nlohmann::ordered_json payload = nlohmann::ordered_json::object()) const;
    bool append_tensor_local_metric(nlohmann::ordered_json payload) const;
    bool append_exact_eval(nlohmann::ordered_json payload) const;
    bool append_row(const char * row_type, nlohmann::ordered_json payload) const;

private:
    std::string path_;
    mutable std::mutex mutex_;
};
