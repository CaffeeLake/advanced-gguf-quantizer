#include "quantize-selector-runtime.h"

#include "quantize-options.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cinttypes>
#include <sstream>
#include <unordered_map>
#include <utility>

#if defined(__linux__)
#include <sys/sysinfo.h>
#endif

static constexpr double NVFP4_SELECTOR_LOGITS_FALLBACK_GIB = 4.0;
static constexpr double NVFP4_SELECTOR_LOGITS_RESERVE_MIN_GIB = 4.0;
static constexpr double NVFP4_SELECTOR_LOGITS_RESERVE_MAX_GIB = 16.0;
static constexpr double NVFP4_SELECTOR_LOGITS_RESERVE_FRACTION = 0.20;
static constexpr int64_t NVFP4_SELECTOR_PROGRESS_HEARTBEAT_SEC = 30;

static std::atomic<int32_t> g_nvfp4_selector_kld_threads_override{0};
static std::atomic_bool g_nvfp4_selector_skip_requested{false};
static std::string g_nvfp4_selector_skip_file;
static std::mutex g_quantize_control_mutex;
static std::unordered_map<std::string, std::string> g_quantize_controls;

scoped_quantize_control::scoped_quantize_control(const char * key_, const char * value) : key(key_) {
    std::lock_guard<std::mutex> lock(g_quantize_control_mutex);
    const auto old = g_quantize_controls.find(key);
    if (old != g_quantize_controls.end()) {
        had_old = true;
        old_value = old->second;
    }
    if (value != nullptr) {
        g_quantize_controls[key] = value;
    } else {
        g_quantize_controls.erase(key);
    }
}

scoped_quantize_control::~scoped_quantize_control() {
    std::lock_guard<std::mutex> lock(g_quantize_control_mutex);
    if (had_old) {
        g_quantize_controls[key] = old_value;
    } else {
        g_quantize_controls.erase(key);
    }
}

void quantize_control_clear() {
    std::lock_guard<std::mutex> lock(g_quantize_control_mutex);
    g_quantize_controls.clear();
}

void quantize_control_set(const char * key, const char * value) {
    if (key == nullptr || key[0] == '\0') {
        return;
    }
    std::lock_guard<std::mutex> lock(g_quantize_control_mutex);
    if (value != nullptr) {
        g_quantize_controls[key] = value;
    } else {
        g_quantize_controls.erase(key);
    }
}

void quantize_control_unset(const char * key) {
    if (key == nullptr || key[0] == '\0') {
        return;
    }
    std::lock_guard<std::mutex> lock(g_quantize_control_mutex);
    g_quantize_controls.erase(key);
}

bool quantize_control_has(const char * key) {
    if (key == nullptr || key[0] == '\0') {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_quantize_control_mutex);
    return g_quantize_controls.find(key) != g_quantize_controls.end();
}

std::string quantize_control_string(const char * key, std::string fallback) {
    if (key == nullptr || key[0] == '\0') {
        return fallback;
    }
    std::lock_guard<std::mutex> lock(g_quantize_control_mutex);
    const auto value = g_quantize_controls.find(key);
    if (value == g_quantize_controls.end() || value->second.empty()) {
        return fallback;
    }
    return value->second;
}

int64_t quantize_control_i64(const char * key, int64_t fallback) {
    const std::string value = quantize_control_string(key);
    if (value.empty()) {
        return fallback;
    }
    char * end = nullptr;
    const long long parsed = strtoll(value.c_str(), &end, 10);
    if (end == value.c_str() || (end != nullptr && *end != '\0')) {
        return fallback;
    }
    return (int64_t) parsed;
}

double quantize_control_f64(const char * key, double fallback) {
    const std::string value = quantize_control_string(key);
    if (value.empty()) {
        return fallback;
    }
    char * end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || (end != nullptr && *end != '\0') || !std::isfinite(parsed)) {
        return fallback;
    }
    return parsed;
}

bool quantize_control_bool(const char * key, bool fallback) {
    const std::string value = quantize_control_string(key);
    if (value.empty()) {
        return fallback;
    }
    if (striequals(value.c_str(), "1") || striequals(value.c_str(), "on") ||
            striequals(value.c_str(), "true") || striequals(value.c_str(), "yes")) {
        return true;
    }
    if (striequals(value.c_str(), "0") || striequals(value.c_str(), "off") ||
            striequals(value.c_str(), "false") || striequals(value.c_str(), "no")) {
        return false;
    }
    return fallback;
}

ggml_type quantize_control_type(const char * key, ggml_type fallback) {
    const std::string value = quantize_control_string(key);
    if (value.empty()) {
        return fallback;
    }

    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_type type = (ggml_type) i;
        const char * name = ggml_type_name(type);
        if (name != nullptr && striequals(name, value.c_str())) {
            return type;
        }
    }

    fprintf(stderr, "%s: invalid ggml_type '%s' in %s, using %s\n",
        __func__, value.c_str(), key, ggml_type_name(fallback));
    return fallback;
}

int32_t nvfp4_selector_kld_threads_override() {
    return g_nvfp4_selector_kld_threads_override.load(std::memory_order_acquire);
}

void nvfp4_selector_set_kld_threads_override(int32_t threads) {
    g_nvfp4_selector_kld_threads_override.store(threads, std::memory_order_release);
}

void nvfp4_selector_set_skip_state(std::string skip_file, bool skip_requested) {
    g_nvfp4_selector_skip_file = std::move(skip_file);
    g_nvfp4_selector_skip_requested.store(skip_requested, std::memory_order_release);
}

bool nvfp4_selector_skip_requested(const char * phase) {
    if (g_nvfp4_selector_skip_requested.load(std::memory_order_acquire)) {
        return true;
    }
    if (g_nvfp4_selector_skip_file.empty()) {
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::exists(g_nvfp4_selector_skip_file, ec)) {
        return false;
    }
    bool expected = false;
    if (g_nvfp4_selector_skip_requested.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel)) {
        fprintf(stderr,
            "nvfp4_selector: skip remaining tuning requested%s%s via %s; using best available policy after current phase\n",
            phase != nullptr && phase[0] != '\0' ? " during " : "",
            phase != nullptr && phase[0] != '\0' ? phase : "",
            g_nvfp4_selector_skip_file.c_str());
    }
    return true;
}

std::string nvfp4_selector_format_duration(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
        return "unknown";
    }
    const int64_t total = (int64_t) std::llround(seconds);
    const int64_t hours = total / 3600;
    const int64_t minutes = (total / 60) % 60;
    const int64_t secs = total % 60;
    char buf[64];
    if (hours > 0) {
        snprintf(buf, sizeof(buf), "%" PRId64 "h%02" PRId64 "m%02" PRId64 "s", hours, minutes, secs);
    } else if (minutes > 0) {
        snprintf(buf, sizeof(buf), "%" PRId64 "m%02" PRId64 "s", minutes, secs);
    } else {
        snprintf(buf, sizeof(buf), "%" PRId64 "s", secs);
    }
    return buf;
}

double nvfp4_selector_host_mem_available_gib() {
#if defined(__linux__)
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string key;
        uint64_t value_kib = 0;
        std::string unit;
        while (meminfo >> key >> value_kib >> unit) {
            if (key == "MemAvailable:") {
                return (double) value_kib / (1024.0 * 1024.0);
            }
        }
    }

    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        const uint64_t free_units = (uint64_t) info.freeram + (uint64_t) info.bufferram;
        const double bytes = (double) free_units * (double) info.mem_unit;
        return bytes / (1024.0 * 1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

nvfp4_selector_logits_budget nvfp4_selector_default_logits_budget() {
    nvfp4_selector_logits_budget budget;
    budget.available_gib = nvfp4_selector_host_mem_available_gib();
    if (budget.available_gib <= 0.0 || !std::isfinite(budget.available_gib)) {
        budget.max_gib = NVFP4_SELECTOR_LOGITS_FALLBACK_GIB;
        return budget;
    }

    budget.reserve_gib = std::min(
        NVFP4_SELECTOR_LOGITS_RESERVE_MAX_GIB,
        std::max(NVFP4_SELECTOR_LOGITS_RESERVE_MIN_GIB,
            budget.available_gib * NVFP4_SELECTOR_LOGITS_RESERVE_FRACTION));
    budget.max_gib = std::max(NVFP4_SELECTOR_LOGITS_MIN_GIB, budget.available_gib - budget.reserve_gib);
    return budget;
}

nvfp4_selector_progress_heartbeat::nvfp4_selector_progress_heartbeat(std::string label, int64_t total) :
    label_(std::move(label)),
    total_(std::max<int64_t>(0, total)),
    start_(std::chrono::steady_clock::now()),
    last_print_(start_) {
    worker_ = std::thread([this]() { this->run(); });
}

nvfp4_selector_progress_heartbeat::~nvfp4_selector_progress_heartbeat() {
    stop();
}

void nvfp4_selector_progress_heartbeat::update(int64_t completed, int64_t total, std::string detail, bool print_now) {
    std::lock_guard<std::mutex> lock(mu_);
    completed_ = std::max<int64_t>(0, completed);
    if (total >= 0) {
        total_ = total;
    }
    if (!detail.empty()) {
        detail_ = std::move(detail);
    }
    const auto now = std::chrono::steady_clock::now();
    if (print_now || now - last_print_ >= std::chrono::seconds(NVFP4_SELECTOR_PROGRESS_HEARTBEAT_SEC)) {
        print_locked(now);
    }
}

void nvfp4_selector_progress_heartbeat::update(int64_t completed, std::string detail, bool print_now) {
    update(completed, -1, std::move(detail), print_now);
}

void nvfp4_selector_progress_heartbeat::detail(std::string detail, bool print_now) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!detail.empty()) {
        detail_ = std::move(detail);
    }
    const auto now = std::chrono::steady_clock::now();
    if (print_now || now - last_print_ >= std::chrono::seconds(NVFP4_SELECTOR_PROGRESS_HEARTBEAT_SEC)) {
        print_locked(now);
    }
}

void nvfp4_selector_progress_heartbeat::eta_hint(std::string eta, bool print_now) {
    std::lock_guard<std::mutex> lock(mu_);
    eta_hint_ = std::move(eta);
    if (print_now) {
        print_locked(std::chrono::steady_clock::now());
    }
}

void nvfp4_selector_progress_heartbeat::finish(std::string detail) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!detail.empty()) {
            detail_ = std::move(detail);
        }
        if (total_ > 0) {
            completed_ = total_;
        }
        finished_ = true;
        print_locked(std::chrono::steady_clock::now());
    }
    stop();
}

void nvfp4_selector_progress_heartbeat::stop() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stopping_) {
            return;
        }
        stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void nvfp4_selector_progress_heartbeat::run() {
    std::unique_lock<std::mutex> lock(mu_);
    while (!stopping_) {
        cv_.wait_for(lock, std::chrono::seconds(5));
        if (stopping_) {
            break;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_print_ >= std::chrono::seconds(NVFP4_SELECTOR_PROGRESS_HEARTBEAT_SEC)) {
            print_locked(now);
        }
    }
}

void nvfp4_selector_progress_heartbeat::print_locked(std::chrono::steady_clock::time_point now) {
    const double elapsed_s = std::chrono::duration<double>(now - start_).count();
    std::string eta = eta_hint_.empty() ? "TBD..." : eta_hint_;
    if (total_ > 0 && completed_ > 0 && completed_ < total_ && eta_hint_.empty()) {
        eta = nvfp4_selector_format_duration(elapsed_s * (double) (total_ - completed_) / (double) completed_);
    } else if (total_ > 0 && completed_ >= total_) {
        eta = finished_ ? "done" : "TBD...";
    }
    if (total_ > 0) {
        const double pct = 100.0 * (double) std::min(completed_, total_) / (double) total_;
        fprintf(stderr,
            "%s progress: %" PRId64 "/%" PRId64 " %.1f%% elapsed=%s eta=%s%s%s\n",
            label_.c_str(),
            completed_,
            total_,
            pct,
            nvfp4_selector_format_duration(elapsed_s).c_str(),
            eta.c_str(),
            detail_.empty() ? "" : " ",
            detail_.empty() ? "" : detail_.c_str());
    } else {
        fprintf(stderr,
            "%s progress: elapsed=%s eta=%s%s%s\n",
            label_.c_str(),
            nvfp4_selector_format_duration(elapsed_s).c_str(),
            eta.c_str(),
            detail_.empty() ? "" : " ",
            detail_.empty() ? "" : detail_.c_str());
    }
    last_print_ = now;
}
