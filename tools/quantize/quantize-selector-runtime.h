#pragma once

#include "llama.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

struct scoped_quantize_control {
    std::string key;
    bool had_old = false;
    std::string old_value;

    scoped_quantize_control(const char * key, const char * value);
    ~scoped_quantize_control();
};

void quantize_control_clear();
void quantize_control_set(const char * key, const char * value);
void quantize_control_unset(const char * key);
bool quantize_control_has(const char * key);
std::string quantize_control_string(const char * key, std::string fallback = {});
int64_t quantize_control_i64(const char * key, int64_t fallback);
double quantize_control_f64(const char * key, double fallback);
bool quantize_control_bool(const char * key, bool fallback);
ggml_type quantize_control_type(const char * key, ggml_type fallback);

int32_t nvfp4_selector_kld_threads_override();
void nvfp4_selector_set_kld_threads_override(int32_t threads);
void nvfp4_selector_set_skip_state(std::string skip_file, bool skip_requested);
bool nvfp4_selector_skip_requested(const char * phase);

struct nvfp4_selector_logits_budget {
    double max_gib = 4.0;
    double available_gib = 0.0;
    double reserve_gib = 0.0;
};

static constexpr double NVFP4_SELECTOR_LOGITS_MIN_GIB = 0.75;
static constexpr bool NVFP4_SELECTOR_DEDUP_SURVEY_DEFAULT = true;
static constexpr bool NVFP4_SELECTOR_DEDUP_EVAL_DEFAULT = true;
static constexpr double NVFP4_SELECTOR_CORRECTION_DENOM_DEFAULT = 6.0 * 448.0;

std::string nvfp4_selector_format_duration(double seconds);
double nvfp4_selector_host_mem_available_gib();
nvfp4_selector_logits_budget nvfp4_selector_default_logits_budget();

class nvfp4_selector_progress_heartbeat {
public:
    explicit nvfp4_selector_progress_heartbeat(std::string label, int64_t total = 0);
    ~nvfp4_selector_progress_heartbeat();

    nvfp4_selector_progress_heartbeat(const nvfp4_selector_progress_heartbeat &) = delete;
    nvfp4_selector_progress_heartbeat & operator=(const nvfp4_selector_progress_heartbeat &) = delete;

    void update(int64_t completed, int64_t total, std::string detail = {}, bool print_now = false);
    void update(int64_t completed, std::string detail = {}, bool print_now = false);
    void detail(std::string detail, bool print_now = false);
    void finish(std::string detail = {});

private:
    void stop();
    void run();
    void print_locked(std::chrono::steady_clock::time_point now);

    std::string label_;
    int64_t completed_ = 0;
    int64_t total_ = 0;
    std::string detail_;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_print_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::thread worker_;
    bool stopping_ = false;
    bool finished_ = false;
};
