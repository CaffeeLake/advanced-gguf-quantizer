#include "run_monitor.h"

#include "shell_workflow.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <ctime>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

namespace bq {
namespace {

static std::string trim_copy_simple(const std::string & value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace((unsigned char) value[begin])) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace((unsigned char) value[end - 1])) {
        --end;
    }
    return value.substr(begin, end - begin);
}

static std::string format_elapsed_seconds(int64_t seconds) {
    if (seconds < 0) {
        seconds = 0;
    }
    const int64_t hours = seconds / 3600;
    const int64_t minutes = (seconds / 60) % 60;
    const int64_t secs = seconds % 60;
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

} // namespace

std::string now_iso_like() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return buf;
}

std::string read_text_file_trimmed(const std::filesystem::path & path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    std::ostringstream out;
    out << in.rdbuf();
    return trim_copy_simple(out.str());
}

std::vector<std::string> read_last_lines(const std::filesystem::path & path, size_t max_lines) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
        if (lines.size() > max_lines) {
            lines.erase(lines.begin(), lines.begin() + (std::ptrdiff_t) (lines.size() - max_lines));
        }
    }
    return lines;
}

uint64_t file_size_or_zero(const std::string & path) {
    if (path.empty()) {
        return 0;
    }
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(path, ec);
    return ec ? 0 : size;
}

std::string shell_extract_eta_progress(const std::string & line) {
    const std::string progress_marker = " progress:";
    const size_t progress_pos = line.find(progress_marker);
    if (progress_pos == std::string::npos || line.find("eta=") == std::string::npos) {
        return {};
    }
    std::string label = trim_copy_simple(line.substr(0, progress_pos));
    std::string body = trim_copy_simple(line.substr(progress_pos + progress_marker.size()));
    const size_t detail_pos = body.find(" chunk ");
    if (detail_pos != std::string::npos) {
        body = body.substr(0, detail_pos);
    }
    const size_t complete_pos = body.find(" complete");
    if (complete_pos != std::string::npos) {
        body = body.substr(0, complete_pos);
    }
    if (label.size() > 34) {
        label = "..." + label.substr(label.size() - 31);
    }
    if (body.size() > 92) {
        body = body.substr(0, 89) + "...";
    }
    return label.empty() ? body : label + " " + body;
}

std::string shell_latest_progress_line(const std::vector<std::string> & log_lines) {
    for (auto it = log_lines.rbegin(); it != log_lines.rend(); ++it) {
        const std::string progress = shell_extract_eta_progress(*it);
        if (!progress.empty()) {
            return progress;
        }
    }
    return {};
}

std::string shell_executable_path() {
#if !defined(_WIN32)
    std::error_code ec;
    const std::filesystem::path path = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec && !path.empty()) {
        return path.string();
    }
#endif
    return PRODUCT_COMMAND;
}

bool shell_pid_alive(const std::filesystem::path & pid_file) {
    const std::string pid = read_text_file_trimmed(pid_file);
    if (pid.empty() || !std::all_of(pid.begin(), pid.end(), [](unsigned char c) { return std::isdigit(c); })) {
        return false;
    }
#if !defined(_WIN32)
    std::error_code ec;
    return std::filesystem::exists(std::filesystem::path("/proc") / pid, ec);
#else
    return true;
#endif
}

bool shell_run_finished(const std::filesystem::path & run_dir, std::string * rc_out) {
    const std::vector<std::string> events = read_last_lines(run_dir / "run.jsonl", 80);
    for (auto it = events.rbegin(); it != events.rend(); ++it) {
        const std::string & line = *it;
        if (line.find("\"event\":\"finished\"") == std::string::npos) {
            continue;
        }
        if (rc_out != nullptr) {
            const std::string key = "\"return_code\":";
            const size_t pos = line.find(key);
            if (pos != std::string::npos) {
                size_t begin = pos + key.size();
                size_t end = begin;
                while (end < line.size() && (std::isdigit((unsigned char) line[end]) || line[end] == '-')) {
                    ++end;
                }
                *rc_out = line.substr(begin, end - begin);
            }
        }
        return true;
    }
    return false;
}

bool is_tool_status_line(const std::string & line) {
    return line.find(std::string(PRODUCT_COMMAND) + " status:") != std::string::npos;
}

std::string shell_infer_run_phase(const std::vector<std::string> & log_lines) {
    for (auto it = log_lines.rbegin(); it != log_lines.rend(); ++it) {
        const std::string & line = *it;
        if (line.empty() || is_tool_status_line(line)) {
            continue;
        }
        if (line.find("selector Best[") != std::string::npos) {
            return "Selecting measured best result";
        }
        if (line.find("selector stage-b policy=") != std::string::npos) {
            return "Stage-B measured candidate measured";
        }
        if (line.find("selector stage-b start") != std::string::npos) {
            return "Stage-B measured candidate running";
        }
        if (line.find("selector kld eval chunk") != std::string::npos) {
            return "Stage-B PPL/KLD chunk scoring";
        }
        if (line.find("selector stage-b baseline") != std::string::npos) {
            return "Stage-B measured baseline measured";
        }
        if (line.find("MXFP6_E2M3 scale refine") != std::string::npos ||
                line.find("selector-mxfp6-scale-refine") != std::string::npos) {
            return "MXFP6 tensor-scale refinement";
        }
        if (line.find("selector rescue pass inspecting") != std::string::npos) {
            return "Quality repair inspecting completed GGUF";
        }
        if (line.find("selector rescue scan") != std::string::npos ||
                line.find("selector rescue rank") != std::string::npos) {
            return "Quality repair ranking tensor edits";
        }
        if (line.find("selector rescue apply") != std::string::npos ||
                line.find("selector rescue chose") != std::string::npos ||
                line.find("selector rescue patched") != std::string::npos) {
            return "Quality repair applying tensor edits";
        }
        if (line.find("stage-b patch complete") != std::string::npos ||
                line.find("runtime patch") != std::string::npos) {
            return "Runtime patch evaluation";
        }
        if (line.find("selector stage-a-survey") != std::string::npos) {
            return "Stage-A full-tensor proxy survey";
        }
        if (line.find("selector stage-a") != std::string::npos || line.find("selector refine") != std::string::npos) {
            return "Stage-A candidate search";
        }
        if (line.find("converting to") != std::string::npos) {
            return "Writing quantized GGUF tensors";
        }
        if (line.find("llama_context") != std::string::npos || line.find("llama_model_load") != std::string::npos) {
            return "Loading model/eval context";
        }
        return line.size() > 72 ? line.substr(0, 69) + "..." : line;
    }
    return "Starting";
}

std::vector<std::string> shell_recent_meaningful_log_lines(
        const std::filesystem::path & log_path,
        size_t max_lines) {
    std::vector<std::string> meaningful;
    for (const std::string & line : read_last_lines(log_path, 240)) {
        const std::string trimmed = trim_copy_simple(line);
        if (trimmed.empty() || is_tool_status_line(trimmed)) {
            continue;
        }
        meaningful.push_back(trimmed);
    }
    if (meaningful.size() > max_lines) {
        meaningful.erase(meaningful.begin(), meaningful.begin() + (meaningful.size() - max_lines));
    }
    return meaningful;
}

RunStatusThread::RunStatusThread(std::filesystem::path run_dir, std::string output, bool enabled)
    : run_dir(std::move(run_dir)),
      output(std::move(output)) {
    if (enabled) {
        worker = std::thread([this]() { loop(); });
    }
}

RunStatusThread::~RunStatusThread() {
    stop();
}

void RunStatusThread::stop() {
    done.store(true);
    if (worker.joinable()) {
        worker.join();
    }
}

void RunStatusThread::loop() {
    const auto start = std::chrono::steady_clock::now();
    const auto status_interval = std::chrono::seconds(10);
    const auto stderr_interval = std::chrono::seconds(30);
    auto last_stderr = start - stderr_interval;
    uint64_t last_stderr_output_bytes = std::numeric_limits<uint64_t>::max();
    while (!done.load()) {
        std::this_thread::sleep_for(status_interval);
        if (done.load()) {
            break;
        }
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - start).count();
        const uint64_t output_bytes = file_size_or_zero(output);
        if (now - last_stderr >= stderr_interval || output_bytes != last_stderr_output_bytes) {
            std::cerr << "\n" << PRODUCT_COMMAND << " status: elapsed="
                      << format_elapsed_seconds(elapsed)
                      << " output=" << mib_string(output_bytes) << " MiB\n";
            last_stderr = now;
            last_stderr_output_bytes = output_bytes;
        }

        std::ofstream events(run_dir / "run.jsonl", std::ios::app);
        if (events) {
            events << "{\"event\":\"status\",\"elapsed_sec\":" << elapsed
                   << ",\"output_bytes\":" << output_bytes
                   << "}\n";
        }
    }
}

} // namespace bq
