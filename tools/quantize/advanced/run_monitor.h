#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace bq {

std::string now_iso_like();
std::string read_text_file_trimmed(const std::filesystem::path & path);
std::vector<std::string> read_last_lines(const std::filesystem::path & path, size_t max_lines);
uint64_t file_size_or_zero(const std::string & path);

std::string shell_extract_eta_progress(const std::string & line);
std::string shell_latest_progress_line(const std::vector<std::string> & log_lines);
std::string shell_executable_path();
bool shell_pid_alive(const std::filesystem::path & pid_file);
bool shell_run_finished(const std::filesystem::path & run_dir, std::string * rc_out = nullptr);
bool is_tool_status_line(const std::string & line);
std::string shell_infer_run_phase(const std::vector<std::string> & log_lines);
std::vector<std::string> shell_recent_meaningful_log_lines(
        const std::filesystem::path & log_path,
        size_t max_lines);

class RunStatusThread {
public:
    RunStatusThread(std::filesystem::path run_dir, std::string output, bool enabled);
    ~RunStatusThread();

    void stop();

private:
    void loop();

    std::filesystem::path run_dir;
    std::string output;
    std::atomic_bool done{false};
    std::thread worker;
};

} // namespace bq
