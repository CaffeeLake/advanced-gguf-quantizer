#pragma once

#include "recipe.h"
#include "tui.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace bq {

inline constexpr const char * PRODUCT_COMMAND = "advanced-gguf-quantizer";
inline constexpr const char * PRODUCT_NAME = "advanced-gguf-quantizer";
inline constexpr const char * PRODUCT_VERSION = "0.2.0-dev";
inline constexpr const char * PRODUCT_CONFIG_DIR = "advanced-gguf-quantizer";

std::string mib_string(uint64_t bytes);

struct KldBaseInfo {
    bool valid = false;
    bool complete = false;
    int32_t n_ctx = 0;
    int32_t n_vocab = 0;
    int32_t n_chunks = 0;
    int32_t available_chunks = 0;
    uint64_t file_bytes = 0;
    uint64_t expected_bytes = 0;
    std::string error;
};

KldBaseInfo read_kld_base_info(const std::string & path);
std::string format_kld_info(const KldBaseInfo & info);

std::filesystem::path shell_settings_dir();
std::string default_project_path();
std::string default_config_path();

struct ShellState {
    std::string project_path;
    std::string last_recipe = default_config_path();
    std::string input_model;
    std::string output_model;
    std::string variant;
    std::string run_dir;
    std::string run_log;
    std::string run_pid_file;
    std::string run_phase;
    std::string run_progress;
    std::string precision_mode;
    std::string status = "Ready";
    Recipe recipe;
    bool have_recipe = false;
    bool quit = false;
};

tui::ProductInfo shell_product();
std::string display_path(const std::string & value);
tui::Style shell_status_style(const std::string & status);
std::vector<tui::StatusItem> shell_status_items(const ShellState & state);
std::vector<tui::StatusItem> shell_status_items_compact(const ShellState & state);
void shell_print_status(const ShellState & state, const tui::TerminalCapabilities & caps);
void shell_clear(const tui::TerminalCapabilities & caps);
std::vector<tui::MenuOption> shell_home_menu_options();
std::vector<std::string> shell_preflight_errors(const ShellState & state);
bool shell_ready_to_quantize(const ShellState & state);
std::vector<tui::MenuOption> shell_project_menu_options(const ShellState & state);
std::vector<tui::MenuOption> shell_options_menu_options(const ShellState & state);

} // namespace bq
