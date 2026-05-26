#include "utils.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace bq {

void write_text_file(const std::filesystem::path & path, const std::string & text) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write " + path.string());
    }
    out << text;
}

std::string shellish_args(const std::vector<std::string> & args) {
    std::string out;
    for (const std::string & arg : args) {
        if (!out.empty()) {
            out.push_back(' ');
        }
        bool simple = !arg.empty();
        for (const char c : arg) {
            if (!(std::isalnum((unsigned char) c) || c == '_' || c == '-' || c == '.' || c == '/' || c == ':' || c == '=' || c == ',')) {
                simple = false;
                break;
            }
        }
        if (simple) {
            out += arg;
        } else {
            out.push_back('\'');
            for (const char c : arg) {
                if (c == '\'') {
                    out += "'\\''";
                } else {
                    out.push_back(c);
                }
            }
            out.push_back('\'');
        }
    }
    return out;
}

std::string shellish_env(const std::vector<std::pair<std::string, std::string>> & env) {
    std::vector<std::string> args;
    args.reserve(env.size());
    for (const auto & item : env) {
        args.push_back(item.first + "=" + item.second);
    }
    return shellish_args(args);
}

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out.push_back(c); break;
        }
    }
    return out;
}

int default_worker_threads() {
    const unsigned int detected = std::thread::hardware_concurrency();
    if (detected <= 2) {
        return std::max(1U, detected);
    }
    // Leave real headroom for the OS, CUDA workers, file cache, and browser/UI
    // responsiveness during multi-GB GGUF runs. Users can still raise recipe
    // thread fields explicitly on dedicated machines.
    return (int) std::min(16U, std::max(1U, detected - 4U));
}

void set_if_empty(std::string & value, const std::string & fallback) {
    if (value.empty()) {
        value = fallback;
    }
}

bool string_is_auto(const std::string & value) {
    return value.empty() || value == "auto" || value == "all" || value == "full";
}

int parse_int_or_default(const std::string & value, int fallback) {
    if (value.empty()) {
        return fallback;
    }
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str()) {
        return fallback;
    }
    return (int) parsed;
}

}
