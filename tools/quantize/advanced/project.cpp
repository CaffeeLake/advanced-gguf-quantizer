#include "project.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace bq {
namespace {

static std::string json_escape(const std::string & value) {
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

static std::string now_string() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", &tm);
    return buf;
}

static std::string trim(std::string value) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

static std::string extract_json_string_field(const std::string & line, const std::string & key) {
    const std::string needle = "\"" + key + "\":\"";
    const size_t start = line.find(needle);
    if (start == std::string::npos) {
        return {};
    }
    std::string out;
    for (size_t i = start + needle.size(); i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            return out;
        }
        if (c == '\\' && i + 1 < line.size()) {
            const char escaped = line[++i];
            switch (escaped) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                default:   out.push_back(escaped); break;
            }
        } else {
            out.push_back(c);
        }
    }
    return {};
}

static std::string extract_json_object_field(const std::string & line, const std::string & key) {
    const std::string needle = "\"" + key + "\":";
    const size_t key_pos = line.find(needle);
    if (key_pos == std::string::npos) {
        return {};
    }
    size_t pos = key_pos + needle.size();
    while (pos < line.size() && std::isspace((unsigned char) line[pos])) {
        ++pos;
    }
    if (pos >= line.size() || line[pos] != '{') {
        return {};
    }

    bool in_string = false;
    bool escape = false;
    int depth = 0;
    const size_t start = pos;
    for (; pos < line.size(); ++pos) {
        const char c = line[pos];
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) {
                return line.substr(start, pos - start + 1);
            }
        }
    }
    return {};
}

static void ensure_parent(const std::string & path) {
    const std::filesystem::path fs_path(path);
    if (fs_path.has_parent_path()) {
        std::filesystem::create_directories(fs_path.parent_path());
    }
}

static std::ofstream append(const std::string & path) {
    ensure_parent(path);
    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to append project file: " + path);
    }
    return out;
}

static void write_string_field(std::ostream & out, const char * key, const std::string & value, bool comma = true) {
    out << "\"" << key << "\":\"" << json_escape(value) << "\"";
    if (comma) {
        out << ",";
    }
}

static bool has_quality_inputs(const ProjectQualityInputs & quality) {
    return !quality.bf16_reference.empty() ||
        !quality.kld_base.empty() ||
        !quality.eval_corpus.empty() ||
        !quality.calibration_corpus.empty() ||
        !quality.imatrix.empty();
}

static void write_file_ref(std::ostream & out, const char * key, const std::string & path, bool comma = true) {
    std::error_code ec;
    const bool exists = !path.empty() && std::filesystem::exists(path, ec);
    const uint64_t bytes = exists ? std::filesystem::file_size(path, ec) : 0;
    out << "\"" << key << "\":{";
    write_string_field(out, "path", path);
    out << "\"exists\":" << (exists ? "true" : "false") << ",";
    out << "\"bytes\":" << (ec ? 0 : bytes);
    out << "}";
    if (comma) {
        out << ",";
    }
}

static void write_quality_inputs(std::ostream & out, const ProjectQualityInputs & quality, bool comma = true) {
    out << "\"quality_inputs\":{";
    write_file_ref(out, "bf16_reference", quality.bf16_reference);
    write_file_ref(out, "kld_base", quality.kld_base);
    write_file_ref(out, "eval_corpus", quality.eval_corpus);
    write_file_ref(out, "calibration_corpus", quality.calibration_corpus);
    write_file_ref(out, "imatrix", quality.imatrix);
    out << "\"kld_command_uses_runtime_defaults\":true";
    out << "}";
    if (comma) {
        out << ",";
    }
}

} // namespace

void project_init_file(const std::string & path, const ProjectInit & init) {
    ensure_parent(path);
    if (std::filesystem::exists(path)) {
        throw std::runtime_error("project already exists: " + path);
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write project file: " + path);
    }
    out << "{";
    write_string_field(out, "schema", "advanced-gguf-quantizer-project-v1");
    write_string_field(out, "event", "project_init");
    write_string_field(out, "created_at", now_string());
    write_string_field(out, "name", init.name);
    write_string_field(out, "recipe", init.recipe);
    write_string_field(out, "input", init.input);
    out << "\"baseline\":{";
    write_string_field(out, "bf16_reference", init.bf16_reference);
    write_string_field(out, "kld_base", init.kld_base);
    write_string_field(out, "corpus", init.corpus);
    write_string_field(out, "calibration_corpus", init.calibration_corpus);
    write_string_field(out, "imatrix", init.imatrix, false);
    out << "},";
    ProjectQualityInputs quality;
    quality.bf16_reference = init.bf16_reference;
    quality.kld_base = init.kld_base;
    quality.eval_corpus = init.corpus;
    quality.calibration_corpus = init.calibration_corpus;
    quality.imatrix = init.imatrix;
    write_quality_inputs(out, quality);
    out << "\"notes\":\"JSONL event log. Tensor type changes usually require GGUF rewrite; same-size payload replacement is the only true in-place edit path.\"";
    out << "}\n";
}

void project_append_candidate_manifest(const std::string & path, const std::string & manifest_path) {
    std::ifstream in(manifest_path);
    if (!in) {
        throw std::runtime_error("failed to open candidate manifest: " + manifest_path);
    }

    std::ofstream out = append(path);
    std::string line;
    uint64_t count = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        out << "{";
        write_string_field(out, "schema", "advanced-gguf-quantizer-project-v1");
        write_string_field(out, "event", "candidate");
        write_string_field(out, "created_at", now_string());
        write_string_field(out, "manifest", manifest_path);
        out << "\"candidate\":" << line << "}\n";
        ++count;
    }
    out << "{";
    write_string_field(out, "schema", "advanced-gguf-quantizer-project-v1");
    write_string_field(out, "event", "candidate_manifest_added");
    write_string_field(out, "created_at", now_string());
    write_string_field(out, "manifest", manifest_path);
    out << "\"count\":" << count << "}\n";
}

void project_append_run_event(
        const std::string & path,
        const std::string & event,
        const std::string & variant,
        const std::string & recipe,
        const std::string & output,
        const std::string & run_dir,
        int return_code,
        uint64_t output_bytes,
        const ProjectQualityInputs & quality) {
    if (path.empty()) {
        return;
    }
    std::ofstream out = append(path);
    out << "{";
    write_string_field(out, "schema", "advanced-gguf-quantizer-project-v1");
    write_string_field(out, "event", event);
    write_string_field(out, "created_at", now_string());
    write_string_field(out, "variant", variant);
    write_string_field(out, "recipe", recipe);
    write_string_field(out, "output", output);
    write_string_field(out, "run_dir", run_dir);
    out << "\"return_code\":" << return_code << ",";
    out << "\"output_bytes\":" << output_bytes << ",";
    if (has_quality_inputs(quality)) {
        write_string_field(out, "bf16_reference", quality.bf16_reference);
        write_string_field(out, "kld_base", quality.kld_base);
        write_string_field(out, "eval_corpus", quality.eval_corpus);
        write_string_field(out, "calibration_corpus", quality.calibration_corpus);
        write_string_field(out, "imatrix", quality.imatrix);
        write_quality_inputs(out, quality);
    }
    out << "\"patch\":{\"can_continue\":true,\"true_in_place_requires_same_size_payload\":true,\"type_or_metadata_change_requires_rewrite\":true}";
    out << "}\n";
}

void project_append_metric_event(
        const std::string & path,
        const std::string & variant,
        const std::string & metric_json) {
    std::ofstream out = append(path);
    out << "{";
    write_string_field(out, "schema", "advanced-gguf-quantizer-project-v1");
    write_string_field(out, "event", "metrics");
    write_string_field(out, "created_at", now_string());
    write_string_field(out, "variant", variant);
    out << "\"metrics\":" << (metric_json.empty() ? "{}" : metric_json) << "}\n";
}

uint64_t project_export_metrics(const std::string & path, std::ostream & out) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open project: " + path);
    }

    std::string line;
    uint64_t count = 0;
    while (std::getline(in, line)) {
        if (line.find("\"event\":\"metrics\"") == std::string::npos) {
            continue;
        }
        const std::string variant = extract_json_string_field(line, "variant");
        const std::string metrics = extract_json_object_field(line, "metrics");
        if (variant.empty() || metrics.size() < 2) {
            continue;
        }
        const std::string body = trim(metrics.substr(1, metrics.size() - 2));
        out << "{\"variant\":\"" << json_escape(variant)
            << "\",\"source_project\":\"" << json_escape(path) << "\"";
        if (!body.empty()) {
            out << "," << body;
        }
        out << "}\n";
        ++count;
    }
    return count;
}

void project_print_summary(const std::string & path, std::ostream & out) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open project: " + path);
    }

    std::string line;
    uint64_t count = 0;
    uint64_t candidates = 0;
    uint64_t runs = 0;
    uint64_t metrics = 0;
    std::vector<std::string> tail;
    while (std::getline(in, line)) {
        ++count;
        if (line.find("\"event\":\"candidate\"") != std::string::npos) {
            ++candidates;
        } else if (line.find("\"event\":\"run_") != std::string::npos) {
            ++runs;
        } else if (line.find("\"event\":\"metrics\"") != std::string::npos) {
            ++metrics;
        }
        tail.push_back(line);
        if (tail.size() > 8) {
            tail.erase(tail.begin());
        }
    }

    out << "project: " << path << "\n";
    out << "events: " << count << ", candidates: " << candidates << ", run events: " << runs << ", metric events: " << metrics << "\n";
    out << "recent events:\n";
    for (const std::string & item : tail) {
        out << "  " << item << "\n";
    }
}

} // namespace bq
