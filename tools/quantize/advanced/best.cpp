#include "best.h"

#include "candidate_metrics.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bq {
namespace {

static void best_usage() {
    std::cout <<
        "usage:\n"
        "  advanced-gguf-quantizer best <candidates.jsonl|candidates.csv> [options]\n\n"
        "options:\n"
        "  --format auto|jsonl|csv       input format (default: auto)\n"
        "  --output auto|jsonl|csv       output format (default: input format)\n"
        "  --min <field>                 lower is better objective; repeatable\n"
        "  --max <field>                 higher is better objective; repeatable\n"
        "  --metric <field[:min|max]>    objective shorthand; repeatable\n\n"
        "  --proxy-ok                    allow proxy/custom metrics instead of real PPL/KLD evidence\n"
        "  --real-ppl-kld                require real PPL plus KLD metrics and seed full quality objectives (default)\n"
        "  --quality-only                keep size/speed out of the quality comparison (default with real PPL/KLD)\n"
        "  --report <path>               write a JSON best-candidate report\n\n"
        "By default this ranks real PPL/KLD quality evidence. Use --proxy-ok only for diagnostics.\n"
        "Use '-' as the candidate path to read from stdin.\n";
}

static std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return value;
}

static ObjectiveDirection parse_direction(const std::string & value) {
    const std::string lower = lower_copy(value);
    if (lower == "min" || lower == "minimize" || lower == "lower") {
        return ObjectiveDirection::Minimize;
    }
    if (lower == "max" || lower == "maximize" || lower == "higher") {
        return ObjectiveDirection::Maximize;
    }
    throw std::runtime_error("unknown objective direction: " + value);
}

static void add_objective(std::vector<CandidateObjective> & objectives, const std::string & field, ObjectiveDirection direction) {
    if (field.empty()) {
        throw std::runtime_error("objective field cannot be empty");
    }

    for (CandidateObjective & objective : objectives) {
        if (objective.field == field) {
            objective.direction = direction;
            return;
        }
    }
    objectives.push_back({ field, direction });
}

static bool numeric_for_all_records(const CandidateTable & table, const std::string & field) {
    if (table.records.empty()) {
        return false;
    }
    for (const CandidateRecord & record : table.records) {
        if (record.numeric.find(field) == record.numeric.end()) {
            return false;
        }
    }
    return true;
}

static std::string find_numeric_alias(const CandidateTable & table, const std::vector<std::string> & aliases) {
    for (const std::string & alias : aliases) {
        for (const std::string & field : table.field_order) {
            if (lower_copy(field) == alias && numeric_for_all_records(table, field)) {
                return field;
            }
        }
    }
    return {};
}

static void add_alias_objective(
        const CandidateTable & table,
        std::vector<CandidateObjective> & objectives,
        const std::vector<std::string> & aliases,
        ObjectiveDirection direction) {
    const std::string field = find_numeric_alias(table, aliases);
    if (!field.empty()) {
        add_objective(objectives, field, direction);
    }
}

static void add_real_ppl_kld_objectives(
        const CandidateTable & table,
        std::vector<CandidateObjective> & objectives,
        bool quality_only) {
    const std::string ppl = find_numeric_alias(table, { "ppl", "perplexity" });
    const std::string mean_kld = find_numeric_alias(table, { "mean_kld", "kld", "kl_divergence", "kl" });
    const std::string max_kld = find_numeric_alias(table, { "max_kld", "kld_max", "maximum_kld" });
    const std::string p99_kld = find_numeric_alias(table, { "p99_kld", "kld_p99", "p99" });
    const std::string p999_kld = find_numeric_alias(table, { "p999_kld", "kld_p999", "p999", "p99_9_kld" });

    if (ppl.empty()) {
        throw std::runtime_error("--real-ppl-kld requires a numeric ppl or perplexity field for every candidate");
    }
    if (mean_kld.empty()) {
        throw std::runtime_error("--real-ppl-kld requires a numeric mean_kld or kld field for every candidate");
    }
    if (max_kld.empty() && p99_kld.empty() && p999_kld.empty()) {
        throw std::runtime_error("--real-ppl-kld requires at least one numeric KLD tail field: max_kld, p99_kld, or p999_kld");
    }

    add_objective(objectives, ppl, ObjectiveDirection::Minimize);
    add_alias_objective(table, objectives, { "ppl_ratio", "perplexity_ratio" }, ObjectiveDirection::Minimize);
    add_objective(objectives, mean_kld, ObjectiveDirection::Minimize);
    add_alias_objective(table, objectives, { "p95_kld", "kld_p95", "p95" }, ObjectiveDirection::Minimize);
    add_alias_objective(table, objectives, { "p99_kld", "kld_p99", "p99" }, ObjectiveDirection::Minimize);
    add_alias_objective(table, objectives, { "p999_kld", "kld_p999", "p999", "p99_9_kld" }, ObjectiveDirection::Minimize);
    add_alias_objective(table, objectives, { "kld_tail_mean", "tail_kld_mean", "cvar99_kld", "kld_cvar99" }, ObjectiveDirection::Minimize);
    add_alias_objective(table, objectives, { "max_kld", "kld_max", "maximum_kld" }, ObjectiveDirection::Minimize);
    add_alias_objective(table, objectives, { "rms_delta_p", "rms_delta", "rms_dp" }, ObjectiveDirection::Minimize);
    add_alias_objective(table, objectives, { "same_top", "same_top_pct", "same_top_percent" }, ObjectiveDirection::Maximize);
    add_alias_objective(table, objectives, { "top_flip_weight", "top_flip_w" }, ObjectiveDirection::Minimize);
    add_alias_objective(table, objectives, { "top_prob_rmse", "top_p_rmse" }, ObjectiveDirection::Minimize);
    add_alias_objective(table, objectives, { "entropy_rmse", "entropy_drift" }, ObjectiveDirection::Minimize);
    if (!quality_only) {
        add_alias_objective(table, objectives, { "bpw", "size_mb", "size_bytes", "weight_bytes" }, ObjectiveDirection::Minimize);
        add_alias_objective(table, objectives, { "pp_tok_s", "pp_tps", "tg_tok_s", "tg_tps", "tok_s" }, ObjectiveDirection::Maximize);
    }
}

static void add_metric_objective(std::vector<CandidateObjective> & objectives, const std::string & spec) {
    const size_t colon = spec.rfind(':');
    if (colon == std::string::npos) {
        add_objective(objectives, spec, ObjectiveDirection::Minimize);
        return;
    }

    const std::string field = spec.substr(0, colon);
    const std::string direction = spec.substr(colon + 1);
    add_objective(objectives, field, parse_direction(direction));
}

static std::string objective_summary(const std::vector<CandidateObjective> & objectives) {
    std::string out;
    for (size_t i = 0; i < objectives.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += objectives[i].direction == ObjectiveDirection::Minimize ? "min " : "max ";
        out += objectives[i].field;
    }
    return out;
}

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

static bool objective_better_or_equal(double a, double b, ObjectiveDirection direction) {
    return direction == ObjectiveDirection::Minimize ? a <= b : a >= b;
}

static bool objective_strictly_better(double a, double b, ObjectiveDirection direction) {
    return direction == ObjectiveDirection::Minimize ? a < b : a > b;
}

static bool record_dominates(
        const CandidateRecord & a,
        const CandidateRecord & b,
        const std::vector<CandidateObjective> & objectives) {
    bool any_strict = false;
    for (const CandidateObjective & objective : objectives) {
        const double av = a.numeric.at(objective.field);
        const double bv = b.numeric.at(objective.field);
        if (!objective_better_or_equal(av, bv, objective.direction)) {
            return false;
        }
        any_strict = any_strict || objective_strictly_better(av, bv, objective.direction);
    }
    return any_strict;
}

static void append_candidate_summary(
        std::ostringstream & out,
        const CandidateTable & table,
        const std::vector<CandidateObjective> & objectives,
        const CandidateRecord & record) {
    out << "{\"ordinal\": " << record.ordinal
        << ", \"line\": " << record.line_number
        << ", \"fields\": {";
    bool first = true;
    for (const std::string & field : table.field_order) {
        const auto it = record.fields.find(field);
        if (it == record.fields.end()) {
            continue;
        }
        if (!first) {
            out << ", ";
        }
        first = false;
        out << "\"" << json_escape(field) << "\": \"" << json_escape(it->second) << "\"";
    }
    out << "}, \"objectives\": {";
    first = true;
    for (const CandidateObjective & objective : objectives) {
        const auto it = record.numeric.find(objective.field);
        if (it == record.numeric.end()) {
            continue;
        }
        if (!first) {
            out << ", ";
        }
        first = false;
        out << "\"" << json_escape(objective.field) << "\": " << it->second;
    }
    out << "}}";
}

static std::string best_report_json(
        const CandidateTable & table,
        const std::vector<CandidateObjective> & objectives,
        const std::vector<size_t> & selected,
        bool real_ppl_kld) {
    const std::set<size_t> selected_lookup(selected.begin(), selected.end());
    std::ostringstream out;
    out << "{\n";
    out << "  \"candidate_count\": " << table.records.size() << ",\n";
    out << "  \"non_dominated_count\": " << selected.size() << ",\n";
    out << "  \"selection_rule\": \"best_non_dominated\",\n";
    out << "  \"quality_mode\": \"" << (real_ppl_kld ? "real_ppl_kld" : "custom") << "\",\n";
    out << "  \"objectives\": [";
    for (size_t i = 0; i < objectives.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << "\n    {\"field\": \"" << json_escape(objectives[i].field)
            << "\", \"direction\": \""
            << (objectives[i].direction == ObjectiveDirection::Minimize ? "min" : "max")
            << "\"}";
    }
    out << "\n  ],\n";
    out << "  \"best_candidates\": [";
    for (size_t i = 0; i < selected.size(); ++i) {
        const CandidateRecord & record = table.records.at(selected[i]);
        if (i > 0) {
            out << ",";
        }
        out << "\n    ";
        append_candidate_summary(out, table, objectives, record);
    }
    out << "\n  ],\n";
    out << "  \"dominated\": [";
    bool first_dominated = true;
    for (size_t i = 0; i < table.records.size(); ++i) {
        if (selected_lookup.find(i) != selected_lookup.end()) {
            continue;
        }
        if (!first_dominated) {
            out << ",";
        }
        first_dominated = false;
        const CandidateRecord & record = table.records.at(i);
        out << "\n    {\"candidate\": ";
        append_candidate_summary(out, table, objectives, record);
        out << ", \"dominated_by\": [";
        bool first_dominator = true;
        for (const CandidateRecord & other : table.records) {
            if (other.ordinal == record.ordinal || !record_dominates(other, record, objectives)) {
                continue;
            }
            if (!first_dominator) {
                out << ", ";
            }
            first_dominator = false;
            out << other.ordinal;
        }
        out << "]}";
    }
    out << "\n  ]\n";
    out << "}\n";
    return out.str();
}

static void write_text_file(const std::string & path, const std::string & text) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write report: " + path);
    }
    out << text;
}

static void validate_objectives(const CandidateTable & table, const std::vector<CandidateObjective> & objectives) {
    if (table.records.empty()) {
        throw std::runtime_error("no candidate records found");
    }
    if (objectives.empty()) {
        throw std::runtime_error("no numeric objective fields found; pass --min/--max explicitly");
    }

    for (const CandidateObjective & objective : objectives) {
        for (const CandidateRecord & record : table.records) {
            if (record.numeric.find(objective.field) == record.numeric.end()) {
                throw std::runtime_error(
                    "objective '" + objective.field + "' is missing or non-numeric at line " +
                    std::to_string(record.line_number));
            }
        }
    }
}

} // namespace

int best_main(int argc, char ** argv) {
    if (argc < 1) {
        best_usage();
        return 1;
    }

    std::string input_path;
    CandidateInputFormat input_format = CandidateInputFormat::Auto;
    CandidateInputFormat output_format = CandidateInputFormat::Auto;
    std::string report_path;
    bool real_ppl_kld = true;
    bool quality_only = true;
    std::vector<CandidateObjective> objectives;

    for (int i = 0; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h" || arg == "help") {
            best_usage();
            return 0;
        }
        if (arg == "--format" && i + 1 < argc) {
            input_format = parse_candidate_input_format(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_format = parse_candidate_input_format(argv[++i]);
        } else if (arg == "--report" && i + 1 < argc) {
            report_path = argv[++i];
        } else if (arg == "--real-ppl-kld") {
            real_ppl_kld = true;
        } else if (arg == "--quality-only") {
            quality_only = true;
        } else if (arg == "--proxy-ok") {
            real_ppl_kld = false;
            quality_only = false;
        } else if (arg == "--min" && i + 1 < argc) {
            add_objective(objectives, argv[++i], ObjectiveDirection::Minimize);
        } else if (arg == "--max" && i + 1 < argc) {
            add_objective(objectives, argv[++i], ObjectiveDirection::Maximize);
        } else if (arg == "--metric" && i + 1 < argc) {
            add_metric_objective(objectives, argv[++i]);
        } else if (input_path.empty() && arg == "-") {
            input_path = arg;
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("unknown best argument: " + arg);
        } else if (input_path.empty()) {
            input_path = arg;
        } else {
            throw std::runtime_error("unexpected best argument: " + arg);
        }
    }

    if (input_path.empty()) {
        throw std::runtime_error("best requires a candidate file path");
    }

    CandidateTable table = load_candidate_table(input_path, input_format);
    if (real_ppl_kld) {
        add_real_ppl_kld_objectives(table, objectives, quality_only);
    }
    if (objectives.empty()) {
        objectives = infer_candidate_objectives(table);
    }
    validate_objectives(table, objectives);

    const std::vector<size_t> best_candidates = best_set_indices(table.records, objectives);
    std::cerr << "best: " << best_candidates.size() << " of " << table.records.size()
              << " candidates non-dominated using " << objective_summary(objectives) << "\n";
    if (!report_path.empty()) {
        write_text_file(report_path, best_report_json(table, objectives, best_candidates, real_ppl_kld));
        std::cerr << "report: " << report_path << "\n";
    }

    if (output_format == CandidateInputFormat::Auto) {
        output_format = table.format == CandidateInputFormat::Auto ? CandidateInputFormat::Csv : table.format;
    }
    if (output_format == CandidateInputFormat::Jsonl) {
        std::cout << format_candidate_jsonl(table, best_candidates);
    } else if (output_format == CandidateInputFormat::Csv) {
        std::cout << format_candidate_csv(table, best_candidates);
    } else {
        throw std::runtime_error("best output format auto-resolution failed");
    }

    return 0;
}

} // namespace bq
