#include "candidate_metrics.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace bq {
namespace {

static std::string trim(std::string value) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

static std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return value;
}

static bool parse_numeric(const std::string & value, double & out) {
    const std::string trimmed = trim(value);
    if (trimmed.empty()) {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    out = std::strtod(trimmed.c_str(), &end);
    if (end == trimmed.c_str() || *end != '\0' || errno == ERANGE) {
        return false;
    }
    return std::isfinite(out);
}

static void add_field_order(CandidateTable & table, const std::string & field) {
    if (std::find(table.field_order.begin(), table.field_order.end(), field) == table.field_order.end()) {
        table.field_order.push_back(field);
    }
}

static std::vector<std::string> read_lines(const std::string & path) {
    std::vector<std::string> lines;
    std::string line;
    if (path == "-") {
        while (std::getline(std::cin, line)) {
            lines.push_back(line);
        }
        return lines;
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open candidate file: " + path);
    }
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

static bool has_extension(const std::string & path, const char * extension) {
    if (path == "-") {
        return false;
    }
    return lower_copy(std::filesystem::path(path).extension().string()) == extension;
}

static CandidateInputFormat detect_format(const std::string & path, const std::vector<std::string> & lines) {
    if (has_extension(path, ".jsonl") || has_extension(path, ".ndjson") || has_extension(path, ".json")) {
        return CandidateInputFormat::Jsonl;
    }
    if (has_extension(path, ".csv")) {
        return CandidateInputFormat::Csv;
    }

    for (const std::string & line : lines) {
        const std::string t = trim(line);
        if (t.empty()) {
            continue;
        }
        return t.front() == '{' ? CandidateInputFormat::Jsonl : CandidateInputFormat::Csv;
    }

    return CandidateInputFormat::Csv;
}

class JsonObjectParser {
public:
    JsonObjectParser(const std::string & line, size_t line_number) : text(line), line_no(line_number) {}

    std::vector<std::pair<std::string, std::string>> parse() {
        skip_ws();
        expect('{');
        std::vector<std::pair<std::string, std::string>> fields;
        skip_ws();
        if (consume('}')) {
            finish();
            return fields;
        }

        while (true) {
            skip_ws();
            const std::string key = parse_string();
            skip_ws();
            expect(':');
            const std::string value = parse_value();
            fields.push_back({ key, value });
            skip_ws();
            if (consume('}')) {
                finish();
                return fields;
            }
            expect(',');
        }
    }

private:
    const std::string & text;
    size_t line_no = 0;
    size_t pos = 0;

    [[noreturn]] void fail(const std::string & message) const {
        throw std::runtime_error("line " + std::to_string(line_no) + ": " + message);
    }

    void skip_ws() {
        while (pos < text.size() && std::isspace((unsigned char) text[pos])) {
            ++pos;
        }
    }

    bool consume(char expected) {
        if (pos < text.size() && text[pos] == expected) {
            ++pos;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    void finish() {
        skip_ws();
        if (pos != text.size()) {
            fail("unexpected trailing JSON text");
        }
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos < text.size()) {
            const char c = text[pos++];
            if (c == '"') {
                return out;
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (pos >= text.size()) {
                fail("unterminated JSON escape");
            }
            const char escaped = text[pos++];
            switch (escaped) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u':
                    if (pos + 4 > text.size()) {
                        fail("short JSON unicode escape");
                    }
                    out += "\\u";
                    out += text.substr(pos, 4);
                    pos += 4;
                    break;
                default:
                    fail("invalid JSON escape");
            }
        }
        fail("unterminated JSON string");
    }

    std::string parse_number_text() {
        const size_t start = pos;
        if (pos < text.size() && (text[pos] == '-' || text[pos] == '+')) {
            ++pos;
        }
        while (pos < text.size() && std::isdigit((unsigned char) text[pos])) {
            ++pos;
        }
        if (pos < text.size() && text[pos] == '.') {
            ++pos;
            while (pos < text.size() && std::isdigit((unsigned char) text[pos])) {
                ++pos;
            }
        }
        if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
            ++pos;
            if (pos < text.size() && (text[pos] == '-' || text[pos] == '+')) {
                ++pos;
            }
            while (pos < text.size() && std::isdigit((unsigned char) text[pos])) {
                ++pos;
            }
        }
        if (start == pos) {
            fail("expected JSON number");
        }
        return text.substr(start, pos - start);
    }

    std::string parse_literal(const char * literal) {
        const std::string value(literal);
        if (text.compare(pos, value.size(), value) != 0) {
            fail("expected JSON literal");
        }
        pos += value.size();
        return value;
    }

    std::string parse_compound_text() {
        const size_t start = pos;
        std::vector<char> stack;
        stack.push_back(text[pos] == '{' ? '}' : ']');
        ++pos;
        bool in_string = false;
        bool escape = false;
        while (pos < text.size() && !stack.empty()) {
            const char c = text[pos++];
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
                stack.push_back('}');
            } else if (c == '[') {
                stack.push_back(']');
            } else if (c == stack.back()) {
                stack.pop_back();
            }
        }
        if (!stack.empty()) {
            fail("unterminated JSON compound value");
        }
        return text.substr(start, pos - start);
    }

    std::string parse_value() {
        skip_ws();
        if (pos >= text.size()) {
            fail("expected JSON value");
        }
        const char c = text[pos];
        if (c == '"') {
            return parse_string();
        }
        if (c == '{' || c == '[') {
            return parse_compound_text();
        }
        if (c == 't') {
            return parse_literal("true");
        }
        if (c == 'f') {
            return parse_literal("false");
        }
        if (c == 'n') {
            return parse_literal("null");
        }
        return parse_number_text();
    }
};

static std::vector<std::string> parse_csv_row(const std::string & line, size_t line_number) {
    std::vector<std::string> row;
    std::string field;
    bool in_quotes = false;
    bool quoted_field = false;

    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    field.push_back('"');
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                field.push_back(c);
            }
            continue;
        }

        if (c == ',' && !in_quotes) {
            row.push_back(field);
            field.clear();
            quoted_field = false;
        } else if (c == '"' && field.empty() && !quoted_field) {
            in_quotes = true;
            quoted_field = true;
        } else {
            field.push_back(c);
        }
    }

    if (in_quotes) {
        throw std::runtime_error("line " + std::to_string(line_number) + ": unterminated quoted CSV field");
    }
    row.push_back(field);
    return row;
}

static void strip_utf8_bom(std::string & value) {
    if (value.size() >= 3 &&
        (unsigned char) value[0] == 0xEF &&
        (unsigned char) value[1] == 0xBB &&
        (unsigned char) value[2] == 0xBF) {
        value.erase(0, 3);
    }
}

static CandidateTable load_jsonl_table(const std::vector<std::string> & lines) {
    CandidateTable table;
    table.format = CandidateInputFormat::Jsonl;

    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string trimmed = trim(lines[i]);
        if (trimmed.empty()) {
            continue;
        }

        JsonObjectParser parser(trimmed, i + 1);
        const auto fields = parser.parse();

        CandidateRecord record;
        record.ordinal = table.records.size();
        record.line_number = i + 1;
        record.raw = trimmed;
        for (const auto & field : fields) {
            add_field_order(table, field.first);
            record.fields[field.first] = field.second;
            double numeric = 0.0;
            if (parse_numeric(field.second, numeric)) {
                record.numeric[field.first] = numeric;
            }
        }
        table.records.push_back(std::move(record));
    }

    return table;
}

static CandidateTable load_csv_table(const std::vector<std::string> & lines) {
    CandidateTable table;
    table.format = CandidateInputFormat::Csv;

    bool have_header = false;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (trim(lines[i]).empty()) {
            continue;
        }

        std::vector<std::string> row = parse_csv_row(lines[i], i + 1);
        if (!have_header) {
            if (!row.empty()) {
                strip_utf8_bom(row[0]);
            }
            for (std::string & field : row) {
                field = trim(field);
                if (field.empty()) {
                    throw std::runtime_error("line " + std::to_string(i + 1) + ": empty CSV header field");
                }
                add_field_order(table, field);
            }
            have_header = true;
            continue;
        }

        if (row.size() != table.field_order.size()) {
            throw std::runtime_error(
                "line " + std::to_string(i + 1) + ": expected " +
                std::to_string(table.field_order.size()) + " CSV fields, got " + std::to_string(row.size()));
        }

        CandidateRecord record;
        record.ordinal = table.records.size();
        record.line_number = i + 1;
        record.raw = lines[i];
        for (size_t col = 0; col < row.size(); ++col) {
            const std::string & field = table.field_order[col];
            record.fields[field] = row[col];
            double numeric = 0.0;
            if (parse_numeric(row[col], numeric)) {
                record.numeric[field] = numeric;
            }
        }
        table.records.push_back(std::move(record));
    }

    if (!have_header) {
        throw std::runtime_error("CSV candidate file is empty");
    }
    return table;
}

static bool field_is_numeric_for_all(const CandidateTable & table, const std::string & field) {
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

static bool contains_name(const std::set<std::string> & names, const std::string & field) {
    return names.find(lower_copy(field)) != names.end();
}

static bool is_metadata_numeric_field(const std::string & field) {
    static const std::set<std::string> names = {
        "candidate", "candidate_id", "id", "index", "line", "line_no", "line_number",
        "rank", "seed", "trial",
    };
    return contains_name(names, field);
}

static std::vector<std::string> numeric_fields_for_all(const CandidateTable & table) {
    std::vector<std::string> fields;
    for (const std::string & field : table.field_order) {
        if (field_is_numeric_for_all(table, field)) {
            fields.push_back(field);
        }
    }
    return fields;
}

static bool better_or_equal(double a, double b, ObjectiveDirection direction) {
    return direction == ObjectiveDirection::Minimize ? a <= b : a >= b;
}

static bool strictly_better(double a, double b, ObjectiveDirection direction) {
    return direction == ObjectiveDirection::Minimize ? a < b : a > b;
}

static bool dominates(const CandidateRecord & a, const CandidateRecord & b, const std::vector<CandidateObjective> & objectives) {
    bool any_strict = false;
    for (const CandidateObjective & objective : objectives) {
        const double av = a.numeric.at(objective.field);
        const double bv = b.numeric.at(objective.field);
        if (!better_or_equal(av, bv, objective.direction)) {
            return false;
        }
        any_strict = any_strict || strictly_better(av, bv, objective.direction);
    }
    return any_strict;
}

static std::string csv_escape(const std::string & value) {
    bool quote = false;
    for (const char c : value) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            quote = true;
            break;
        }
    }
    if (!quote) {
        return value;
    }

    std::string out = "\"";
    for (const char c : value) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
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

} // namespace

CandidateInputFormat parse_candidate_input_format(const std::string & value) {
    const std::string lower = lower_copy(value);
    if (lower == "auto") {
        return CandidateInputFormat::Auto;
    }
    if (lower == "jsonl" || lower == "ndjson" || lower == "json") {
        return CandidateInputFormat::Jsonl;
    }
    if (lower == "csv") {
        return CandidateInputFormat::Csv;
    }
    throw std::runtime_error("unknown candidate format: " + value);
}

std::string candidate_input_format_name(CandidateInputFormat format) {
    switch (format) {
        case CandidateInputFormat::Auto:  return "auto";
        case CandidateInputFormat::Jsonl: return "jsonl";
        case CandidateInputFormat::Csv:   return "csv";
    }
    return "auto";
}

CandidateTable load_candidate_table(const std::string & path, CandidateInputFormat requested_format) {
    const std::vector<std::string> lines = read_lines(path);
    const CandidateInputFormat format =
        requested_format == CandidateInputFormat::Auto ? detect_format(path, lines) : requested_format;

    if (format == CandidateInputFormat::Jsonl) {
        return load_jsonl_table(lines);
    }
    if (format == CandidateInputFormat::Csv) {
        return load_csv_table(lines);
    }
    throw std::runtime_error("candidate format auto-detection failed");
}

std::vector<CandidateObjective> infer_candidate_objectives(const CandidateTable & table) {
    static const std::set<std::string> minimize_names = {
        "bpw", "bytes", "cost", "error", "kld", "latency_ms", "load_ms", "loss", "loss_delta",
        "max_kld", "mean_kld", "memory_bytes", "mse", "p99", "p99_kld", "p999", "p999_kld",
        "perplexity", "ppl", "ppl_delta", "quality_cost", "rmse", "size_bytes", "size_mb",
        "time_ms", "vram_bytes", "vram_mb", "weight_bytes",
    };
    static const std::set<std::string> maximize_names = {
        "memory_saved_bytes", "pp_tok_s", "pp_tps", "quality_score", "savings_bytes",
        "speed_tps", "tg_tok_s", "tg_tps", "throughput", "tok_s", "tokens_per_second",
    };

    std::vector<CandidateObjective> objectives;
    std::set<std::string> seen;
    const std::vector<std::string> numeric_fields = numeric_fields_for_all(table);
    for (const std::string & field : numeric_fields) {
        if (contains_name(minimize_names, field) && seen.insert(field).second) {
            objectives.push_back({ field, ObjectiveDirection::Minimize });
        } else if (contains_name(maximize_names, field) && seen.insert(field).second) {
            objectives.push_back({ field, ObjectiveDirection::Maximize });
        }
    }

    if (!objectives.empty()) {
        return objectives;
    }

    for (const std::string & field : numeric_fields) {
        if (!is_metadata_numeric_field(field)) {
            objectives.push_back({ field, ObjectiveDirection::Minimize });
        }
    }
    return objectives;
}

std::vector<size_t> best_set_indices(
        const std::vector<CandidateRecord> & records,
        const std::vector<CandidateObjective> & objectives) {
    std::vector<size_t> best_set;
    for (size_t i = 0; i < records.size(); ++i) {
        bool dominated = false;
        for (size_t j = 0; j < records.size(); ++j) {
            if (i != j && dominates(records[j], records[i], objectives)) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            best_set.push_back(i);
        }
    }
    return best_set;
}

std::string format_candidate_csv(const CandidateTable & table, const std::vector<size_t> & indices) {
    std::ostringstream out;
    for (size_t col = 0; col < table.field_order.size(); ++col) {
        if (col > 0) {
            out << ',';
        }
        out << csv_escape(table.field_order[col]);
    }
    out << '\n';

    for (const size_t index : indices) {
        const CandidateRecord & record = table.records.at(index);
        for (size_t col = 0; col < table.field_order.size(); ++col) {
            if (col > 0) {
                out << ',';
            }
            const auto it = record.fields.find(table.field_order[col]);
            if (it != record.fields.end()) {
                out << csv_escape(it->second);
            }
        }
        out << '\n';
    }
    return out.str();
}

std::string format_candidate_jsonl(const CandidateTable & table, const std::vector<size_t> & indices) {
    std::ostringstream out;
    for (const size_t index : indices) {
        const CandidateRecord & record = table.records.at(index);
        if (table.format == CandidateInputFormat::Jsonl && !record.raw.empty()) {
            out << record.raw << '\n';
            continue;
        }

        out << '{';
        bool first = true;
        for (const std::string & field : table.field_order) {
            const auto it = record.fields.find(field);
            if (it == record.fields.end()) {
                continue;
            }
            if (!first) {
                out << ',';
            }
            first = false;
            out << '"' << json_escape(field) << "\":\"" << json_escape(it->second) << '"';
        }
        out << "}\n";
    }
    return out.str();
}

} // namespace bq
