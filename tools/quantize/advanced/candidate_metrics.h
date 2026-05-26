#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace bq {

enum class CandidateInputFormat {
    Auto,
    Jsonl,
    Csv,
};

enum class ObjectiveDirection {
    Minimize,
    Maximize,
};

struct CandidateObjective {
    std::string field;
    ObjectiveDirection direction = ObjectiveDirection::Minimize;
};

struct CandidateRecord {
    size_t ordinal = 0;
    size_t line_number = 0;
    std::string raw;
    std::map<std::string, std::string> fields;
    std::map<std::string, double> numeric;
};

struct CandidateTable {
    CandidateInputFormat format = CandidateInputFormat::Auto;
    std::vector<std::string> field_order;
    std::vector<CandidateRecord> records;
};

CandidateInputFormat parse_candidate_input_format(const std::string & value);
std::string candidate_input_format_name(CandidateInputFormat format);

CandidateTable load_candidate_table(const std::string & path, CandidateInputFormat requested_format);
std::vector<CandidateObjective> infer_candidate_objectives(const CandidateTable & table);
std::vector<size_t> best_set_indices(
    const std::vector<CandidateRecord> & records,
    const std::vector<CandidateObjective> & objectives);

std::string format_candidate_csv(const CandidateTable & table, const std::vector<size_t> & indices);
std::string format_candidate_jsonl(const CandidateTable & table, const std::vector<size_t> & indices);

} // namespace bq
