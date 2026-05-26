#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

namespace bq {

struct ProjectQualityInputs {
    std::string bf16_reference;
    std::string kld_base;
    std::string eval_corpus;
    std::string calibration_corpus;
    std::string imatrix;
};

struct ProjectInit {
    std::string name;
    std::string recipe;
    std::string input;
    std::string bf16_reference;
    std::string kld_base;
    std::string corpus;
    std::string calibration_corpus;
    std::string imatrix;
};

void project_init_file(const std::string & path, const ProjectInit & init);
void project_append_candidate_manifest(const std::string & path, const std::string & manifest_path);
void project_append_run_event(
        const std::string & path,
        const std::string & event,
        const std::string & variant,
        const std::string & recipe,
        const std::string & output,
        const std::string & run_dir,
        int return_code,
        uint64_t output_bytes,
        const ProjectQualityInputs & quality = {});
void project_append_metric_event(
        const std::string & path,
        const std::string & variant,
        const std::string & metric_json);
uint64_t project_export_metrics(const std::string & path, std::ostream & out);
void project_print_summary(const std::string & path, std::ostream & out);

} // namespace bq
