#pragma once

#include "recipe.h"

#include <string>
#include <vector>

namespace bq {

struct QuantizeRunPlan {
    std::vector<std::string> argv;
    std::string input;
    std::string output;
    std::string ftype;
    int threads = 0;
    bool dry_run = false;
};

QuantizeRunPlan make_quantize_run_plan(const Recipe & recipe, bool force_dry_run);

} // namespace bq
