#include "run_plan.h"

namespace bq {

QuantizeRunPlan make_quantize_run_plan(const Recipe & recipe, bool force_dry_run) {
    QuantizeRunPlan plan;
    plan.argv = build_quantize_args(recipe, force_dry_run);
    plan.input = recipe.io.input;
    plan.output = recipe.io.output;
    plan.ftype = recipe.base.copy_only ? "COPY" :
        (!recipe.target.precision_mode.empty() ? recipe.target.precision_mode : recipe.base.ftype);
    plan.threads = recipe.base.threads;
    plan.dry_run = force_dry_run || recipe.base.dry_run;
    return plan;
}

} // namespace bq
