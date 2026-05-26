#include "quantize-mxfp6.h"

#include "quantize-options.h"
#include "quantize-selector-runtime.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

std::vector<float> quantize_mxfp6_scale_candidates() {
    std::vector<float> out;
    const std::string configured = quantize_control_string("LLAMA_MXFP6_SELECTOR_SCALE_CANDIDATES");
    if (!configured.empty()) {
        std::stringstream ss(configured);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim_copy(token);
            if (token.empty()) {
                continue;
            }
            try {
                const float v = std::stof(token);
                if (v > 0.0f && std::isfinite(v)) {
                    out.push_back(v);
                }
            } catch (const std::exception &) {
            }
        }
    }
    if (out.empty()) {
        // Keep the default search compact: +/-1/16 and +/-1/8 nudges usually
        // move PPL, while +/-3/8 catches useful outliers found in Qwen runs.
        const std::vector<float> exponents = {
            -0.375f, -0.25f, -0.125f, -0.0625f, 0.0f, 0.0625f, 0.125f, 0.25f, 0.375f,
        };
        for (float e : exponents) {
            out.push_back(std::exp2(e));
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end(), [](float a, float b) {
        return std::fabs(a - b) <= 1e-6f * std::max(std::fabs(a), std::fabs(b));
    }), out.end());
    return out;
}
