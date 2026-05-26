#pragma once

#include "../ggml/include/ggml-cuda.h"

#include <cstdint>
#include <string>

// Parsed tensor override passed from the CLI front-end into the quantizer core.
struct tensor_type_option {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
    bool has_nvfp4_cfg = false;
    nvfp4_cuda_runtime_cfg nvfp4_cfg = {
        NVFP4_CUDA_CHOOSE46_ADAPTIVE,
        8,
        1,
        0,
        448.0f,
        256.0f,
    };
    int64_t nvfp4_sample_blocks = 0;
    std::string nvfp4_policy_name;
    bool has_mxfp6_scale_mul = false;
    float mxfp6_e2m3_scale_mul = 1.0f;
    std::string mxfp6_policy_name;
};
