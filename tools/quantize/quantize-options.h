#pragma once

#include "../../src/llama-quant-types.h"

#include "llama.h"

#include <string>
#include <vector>

bool striequals(const char * a, const char * b);
std::string trim_copy(std::string value);

bool try_parse_ftype(const std::string & ftype_str_in, llama_ftype & ftype, std::string & ftype_str_out);
bool ftype_is_nvfp4_mxfp6_alias(const std::string & ftype_str);
[[noreturn]] void usage(const char * executable);

ggml_type parse_ggml_type(const char * arg);
const char * nvfp4_choose46_mode_name(int32_t mode);
bool parse_nvfp4_preset(const std::string & value, nvfp4_cuda_runtime_cfg & out_cfg, std::string * out_name = nullptr);
bool parse_tensor_type_nvfp4_cfg(const std::string & spec, tensor_type_option & out);
bool parse_tensor_type(const char * data, std::vector<tensor_type_option> & tensor_type);
bool parse_tensor_type_file(const char * filename, std::vector<tensor_type_option> & tensor_type);
bool parse_layer_prune(const char * data, std::vector<int> & prune_layers);
bool parse_on_off_value(const std::string & value, bool & out);
bool parse_nvfp4_input_scale_policy(const std::string & value, int32_t & out);
bool parse_nvfp4_scale_denom(const std::string & value, float & out);
std::string format_tensor_type_value(const tensor_type_option & opt);
