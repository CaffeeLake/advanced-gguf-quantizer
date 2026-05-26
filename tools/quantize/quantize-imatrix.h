#pragma once

#include <string>
#include <unordered_map>
#include <vector>

int prepare_imatrix(
        const std::string & imatrix_file,
        std::vector<std::string> & imatrix_dataset,
        const std::vector<std::string> & included_weights,
        const std::vector<std::string> & excluded_weights,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data);
