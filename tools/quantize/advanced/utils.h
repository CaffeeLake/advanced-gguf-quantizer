#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace bq {

void write_text_file(const std::filesystem::path & path, const std::string & text);

std::string shellish_args(const std::vector<std::string> & args);
std::string shellish_env(const std::vector<std::pair<std::string, std::string>> & env);
std::string json_escape(const std::string & value);

int default_worker_threads();
void set_if_empty(std::string & value, const std::string & fallback);
bool string_is_auto(const std::string & value);
int parse_int_or_default(const std::string & value, int fallback);

}
