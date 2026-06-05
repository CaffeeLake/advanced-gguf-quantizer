#include "quantize-imatrix.h"

#include "imatrix-loader.h"

#include <cstdio>
#include <cstdlib>
#include <utility>

static int load_imatrix(const std::string & imatrix_file, std::vector<std::string> & imatrix_datasets, std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    common_imatrix loaded;
    if (!common_imatrix_load(imatrix_file, loaded)) {
        fprintf(stderr, "%s: failed to load imatrix from '%s'\n", __func__, imatrix_file.c_str());
        exit(1);
    }

    if (!loaded.is_legacy && !loaded.has_metadata) {
        fprintf(stderr, "%s: missing imatrix metadata in file %s\n", __func__, imatrix_file.c_str());
        exit(1);
    }

    for (const auto & [name, entry] : loaded.entries) {
        auto & e = imatrix_data[name];
        e.resize(entry.sums.size());

        if (loaded.is_legacy) {
            const int64_t ncall = entry.counts.empty() ? 0 : entry.counts[0];
            if (ncall > 0) {
                for (size_t i = 0; i < entry.sums.size(); ++i) {
                    e[i] = entry.sums[i] / ncall;
                }
            } else {
                for (size_t i = 0; i < entry.sums.size(); ++i) {
                    e[i] = entry.sums[i];
                }
            }

            if (getenv("LLAMA_TRACE")) {
                printf("%s: loaded data (size = %6d, ncall = %6d) for '%s'\n",
                        __func__, int(e.size()), int(ncall), name.c_str());
            }
        } else {
            const int64_t ncounts = entry.counts.size();
            const int64_t ne0     = (int64_t) entry.sums.size() / ncounts;

            float max_count = 0.0f;
            for (int64_t j = 0; j < ncounts; ++j) {
                const float count = (float) entry.counts[j];
                if (count > 0.0f) {
                    for (int64_t i = 0; i < ne0; ++i) {
                        e[j*ne0 + i] = entry.sums[j*ne0 + i] / count;
                    }
                } else {
                    for (int64_t i = 0; i < ne0; ++i) {
                        e[j*ne0 + i] = 1;
                    }
                }
                if (count > max_count) {
                    max_count = count;
                }
            }

            if (getenv("LLAMA_TRACE")) {
                printf("%s: loaded data (size = %6d, n_tokens = %6d, n_chunks = %6d) for '%s'\n",
                        __func__, int(e.size()), int(max_count), int(max_count / loaded.chunk_size), name.c_str());
            }
        }
    }

    imatrix_datasets = std::move(loaded.datasets);

    if (!imatrix_datasets.empty()) {
        printf("%s: imatrix datasets=['%s'", __func__, imatrix_datasets[0].c_str());
        for (size_t i = 1; i < imatrix_datasets.size(); ++i) {
            printf(", '%s'", imatrix_datasets[i].c_str());
        }
        printf("]\n");
    }

    printf("%s: loaded %d importance matrix entries from %s computed on %d chunks\n", __func__, int(imatrix_data.size()), imatrix_file.c_str(), loaded.chunk_count);

    return loaded.chunk_count;
}

int prepare_imatrix(const std::string & imatrix_file,
        std::vector<std::string> & imatrix_dataset,
        const std::vector<std::string> & included_weights,
        const std::vector<std::string> & excluded_weights,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    int m_last_call = -1;
    if (!imatrix_file.empty()) {
        m_last_call = load_imatrix(imatrix_file, imatrix_dataset, imatrix_data);
    }
    if (imatrix_data.empty()) {
        return m_last_call;
    }
    if (!excluded_weights.empty()) {
        for (const auto & name : excluded_weights) {
            for (auto it = imatrix_data.begin(); it != imatrix_data.end();) {
                auto pos = it->first.find(name);
                if (pos != std::string::npos) {
                    it = imatrix_data.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
    if (!included_weights.empty()) {
        std::unordered_map<std::string, std::vector<float>> tmp;
        for (const auto & name : included_weights) {
            for (auto & e : imatrix_data) {
                auto pos = e.first.find(name);
                if (pos != std::string::npos) {
                    tmp.emplace(std::move(e));
                }
            }
        }
        imatrix_data = std::move(tmp);
    }
    if (!imatrix_data.empty()) {
        printf("%s: have %d importance matrix entries\n", __func__, int(imatrix_data.size()));
    }
    return m_last_call;
}
