#include "quantize.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

static bool is_advanced_subcommand(const char * arg) {
    if (arg == nullptr) {
        return false;
    }
    static const char * const commands[] = {
        "recipe",
        "candidates",
        "best",
        "inspect",
        "kld-info",
        "kld-command",
        "imatrix-command",
        "layer-policy",
        "plan",
        "what-if",
        "project",
        "size",
        "run",
        "shell",
        "wizard",
    };
    for (const char * command : commands) {
        if (std::strcmp(arg, command) == 0) {
            return true;
        }
    }
    return false;
}

static int dispatch_advanced_quantizer(int argc, char ** argv) {
    const std::filesystem::path self = argv[0] != nullptr ? argv[0] : "";
    const std::filesystem::path tool = self.has_parent_path()
        ? self.parent_path() / "advanced-gguf-quantizer"
        : std::filesystem::path("advanced-gguf-quantizer");

#if !defined(_WIN32)
    std::vector<std::string> owned;
    owned.reserve((size_t) argc);
    owned.push_back(tool.string());
    for (int i = 1; i < argc; ++i) {
        owned.emplace_back(argv[i]);
    }

    std::vector<char *> forwarded;
    forwarded.reserve(owned.size() + 1);
    for (std::string & arg : owned) {
        forwarded.push_back(arg.data());
    }
    forwarded.push_back(nullptr);

    execv(tool.c_str(), forwarded.data());
    std::cerr
        << "llama-quantize: failed to dispatch advanced subcommand to "
        << tool << "\n";
    return 127;
#else
    (void) argc;
    std::cerr
        << "llama-quantize: advanced subcommand dispatch needs "
        << tool << " on PATH or beside llama-quantize\n";
    return 127;
#endif
}

int main(int argc, char ** argv) {
    if (argc > 1 && is_advanced_subcommand(argv[1])) {
        return dispatch_advanced_quantizer(argc, argv);
    }
    return llama_quantize(argc, argv);
}
