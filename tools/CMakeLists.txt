# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# retained product tools

add_subdirectory(completion)
add_subdirectory(imatrix)
add_subdirectory(perplexity)
add_subdirectory(quantize)

if (ADVANCED_GGUF_QUANTIZER_BUILD_BENCH)
    add_subdirectory(llama-bench)
endif()

if (ADVANCED_GGUF_QUANTIZER_BUILD_FIT_PARAMS)
    add_subdirectory(fit-params)
endif()

if (TARGET llama-quantize AND TARGET advanced-gguf-quantizer)
    add_dependencies(llama-quantize advanced-gguf-quantizer)
endif()
