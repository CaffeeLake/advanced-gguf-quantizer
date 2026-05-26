cmake_minimum_required(VERSION 3.14...3.28) # for add_link_options and implicit target directories.
project("advanced-gguf-quantizer" C CXX)
include(CheckIncludeFileCXX)

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)

    include(git-vars)

    # configure project version
    # TODO
else()
    set(LLAMA_STANDALONE OFF)
endif()

option(LLAMA_USE_SYSTEM_GGML "Use system libggml" OFF)

option(LLAMA_WASM_MEM64 "advanced-gguf-quantizer: use 64-bit memory in WASM builds" ON)

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    # Use 64-bit memory to support backend_get_memory queries
    # TODO: analyze performance impact, see https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using
    if (LLAMA_WASM_MEM64)
      add_compile_options("-sMEMORY64=1")
      add_link_options("-sMEMORY64=1")
    endif()
    add_link_options("-sALLOW_MEMORY_GROWTH=1")

    option(LLAMA_WASM_SINGLE_FILE "llama: embed WASM inside the generated llama.js" OFF)
    option(LLAMA_BUILD_HTML "advanced-gguf-quantizer: build HTML file" ON)
    if (LLAMA_BUILD_HTML)
        set(CMAKE_EXECUTABLE_SUFFIX ".html")
    endif()
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

if (LLAMA_STANDALONE)
    # enable parallel builds for msbuild
    list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true)
    list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(LLAMA_TOOLS_INSTALL_DEFAULT OFF)
else()
    set(LLAMA_TOOLS_INSTALL_DEFAULT ${LLAMA_STANDALONE})
endif()

#
# option list
#

# debug
option(LLAMA_ALL_WARNINGS           "advanced-gguf-quantizer: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY "advanced-gguf-quantizer: enable all compiler warnings in 3rd party libs" OFF)

# build
option(LLAMA_FATAL_WARNINGS "advanced-gguf-quantizer: enable -Werror flag" OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD    "advanced-gguf-quantizer: enable thread sanitizer"    OFF)
option(LLAMA_SANITIZE_ADDRESS   "advanced-gguf-quantizer: enable address sanitizer"   OFF)
option(LLAMA_SANITIZE_UNDEFINED "advanced-gguf-quantizer: enable undefined sanitizer" OFF)

# utils
option(LLAMA_BUILD_COMMON "advanced-gguf-quantizer: build retained common utils library" ${LLAMA_STANDALONE})

# extra artifacts
option(LLAMA_BUILD_TESTS     "advanced-gguf-quantizer: build focused tests"                                                   ${LLAMA_STANDALONE})
option(LLAMA_BUILD_TOOLS     "advanced-gguf-quantizer: build retained tools"                                                   ${LLAMA_STANDALONE})
option(LLAMA_BUILD_EXAMPLES  "advanced-gguf-quantizer: build upstream examples when standalone mode is disabled"               ${LLAMA_STANDALONE})
option(ADVANCED_GGUF_QUANTIZER_STANDALONE "advanced-gguf-quantizer: build only the GGUF quantization tool surface" ON)
option(ADVANCED_GGUF_QUANTIZER_BUILD_BENCH "advanced-gguf-quantizer: build optional llama-bench evidence helper" OFF)
option(ADVANCED_GGUF_QUANTIZER_BUILD_FIT_PARAMS "advanced-gguf-quantizer: build optional llama-fit-params sizing helper" OFF)

if (ADVANCED_GGUF_QUANTIZER_STANDALONE)
    set(LLAMA_BUILD_COMMON  ON  CACHE BOOL "advanced-gguf-quantizer: build retained common utils library" FORCE)
    set(LLAMA_BUILD_TOOLS   ON  CACHE BOOL "advanced-gguf-quantizer: build retained tools" FORCE)
    set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "advanced-gguf-quantizer: build focused tests" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "advanced-gguf-quantizer: build upstream examples" FORCE)
    set(LLAMA_OPENSSL       OFF CACHE BOOL "advanced-gguf-quantizer: use openssl for non-standalone HTTPS helpers" FORCE)
    set(LLAMA_USE_SYSTEM_GGML OFF CACHE BOOL "Use system libggml" FORCE)
endif()

option(LLAMA_TOOLS_INSTALL "advanced-gguf-quantizer: install retained tools" ${LLAMA_TOOLS_INSTALL_DEFAULT})
option(LLAMA_TESTS_INSTALL "advanced-gguf-quantizer: install focused tests" ON)

# 3rd party libs
option(LLAMA_OPENSSL    "advanced-gguf-quantizer: use openssl for non-standalone HTTPS helpers" ON)
option(LLAMA_LLGUIDANCE "llama-common: include LLGuidance library for structured output in retained common utils" OFF)


# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

if (NOT DEFINED LLAMA_BUILD_NUMBER)
    set(LLAMA_BUILD_NUMBER        ${BUILD_NUMBER})
endif()
if (NOT DEFINED LLAMA_BUILD_COMMIT)
    set(LLAMA_BUILD_COMMIT        ${BUILD_COMMIT})
endif()
set(LLAMA_INSTALL_VERSION 0.0.${LLAMA_BUILD_NUMBER})

# override ggml options
set(GGML_ALL_WARNINGS   ${LLAMA_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${LLAMA_FATAL_WARNINGS})

# change the default for these ggml options
if (NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif()

if (NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()

# transition helpers
function (llama_option_depr TYPE OLD)
    if (${OLD})
        set(NEW "${ARGV2}")
        if(NEW)
            message(${TYPE} "${OLD} is deprecated, use ${NEW} instead")
            set(${NEW} ON PARENT_SCOPE)
        else()
            message(${TYPE} "${OLD} is deprecated and will be ignored")
        endif()
    endif()
endfunction()

llama_option_depr(FATAL_ERROR LLAMA_CUBLAS              GGML_CUDA)
llama_option_depr(WARNING     LLAMA_CUDA                GGML_CUDA)
llama_option_depr(WARNING     LLAMA_METAL               GGML_METAL)
llama_option_depr(WARNING     LLAMA_METAL_EMBED_LIBRARY GGML_METAL_EMBED_LIBRARY)
llama_option_depr(WARNING     LLAMA_NATIVE              GGML_NATIVE)
llama_option_depr(WARNING     LLAMA_RPC                 GGML_RPC)
llama_option_depr(WARNING     LLAMA_SYCL                GGML_SYCL)
llama_option_depr(WARNING     LLAMA_SYCL_F16            GGML_SYCL_F16)
llama_option_depr(WARNING     LLAMA_CANN                GGML_CANN)
llama_option_depr(WARNING     LLAMA_CURL)

if (ADVANCED_GGUF_QUANTIZER_STANDALONE)
    # This tool is a CUDA Blackwell quantizer: CPU threads feed the work and
    # CUDA runs the accelerated NVFP4/MXFP6 quant/eval kernels. Keep other
    # backends out of the standalone build until they have matching kernels.
    set(GGML_CPU               ON  CACHE BOOL "ggml: enable CPU backend" FORCE)
    set(GGML_CUDA              ON  CACHE BOOL "ggml: use CUDA" FORCE)
    set(GGML_BACKEND_DL        OFF CACHE BOOL "ggml: build backends as dynamic libraries" FORCE)
    set(GGML_BACKEND_DIR       ""  CACHE PATH "ggml: directory to load dynamic backends from" FORCE)
    set(GGML_CPU_ALL_VARIANTS  OFF CACHE BOOL "ggml: build all CPU backend variants" FORCE)
    set(GGML_ACCELERATE        OFF CACHE BOOL "ggml: enable Accelerate framework" FORCE)
    set(GGML_BLAS              OFF CACHE BOOL "ggml: use BLAS" FORCE)
    set(GGML_METAL             OFF CACHE BOOL "ggml: use Metal" FORCE)
    set(GGML_METAL_EMBED_LIBRARY OFF CACHE BOOL "ggml: embed Metal library" FORCE)
    set(GGML_HIP               OFF CACHE BOOL "ggml: use HIP" FORCE)
    set(GGML_MUSA              OFF CACHE BOOL "ggml: use MUSA" FORCE)
    set(GGML_VULKAN            OFF CACHE BOOL "ggml: use Vulkan" FORCE)
    set(GGML_WEBGPU            OFF CACHE BOOL "ggml: use WebGPU" FORCE)
    set(GGML_ZDNN              OFF CACHE BOOL "ggml: use zDNN" FORCE)
    set(GGML_VIRTGPU           OFF CACHE BOOL "ggml: use VirtGPU" FORCE)
    set(GGML_VIRTGPU_BACKEND   OFF CACHE BOOL "ggml: build VirtGPU backend" FORCE)
    set(GGML_RPC               OFF CACHE BOOL "ggml: use RPC" FORCE)
    set(GGML_SYCL              OFF CACHE BOOL "ggml: use SYCL" FORCE)
    set(GGML_OPENCL            OFF CACHE BOOL "ggml: use OpenCL" FORCE)
    set(GGML_OPENVINO          OFF CACHE BOOL "ggml: use OpenVINO" FORCE)
    set(GGML_CANN              OFF CACHE BOOL "ggml: use CANN" FORCE)
    set(GGML_HEXAGON           OFF CACHE BOOL "ggml: enable Hexagon backend" FORCE)
    set(GGML_ZENDNN            OFF CACHE BOOL "ggml: use ZenDNN" FORCE)
endif()

include("cmake/license.cmake")
license_add_file("${PROJECT_NAME}" "LICENSE")

#
# 3rd-party
#

if (LLAMA_USE_SYSTEM_GGML)
    message(STATUS "Using system-provided libggml, skipping ggml build")
    find_package(ggml REQUIRED)
    add_library(ggml ALIAS ggml::ggml)
endif()

if (NOT TARGET ggml AND NOT LLAMA_USE_SYSTEM_GGML)
    set(GGML_BUILD_NUMBER ${LLAMA_BUILD_NUMBER})
    set(GGML_BUILD_COMMIT ${LLAMA_BUILD_COMMIT})
    add_subdirectory(ggml)
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

#
# build the library
#

add_subdirectory(src)

#
# utils, programs, examples and tests
#

if (LLAMA_BUILD_COMMON)
    add_subdirectory(common)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# Automatically add all files from the 'licenses' directory
file(GLOB EXTRA_LICENSES "${CMAKE_SOURCE_DIR}/licenses/LICENSE-*")

foreach(FILE_PATH ${EXTRA_LICENSES})
    get_filename_component(FILE_NAME "${FILE_PATH}" NAME)
    string(REGEX REPLACE "^LICENSE-" "" NAME "${FILE_NAME}")
    license_add_file("${NAME}" "${FILE_PATH}")
endforeach()

if (LLAMA_BUILD_COMMON)
    license_generate(llama-common)
endif()

if (NOT ADVANCED_GGUF_QUANTIZER_STANDALONE)
    #
    # install
    #

    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
    set(LLAMA_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
    set(LLAMA_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

    set(LLAMA_PUBLIC_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/llama.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/llama-cpp.h)

    set_target_properties(llama
        PROPERTIES
            PUBLIC_HEADER "${LLAMA_PUBLIC_HEADERS}")

    install(TARGETS llama LIBRARY PUBLIC_HEADER)

    if (LLAMA_BUILD_COMMON)
        install(TARGETS llama-common LIBRARY)
    endif()

    configure_package_config_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/advanced-gguf-quantizer-config.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/advanced-gguf-quantizer-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/advanced-gguf-quantizer
        PATH_VARS LLAMA_INCLUDE_INSTALL_DIR
                  LLAMA_LIB_INSTALL_DIR
                  LLAMA_BIN_INSTALL_DIR )

    write_basic_package_version_file(
            ${CMAKE_CURRENT_BINARY_DIR}/advanced-gguf-quantizer-version.cmake
        VERSION ${LLAMA_INSTALL_VERSION}
        COMPATIBILITY SameMajorVersion)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/advanced-gguf-quantizer-config.cmake
                  ${CMAKE_CURRENT_BINARY_DIR}/advanced-gguf-quantizer-version.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/advanced-gguf-quantizer)

    configure_file(cmake/advanced-gguf-quantizer.pc.in
            "${CMAKE_CURRENT_BINARY_DIR}/advanced-gguf-quantizer.pc"
            @ONLY)

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/advanced-gguf-quantizer.pc"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()
