set(TARGET advanced-gguf-quantizer)

add_executable(${TARGET}
    candidate_metrics.cpp
    best.cpp
    inspect.cpp
    main.cpp
    project.cpp
    recipe.cpp
    run_monitor.cpp
    run_plan.cpp
    shell_workflow.cpp
    terminal_ui.cpp
    tui.cpp
    utils.cpp
)

target_link_libraries(${TARGET} PRIVATE llama-quantize-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS ${TARGET} RUNTIME)
endif()

if(LLAMA_BUILD_TESTS)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND AND UNIX)
        add_test(
            NAME advanced-gguf-quantizer-tui-smoke
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/capture_advanced_gguf_quantizer_tui.py
                $<TARGET_FILE:${TARGET}>
                --out ${CMAKE_CURRENT_BINARY_DIR}/tui-smoke
                --case main_menu_simplicity
                --case bare_esc_exits_main_menu
                --timeout 5
                --rows 24
                --cols 100
                --cwd ${CMAKE_CURRENT_BINARY_DIR}
        )
        set_property(TEST advanced-gguf-quantizer-tui-smoke PROPERTY LABELS quantizer tui)
    endif()
endif()
