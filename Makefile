define newline


endef

$(error Build system changed:$(newline)\
The Makefile build has been replaced by CMake.$(newline)$(newline)\
For build instructions see:$(newline)\
README.md$(newline)$(newline)\
Quick start: cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(newline)${newline})
