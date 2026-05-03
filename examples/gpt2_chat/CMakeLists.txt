# Example: minimal GPT-2 chat demo. Disabled by default; enable with
#   cmake -DFDEEP_BUILD_GPT2_CHAT=ON ..
# The example needs the rebuilt model JSON and the GPT-2 vocab/merges, see
# keras_export/save_gpt2_backbone_for_fdeep.py and the README in this directory.

option(FDEEP_BUILD_GPT2_CHAT "Build the GPT-2 chat demo example" OFF)

if(FDEEP_BUILD_GPT2_CHAT)
    add_executable(gpt2_chat main.cpp)
    target_link_libraries(gpt2_chat PRIVATE fdeep)
    if(NOT MSVC)
        target_compile_options(gpt2_chat PRIVATE -O3)
    endif()
endif()
