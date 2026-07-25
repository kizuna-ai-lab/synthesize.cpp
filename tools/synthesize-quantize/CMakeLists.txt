add_executable(synthesize-quantize
    main.cpp
    policy.cpp
    quantize.cpp)
target_include_directories(synthesize-quantize PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-quantize PRIVATE synthesize ggml)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(synthesize-quantize PRIVATE
        -Wall
        -Wextra
        -Wshadow
        -Wpedantic)
endif()
