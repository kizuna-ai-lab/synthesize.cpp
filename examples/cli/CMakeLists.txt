add_executable(synthesize-cli
    main.cpp
    cli.cpp)

target_link_libraries(synthesize-cli PRIVATE synthesize)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(synthesize-cli PRIVATE
        -Wall
        -Wextra
        -Wshadow
        -Wpedantic)
endif()
