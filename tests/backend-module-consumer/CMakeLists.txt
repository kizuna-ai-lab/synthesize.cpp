cmake_minimum_required(VERSION 3.24)
project(synthesize-backend-module-consumer LANGUAGES C)

find_package(synthesize CONFIG REQUIRED)

add_executable(synthesize-backend-module-consumer
    ../backend-module-consumer.c)
target_link_libraries(synthesize-backend-module-consumer
    PRIVATE synthesize::synthesize)
