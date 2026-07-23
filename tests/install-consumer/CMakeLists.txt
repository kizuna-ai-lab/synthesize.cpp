cmake_minimum_required(VERSION 3.24)
project(synthesize-installed-consumer LANGUAGES C)

find_package(synthesize CONFIG REQUIRED)

add_executable(synthesize-installed-consumer main.c)
target_link_libraries(synthesize-installed-consumer PRIVATE synthesize::synthesize)
