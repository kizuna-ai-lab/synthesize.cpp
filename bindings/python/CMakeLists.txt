cmake_minimum_required(VERSION 3.26)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../include/synthesize.h" _synth_header)
foreach(_synth_version_component MAJOR MINOR PATCH)
    string(REGEX MATCH
        "define +SYNTH_VERSION_${_synth_version_component} +([0-9]+)"
        _synth_version_match
        "${_synth_header}")
    set(SYNTH_VERSION_${_synth_version_component} "${CMAKE_MATCH_1}")
    if(NOT SYNTH_VERSION_${_synth_version_component} MATCHES "^[0-9]+$")
        message(FATAL_ERROR
            "Could not parse SYNTH_VERSION_${_synth_version_component} from include/synthesize.h")
    endif()
endforeach()

project(synthesize_python_adapter
    VERSION
        ${SYNTH_VERSION_MAJOR}.${SYNTH_VERSION_MINOR}.${SYNTH_VERSION_PATCH}
    LANGUAGES C)

file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/../../include/synthesize.h"
    SYNTH_ADAPTER_PUBLIC_HEADER_HASH)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/_build_contract.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/_build_contract.py"
    @ONLY)

find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter Development.SABIModule)

Python3_add_library(_native MODULE USE_SABI 3.11 WITH_SOABI
    src/native_loader.c)
target_compile_features(_native PRIVATE c_std_11)
target_include_directories(_native PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_link_libraries(_native PRIVATE Python3::SABIModule ${CMAKE_DL_LIBS})
set_target_properties(_native PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)

install(TARGETS _native
    LIBRARY DESTINATION synthesize_cpp COMPONENT wheel
    RUNTIME DESTINATION synthesize_cpp COMPONENT wheel)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/_build_contract.py"
    DESTINATION synthesize_cpp COMPONENT wheel)
