foreach(_synth_required_variable
        SYNTH_SOURCE_DIR SYNTH_BINARY_DIR SYNTH_INSTALL_TEST_ROOT
        SYNTH_TEST_GENERATOR SYNTH_TEST_C_COMPILER SYNTH_TEST_SHARED
        SYNTH_TEST_CUDA SYNTH_TEST_SANITIZE)
    if(NOT DEFINED ${_synth_required_variable} OR
       "${${_synth_required_variable}}" STREQUAL "")
        message(FATAL_ERROR
            "check-installed-sdk requires ${_synth_required_variable}")
    endif()
endforeach()
if(SYNTH_TEST_CUDA AND
   (NOT DEFINED SYNTH_TEST_CUDA_TOOLKIT_ROOT OR
    "${SYNTH_TEST_CUDA_TOOLKIT_ROOT}" STREQUAL ""))
    message(FATAL_ERROR
        "CUDA installed SDK test requires SYNTH_TEST_CUDA_TOOLKIT_ROOT")
endif()

set(_synth_stage "${SYNTH_INSTALL_TEST_ROOT}/stage")
set(_synth_consumer_build "${SYNTH_INSTALL_TEST_ROOT}/consumer-build")
set(_synth_cuda_loader_environment "")
if(SYNTH_TEST_CUDA)
    set(_synth_cuda_library_path
        "${SYNTH_TEST_CUDA_TOOLKIT_ROOT}/lib64")
    set(_synth_cuda_loader_environment
        "LD_LIBRARY_PATH=${_synth_cuda_library_path}:$ENV{LD_LIBRARY_PATH}")
endif()
file(REMOVE_RECURSE "${SYNTH_INSTALL_TEST_ROOT}")
file(MAKE_DIRECTORY "${SYNTH_INSTALL_TEST_ROOT}")

execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${SYNTH_BINARY_DIR}"
        --prefix "${_synth_stage}"
    RESULT_VARIABLE _synth_install_status
    OUTPUT_VARIABLE _synth_install_output
    ERROR_VARIABLE _synth_install_error)
if(NOT _synth_install_status EQUAL 0)
    message(FATAL_ERROR
        "SDK install failed (${_synth_install_status}):\n"
        "${_synth_install_output}\n${_synth_install_error}")
endif()

foreach(_synth_required_path
        "${_synth_stage}/include/synthesize.h"
        "${_synth_stage}/lib/cmake/synthesize/synthesize-config.cmake"
        "${_synth_stage}/lib/cmake/synthesize/synthesize-config-version.cmake"
        "${_synth_stage}/lib/cmake/synthesize/synthesize-targets.cmake"
        "${_synth_stage}/lib/pkgconfig/synthesize.pc")
    if(NOT EXISTS "${_synth_required_path}")
        message(FATAL_ERROR
            "installed SDK is missing ${_synth_required_path}")
    endif()
endforeach()

set(_synth_consumer_configure_command
    ${CMAKE_COMMAND}
        -S "${SYNTH_SOURCE_DIR}/tests/install-consumer"
        -B "${_synth_consumer_build}"
        -G "${SYNTH_TEST_GENERATOR}"
        -DCMAKE_C_COMPILER=${SYNTH_TEST_C_COMPILER}
        -DCMAKE_PREFIX_PATH=${_synth_stage})
if(SYNTH_TEST_CUDA)
    list(APPEND _synth_consumer_configure_command
        -DCUDAToolkit_ROOT=${SYNTH_TEST_CUDA_TOOLKIT_ROOT})
endif()
if(SYNTH_TEST_SANITIZE)
    list(APPEND _synth_consumer_configure_command
        -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined)
endif()
execute_process(
    COMMAND ${_synth_consumer_configure_command}
    RESULT_VARIABLE _synth_configure_status
    OUTPUT_VARIABLE _synth_configure_output
    ERROR_VARIABLE _synth_configure_error)
if(NOT _synth_configure_status EQUAL 0)
    message(FATAL_ERROR
        "installed SDK consumer configure failed (${_synth_configure_status}):\n"
        "${_synth_configure_output}\n${_synth_configure_error}")
endif()

set(_synth_consumer_build_command
    ${CMAKE_COMMAND} --build "${_synth_consumer_build}")
if(SYNTH_TEST_CUDA)
    set(_synth_consumer_build_command
        ${CMAKE_COMMAND} -E env
        "${_synth_cuda_loader_environment}"
        ${_synth_consumer_build_command})
endif()
execute_process(
    COMMAND ${_synth_consumer_build_command}
    RESULT_VARIABLE _synth_build_status
    OUTPUT_VARIABLE _synth_build_output
    ERROR_VARIABLE _synth_build_error)
if(NOT _synth_build_status EQUAL 0)
    message(FATAL_ERROR
        "installed SDK consumer build failed (${_synth_build_status}):\n"
        "${_synth_build_output}\n${_synth_build_error}")
endif()

set(_synth_consumer_run_command
    "${_synth_consumer_build}/synthesize-installed-consumer")
if(SYNTH_TEST_CUDA)
    set(_synth_consumer_run_command
        ${CMAKE_COMMAND} -E env
        "${_synth_cuda_loader_environment}"
        ${_synth_consumer_run_command})
endif()
execute_process(
    COMMAND ${_synth_consumer_run_command}
    RESULT_VARIABLE _synth_run_status
    OUTPUT_VARIABLE _synth_run_output
    ERROR_VARIABLE _synth_run_error)
if(NOT _synth_run_status EQUAL 0)
    message(FATAL_ERROR
        "installed SDK consumer run failed (${_synth_run_status}):\n"
        "${_synth_run_output}\n${_synth_run_error}")
endif()
if(SYNTH_TEST_CUDA AND UNIX AND NOT APPLE)
    find_program(_synth_ldd NAMES ldd REQUIRED)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
            "${_synth_cuda_loader_environment}"
            "${_synth_ldd}"
            "${_synth_consumer_build}/synthesize-installed-consumer"
        RESULT_VARIABLE _synth_ldd_status
        OUTPUT_VARIABLE _synth_ldd_output
        ERROR_VARIABLE _synth_ldd_error)
    if(NOT _synth_ldd_status EQUAL 0)
        message(FATAL_ERROR
            "installed SDK consumer ldd failed (${_synth_ldd_status}):\n"
            "${_synth_ldd_output}\n${_synth_ldd_error}")
    endif()
    foreach(_synth_cuda_soname libcudart.so.13 libcublas.so.13)
        string(FIND "${_synth_ldd_output}"
            "${_synth_cuda_soname} => ${_synth_cuda_library_path}/"
            _synth_cuda_resolution)
        if(_synth_cuda_resolution EQUAL -1)
            message(FATAL_ERROR
                "${_synth_cuda_soname} was not resolved from the requested "
                "CUDA 13.3 root ${SYNTH_TEST_CUDA_TOOLKIT_ROOT}:\n"
                "${_synth_ldd_output}")
        endif()
    endforeach()
endif()

find_program(_synth_pkg_config NAMES pkg-config REQUIRED)
set(_synth_pkg_config_args --cflags --libs)
if(NOT SYNTH_TEST_SHARED)
    list(APPEND _synth_pkg_config_args --static)
endif()
list(APPEND _synth_pkg_config_args synthesize)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
        "PKG_CONFIG_PATH=${_synth_stage}/lib/pkgconfig"
        "${_synth_pkg_config}" ${_synth_pkg_config_args}
    RESULT_VARIABLE _synth_pkg_config_status
    OUTPUT_VARIABLE _synth_pkg_config_flags
    ERROR_VARIABLE _synth_pkg_config_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _synth_pkg_config_status EQUAL 0)
    message(FATAL_ERROR
        "pkg-config query failed (${_synth_pkg_config_status}):\n"
        "${_synth_pkg_config_error}")
endif()
separate_arguments(_synth_pkg_config_flags NATIVE_COMMAND
    "${_synth_pkg_config_flags}")

set(_synth_pkg_consumer
    "${SYNTH_INSTALL_TEST_ROOT}/synthesize-pkg-config-consumer")
set(_synth_pkg_build_command
    "${SYNTH_TEST_C_COMPILER}"
    "${SYNTH_SOURCE_DIR}/tests/install-consumer/main.c"
    -o "${_synth_pkg_consumer}"
    ${_synth_pkg_config_flags})
if(SYNTH_TEST_SANITIZE)
    list(APPEND _synth_pkg_build_command -fsanitize=address,undefined)
endif()
if(SYNTH_TEST_CUDA)
    set(_synth_pkg_build_command
        ${CMAKE_COMMAND} -E env
        "LIBRARY_PATH=${_synth_cuda_library_path}:$ENV{LIBRARY_PATH}"
        "${_synth_cuda_loader_environment}"
        ${_synth_pkg_build_command})
endif()
execute_process(
    COMMAND ${_synth_pkg_build_command}
    RESULT_VARIABLE _synth_pkg_build_status
    OUTPUT_VARIABLE _synth_pkg_build_output
    ERROR_VARIABLE _synth_pkg_build_error)
if(NOT _synth_pkg_build_status EQUAL 0)
    message(FATAL_ERROR
        "pkg-config consumer build failed (${_synth_pkg_build_status}):\n"
        "${_synth_pkg_build_output}\n${_synth_pkg_build_error}")
endif()

set(_synth_runtime_library_path "${_synth_stage}/lib")
if(SYNTH_TEST_CUDA)
    string(APPEND _synth_runtime_library_path
        ":${SYNTH_TEST_CUDA_TOOLKIT_ROOT}/lib64")
endif()
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
        "LD_LIBRARY_PATH=${_synth_runtime_library_path}:$ENV{LD_LIBRARY_PATH}"
        "DYLD_LIBRARY_PATH=${_synth_stage}/lib"
        "${_synth_pkg_consumer}"
    RESULT_VARIABLE _synth_pkg_run_status
    OUTPUT_VARIABLE _synth_pkg_run_output
    ERROR_VARIABLE _synth_pkg_run_error)
if(NOT _synth_pkg_run_status EQUAL 0)
    message(FATAL_ERROR
        "pkg-config consumer run failed (${_synth_pkg_run_status}):\n"
        "${_synth_pkg_run_output}\n${_synth_pkg_run_error}")
endif()
