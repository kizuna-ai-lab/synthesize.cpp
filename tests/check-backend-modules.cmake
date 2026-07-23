foreach(_synth_required_variable
        SYNTH_SOURCE_DIR SYNTH_BACKEND_MODULE_TEST_ROOT SYNTH_TEST_GENERATOR
        SYNTH_TEST_C_COMPILER SYNTH_TEST_CXX_COMPILER)
    if(NOT DEFINED ${_synth_required_variable} OR
       "${${_synth_required_variable}}" STREQUAL "")
        message(FATAL_ERROR
            "check-backend-modules requires ${_synth_required_variable}")
    endif()
endforeach()

set(_synth_build "${SYNTH_BACKEND_MODULE_TEST_ROOT}/build")
set(_synth_stage "${SYNTH_BACKEND_MODULE_TEST_ROOT}/stage")
set(_synth_empty "${SYNTH_BACKEND_MODULE_TEST_ROOT}/empty")
set(_synth_consumer_build "${SYNTH_BACKEND_MODULE_TEST_ROOT}/consumer-build")
file(REMOVE_RECURSE "${SYNTH_BACKEND_MODULE_TEST_ROOT}")
file(MAKE_DIRECTORY "${_synth_empty}")

execute_process(
    COMMAND ${CMAKE_COMMAND}
        -S "${SYNTH_SOURCE_DIR}"
        -B "${_synth_build}"
        -G "${SYNTH_TEST_GENERATOR}"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${SYNTH_TEST_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${SYNTH_TEST_CXX_COMPILER}
        -DCMAKE_INSTALL_PREFIX=${_synth_stage}
        -DSYNTH_BUILD_SHARED=ON
        -DSYNTH_GGML_BACKEND_DL=ON
        -DSYNTH_BUILD_CLI=OFF
        -DSYNTH_BUILD_TESTS=OFF
        -DSYNTH_BUILD_PYTHON_TESTS=OFF
        -DSYNTH_BUILD_INTEGRATION_TESTS=OFF
        -DSYNTH_CUDA=OFF
        -DSYNTH_METAL=OFF
        -DSYNTH_VULKAN=OFF
        -DGGML_NATIVE=OFF
    RESULT_VARIABLE _synth_configure_status
    OUTPUT_VARIABLE _synth_configure_output
    ERROR_VARIABLE _synth_configure_error)
if(NOT _synth_configure_status EQUAL 0)
    message(FATAL_ERROR
        "dynamic backend configure failed (${_synth_configure_status}):\n"
        "${_synth_configure_output}\n${_synth_configure_error}")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${_synth_build}" --parallel 8
    RESULT_VARIABLE _synth_build_status
    OUTPUT_VARIABLE _synth_build_output
    ERROR_VARIABLE _synth_build_error)
if(NOT _synth_build_status EQUAL 0)
    message(FATAL_ERROR
        "dynamic backend build failed (${_synth_build_status}):\n"
        "${_synth_build_output}\n${_synth_build_error}")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${_synth_build}"
    RESULT_VARIABLE _synth_install_status
    OUTPUT_VARIABLE _synth_install_output
    ERROR_VARIABLE _synth_install_error)
if(NOT _synth_install_status EQUAL 0)
    message(FATAL_ERROR
        "dynamic backend install failed (${_synth_install_status}):\n"
        "${_synth_install_output}\n${_synth_install_error}")
endif()

set(_synth_module_dir "${_synth_stage}/lib/synthesize/backends")
if(WIN32)
    set(_synth_library_dir "${_synth_stage}/bin")
    set(_synth_module_dir "${_synth_stage}/bin/synthesize/backends")
else()
    set(_synth_library_dir "${_synth_stage}/lib")
endif()
if(NOT IS_DIRECTORY "${_synth_module_dir}")
    message(FATAL_ERROR "dynamic backend module directory is missing: ${_synth_module_dir}")
endif()
file(GLOB _synth_cpu_modules "${_synth_module_dir}/*ggml-cpu*")
list(LENGTH _synth_cpu_modules _synth_cpu_module_count)
if(_synth_cpu_module_count EQUAL 0)
    message(FATAL_ERROR "dynamic CPU backend module is missing from ${_synth_module_dir}")
endif()
list(GET _synth_cpu_modules 0 _synth_cpu_module)

execute_process(
    COMMAND ${CMAKE_COMMAND}
        -S "${SYNTH_SOURCE_DIR}/tests/backend-module-consumer"
        -B "${_synth_consumer_build}"
        -G "${SYNTH_TEST_GENERATOR}"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${SYNTH_TEST_C_COMPILER}
        -DCMAKE_PREFIX_PATH=${_synth_stage}
    RESULT_VARIABLE _synth_consumer_build_status
    OUTPUT_VARIABLE _synth_consumer_build_output
    ERROR_VARIABLE _synth_consumer_build_error)
if(NOT _synth_consumer_build_status EQUAL 0)
    message(FATAL_ERROR
        "dynamic backend consumer configure failed (${_synth_consumer_build_status}):\n"
        "${_synth_consumer_build_output}\n${_synth_consumer_build_error}")
endif()
execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${_synth_consumer_build}" --config Release
    RESULT_VARIABLE _synth_consumer_build_status
    OUTPUT_VARIABLE _synth_consumer_build_output
    ERROR_VARIABLE _synth_consumer_build_error)
if(NOT _synth_consumer_build_status EQUAL 0)
    message(FATAL_ERROR
        "dynamic backend consumer build failed (${_synth_consumer_build_status}):\n"
        "${_synth_consumer_build_output}\n${_synth_consumer_build_error}")
endif()
set(_synth_consumer "${_synth_consumer_build}/synthesize-backend-module-consumer")
if(WIN32)
    set(_synth_consumer "${_synth_consumer_build}/Release/synthesize-backend-module-consumer.exe")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
        "LD_LIBRARY_PATH=${_synth_library_dir}:$ENV{LD_LIBRARY_PATH}"
        "DYLD_LIBRARY_PATH=${_synth_library_dir}:$ENV{DYLD_LIBRARY_PATH}"
        "GGML_BACKEND_PATH=${_synth_cpu_module}"
        "${_synth_consumer}" "${_synth_empty}" "${_synth_module_dir}"
    RESULT_VARIABLE _synth_consumer_status
    OUTPUT_VARIABLE _synth_consumer_output
    ERROR_VARIABLE _synth_consumer_error)
if(NOT _synth_consumer_status EQUAL 0)
    message(FATAL_ERROR
        "dynamic backend consumer failed (${_synth_consumer_status}):\n"
        "${_synth_consumer_output}\n${_synth_consumer_error}")
endif()
