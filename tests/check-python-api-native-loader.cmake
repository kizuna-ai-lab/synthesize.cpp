foreach(_synth_required_variable
        SYNTH_SOURCE_DIR SYNTH_API_LOADER_TEST_ROOT SYNTH_TEST_GENERATOR
        SYNTH_TEST_C_COMPILER SYNTH_TEST_CXX_COMPILER SYNTH_UV_EXECUTABLE)
    if(NOT DEFINED ${_synth_required_variable} OR
       "${${_synth_required_variable}}" STREQUAL "")
        message(FATAL_ERROR
            "check-python-api-native-loader requires ${_synth_required_variable}")
    endif()
endforeach()

set(_synth_provider_build "${SYNTH_API_LOADER_TEST_ROOT}/provider-build")
set(_synth_provider_stage "${SYNTH_API_LOADER_TEST_ROOT}/provider-stage")
set(_synth_api_build "${SYNTH_API_LOADER_TEST_ROOT}/api-build")
set(_synth_api_stage "${SYNTH_API_LOADER_TEST_ROOT}/api-stage")
set(_synth_bad_library "${SYNTH_API_LOADER_TEST_ROOT}/libmissing-synth.so")
file(REMOVE_RECURSE "${SYNTH_API_LOADER_TEST_ROOT}")

execute_process(
    COMMAND ${SYNTH_UV_EXECUTABLE} run
        --project ${SYNTH_SOURCE_DIR}/scripts/envs/vits
        --locked python -c "import sys; print(sys.executable)"
    RESULT_VARIABLE _synth_python_status
    OUTPUT_VARIABLE _synth_python
    ERROR_VARIABLE _synth_python_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _synth_python_status EQUAL 0)
    message(FATAL_ERROR "could not resolve locked Python: ${_synth_python_error}")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND}
        -S "${SYNTH_SOURCE_DIR}"
        -B "${_synth_provider_build}"
        -G "${SYNTH_TEST_GENERATOR}"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${SYNTH_TEST_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${SYNTH_TEST_CXX_COMPILER}
        -DSYNTH_BUILD_PYTHON_PROVIDER=ON
        -DSYNTH_PYTHON_PROVIDER_ID=default
        -DSYNTH_BUILD_SHARED=ON
        -DSYNTH_BUILD_CLI=OFF
        -DSYNTH_BUILD_TESTS=OFF
        -DSYNTH_BUILD_PYTHON_TESTS=OFF
        -DSYNTH_BUILD_INTEGRATION_TESTS=OFF
        -DSYNTH_GGML_BACKEND_DL=ON
        -DSYNTH_CUDA=OFF
        -DSYNTH_METAL=OFF
        -DSYNTH_VULKAN=OFF
        -DGGML_NATIVE=OFF
    RESULT_VARIABLE _synth_provider_configure_status
    OUTPUT_VARIABLE _synth_provider_configure_output
    ERROR_VARIABLE _synth_provider_configure_error)
if(NOT _synth_provider_configure_status EQUAL 0)
    message(FATAL_ERROR
        "Provider configure failed:\n${_synth_provider_configure_output}\n"
        "${_synth_provider_configure_error}")
endif()
execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${_synth_provider_build}"
        --target synthesize --parallel 4
    RESULT_VARIABLE _synth_provider_build_status)
if(NOT _synth_provider_build_status EQUAL 0)
    message(FATAL_ERROR "Provider build failed")
endif()
execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${_synth_provider_build}"
        --prefix "${_synth_provider_stage}" --component wheel
    RESULT_VARIABLE _synth_provider_install_status)
if(NOT _synth_provider_install_status EQUAL 0)
    message(FATAL_ERROR "Provider install failed")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND}
        -S "${SYNTH_SOURCE_DIR}/bindings/python"
        -B "${_synth_api_build}"
        -G "${SYNTH_TEST_GENERATOR}"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${SYNTH_TEST_C_COMPILER}
        -DPython3_EXECUTABLE=${_synth_python}
    RESULT_VARIABLE _synth_api_configure_status
    OUTPUT_VARIABLE _synth_api_configure_output
    ERROR_VARIABLE _synth_api_configure_error)
if(NOT _synth_api_configure_status EQUAL 0)
    message(FATAL_ERROR
        "API loader configure failed:\n${_synth_api_configure_output}\n"
        "${_synth_api_configure_error}")
endif()
execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${_synth_api_build}" --parallel 4
    RESULT_VARIABLE _synth_api_build_status)
if(NOT _synth_api_build_status EQUAL 0)
    message(FATAL_ERROR "API loader build failed")
endif()
execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${_synth_api_build}"
        --prefix "${_synth_api_stage}" --component wheel
    RESULT_VARIABLE _synth_api_install_status)
if(NOT _synth_api_install_status EQUAL 0)
    message(FATAL_ERROR "API loader install failed")
endif()

execute_process(
    COMMAND ${SYNTH_TEST_C_COMPILER} -shared -fPIC
        "${SYNTH_SOURCE_DIR}/tests/fixtures/missing_synth_symbols.c"
        -o "${_synth_bad_library}"
    RESULT_VARIABLE _synth_bad_library_status)
if(NOT _synth_bad_library_status EQUAL 0)
    message(FATAL_ERROR "missing-symbol fixture build failed")
endif()

execute_process(
    COMMAND ${SYNTH_UV_EXECUTABLE} run
        --project ${SYNTH_SOURCE_DIR}/scripts/envs/vits
        --locked python
        ${SYNTH_SOURCE_DIR}/tests/python/api_native_loader_smoke.py
        --extension-dir ${_synth_api_stage}/synthesize_cpp
        --good-library ${_synth_provider_stage}/_native/libsynthesize.so
        --missing-symbol-library ${_synth_bad_library}
    RESULT_VARIABLE _synth_smoke_status
    OUTPUT_VARIABLE _synth_smoke_output
    ERROR_VARIABLE _synth_smoke_error)
if(NOT _synth_smoke_status EQUAL 0)
    message(FATAL_ERROR
        "API loader smoke failed:\n${_synth_smoke_output}\n${_synth_smoke_error}")
endif()
