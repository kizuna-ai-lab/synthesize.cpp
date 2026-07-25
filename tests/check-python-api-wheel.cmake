foreach(_synth_required_variable
        SYNTH_SOURCE_DIR SYNTH_API_WHEEL_TEST_ROOT SYNTH_UV_EXECUTABLE
        SYNTH_TEST_C_COMPILER)
    if(NOT DEFINED ${_synth_required_variable} OR
       "${${_synth_required_variable}}" STREQUAL "")
        message(FATAL_ERROR
            "check-python-api-wheel requires ${_synth_required_variable}")
    endif()
endforeach()

set(_synth_wheelhouse "${SYNTH_API_WHEEL_TEST_ROOT}/wheelhouse")
set(_synth_venv "${SYNTH_API_WHEEL_TEST_ROOT}/venv")
set(_synth_bad_library "${SYNTH_API_WHEEL_TEST_ROOT}/libmissing-synth.so")
file(REMOVE_RECURSE "${SYNTH_API_WHEEL_TEST_ROOT}")
file(MAKE_DIRECTORY "${_synth_wheelhouse}")

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
    COMMAND ${SYNTH_UV_EXECUTABLE} build --wheel
        --python ${_synth_python}
        --out-dir ${_synth_wheelhouse}
        ${SYNTH_SOURCE_DIR}
    RESULT_VARIABLE _synth_provider_build_status
    OUTPUT_VARIABLE _synth_provider_build_output
    ERROR_VARIABLE _synth_provider_build_error)
if(NOT _synth_provider_build_status EQUAL 0)
    message(FATAL_ERROR
        "default Provider wheel build failed:\n${_synth_provider_build_output}\n"
        "${_synth_provider_build_error}")
endif()

execute_process(
    COMMAND ${SYNTH_UV_EXECUTABLE} build --wheel
        --python ${_synth_python}
        --out-dir ${_synth_wheelhouse}
        ${SYNTH_SOURCE_DIR}/bindings/python
    RESULT_VARIABLE _synth_api_build_status
    OUTPUT_VARIABLE _synth_api_build_output
    ERROR_VARIABLE _synth_api_build_error)
if(NOT _synth_api_build_status EQUAL 0)
    message(FATAL_ERROR
        "API wheel build failed:\n${_synth_api_build_output}\n"
        "${_synth_api_build_error}")
endif()

file(GLOB _synth_provider_wheels
    "${_synth_wheelhouse}/synthesize_cpp_native-0.1.0-py3-none-*.whl")
file(GLOB _synth_api_wheels
    "${_synth_wheelhouse}/synthesize_cpp-0.1.0-cp311-abi3-*.whl")
list(LENGTH _synth_provider_wheels _synth_provider_wheel_count)
list(LENGTH _synth_api_wheels _synth_api_wheel_count)
if(NOT _synth_provider_wheel_count EQUAL 1 OR
   NOT _synth_api_wheel_count EQUAL 1)
    message(FATAL_ERROR
        "expected one Provider and one cp311-abi3 API wheel; "
        "Provider=${_synth_provider_wheels}; API=${_synth_api_wheels}")
endif()
list(GET _synth_provider_wheels 0 _synth_provider_wheel)
list(GET _synth_api_wheels 0 _synth_api_wheel)

execute_process(
    COMMAND ${SYNTH_UV_EXECUTABLE} run
        --project ${SYNTH_SOURCE_DIR}/scripts/envs/vits
        --locked python
        ${SYNTH_SOURCE_DIR}/tests/python/validate_python_api_wheels.py
        --api-wheel ${_synth_api_wheel}
        --provider-wheel ${_synth_provider_wheel}
    RESULT_VARIABLE _synth_validate_status
    OUTPUT_VARIABLE _synth_validate_output
    ERROR_VARIABLE _synth_validate_error)
if(NOT _synth_validate_status EQUAL 0)
    message(FATAL_ERROR
        "wheel contract validation failed:\n${_synth_validate_output}\n"
        "${_synth_validate_error}")
endif()

execute_process(
    COMMAND ${SYNTH_UV_EXECUTABLE} venv --python ${_synth_python} ${_synth_venv}
    RESULT_VARIABLE _synth_venv_status
    OUTPUT_VARIABLE _synth_venv_output
    ERROR_VARIABLE _synth_venv_error)
if(NOT _synth_venv_status EQUAL 0)
    message(FATAL_ERROR "venv creation failed: ${_synth_venv_error}")
endif()
set(_synth_venv_python "${_synth_venv}/bin/python")
execute_process(
    COMMAND ${SYNTH_UV_EXECUTABLE} pip install
        --python ${_synth_venv_python} --no-deps
        ${_synth_provider_wheel} ${_synth_api_wheel}
    RESULT_VARIABLE _synth_install_status
    OUTPUT_VARIABLE _synth_install_output
    ERROR_VARIABLE _synth_install_error)
if(NOT _synth_install_status EQUAL 0)
    message(FATAL_ERROR
        "wheel installation failed:\n${_synth_install_output}\n"
        "${_synth_install_error}")
endif()

file(GLOB _synth_extensions
    "${_synth_venv}/lib/python*/site-packages/synthesize_cpp/_native*.so")
file(GLOB _synth_provider_libraries
    "${_synth_venv}/lib/python*/site-packages/synthesize_cpp_native/_native/libsynthesize.so")
list(LENGTH _synth_extensions _synth_extension_count)
list(LENGTH _synth_provider_libraries _synth_provider_library_count)
if(NOT _synth_extension_count EQUAL 1 OR
   NOT _synth_provider_library_count EQUAL 1)
    message(FATAL_ERROR
        "installed wheel contents are incomplete: extension=${_synth_extensions}; "
        "Provider=${_synth_provider_libraries}")
endif()
list(GET _synth_extensions 0 _synth_extension)
get_filename_component(_synth_extension_dir "${_synth_extension}" DIRECTORY)
list(GET _synth_provider_libraries 0 _synth_provider_library)

execute_process(
    COMMAND ${SYNTH_TEST_C_COMPILER} -shared -fPIC
        ${SYNTH_SOURCE_DIR}/tests/fixtures/missing_synth_symbols.c
        -o ${_synth_bad_library}
    RESULT_VARIABLE _synth_bad_library_status)
if(NOT _synth_bad_library_status EQUAL 0)
    message(FATAL_ERROR "missing-symbol fixture build failed")
endif()

execute_process(
    COMMAND ${_synth_venv_python}
        ${SYNTH_SOURCE_DIR}/tests/python/api_native_loader_smoke.py
        --extension-dir ${_synth_extension_dir}
        --good-library ${_synth_provider_library}
        --missing-symbol-library ${_synth_bad_library}
    RESULT_VARIABLE _synth_native_smoke_status
    OUTPUT_VARIABLE _synth_native_smoke_output
    ERROR_VARIABLE _synth_native_smoke_error)
if(NOT _synth_native_smoke_status EQUAL 0)
    message(FATAL_ERROR
        "installed native loader smoke failed:\n${_synth_native_smoke_output}\n"
        "${_synth_native_smoke_error}")
endif()

execute_process(
    COMMAND ${_synth_venv_python}
        ${SYNTH_SOURCE_DIR}/tests/python/api_wheel_runtime_smoke.py
    RESULT_VARIABLE _synth_runtime_status
    OUTPUT_VARIABLE _synth_runtime_output
    ERROR_VARIABLE _synth_runtime_error)
if(NOT _synth_runtime_status EQUAL 0)
    message(FATAL_ERROR
        "installed API runtime smoke failed:\n${_synth_runtime_output}\n"
        "${_synth_runtime_error}")
endif()

execute_process(
    COMMAND ${_synth_venv_python}
        ${SYNTH_SOURCE_DIR}/tests/python/api_wheel_lifecycle_smoke.py
        --model ${SYNTH_SOURCE_DIR}/models/vits-ljspeech/vits-ljspeech-F32.gguf
    RESULT_VARIABLE _synth_lifecycle_status
    OUTPUT_VARIABLE _synth_lifecycle_output
    ERROR_VARIABLE _synth_lifecycle_error)
if(NOT _synth_lifecycle_status EQUAL 0)
    message(FATAL_ERROR
        "installed API lifecycle smoke failed:\n${_synth_lifecycle_output}\n"
        "${_synth_lifecycle_error}")
endif()

execute_process(
    COMMAND ${_synth_venv_python}
        ${SYNTH_SOURCE_DIR}/tests/python/api_wheel_synthesis_smoke.py
        --model ${SYNTH_SOURCE_DIR}/models/vits-ljspeech/vits-ljspeech-F32.gguf
        --tokens ${SYNTH_SOURCE_DIR}/build/goldens/vits/vits-ljspeech/ljs-minimal/input/token_ids.i32
    RESULT_VARIABLE _synth_synthesis_status
    OUTPUT_VARIABLE _synth_synthesis_output
    ERROR_VARIABLE _synth_synthesis_error)
if(NOT _synth_synthesis_status EQUAL 0)
    message(FATAL_ERROR
        "installed API synthesis smoke failed:\n${_synth_synthesis_output}\n"
        "${_synth_synthesis_error}")
endif()

# Every Model Family the build can see is synthesized through the installed
# wheel. Adding a family means adding a block here, not a script: the smoke
# itself asserts only what holds for all of them.
#
# A family that reaches the C interface without reaching the binding would
# otherwise ship untested through the wrapper that most callers actually use.
function(synth_family_smoke label model tokens sample_rate samples_per_frame voice seed_changes_length)
    if(NOT EXISTS "${model}")
        message(STATUS "skipping the ${label} binding smoke: ${model} is absent")
        return()
    endif()
    set(voice_arguments "")
    if(NOT voice STREQUAL "")
        set(voice_arguments --voice "${voice}")
    endif()
    set(seed_arguments "")
    if(seed_changes_length)
        set(seed_arguments --seed-changes-length)
    endif()
    execute_process(
        COMMAND ${_synth_venv_python}
            ${SYNTH_SOURCE_DIR}/tests/python/api_wheel_family_smoke.py
            --model "${model}"
            --tokens "${tokens}"
            --sample-rate "${sample_rate}"
            --samples-per-frame "${samples_per_frame}"
            ${voice_arguments}
            ${seed_arguments}
        RESULT_VARIABLE _synth_family_status
        OUTPUT_VARIABLE _synth_family_output
        ERROR_VARIABLE _synth_family_error)
    if(NOT _synth_family_status EQUAL 0)
        message(FATAL_ERROR
            "${label} binding smoke failed:\n${_synth_family_output}\n${_synth_family_error}")
    endif()
endfunction()

synth_family_smoke("VITS"
    "${SYNTH_SOURCE_DIR}/models/vits-ljspeech/vits-ljspeech-F32.gguf"
    "0,156,0,47,0,102,0,4,0" 22050 256 "" TRUE)
synth_family_smoke("Kokoro"
    "${SYNTH_SOURCE_DIR}/models/kokoro-v1-0/kokoro-v1-0-F16.gguf"
    "0,50,83,54,156,31,3,16,65,156,87,123,54,46,5,0" 24000 600 "af_heart" FALSE)
