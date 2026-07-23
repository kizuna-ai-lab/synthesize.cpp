foreach(_synth_required_variable
        SYNTH_SOURCE_DIR SYNTH_PROVIDER_TEST_ROOT SYNTH_TEST_GENERATOR
        SYNTH_TEST_C_COMPILER SYNTH_TEST_CXX_COMPILER
        SYNTH_TEST_SYSTEM_PROCESSOR)
    if(NOT DEFINED ${_synth_required_variable} OR
       "${${_synth_required_variable}}" STREQUAL "")
        message(FATAL_ERROR
            "check-python-provider-stage requires ${_synth_required_variable}")
    endif()
endforeach()

set(_synth_build "${SYNTH_PROVIDER_TEST_ROOT}/build")
set(_synth_stage "${SYNTH_PROVIDER_TEST_ROOT}/stage")
file(REMOVE_RECURSE "${SYNTH_PROVIDER_TEST_ROOT}")

execute_process(
    COMMAND ${CMAKE_COMMAND}
        -S "${SYNTH_SOURCE_DIR}"
        -B "${_synth_build}"
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
    RESULT_VARIABLE _synth_configure_status
    OUTPUT_VARIABLE _synth_configure_output
    ERROR_VARIABLE _synth_configure_error)
if(NOT _synth_configure_status EQUAL 0)
    message(FATAL_ERROR
        "provider fixture configure failed (${_synth_configure_status}):\n"
        "${_synth_configure_output}\n${_synth_configure_error}")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${_synth_build}"
        --target synthesize --parallel 4
    RESULT_VARIABLE _synth_build_status
    OUTPUT_VARIABLE _synth_build_output
    ERROR_VARIABLE _synth_build_error)
if(NOT _synth_build_status EQUAL 0)
    message(FATAL_ERROR
        "provider fixture build failed (${_synth_build_status}):\n"
        "${_synth_build_output}\n${_synth_build_error}")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${_synth_build}"
        --prefix "${_synth_stage}" --component wheel
    RESULT_VARIABLE _synth_install_status
    OUTPUT_VARIABLE _synth_install_output
    ERROR_VARIABLE _synth_install_error)
if(NOT _synth_install_status EQUAL 0)
    message(FATAL_ERROR
        "provider fixture install failed (${_synth_install_status}):\n"
        "${_synth_install_output}\n${_synth_install_error}")
endif()

set(_synth_native "${_synth_stage}/_native")
foreach(_synth_required_path
        "${_synth_stage}/_contract.py"
        "${_synth_native}/contract.json"
        "${_synth_native}/libsynthesize.so"
        "${_synth_native}/libggml.so"
        "${_synth_native}/libggml-base.so")
    if(NOT EXISTS "${_synth_required_path}")
        message(FATAL_ERROR
            "provider wheel component is missing ${_synth_required_path}")
    endif()
endforeach()
if(SYNTH_TEST_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(_synth_cpu_variants
        x64
        sse42
        sandybridge
        ivybridge
        piledriver
        haswell
        skylakex
        cannonlake
        cascadelake
        icelake
        cooperlake
        zen4
        alderlake
        sapphirerapids)
    foreach(_synth_cpu_variant IN LISTS _synth_cpu_variants)
        set(_synth_cpu_module
            "${_synth_native}/synthesize/backends/libggml-cpu-${_synth_cpu_variant}.so")
        if(NOT EXISTS "${_synth_cpu_module}")
            message(FATAL_ERROR
                "x86 Provider wheel component is missing ${_synth_cpu_module}")
        endif()
    endforeach()
    if(EXISTS "${_synth_native}/synthesize/backends/libggml-cpu.so")
        message(FATAL_ERROR
            "x86 Provider wheel component contains a non-dispatched CPU module")
    endif()
else()
    set(_synth_cpu_module
        "${_synth_native}/synthesize/backends/libggml-cpu.so")
    if(NOT EXISTS "${_synth_cpu_module}")
        message(FATAL_ERROR
            "Provider wheel component is missing ${_synth_cpu_module}")
    endif()
endif()

if(EXISTS "${_synth_native}/libggml-cpu.so")
    message(FATAL_ERROR
        "provider wheel component installed the CPU module beside the core library")
endif()

foreach(_synth_forbidden_path
        "${_synth_stage}/include"
        "${_synth_stage}/lib/cmake"
        "${_synth_stage}/lib/pkgconfig")
    if(EXISTS "${_synth_forbidden_path}")
        message(FATAL_ERROR
            "provider wheel component leaked SDK path ${_synth_forbidden_path}")
    endif()
endforeach()

file(GLOB _synth_versioned_libraries
    "${_synth_native}/*.so.*"
    "${_synth_native}/synthesize/backends/*.so.*")
if(_synth_versioned_libraries)
    message(FATAL_ERROR
        "provider wheel component contains versioned/symlink duplicates: "
        "${_synth_versioned_libraries}")
endif()

file(READ "${_synth_native}/contract.json" _synth_contract)
string(JSON _synth_contract_version GET "${_synth_contract}" version)
string(JSON _synth_contract_provider GET "${_synth_contract}" provider_id)
string(JSON _synth_contract_backend_count
    LENGTH "${_synth_contract}" backends)
string(JSON _synth_contract_backend GET "${_synth_contract}" backends 0)
string(JSON _synth_contract_hash GET "${_synth_contract}" header_hash)
file(SHA256 "${SYNTH_SOURCE_DIR}/include/synthesize.h" _synth_header_hash)
if(NOT _synth_contract_version STREQUAL "0.1.0" OR
   NOT _synth_contract_provider STREQUAL "default" OR
   NOT _synth_contract_backend_count EQUAL 1 OR
   NOT _synth_contract_backend STREQUAL "cpu" OR
   NOT _synth_contract_hash STREQUAL _synth_header_hash)
    message(FATAL_ERROR
        "provider contract mismatch: ${_synth_contract}")
endif()

file(READ "${_synth_stage}/_contract.py" _synth_python_contract)
foreach(_synth_expected_fragment
        "VERSION = \"0.1.0\""
        "PUBLIC_HEADER_HASH = \"${_synth_header_hash}\""
        "PROVIDER_ID = \"default\""
        "BACKENDS = (\"cpu\",)")
    string(FIND "${_synth_python_contract}" "${_synth_expected_fragment}"
        _synth_fragment_position)
    if(_synth_fragment_position EQUAL -1)
        message(FATAL_ERROR
            "generated Python contract is missing ${_synth_expected_fragment}:\n"
            "${_synth_python_contract}")
    endif()
endforeach()
