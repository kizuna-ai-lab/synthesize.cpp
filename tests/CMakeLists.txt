function(synth_add_unit_test target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE synthesize ggml)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "unit")
endfunction()

add_executable(synthesize-abi-smoke abi_smoke.c)
target_link_libraries(synthesize-abi-smoke PRIVATE synthesize)
add_test(NAME synthesize-abi-smoke COMMAND synthesize-abi-smoke)
set_tests_properties(synthesize-abi-smoke PROPERTIES LABELS "unit;abi")
add_executable(synthesize-abi-initializer-bounds-test abi_initializer_bounds_test.c)
target_link_libraries(synthesize-abi-initializer-bounds-test PRIVATE synthesize)
add_test(NAME synthesize-abi-initializer-bounds-test COMMAND synthesize-abi-initializer-bounds-test)
set_tests_properties(synthesize-abi-initializer-bounds-test PROPERTIES LABELS "unit;abi")
add_executable(synthesize-public-api-test public_api_test.c)
target_link_libraries(synthesize-public-api-test PRIVATE synthesize)
add_test(NAME synthesize-public-api-test COMMAND synthesize-public-api-test)
set_tests_properties(synthesize-public-api-test PROPERTIES LABELS "unit;abi")
add_executable(synthesize-voice-profile-api-test voice_profile_api_test.c)
target_link_libraries(synthesize-voice-profile-api-test PRIVATE synthesize)
add_test(NAME synthesize-voice-profile-api-test COMMAND synthesize-voice-profile-api-test)
set_tests_properties(synthesize-voice-profile-api-test PROPERTIES LABELS "unit;abi;voice-profile")
add_executable(synthesize-backend-device-test backend_device_test.c)
target_link_libraries(synthesize-backend-device-test PRIVATE synthesize)
add_test(NAME synthesize-backend-device-test COMMAND synthesize-backend-device-test)
set_tests_properties(synthesize-backend-device-test PROPERTIES LABELS "unit;abi;backend")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/backend-modules")
add_executable(synthesize-backend-module-test backend_module_test.c)
target_link_libraries(synthesize-backend-module-test PRIVATE synthesize)
add_test(NAME synthesize-backend-module-test
    COMMAND synthesize-backend-module-test
        "${CMAKE_CURRENT_BINARY_DIR}/fixtures/backend-modules")
set_tests_properties(synthesize-backend-module-test PROPERTIES LABELS "unit;abi;backend")
add_test(NAME synthesize-backend-module-dl-test
    COMMAND ${CMAKE_COMMAND}
        -DSYNTH_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DSYNTH_BACKEND_MODULE_TEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/fixtures/backend-module-dl
        -DSYNTH_TEST_GENERATOR=${CMAKE_GENERATOR}
        -DSYNTH_TEST_C_COMPILER=${CMAKE_C_COMPILER}
        -DSYNTH_TEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check-backend-modules.cmake)
set_tests_properties(synthesize-backend-module-dl-test PROPERTIES
    LABELS "unit;abi;backend;packaging"
    RESOURCE_LOCK synth_dynamic_backend_fixture
    TIMEOUT 180)
set(_synth_test_cuda_toolkit_root "")
if(SYNTH_CUDA)
    get_filename_component(_synth_test_cuda_toolkit_root
        "${CUDAToolkit_BIN_DIR}" DIRECTORY)
endif()
add_test(NAME synthesize-installed-sdk-test
    COMMAND ${CMAKE_COMMAND}
        -DSYNTH_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DSYNTH_BINARY_DIR=${CMAKE_BINARY_DIR}
        -DSYNTH_INSTALL_TEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/fixtures/installed-sdk
        -DSYNTH_TEST_GENERATOR=${CMAKE_GENERATOR}
        -DSYNTH_TEST_C_COMPILER=${CMAKE_C_COMPILER}
        -DSYNTH_TEST_SHARED=$<BOOL:${SYNTH_BUILD_SHARED}>
        -DSYNTH_TEST_CUDA=$<BOOL:${SYNTH_CUDA}>
        -DSYNTH_TEST_SANITIZE=$<BOOL:${SYNTH_SANITIZE}>
        -DSYNTH_TEST_CUDA_TOOLKIT_ROOT=${_synth_test_cuda_toolkit_root}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check-installed-sdk.cmake)
set_tests_properties(synthesize-installed-sdk-test PROPERTIES
    LABELS "unit;abi;packaging"
    TIMEOUT 120)
add_test(NAME synthesize-python-provider-stage-test
    COMMAND ${CMAKE_COMMAND}
        -DSYNTH_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DSYNTH_PROVIDER_TEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/fixtures/python-provider
        -DSYNTH_TEST_GENERATOR=${CMAKE_GENERATOR}
        -DSYNTH_TEST_C_COMPILER=${CMAKE_C_COMPILER}
        -DSYNTH_TEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DSYNTH_TEST_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check-python-provider-stage.cmake)
set_tests_properties(synthesize-python-provider-stage-test PROPERTIES
    LABELS "unit;packaging;python"
    RESOURCE_LOCK synth_dynamic_backend_fixture
    TIMEOUT 180)
if(SYNTH_BUILD_PYTHON_TESTS)
    find_program(SYNTH_UV_EXECUTABLE uv REQUIRED)
    add_test(NAME synthesize-python-api-native-loader-test
        COMMAND ${CMAKE_COMMAND}
            -DSYNTH_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DSYNTH_API_LOADER_TEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/fixtures/python-api-loader
            -DSYNTH_TEST_GENERATOR=${CMAKE_GENERATOR}
            -DSYNTH_TEST_C_COMPILER=${CMAKE_C_COMPILER}
            -DSYNTH_TEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DSYNTH_UV_EXECUTABLE=${SYNTH_UV_EXECUTABLE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check-python-api-native-loader.cmake)
    set_tests_properties(synthesize-python-api-native-loader-test PROPERTIES
        LABELS "unit;abi;packaging;python"
        RESOURCE_LOCK synth_dynamic_backend_fixture
        TIMEOUT 180)
    add_test(NAME synthesize-python-api-wheel-test
        COMMAND ${CMAKE_COMMAND}
            -DSYNTH_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DSYNTH_API_WHEEL_TEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/fixtures/python-api-wheel
            -DSYNTH_UV_EXECUTABLE=${SYNTH_UV_EXECUTABLE}
            -DSYNTH_TEST_C_COMPILER=${CMAKE_C_COMPILER}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check-python-api-wheel.cmake)
    set_tests_properties(synthesize-python-api-wheel-test PROPERTIES
        LABELS "unit;abi;packaging;python"
        TIMEOUT 240)
    # Family-independent contract check over every committed Golden Manifest.
    # It runs in the VITS environment only because that environment currently
    # owns the repository's locked Python tooling; the test itself is not
    # VITS-specific and validates whatever families exist under tests/golden/.
    add_test(
        NAME synthesize-golden-manifest-contract
        COMMAND ${SYNTH_UV_EXECUTABLE} run
            --project ${CMAKE_SOURCE_DIR}/scripts/envs/vits
            --locked python -m unittest
            tests.python.test_golden_manifests)
    set_tests_properties(synthesize-golden-manifest-contract PROPERTIES
        LABELS "unit;python;golden"
        TIMEOUT 120
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
if(SYNTH_BUILD_SHARED AND UNIX AND NOT APPLE)
    add_test(NAME synthesize-shared-exports
        COMMAND ${CMAKE_COMMAND}
            -DSYNTH_NM=${CMAKE_NM}
            -DSYNTH_LIBRARY=$<TARGET_FILE:synthesize>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check-shared-exports.cmake)
    set_tests_properties(synthesize-shared-exports PROPERTIES LABELS "unit;abi")
endif()

synth_add_unit_test(synthesize-kokoro-metadata-test kokoro_metadata_test.cpp)
synth_add_unit_test(synthesize-kokoro-lstm-test kokoro_lstm_test.cpp)
synth_add_unit_test(synthesize-kokoro-catalog-test kokoro_catalog_test.cpp)
synth_add_unit_test(synthesize-kokoro-plbert-test kokoro_plbert_test.cpp)
synth_add_unit_test(synthesize-kokoro-duration-test kokoro_duration_test.cpp)
set_tests_properties(synthesize-kokoro-metadata-test synthesize-kokoro-lstm-test
    synthesize-kokoro-catalog-test synthesize-kokoro-plbert-test
    synthesize-kokoro-duration-test
    PROPERTIES LABELS "unit;kokoro")

synth_add_unit_test(synthesize-vits-text-encoder-host-test vits_text_encoder_host_test.cpp)
synth_add_unit_test(synthesize-vits-metadata-test vits_metadata_test.cpp)
synth_add_unit_test(synthesize-vits-weights-test vits_weights_test.cpp)
synth_add_unit_test(synthesize-vits-text-encoder-graph-test vits_text_encoder_graph_test.cpp)
synth_add_unit_test(synthesize-vits-duration-host-test vits_duration_host_test.cpp)
synth_add_unit_test(synthesize-vits-duration-weights-test vits_duration_weights_test.cpp)
synth_add_unit_test(synthesize-vits-duration-graph-test vits_duration_graph_test.cpp)
synth_add_unit_test(synthesize-vits-duration-path-test vits_duration_path_test.cpp)
synth_add_unit_test(synthesize-vits-prior-expansion-host-test vits_prior_expansion_host_test.cpp)
synth_add_unit_test(synthesize-vits-prior-expansion-graph-test vits_prior_expansion_graph_test.cpp)
synth_add_unit_test(synthesize-vits-latent-sampling-host-test vits_latent_sampling_host_test.cpp)
synth_add_unit_test(synthesize-vits-latent-sampling-graph-test vits_latent_sampling_graph_test.cpp)
synth_add_unit_test(synthesize-vits-flow-weights-test vits_flow_weights_test.cpp)
synth_add_unit_test(synthesize-vits-acoustic-flow-host-test vits_acoustic_flow_host_test.cpp)
synth_add_unit_test(synthesize-vits-acoustic-flow-graph-test vits_acoustic_flow_graph_test.cpp)
synth_add_unit_test(synthesize-vits-decoder-weights-test vits_decoder_weights_test.cpp)
synth_add_unit_test(synthesize-vits-waveform-decoder-host-test vits_waveform_decoder_host_test.cpp)
synth_add_unit_test(synthesize-vits-conv-test vits_conv_test.cpp)
synth_add_unit_test(synthesize-vits-transpose-conv-test vits_transpose_conv_test.cpp)
synth_add_unit_test(synthesize-vits-waveform-decoder-graph-test vits_waveform_decoder_graph_test.cpp)
synth_add_unit_test(synthesize-random-stream-test random_stream_test.cpp)
synth_add_unit_test(synthesize-backend-device-internal-test backend_device_internal_test.cpp)
synth_add_unit_test(synthesize-backend-plan-test backend_plan_test.cpp)
target_compile_definitions(synthesize-backend-plan-test PRIVATE
    SYNTH_TEST_CUDA_TF32=$<BOOL:${SYNTH_CUDA_TF32}>)
synth_add_unit_test(synthesize-text-frontend-test text_frontend_test.cpp)
synth_add_unit_test(synthesize-synthesis-request-test synthesis_request_test.cpp)
synth_add_unit_test(synthesize-audio-delivery-test audio_delivery_test.cpp)
synth_add_unit_test(synthesize-vits-runner-voice-test vits_runner_voice_test.cpp)
add_executable(synthesize-quantize-policy-test
    synthesize_quantize_policy_test.cpp
    ${CMAKE_SOURCE_DIR}/tools/synthesize-quantize/policy.cpp)
target_include_directories(synthesize-quantize-policy-test PRIVATE
    ${CMAKE_SOURCE_DIR}/tools/synthesize-quantize
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-quantize-policy-test PRIVATE ggml)
add_test(NAME synthesize-quantize-policy-test COMMAND synthesize-quantize-policy-test)
set_tests_properties(synthesize-quantize-policy-test PROPERTIES LABELS "unit;tools;quantization")
add_executable(synthesize-quantize-test
    synthesize_quantize_test.cpp
    ${CMAKE_SOURCE_DIR}/tools/synthesize-quantize/policy.cpp
    ${CMAKE_SOURCE_DIR}/tools/synthesize-quantize/quantize.cpp)
target_include_directories(synthesize-quantize-test PRIVATE
    ${CMAKE_SOURCE_DIR}/tools/synthesize-quantize
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-quantize-test PRIVATE ggml)
add_test(NAME synthesize-quantize-test
    COMMAND synthesize-quantize-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
set_tests_properties(synthesize-quantize-test PROPERTIES LABELS "unit;tools;quantization")
if(SYNTH_BUILD_TOOLS)
    add_test(NAME synthesize-quantize-cli-help-test
        COMMAND synthesize-quantize --help)
    set_tests_properties(synthesize-quantize-cli-help-test PROPERTIES
        LABELS "unit;tools;quantization;cli"
        PASS_REGULAR_EXPRESSION "F16[|]Q8_MIXED")
endif()
add_executable(synthesize-cli-unit-test
    cli_test.cpp
    ${CMAKE_SOURCE_DIR}/examples/cli/cli.cpp)
target_include_directories(synthesize-cli-unit-test PRIVATE
    ${CMAKE_SOURCE_DIR}/examples/cli ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-cli-unit-test PRIVATE synthesize)
add_test(NAME synthesize-cli-unit-test
    COMMAND synthesize-cli-unit-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
set_tests_properties(synthesize-cli-unit-test PROPERTIES LABELS "unit;cli")
add_executable(synthesize-vits-model-errors-test vits_model_errors_test.cpp)
target_include_directories(synthesize-vits-model-errors-test PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-vits-model-errors-test PRIVATE synthesize ggml)
add_test(NAME synthesize-vits-model-errors-test
    COMMAND synthesize-vits-model-errors-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
set_tests_properties(
    synthesize-vits-text-encoder-host-test
    synthesize-vits-metadata-test
    synthesize-vits-weights-test
    synthesize-vits-text-encoder-graph-test
    synthesize-vits-duration-host-test
    synthesize-vits-duration-weights-test
    synthesize-vits-duration-graph-test
    synthesize-vits-duration-path-test
    synthesize-vits-prior-expansion-host-test
    synthesize-vits-prior-expansion-graph-test
    synthesize-vits-latent-sampling-host-test
    synthesize-vits-latent-sampling-graph-test
    synthesize-vits-flow-weights-test
    synthesize-vits-acoustic-flow-host-test
    synthesize-vits-acoustic-flow-graph-test
    synthesize-vits-decoder-weights-test
    synthesize-vits-waveform-decoder-host-test
    synthesize-vits-conv-test
    synthesize-vits-transpose-conv-test
    synthesize-vits-waveform-decoder-graph-test
    synthesize-random-stream-test
    synthesize-backend-device-internal-test
    synthesize-backend-plan-test
    synthesize-text-frontend-test
    synthesize-synthesis-request-test
    synthesize-audio-delivery-test
    synthesize-vits-runner-voice-test
    synthesize-vits-model-errors-test
    PROPERTIES LABELS "unit;vits")

if(SYNTH_BUILD_PYTHON_TESTS)
    find_program(SYNTH_UV_EXECUTABLE uv REQUIRED)
    # Despite the legacy name this discovers the whole locked Python unit
    # suite, which now covers family-independent packaging and manifest checks
    # plus the VITS and Kokoro converter contracts. The VITS environment owns
    # the repository's locked tooling; a family whose tests need environment
    # specific dependencies gets its own registration.
    add_test(
        NAME synthesize-vits-python-unit
        COMMAND ${SYNTH_UV_EXECUTABLE} run
            --project ${CMAKE_SOURCE_DIR}/scripts/envs/vits
            --locked python -m unittest discover
            -s ${CMAKE_CURRENT_SOURCE_DIR}/python
            -p test_*.py)
    set_tests_properties(synthesize-vits-python-unit PROPERTIES
        LABELS "unit;python;vits;kokoro"
        TIMEOUT 180
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# The runner is a thin adapter shared by the Python Golden validator and manual
# diagnostics. It is intentionally not itself a test.
add_executable(synthesize-vits-text-encoder vits_text_encoder_real.cpp)
target_include_directories(synthesize-vits-text-encoder PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-text-encoder PRIVATE synthesize ggml)
add_executable(synthesize-vits-duration-predictor vits_duration_predictor_real.cpp)
target_include_directories(synthesize-vits-duration-predictor PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-duration-predictor PRIVATE synthesize ggml)
add_executable(synthesize-vits-duration-path vits_duration_path_real.cpp)
target_include_directories(synthesize-vits-duration-path PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-duration-path PRIVATE synthesize ggml)
add_executable(synthesize-vits-prior-expansion vits_prior_expansion_real.cpp)
target_include_directories(synthesize-vits-prior-expansion PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-prior-expansion PRIVATE synthesize ggml)
add_executable(synthesize-vits-latent-sampling vits_latent_sampling_real.cpp)
target_include_directories(synthesize-vits-latent-sampling PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-latent-sampling PRIVATE synthesize ggml)
add_executable(synthesize-vits-acoustic-flow vits_acoustic_flow_real.cpp)
target_include_directories(synthesize-vits-acoustic-flow PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-acoustic-flow PRIVATE synthesize ggml)
add_executable(synthesize-vits-waveform-decoder vits_waveform_decoder_real.cpp)
target_include_directories(synthesize-vits-waveform-decoder PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-waveform-decoder PRIVATE synthesize ggml)
add_executable(synthesize-vits-public-lifecycle-test vits_public_lifecycle_test.cpp)
target_include_directories(synthesize-vits-public-lifecycle-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-vits-public-lifecycle-test PRIVATE synthesize)
add_executable(synthesize-vits-public-synthesis-test vits_public_synthesis_test.cpp)
target_include_directories(synthesize-vits-public-synthesis-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-vits-public-synthesis-test PRIVATE synthesize)
add_executable(synthesize-vits-vctk-public-test vits_vctk_public_test.cpp)
target_include_directories(synthesize-vits-vctk-public-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-vits-vctk-public-test PRIVATE synthesize)

if(SYNTH_BUILD_INTEGRATION_TESTS)
    set(_synth_vits_golden_sentinel
        "${CMAKE_SOURCE_DIR}/build/goldens/vits/vits-ljspeech/ljs-upstream-default/input/token_ids.i32")
    if(NOT EXISTS "${SYNTH_VITS_TEST_MODEL}")
        message(FATAL_ERROR "SYNTH_VITS_TEST_MODEL does not exist: ${SYNTH_VITS_TEST_MODEL}")
    endif()
    if(NOT EXISTS "${_synth_vits_golden_sentinel}")
        message(FATAL_ERROR
            "VITS Golden artifacts are not materialized; run scripts/dump_reference_vits_pytorch.py first")
    endif()
    if(NOT SYNTH_UV_EXECUTABLE)
        find_program(SYNTH_UV_EXECUTABLE uv REQUIRED)
    endif()
    add_test(
        NAME synthesize-vits-public-lifecycle
        COMMAND synthesize-vits-public-lifecycle-test ${SYNTH_VITS_TEST_MODEL})
    set_tests_properties(synthesize-vits-public-lifecycle PROPERTIES
        LABELS "integration;abi;vits"
        TIMEOUT 60)
    add_test(
        NAME synthesize-vits-public-synthesis
        COMMAND synthesize-vits-public-synthesis-test
            ${SYNTH_VITS_TEST_MODEL}
            ${CMAKE_SOURCE_DIR}/build/goldens/vits/vits-ljspeech/ljs-minimal/input/token_ids.i32)
    set_tests_properties(synthesize-vits-public-synthesis PROPERTIES
        LABELS "integration;abi;vits"
        TIMEOUT 300)
    if(SYNTH_BUILD_CLI)
        add_test(
            NAME synthesize-vits-cli
            COMMAND ${CMAKE_COMMAND}
                -DSYNTH_CLI=$<TARGET_FILE:synthesize-cli>
                -DSYNTH_MODEL=${SYNTH_VITS_TEST_MODEL}
                -DSYNTH_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures/cli-real
                -P ${CMAKE_CURRENT_SOURCE_DIR}/check-cli-real.cmake)
        set_tests_properties(synthesize-vits-cli PROPERTIES
            LABELS "integration;cli;vits"
            TIMEOUT 300)
    endif()
    add_test(
        NAME synthesize-vits-text-encoder-golden
        COMMAND ${SYNTH_UV_EXECUTABLE} run
            --project ${CMAKE_SOURCE_DIR}/scripts/envs/vits
            --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-vits-text-encoder.py
            --manifest ${CMAKE_SOURCE_DIR}/tests/golden/vits/vits-ljspeech.manifest.json
            --model ${SYNTH_VITS_TEST_MODEL}
            --runner $<TARGET_FILE:synthesize-vits-text-encoder>
            --threads 1)
    set_tests_properties(synthesize-vits-text-encoder-golden PROPERTIES
        LABELS "integration;vits;golden"
        TIMEOUT 300
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(
        NAME synthesize-vits-duration-predictor-golden
        COMMAND ${SYNTH_UV_EXECUTABLE} run
            --project ${CMAKE_SOURCE_DIR}/scripts/envs/vits
            --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-vits-duration-predictor.py
            --manifest ${CMAKE_SOURCE_DIR}/tests/golden/vits/vits-ljspeech.manifest.json
            --model ${SYNTH_VITS_TEST_MODEL}
            --runner $<TARGET_FILE:synthesize-vits-duration-predictor>
            --threads 1)
    set_tests_properties(synthesize-vits-duration-predictor-golden PROPERTIES
        LABELS "integration;vits;golden"
        TIMEOUT 300
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(
        NAME synthesize-vits-duration-path-golden
        COMMAND ${SYNTH_UV_EXECUTABLE} run
            --project ${CMAKE_SOURCE_DIR}/scripts/envs/vits
            --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-vits-duration-path.py
            --manifest ${CMAKE_SOURCE_DIR}/tests/golden/vits/vits-ljspeech.manifest.json
            --model ${SYNTH_VITS_TEST_MODEL}
            --runner $<TARGET_FILE:synthesize-vits-duration-path>
            --threads 1)
    set_tests_properties(synthesize-vits-duration-path-golden PROPERTIES
        LABELS "integration;vits;golden"
        TIMEOUT 300
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(
        NAME synthesize-vits-prior-expansion-golden
        COMMAND ${SYNTH_UV_EXECUTABLE} run
            --project ${CMAKE_SOURCE_DIR}/scripts/envs/vits
            --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-vits-prior-expansion.py
            --manifest ${CMAKE_SOURCE_DIR}/tests/golden/vits/vits-ljspeech.manifest.json
            --model ${SYNTH_VITS_TEST_MODEL}
            --runner $<TARGET_FILE:synthesize-vits-prior-expansion>
            --threads 1)
    set_tests_properties(synthesize-vits-prior-expansion-golden PROPERTIES
        LABELS "integration;vits;golden"
        TIMEOUT 300
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(
        NAME synthesize-vits-latent-sampling-golden
        COMMAND ${SYNTH_UV_EXECUTABLE} run
            --project ${CMAKE_SOURCE_DIR}/scripts/envs/vits
            --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-vits-latent-sampling.py
            --manifest ${CMAKE_SOURCE_DIR}/tests/golden/vits/vits-ljspeech.manifest.json
            --model ${SYNTH_VITS_TEST_MODEL}
            --runner $<TARGET_FILE:synthesize-vits-latent-sampling>
            --threads 1)
    set_tests_properties(synthesize-vits-latent-sampling-golden PROPERTIES
        LABELS "integration;vits;golden"
        TIMEOUT 300
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(
        NAME synthesize-vits-acoustic-flow-golden
        COMMAND ${SYNTH_UV_EXECUTABLE} run
            --project ${CMAKE_SOURCE_DIR}/scripts/envs/vits
            --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-vits-acoustic-flow.py
            --manifest ${CMAKE_SOURCE_DIR}/tests/golden/vits/vits-ljspeech.manifest.json
            --model ${SYNTH_VITS_TEST_MODEL}
            --runner $<TARGET_FILE:synthesize-vits-acoustic-flow>
            --threads 1)
    set_tests_properties(synthesize-vits-acoustic-flow-golden PROPERTIES
        LABELS "integration;vits;golden"
        TIMEOUT 300
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set(_synth_vits_waveform_threads 1)
    if(SYNTH_SANITIZE)
        # ASan/UBSan makes the full decoder substantially slower. This remains
        # a memory-safety run; the Release one-thread test owns numeric reports.
        set(_synth_vits_waveform_threads 8)
    endif()
    add_test(
        NAME synthesize-vits-waveform-decoder-golden
        COMMAND ${SYNTH_UV_EXECUTABLE} run
            --project ${CMAKE_SOURCE_DIR}/scripts/envs/vits
            --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-vits-waveform-decoder.py
            --manifest ${CMAKE_SOURCE_DIR}/tests/golden/vits/vits-ljspeech.manifest.json
            --model ${SYNTH_VITS_TEST_MODEL}
            --runner $<TARGET_FILE:synthesize-vits-waveform-decoder>
            --threads ${_synth_vits_waveform_threads})
    set_tests_properties(synthesize-vits-waveform-decoder-golden PROPERTIES
        LABELS "integration;vits;golden"
        TIMEOUT 1800
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    set(_synth_vits_vctk_golden_sentinel
        "${CMAKE_SOURCE_DIR}/build/goldens/vits/vits-vctk/vctk-upstream-speaker-004/input/token_ids.i32")
    if(EXISTS "${SYNTH_VITS_VCTK_TEST_MODEL}" AND EXISTS "${_synth_vits_vctk_golden_sentinel}")
        add_test(
            NAME synthesize-vits-vctk-public
            COMMAND synthesize-vits-vctk-public-test
                ${SYNTH_VITS_VCTK_TEST_MODEL}
                ${CMAKE_SOURCE_DIR}/build/goldens/vits/vits-vctk/vctk-minimal/input/token_ids.i32)
        set_tests_properties(synthesize-vits-vctk-public PROPERTIES
            LABELS "integration;abi;vits;voice"
            TIMEOUT 300)

        function(synth_add_vits_vctk_golden_test name script runner timeout threads)
            add_test(
                NAME synthesize-vits-vctk-${name}-golden
                COMMAND ${SYNTH_UV_EXECUTABLE} run
                    --project ${CMAKE_SOURCE_DIR}/scripts/envs/vits
                    --locked python ${CMAKE_SOURCE_DIR}/scripts/${script}
                    --manifest ${CMAKE_SOURCE_DIR}/tests/golden/vits/vits-vctk.manifest.json
                    --model ${SYNTH_VITS_VCTK_TEST_MODEL}
                    --runner $<TARGET_FILE:${runner}>
                    --threads ${threads})
            set_tests_properties(synthesize-vits-vctk-${name}-golden PROPERTIES
                LABELS "integration;vits;golden;voice"
                TIMEOUT ${timeout}
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        endfunction()

        synth_add_vits_vctk_golden_test(
            text-encoder validate-vits-text-encoder.py synthesize-vits-text-encoder 300 1)
        synth_add_vits_vctk_golden_test(
            duration-predictor validate-vits-duration-predictor.py synthesize-vits-duration-predictor 300 1)
        synth_add_vits_vctk_golden_test(
            duration-path validate-vits-duration-path.py synthesize-vits-duration-path 300 1)
        synth_add_vits_vctk_golden_test(
            prior-expansion validate-vits-prior-expansion.py synthesize-vits-prior-expansion 300 1)
        synth_add_vits_vctk_golden_test(
            latent-sampling validate-vits-latent-sampling.py synthesize-vits-latent-sampling 300 1)
        synth_add_vits_vctk_golden_test(
            acoustic-flow validate-vits-acoustic-flow.py synthesize-vits-acoustic-flow 300 1)
        synth_add_vits_vctk_golden_test(
            waveform-decoder validate-vits-waveform-decoder.py synthesize-vits-waveform-decoder
            1800 ${_synth_vits_waveform_threads})
    else()
        message(STATUS
            "VITS VCTK integration tests skipped: local model or Golden artifacts are absent")
    endif()
endif()

add_custom_target(synthesize-check-unit
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L unit
    DEPENDS
        synthesize-abi-smoke
        synthesize-abi-initializer-bounds-test
        synthesize-public-api-test
        synthesize-voice-profile-api-test
        synthesize-backend-device-test
        synthesize-backend-module-test
        synthesize-kokoro-metadata-test
        synthesize-kokoro-lstm-test
        synthesize-kokoro-catalog-test
        synthesize-kokoro-plbert-test
        synthesize-kokoro-duration-test
        synthesize-vits-text-encoder-host-test
        synthesize-vits-metadata-test
        synthesize-vits-weights-test
        synthesize-vits-text-encoder-graph-test
        synthesize-vits-duration-host-test
        synthesize-vits-duration-weights-test
        synthesize-vits-duration-graph-test
        synthesize-vits-duration-path-test
        synthesize-vits-prior-expansion-host-test
        synthesize-vits-prior-expansion-graph-test
        synthesize-vits-latent-sampling-host-test
        synthesize-vits-latent-sampling-graph-test
        synthesize-vits-flow-weights-test
        synthesize-vits-acoustic-flow-host-test
        synthesize-vits-acoustic-flow-graph-test
        synthesize-vits-decoder-weights-test
        synthesize-vits-waveform-decoder-host-test
        synthesize-vits-conv-test
        synthesize-vits-transpose-conv-test
        synthesize-vits-waveform-decoder-graph-test
        synthesize-random-stream-test
        synthesize-backend-device-internal-test
        synthesize-backend-plan-test
        synthesize-text-frontend-test
        synthesize-synthesis-request-test
        synthesize-audio-delivery-test
        synthesize-vits-runner-voice-test
        synthesize-quantize-policy-test
        synthesize-quantize-test
        synthesize-cli-unit-test
        synthesize-vits-model-errors-test
    USES_TERMINAL)
if(SYNTH_BUILD_TOOLS)
    add_dependencies(synthesize-check-unit synthesize-quantize)
endif()
if(SYNTH_BUILD_CLI)
    add_dependencies(synthesize-check-unit synthesize-cli)
endif()

if(SYNTH_BUILD_INTEGRATION_TESTS)
    add_custom_target(synthesize-check-integration
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L integration
        DEPENDS
            synthesize-vits-text-encoder
            synthesize-vits-duration-predictor
            synthesize-vits-duration-path
            synthesize-vits-prior-expansion
            synthesize-vits-latent-sampling
            synthesize-vits-acoustic-flow
            synthesize-vits-waveform-decoder
            synthesize-vits-public-lifecycle-test
            synthesize-vits-public-synthesis-test
            synthesize-vits-vctk-public-test
            $<$<BOOL:${SYNTH_BUILD_CLI}>:synthesize-cli>
        USES_TERMINAL)
endif()
