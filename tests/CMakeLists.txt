# Declaring a test is what makes a check target build it.
#
# Both check targets used to carry a hand-written DEPENDS list, so a test could
# be correct, registered, and never built — it then passed or failed on whatever
# binary happened to be in the tree. That happened twice in this repository:
# once for a unit test and once for a Golden runner, and both times a stale
# local build hid it. These properties are appended to at declaration and
# consumed by the check targets below, so the two can no longer disagree.
define_property(GLOBAL PROPERTY SYNTH_UNIT_TEST_TARGETS
    BRIEF_DOCS "Executables the unit check target must build"
    FULL_DOCS "Appended to by synth_add_unit_test and consumed by synthesize-check-unit")
define_property(GLOBAL PROPERTY SYNTH_INTEGRATION_TARGETS
    BRIEF_DOCS "Executables the integration check target must build"
    FULL_DOCS "Appended to by synth_register_integration_target and consumed by synthesize-check-integration")

function(synth_register_unit_target target)
    set_property(GLOBAL APPEND PROPERTY SYNTH_UNIT_TEST_TARGETS ${target})
endfunction()

# Integration binaries include Golden runners, which are adapters rather than
# tests and so are never picked up by a test-registration helper.
function(synth_register_integration_target target)
    set_property(GLOBAL APPEND PROPERTY SYNTH_INTEGRATION_TARGETS ${target})
endfunction()

function(synth_add_unit_test target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE synthesize ggml)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "unit")
    synth_register_unit_target(${target})
endfunction()

add_executable(synthesize-abi-smoke abi_smoke.c)
target_link_libraries(synthesize-abi-smoke PRIVATE synthesize)
add_test(NAME synthesize-abi-smoke COMMAND synthesize-abi-smoke)
synth_register_unit_target(synthesize-abi-smoke)
set_tests_properties(synthesize-abi-smoke PROPERTIES LABELS "unit;abi")
add_executable(synthesize-abi-initializer-bounds-test abi_initializer_bounds_test.c)
target_link_libraries(synthesize-abi-initializer-bounds-test PRIVATE synthesize)
add_test(NAME synthesize-abi-initializer-bounds-test COMMAND synthesize-abi-initializer-bounds-test)
synth_register_unit_target(synthesize-abi-initializer-bounds-test)
set_tests_properties(synthesize-abi-initializer-bounds-test PROPERTIES LABELS "unit;abi")
add_executable(synthesize-public-api-test public_api_test.c)
target_link_libraries(synthesize-public-api-test PRIVATE synthesize)
add_test(NAME synthesize-public-api-test COMMAND synthesize-public-api-test)
synth_register_unit_target(synthesize-public-api-test)
set_tests_properties(synthesize-public-api-test PROPERTIES LABELS "unit;abi")
add_executable(synthesize-voice-profile-api-test voice_profile_api_test.c)
target_link_libraries(synthesize-voice-profile-api-test PRIVATE synthesize)
add_test(NAME synthesize-voice-profile-api-test COMMAND synthesize-voice-profile-api-test)
synth_register_unit_target(synthesize-voice-profile-api-test)
set_tests_properties(synthesize-voice-profile-api-test PROPERTIES LABELS "unit;abi;voice-profile")
add_executable(synthesize-backend-device-test backend_device_test.c)
target_link_libraries(synthesize-backend-device-test PRIVATE synthesize)
add_test(NAME synthesize-backend-device-test COMMAND synthesize-backend-device-test)
synth_register_unit_target(synthesize-backend-device-test)
set_tests_properties(synthesize-backend-device-test PROPERTIES LABELS "unit;abi;backend")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/backend-modules")
add_executable(synthesize-backend-module-test backend_module_test.c)
target_link_libraries(synthesize-backend-module-test PRIVATE synthesize)
add_test(NAME synthesize-backend-module-test
    COMMAND synthesize-backend-module-test
        "${CMAKE_CURRENT_BINARY_DIR}/fixtures/backend-modules")
synth_register_unit_target(synthesize-backend-module-test)
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

synth_add_unit_test(synthesize-sha256-test sha256_test.cpp)
synth_add_unit_test(synthesize-kokoro-metadata-test kokoro_metadata_test.cpp)
synth_add_unit_test(synthesize-qwen3-tts-metadata-test qwen3_tts_metadata_test.cpp)
synth_add_unit_test(synthesize-qwen3-tts-bpe-test qwen3_tts_bpe_test.cpp)
synth_add_unit_test(synthesize-qwen3-tts-catalog-test qwen3_tts_catalog_test.cpp)
synth_add_unit_test(synthesize-qwen3-tts-voice-required-test qwen3_tts_voice_required_test.cpp)
synth_add_unit_test(synthesize-qwen3-tts-decoder-layer-test qwen3_tts_decoder_layer_test.cpp)
synth_add_unit_test(synthesize-qwen3-tts-talker-test qwen3_tts_talker_test.cpp)
synth_add_unit_test(synthesize-qwen3-tts-codec-ops-test qwen3_tts_codec_ops_test.cpp)
synth_add_unit_test(synthesize-qwen3-tts-codec-test qwen3_tts_codec_test.cpp)
synth_add_unit_test(synthesize-qwen3-tts-code-predictor-test qwen3_tts_code_predictor_test.cpp)
synth_add_unit_test(synthesize-qwen3-tts-sampling-test qwen3_tts_sampling_test.cpp)
synth_add_unit_test(synthesize-omnivoice-metadata-test omnivoice_metadata_test.cpp)
synth_add_unit_test(synthesize-omnivoice-catalog-test omnivoice_catalog_test.cpp)
synth_add_unit_test(synthesize-omnivoice-frontend-test omnivoice_frontend_test.cpp)
synth_add_unit_test(synthesize-omnivoice-generator-test omnivoice_generator_test.cpp)
synth_add_unit_test(synthesize-omnivoice-generator-host-test omnivoice_generator_host_test.cpp)
synth_add_unit_test(synthesize-omnivoice-sampler-test omnivoice_sampler_test.cpp)
synth_add_unit_test(synthesize-omnivoice-candidate-scan-test omnivoice_candidate_scan_test.cpp)
synth_add_unit_test(synthesize-omnivoice-codec-test omnivoice_codec_test.cpp)
synth_add_unit_test(synthesize-omnivoice-resampler-test omnivoice_resampler_test.cpp)
synth_add_unit_test(synthesize-omnivoice-reference-encoder-test omnivoice_reference_encoder_test.cpp)
synth_add_unit_test(synthesize-omnivoice-quantization-test omnivoice_quantization_test.cpp)
synth_add_unit_test(synthesize-omnivoice-serialize-writer-agreement-test omnivoice_serialize_writer_agreement_test.cpp)
synth_add_unit_test(synthesize-kokoro-lstm-test kokoro_lstm_test.cpp)
synth_add_unit_test(synthesize-kokoro-catalog-test kokoro_catalog_test.cpp)
synth_add_unit_test(synthesize-kokoro-plbert-test kokoro_plbert_test.cpp)
synth_add_unit_test(synthesize-kokoro-duration-test kokoro_duration_test.cpp)
synth_add_unit_test(synthesize-kokoro-transpose-conv-test kokoro_transpose_conv_test.cpp)
synth_add_unit_test(synthesize-kokoro-prosody-test kokoro_prosody_test.cpp)
synth_add_unit_test(synthesize-kokoro-text-encoder-test kokoro_text_encoder_test.cpp)
synth_add_unit_test(synthesize-kokoro-source-test kokoro_source_test.cpp)
add_executable(synthesize-kokoro-model-errors-test kokoro_model_errors_test.cpp)
target_include_directories(synthesize-kokoro-model-errors-test PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-kokoro-model-errors-test PRIVATE synthesize ggml)
add_test(NAME synthesize-kokoro-model-errors-test
    COMMAND synthesize-kokoro-model-errors-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
synth_register_unit_target(synthesize-kokoro-model-errors-test)
synth_add_unit_test(synthesize-kokoro-quantization-policy-test kokoro_quantization_policy_test.cpp)
synth_add_unit_test(synthesize-kokoro-istft-test kokoro_istft_test.cpp)
synth_add_unit_test(synthesize-kokoro-decoder-test kokoro_decoder_test.cpp)
set_tests_properties(synthesize-kokoro-metadata-test synthesize-kokoro-lstm-test
    synthesize-kokoro-catalog-test synthesize-kokoro-plbert-test
    synthesize-kokoro-duration-test synthesize-kokoro-transpose-conv-test
    synthesize-kokoro-prosody-test synthesize-kokoro-text-encoder-test
    synthesize-kokoro-source-test synthesize-kokoro-istft-test
    synthesize-kokoro-decoder-test synthesize-kokoro-model-errors-test
    synthesize-kokoro-quantization-policy-test
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
synth_add_unit_test(synthesize-cpu-parallelism-test cpu_parallelism_test.cpp)
synth_add_unit_test(synthesize-backend-device-internal-test backend_device_internal_test.cpp)
synth_add_unit_test(synthesize-backend-plan-test backend_plan_test.cpp)
synth_add_unit_test(synthesize-text-frontend-test text_frontend_test.cpp)
synth_add_unit_test(synthesize-synthesis-request-test synthesis_request_test.cpp)
synth_add_unit_test(synthesize-model-info-test model_info_test.cpp)
synth_add_unit_test(synthesize-audio-delivery-test audio_delivery_test.cpp)
synth_add_unit_test(synthesize-audio-normalizer-test audio_normalizer_test.cpp)
synth_add_unit_test(synthesize-vits-runner-voice-test vits_runner_voice_test.cpp)
add_executable(synthesize-qwen3-tts-quantization-policy-test
    qwen3_tts_quantization_policy_test.cpp
    ${CMAKE_SOURCE_DIR}/tools/synthesize-quantize/policy.cpp)
target_include_directories(synthesize-qwen3-tts-quantization-policy-test PRIVATE
    ${CMAKE_SOURCE_DIR}/tools/synthesize-quantize
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-qwen3-tts-quantization-policy-test PRIVATE synthesize ggml)
add_test(NAME synthesize-qwen3-tts-quantization-policy-test
    COMMAND synthesize-qwen3-tts-quantization-policy-test)
synth_register_unit_target(synthesize-qwen3-tts-quantization-policy-test)
set_tests_properties(synthesize-qwen3-tts-quantization-policy-test PROPERTIES
    LABELS "unit;tools;quantization;qwen3-tts")
add_executable(synthesize-quantize-policy-test
    synthesize_quantize_policy_test.cpp
    ${CMAKE_SOURCE_DIR}/tools/synthesize-quantize/policy.cpp)
target_include_directories(synthesize-quantize-policy-test PRIVATE
    ${CMAKE_SOURCE_DIR}/tools/synthesize-quantize
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-quantize-policy-test PRIVATE synthesize ggml)
add_test(NAME synthesize-quantize-policy-test COMMAND synthesize-quantize-policy-test)
synth_register_unit_target(synthesize-quantize-policy-test)
set_tests_properties(synthesize-quantize-policy-test PROPERTIES LABELS "unit;tools;quantization")
add_executable(synthesize-quantize-test
    synthesize_quantize_test.cpp
    ${CMAKE_SOURCE_DIR}/tools/synthesize-quantize/policy.cpp
    ${CMAKE_SOURCE_DIR}/tools/synthesize-quantize/quantize.cpp)
target_include_directories(synthesize-quantize-test PRIVATE
    ${CMAKE_SOURCE_DIR}/tools/synthesize-quantize
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-quantize-test PRIVATE synthesize ggml)
add_test(NAME synthesize-quantize-test
    COMMAND synthesize-quantize-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
synth_register_unit_target(synthesize-quantize-test)
set_tests_properties(synthesize-quantize-test PROPERTIES LABELS "unit;tools;quantization")
if(SYNTH_BUILD_TOOLS)
    add_test(NAME synthesize-quantize-cli-help-test
        COMMAND synthesize-quantize --help)
    set_tests_properties(synthesize-quantize-cli-help-test PROPERTIES
        LABELS "unit;tools;quantization;cli"
        PASS_REGULAR_EXPRESSION "Q8_CODEC_MIXED[|]Q8[|]Q4_K[|]BF16")
endif()
add_executable(synthesize-cli-unit-test
    cli_test.cpp
    ${CMAKE_SOURCE_DIR}/examples/cli/cli.cpp)
target_include_directories(synthesize-cli-unit-test PRIVATE
    ${CMAKE_SOURCE_DIR}/examples/cli ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-cli-unit-test PRIVATE synthesize)
add_test(NAME synthesize-cli-unit-test
    COMMAND synthesize-cli-unit-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
synth_register_unit_target(synthesize-cli-unit-test)
set_tests_properties(synthesize-cli-unit-test PROPERTIES LABELS "unit;cli")
add_executable(synthesize-vits-model-errors-test vits_model_errors_test.cpp)
target_include_directories(synthesize-vits-model-errors-test PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-vits-model-errors-test PRIVATE synthesize ggml)
add_test(NAME synthesize-vits-model-errors-test
    COMMAND synthesize-vits-model-errors-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
synth_register_unit_target(synthesize-vits-model-errors-test)
add_executable(synthesize-omnivoice-model-errors-test omnivoice_model_errors_test.cpp)
target_include_directories(synthesize-omnivoice-model-errors-test PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-omnivoice-model-errors-test PRIVATE synthesize ggml)
add_test(NAME synthesize-omnivoice-model-errors-test
    COMMAND synthesize-omnivoice-model-errors-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
synth_register_unit_target(synthesize-omnivoice-model-errors-test)
add_executable(synthesize-omnivoice-load-synthetic-test omnivoice_load_synthetic_test.cpp)
target_include_directories(synthesize-omnivoice-load-synthetic-test PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-omnivoice-load-synthetic-test PRIVATE synthesize ggml)
add_test(NAME synthesize-omnivoice-load-synthetic-test
    COMMAND synthesize-omnivoice-load-synthetic-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-synthetic)
synth_register_unit_target(synthesize-omnivoice-load-synthetic-test)
add_executable(synthesize-omnivoice-decode-loop-test omnivoice_decode_loop_test.cpp)
target_include_directories(synthesize-omnivoice-decode-loop-test PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-omnivoice-decode-loop-test PRIVATE synthesize ggml)
add_test(NAME synthesize-omnivoice-decode-loop-test
    COMMAND synthesize-omnivoice-decode-loop-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-synthetic)
synth_register_unit_target(synthesize-omnivoice-decode-loop-test)
# Description Text profiles (Task 15): a genuine unit test against a
# synthetic package (its own fixtures subdirectory, not
# omnivoice-synthetic's -- write_synthetic_package's own file, not one this
# other suite's tests could race against). Nothing here touches the
# frontend or a tensor -- resolve_instruct is pure string processing and
# create_from_description never reaches the frontend either -- so the
# heavier model-guarded harness tests/omnivoice_profile_test.cpp uses for
# Task 14's Reference Audio path (a real forward pass) buys nothing here.
add_executable(synthesize-omnivoice-design-test omnivoice_design_test.cpp)
target_include_directories(synthesize-omnivoice-design-test PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-omnivoice-design-test PRIVATE synthesize ggml)
add_test(NAME synthesize-omnivoice-design-test
    COMMAND synthesize-omnivoice-design-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-design)
synth_register_unit_target(synthesize-omnivoice-design-test)
# Serialized Voice Profiles (Task 16): the GGUF envelope round trip and
# tamper matrix, against a synthetic package -- its own fixtures
# subdirectory, not omnivoice-synthetic's or omnivoice-design's, for the
# same race-avoidance reason those two already keep separate ones.
add_executable(synthesize-omnivoice-serialize-test omnivoice_serialize_test.cpp)
target_include_directories(synthesize-omnivoice-serialize-test PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-omnivoice-serialize-test PRIVATE synthesize ggml)
add_test(NAME synthesize-omnivoice-serialize-test
    COMMAND synthesize-omnivoice-serialize-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-serialize)
synth_register_unit_target(synthesize-omnivoice-serialize-test)
# Task 8 (Plan 4): the per-family Execution Backend gate in
# src/synthesize.cpp's synth_model_load, against a synthetic package -- its
# own fixtures subdirectory for the same race-avoidance reason the other
# synthetic-package suites keep separate ones. A genuine unit test: the
# refusal happens before device resolution, so it holds with no CUDA device
# present at all.
add_executable(synthesize-omnivoice-backend-test omnivoice_backend_test.cpp)
target_include_directories(synthesize-omnivoice-backend-test PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-omnivoice-backend-test PRIVATE synthesize ggml)
add_test(NAME synthesize-omnivoice-backend-test
    COMMAND synthesize-omnivoice-backend-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-backend)
synth_register_unit_target(synthesize-omnivoice-backend-test)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-synthetic)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-design)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-serialize)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-backend)
set_tests_properties(
    synthesize-omnivoice-metadata-test
    synthesize-omnivoice-catalog-test
    synthesize-omnivoice-frontend-test
    synthesize-omnivoice-generator-test
    synthesize-omnivoice-generator-host-test
    synthesize-omnivoice-sampler-test
    synthesize-omnivoice-codec-test
    synthesize-omnivoice-resampler-test
    synthesize-omnivoice-reference-encoder-test
    synthesize-omnivoice-quantization-test
    synthesize-omnivoice-model-errors-test
    synthesize-omnivoice-load-synthetic-test
    synthesize-omnivoice-decode-loop-test
    synthesize-omnivoice-design-test
    synthesize-omnivoice-serialize-test
    synthesize-omnivoice-serialize-writer-agreement-test
    synthesize-omnivoice-backend-test
    PROPERTIES LABELS "unit;omnivoice")
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
    # The OmniVoice converter tests ride this same discovery; their one
    # package-dependent case skips here and runs for real under
    # synthesize-omnivoice-python-unit below.
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
    # The OmniVoice converter's positive generation-defaults test needs the
    # upstream package, which exists only in this family's locked environment;
    # under the VITS discovery above it skips forever. Running the module where
    # the import resolves enforces the "defaults come from the pinned package"
    # rule instead of permanently skipping it.
    add_test(
        NAME synthesize-omnivoice-python-unit
        COMMAND ${SYNTH_UV_EXECUTABLE} run
            --project ${CMAKE_SOURCE_DIR}/scripts/envs/omnivoice
            --locked python -m unittest -v
            tests.python.test_convert_omnivoice)
    set_tests_properties(synthesize-omnivoice-python-unit PROPERTIES
        LABELS "unit;python;omnivoice"
        TIMEOUT 600
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

add_executable(synthesize-qwen3-tts-public-real qwen3_tts_public_real.c)
target_link_libraries(synthesize-qwen3-tts-public-real PRIVATE synthesize)
synth_register_integration_target(synthesize-qwen3-tts-public-real)

add_executable(synthesize-omnivoice-public-real omnivoice_public_real.c)
target_link_libraries(synthesize-omnivoice-public-real PRIVATE synthesize)
synth_register_integration_target(synthesize-omnivoice-public-real)

add_executable(synthesize-qwen3-tts-replay-real qwen3_tts_replay_real.cpp)
target_include_directories(synthesize-qwen3-tts-replay-real PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-qwen3-tts-replay-real PRIVATE synthesize ggml)
synth_register_integration_target(synthesize-qwen3-tts-replay-real)

add_executable(synthesize-omnivoice-replay-real omnivoice_replay_real.cpp)
target_include_directories(synthesize-omnivoice-replay-real PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-omnivoice-replay-real PRIVATE synthesize ggml)
synth_register_integration_target(synthesize-omnivoice-replay-real)

# The runners are thin adapters shared by the Python Golden validators and
# manual diagnostics. They are intentionally not themselves tests.
add_executable(synthesize-kokoro-stages kokoro_stages_real.cpp)
target_include_directories(synthesize-kokoro-stages PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-kokoro-stages PRIVATE synthesize ggml)
synth_register_integration_target(synthesize-kokoro-stages)
add_executable(synthesize-vits-text-encoder vits_text_encoder_real.cpp)
target_include_directories(synthesize-vits-text-encoder PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-text-encoder PRIVATE synthesize ggml)
synth_register_integration_target(synthesize-vits-text-encoder)
add_executable(synthesize-vits-duration-predictor vits_duration_predictor_real.cpp)
target_include_directories(synthesize-vits-duration-predictor PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-duration-predictor PRIVATE synthesize ggml)
synth_register_integration_target(synthesize-vits-duration-predictor)
add_executable(synthesize-vits-duration-path vits_duration_path_real.cpp)
target_include_directories(synthesize-vits-duration-path PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-duration-path PRIVATE synthesize ggml)
synth_register_integration_target(synthesize-vits-duration-path)
add_executable(synthesize-vits-prior-expansion vits_prior_expansion_real.cpp)
target_include_directories(synthesize-vits-prior-expansion PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-prior-expansion PRIVATE synthesize ggml)
synth_register_integration_target(synthesize-vits-prior-expansion)
add_executable(synthesize-vits-latent-sampling vits_latent_sampling_real.cpp)
target_include_directories(synthesize-vits-latent-sampling PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-latent-sampling PRIVATE synthesize ggml)
synth_register_integration_target(synthesize-vits-latent-sampling)
add_executable(synthesize-vits-acoustic-flow vits_acoustic_flow_real.cpp)
target_include_directories(synthesize-vits-acoustic-flow PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-acoustic-flow PRIVATE synthesize ggml)
synth_register_integration_target(synthesize-vits-acoustic-flow)
add_executable(synthesize-vits-waveform-decoder vits_waveform_decoder_real.cpp)
target_include_directories(synthesize-vits-waveform-decoder PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-vits-waveform-decoder PRIVATE synthesize ggml)
synth_register_integration_target(synthesize-vits-waveform-decoder)
add_executable(synthesize-kokoro-public-lifecycle-test kokoro_public_lifecycle_test.cpp)
target_include_directories(synthesize-kokoro-public-lifecycle-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-kokoro-public-lifecycle-test PRIVATE synthesize)
synth_register_integration_target(synthesize-kokoro-public-lifecycle-test)
add_executable(synthesize-vits-public-lifecycle-test vits_public_lifecycle_test.cpp)
target_include_directories(synthesize-vits-public-lifecycle-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-vits-public-lifecycle-test PRIVATE synthesize)
synth_register_integration_target(synthesize-vits-public-lifecycle-test)
add_executable(synthesize-vits-public-synthesis-test vits_public_synthesis_test.cpp)
target_include_directories(synthesize-vits-public-synthesis-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-vits-public-synthesis-test PRIVATE synthesize)
synth_register_integration_target(synthesize-vits-public-synthesis-test)
add_executable(synthesize-vits-vctk-public-test vits_vctk_public_test.cpp)
target_include_directories(synthesize-vits-vctk-public-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-vits-vctk-public-test PRIVATE synthesize)
synth_register_integration_target(synthesize-vits-vctk-public-test)

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
    if(EXISTS "${SYNTH_KOKORO_TEST_MODEL}")
        add_test(
            NAME synthesize-kokoro-public-lifecycle
            COMMAND synthesize-kokoro-public-lifecycle-test ${SYNTH_KOKORO_TEST_MODEL})
        set_tests_properties(synthesize-kokoro-public-lifecycle PROPERTIES
            LABELS "integration;abi;kokoro"
            TIMEOUT 300)
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
                -DSYNTH_TOKEN_IDS=0,156,0,47,0,102,0,4,0
                -DSYNTH_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures/cli-real
                -P ${CMAKE_CURRENT_SOURCE_DIR}/check-cli-real.cmake)
        set_tests_properties(synthesize-vits-cli PROPERTIES
            LABELS "integration;cli;vits"
            TIMEOUT 300)
        if(EXISTS "${SYNTH_KOKORO_TEST_MODEL}")
            # Kokoro's catalog has no package default, so the CLI must be given
            # a Voice; that difference is exactly what this run covers.
            add_test(NAME synthesize-kokoro-cli
                COMMAND ${CMAKE_COMMAND}
                    -DSYNTH_CLI=$<TARGET_FILE:synthesize-cli>
                    -DSYNTH_MODEL=${SYNTH_KOKORO_TEST_MODEL}
                    -DSYNTH_TOKEN_IDS=0,50,83,54,156,31,3,16,65,156,87,123,54,46,5,0
                    -DSYNTH_VOICE=af_heart
                    -DSYNTH_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures/cli-real-kokoro
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/check-cli-real.cmake)
            set_tests_properties(synthesize-kokoro-cli PROPERTIES
                LABELS "integration;cli;kokoro"
                TIMEOUT 600)
        endif()
        if(EXISTS "${SYNTH_OMNIVOICE_TEST_MODEL}")
            # OmniVoice's catalog is empty but carries an unnamed package
            # default (voices.mode "profile-sources", has_package_default
            # true, zero preset ids), so unlike Kokoro above the CLI runs with
            # NO --voice flag. Its package also declares text_utf8 as the
            # only supported input kind -- no token-ids, no phonemes -- so
            # SYNTH_TEXT drives --text instead of SYNTH_TOKEN_IDS driving
            # --token-ids, and the unsupported-input arm is pointed at
            # --phonemes (the kind this package's input_flags lacks) rather
            # than the --text VITS/Kokoro use for the same purpose (the kind
            # their own input_flags lack).
            add_test(NAME synthesize-omnivoice-cli
                COMMAND ${CMAKE_COMMAND}
                    -DSYNTH_CLI=$<TARGET_FILE:synthesize-cli>
                    -DSYNTH_MODEL=${SYNTH_OMNIVOICE_TEST_MODEL}
                    -DSYNTH_TEXT="Sampling follows the seed."
                    -DSYNTH_UNSUPPORTED_FLAG=--phonemes
                    -DSYNTH_UNSUPPORTED_VALUE="HH EH L OW"
                    -DSYNTH_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/fixtures/cli-real-omnivoice
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/check-cli-real.cmake)
            set_tests_properties(synthesize-omnivoice-cli PROPERTIES
                LABELS "integration;cli;omnivoice"
                # Three real forward passes (mask-predict denoising plus codec
                # decode) measured at ~31s total on the 20-CPU development
                # host; 1200s is headroom for a slower or busier machine.
                TIMEOUT 1200)
        endif()
    endif()
    # The repeated-run and resource-cleanup gate docs/backends.md requires of a
    # claimed backend, which no family had in code: every manifest declares
    # `resource_cleanup` on every case and nothing read the field. VITS's
    # recorded evidence was twenty repeated *processes*, which cannot see an
    # in-process leak at all, since exit reclaims everything.
    #
    # One test for all three families. What differs between them is only the
    # input a request carries -- token ids through a symbol-map frontend, UTF-8
    # text through a byte-pair one -- so that arrives as an argument.
    #
    # Cycle count drops under sanitizers because LeakSanitizer decides the same
    # question at allocation granularity, which is strictly better than a bound
    # on resident memory, and each sanitized cycle costs about thirty seconds.
    add_executable(synthesize-public-cleanup-test public_cleanup_test.cpp)
    target_link_libraries(synthesize-public-cleanup-test PRIVATE synthesize)
    synth_register_integration_target(synthesize-public-cleanup-test)
    if(SYNTH_SANITIZE)
        set(_synth_cleanup_cycles 3)
    else()
        set(_synth_cleanup_cycles 12)
    endif()

    function(synth_add_cleanup_test family model frames input)
        if(NOT EXISTS "${model}")
            return()
        endif()
        # An empty optional trailing argument (OmniVoice's voice: its catalog
        # has no preset to name) cannot survive an unquoted ${ARGN} expansion
        # -- CMake's argument splitter silently drops empty list elements
        # there, which would shift a language tag into the voice slot
        # instead. foreach(IN LISTS ...) does not lose it, so the two
        # optional slots are read out positionally and re-quoted individually
        # below, which does preserve an empty one.
        set(voice "")
        set(language "")
        set(_synth_cleanup_optional_index 0)
        foreach(_synth_cleanup_optional IN LISTS ARGN)
            if(_synth_cleanup_optional_index EQUAL 0)
                set(voice "${_synth_cleanup_optional}")
            elseif(_synth_cleanup_optional_index EQUAL 1)
                set(language "${_synth_cleanup_optional}")
            endif()
            math(EXPR _synth_cleanup_optional_index "${_synth_cleanup_optional_index} + 1")
        endforeach()
        add_test(NAME synthesize-${family}-public-cleanup
            COMMAND synthesize-public-cleanup-test
                "${model}" "${_synth_cleanup_cycles}" "${frames}" "${input}" "${voice}" "${language}")
        set_tests_properties(synthesize-${family}-public-cleanup PROPERTIES
            LABELS "integration;abi;${family}"
            TIMEOUT 900)
    endfunction()

    # `max_output_frames` does not mean the same thing to each family: VITS and
    # Kokoro compare it against the PCM frame count, Qwen3-TTS and OmniVoice
    # against their own codec frames. So these numbers are not comparable and
    # none is a default. VITS reads its tokens from the Golden payload rather
    # than a literal: its frontend is a symbol map, and an arbitrary id
    # sequence makes the duration predictor ask for more frames than any limit
    # allows. Kokoro's come from the sequence its own lifecycle test already
    # synthesizes.
    set(_synth_vits_cleanup_tokens
        "${CMAKE_SOURCE_DIR}/build/goldens/vits/vits-ljspeech/ljs-minimal/input/token_ids.i32")
    if(EXISTS "${_synth_vits_cleanup_tokens}")
        synth_add_cleanup_test(vits "${SYNTH_VITS_TEST_MODEL}" 131072
            "tokensfile:${_synth_vits_cleanup_tokens}")
    endif()
    synth_add_cleanup_test(kokoro "${SYNTH_KOKORO_TEST_MODEL}" 131072
        "tokens:32,0,81,83,16,61,53,156,25,16,83,44,156,138,64,16,81,83,16,58" "af_heart")
    synth_add_cleanup_test(qwen3-tts "${SYNTH_QWEN3_TTS_TEST_MODEL}" 46080 "text:Hi." "aiden" "en")
    # This `frames` argument becomes `synth_request_t.max_output_frames`,
    # which the C interface always counts in native PCM frames (docs/c-
    # interface.md) -- src/synthesize.cpp divides it by samples_per_frame
    # (960 here) before comparing it against this family's own codec-frame
    # cap. 720000 matches the package's own declared ceiling exactly
    # (tests/golden/omnivoice/omnivoice-0-6b.manifest.json's
    # package_contract.max_output_frames, itself 750 of this family's own
    # codec frames at its 25 Hz frame rate), so this request does not
    # accidentally clamp tighter than the package already does, and it is
    # comfortably above what the duration estimator settles on for "Hi.".
    # The voice stays empty: OmniVoice's catalog is empty with an unnamed
    # package default, so an unset voice_id must succeed rather than refuse.
    synth_add_cleanup_test(omnivoice "${SYNTH_OMNIVOICE_TEST_MODEL}" 720000 "text:Hi." "" "en")

    # What a limit stop reports back. Registered because it needs the package:
    # docs/c-interface.md makes an output limit one of four cases that must fill
    # the result, and this family reported it as a graph failure with every
    # resolved field zeroed.
    if(EXISTS "${SYNTH_QWEN3_TTS_TEST_MODEL}")
        add_executable(synthesize-qwen3-tts-output-limit-test qwen3_tts_output_limit_test.cpp)
        target_link_libraries(synthesize-qwen3-tts-output-limit-test PRIVATE synthesize)
        synth_register_integration_target(synthesize-qwen3-tts-output-limit-test)
        add_test(NAME synthesize-qwen3-tts-output-limit
            COMMAND synthesize-qwen3-tts-output-limit-test ${SYNTH_QWEN3_TTS_TEST_MODEL})
        set_tests_properties(synthesize-qwen3-tts-output-limit PROPERTIES
            LABELS "integration;abi;qwen3-tts"
            TIMEOUT 600)
    endif()

    # Stage 2 Plan 1's closing check: the real Base package through the
    # public C interface only. Task 9's review proved the capability-snapshot
    # seam line in src/synthesize.cpp (`shared.voice_profile =
    # info.voice_profile;`) had zero coverage at the ABI level -- deleting it
    # left the full unit gate green. This is that coverage: it needs the real
    # package because a synthetic HParams fixture (as
    # tests/qwen3_tts_voice_required_test.cpp already uses) cannot prove the
    # on-disk metadata reaches the public struct, only that the function
    # would do the right thing if it did.
    if(EXISTS "${SYNTH_QWEN3_TTS_BASE_TEST_MODEL}")
        add_executable(synthesize-qwen3-tts-base-load-real qwen3_tts_base_load_real.cpp)
        target_link_libraries(synthesize-qwen3-tts-base-load-real PRIVATE synthesize)
        synth_register_integration_target(synthesize-qwen3-tts-base-load-real)
        add_test(NAME synthesize-qwen3-tts-base-load-real
            COMMAND synthesize-qwen3-tts-base-load-real ${SYNTH_QWEN3_TTS_BASE_TEST_MODEL})
        set_tests_properties(synthesize-qwen3-tts-base-load-real PROPERTIES
            LABELS "integration;abi;qwen3-tts"
            TIMEOUT 300)
    endif()

    # The foundation slice's closing check: the real OmniVoice package through
    # the family loader and then through the public seam. It needs the package
    # because that is the whole point — every earlier omnivoice test runs
    # against a synthetic one, which can only prove the rules agree with
    # themselves.
    if(EXISTS "${SYNTH_OMNIVOICE_TEST_MODEL}")
        add_executable(synthesize-omnivoice-load-real omnivoice_load_real.cpp)
        target_include_directories(synthesize-omnivoice-load-real PRIVATE
            ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(synthesize-omnivoice-load-real PRIVATE synthesize ggml)
        synth_register_integration_target(synthesize-omnivoice-load-real)
        add_test(NAME synthesize-omnivoice-load-real
            COMMAND synthesize-omnivoice-load-real ${SYNTH_OMNIVOICE_TEST_MODEL})
        set_tests_properties(synthesize-omnivoice-load-real PROPERTIES
            LABELS "integration;omnivoice;abi"
            TIMEOUT 600)
    endif()

    # Reference Audio profiles end to end (Plan 3's Task 14): the public
    # create_from_reference dispatch against the real package and its pinned
    # reference clip. Guarded on the wav too -- a locally-materialized asset
    # like the model itself (models/ is gitignored), not something every
    # checkout has merely because the model does.
    set(_synth_omnivoice_reference_wav
        "${CMAKE_SOURCE_DIR}/models/omnivoice-reference-audio/seedtts_ref_en_1.wav")
    if(EXISTS "${SYNTH_OMNIVOICE_TEST_MODEL}" AND EXISTS "${_synth_omnivoice_reference_wav}")
        add_executable(synthesize-omnivoice-profile-test omnivoice_profile_test.cpp)
        target_include_directories(synthesize-omnivoice-profile-test PRIVATE
            ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(synthesize-omnivoice-profile-test PRIVATE synthesize ggml)
        synth_register_integration_target(synthesize-omnivoice-profile-test)
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-profile")
        add_test(NAME synthesize-omnivoice-profile-test
            COMMAND synthesize-omnivoice-profile-test
                ${SYNTH_OMNIVOICE_TEST_MODEL}
                ${_synth_omnivoice_reference_wav}
                ${CMAKE_SOURCE_DIR}/build/goldens/omnivoice/omni-clone-en/ref/tokens.i32
                ${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-profile)
        set_tests_properties(synthesize-omnivoice-profile-test PROPERTIES
            LABELS "integration;omnivoice;abi"
            # One real encode_reference pass (HuBERT semantic branch, DAC
            # acoustic encoder, RVQ) over ~14 s of audio, the silent-clip
            # case's own pass over exactly 1 s, and -- since Task 16's
            # Serialized Profile round trip -- four more full synthesize
            # calls through the 0.6B-parameter generator (original and
            # reloaded, ClonePrompt and DesignInstruct, each a real
            # denoising-loop decode, not just an encode). 600s covered the
            # pre-Task-16 shape comfortably.
            #
            # Fix round 1 (reviewer FINDING 3): the round-trip text was
            # originally a full sentence per call; measured wall time (this
            # binary run directly, unconstrained, uncontended) was
            # 134-246s on a plain build and ~2700s under the sanitizer
            # build (docs/testing.md's required gate for any C/C++
            # inference change -- ASan/UBSan instrumentation on this much
            # real generator work is substantially slower, not just
            # differently-behaved). ~2700s would already have exceeded the
            # 1800s ceiling this task's own first round chose, so the
            # round-trip text was shortened to "Hi." (the same minimal text
            # this file's own cleanup-test invocation already exercises
            # successfully against this real package) to cut real,
            # unnecessary cost rather than only widen the ceiling: the
            # byte-identity claim under test needs matching output, not a
            # long one. Re-measured after that change: 93.87s plain,
            # ~1808s (30:07.97) sanitizer -- a real reduction, but not
            # enough on its own to fit inside 1800s with any margin, so the
            # ceiling below still needs raising. 3600s leaves roughly 2x
            # headroom over the slower, sanitizer-build measurement,
            # consistent with this project's own convention for other
            # real-model integration gates in this same file (several
            # already sit at 1800/3600/7200/14400s).
            TIMEOUT 3600)
    endif()

    # The resampler's real-signal gate: the same
    # synthesize-omnivoice-resampler-test binary the unit gate already built
    # (registered unconditionally above with no arguments), re-invoked here
    # with the golden directory as argv[1] so its sentinel-guarded branch
    # loads ref/pcm_24k.f32 and ref/pcm_16k.f32 and asserts max_abs against
    # the oracle. Needs only the (uncommitted) oracle payload, not the
    # package itself -- unlike every other golden gate in this file, there is
    # no model here to check for.
    set(_synth_omnivoice_resampler_golden_sentinel
        "${CMAKE_SOURCE_DIR}/build/goldens/omnivoice/omni-clone-en/ref/pcm_16k.f32")
    if(EXISTS "${_synth_omnivoice_resampler_golden_sentinel}")
        add_test(
            NAME synthesize-omnivoice-resampler-golden
            COMMAND synthesize-omnivoice-resampler-test
                "${CMAKE_SOURCE_DIR}/build/goldens/omnivoice/omni-clone-en")
        set_tests_properties(synthesize-omnivoice-resampler-golden PROPERTIES
            LABELS "integration;omnivoice"
            TIMEOUT 60)
    endif()

    # OmniVoice's replay validator runs with --check, so it is a gate against
    # the committed tolerances rather than a measurement -- and its exact-token
    # arm fails any greedy case whose grid is not the oracle's, tolerance file
    # or no. The golden sentinel is the oracle payload, deliberately not
    # committed.
    #
    # TIMEOUT 14400: this gate re-runs 17 greedy free-runs, each 32 denoising
    # steps at two forwards per step over the whole canvas -- omni-long-boundary
    # alone is 719 frames and 126 s. The twenty-case sweep takes about seven
    # minutes on the 20-CPU development host (417-420 s across this gate's own
    # runs; 453 s is the worst observed), so four hours is headroom for a slower
    # or busier machine rather than an estimate: a gate that times out reports a
    # failure it did not find.
    set(_synth_omnivoice_golden_sentinel
        "${CMAKE_SOURCE_DIR}/build/goldens/omnivoice/omni-short-en/codes/grid.i32")
    if(EXISTS "${SYNTH_OMNIVOICE_TEST_MODEL}" AND EXISTS "${_synth_omnivoice_golden_sentinel}")
        add_test(
            NAME synthesize-omnivoice-replay-golden
            COMMAND ${SYNTH_UV_EXECUTABLE} run
                --project ${CMAKE_SOURCE_DIR}/scripts/envs/omnivoice
                --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-omnivoice-replay.py
                --manifest ${CMAKE_SOURCE_DIR}/tests/golden/omnivoice/omnivoice-0-6b.manifest.json
                --model ${SYNTH_OMNIVOICE_TEST_MODEL}
                --runner $<TARGET_FILE:synthesize-omnivoice-replay-real>
                --check --profile F32 --backend CPU --stage replay)
        set_tests_properties(synthesize-omnivoice-replay-golden PROPERTIES
            LABELS "integration;omnivoice;golden"
            TIMEOUT 14400
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()

    # Plan 4 Task 3 measured two codec-only Quantization Profiles
    # (Q8_CODEC_MIXED, F16_CODEC) against this same gate and BOTH failed the
    # cloning path's RVQ
    # encode grid -- 1023/2808 and 103/2808 positions respectively, neither a
    # knife-edge margin call (docs/porting/families/omnivoice.md's
    # "Quantization Profile Shape"). Neither ships, and no per-profile golden
    # gate is registered for either, deliberately --
    # a permanently-red registered test is not an acceptable end state for
    # the branch, and the measured evidence for both is recorded in that doc
    # and in reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md instead
    # of as a failing gate here.
    #
    # Amended 2026-08-09: this comment used to say "This family ships
    # F32-only", which stopped being true when the generator-half F16 and Q8
    # profiles were published on 2026-08-09. They register no per-profile
    # golden gate either, for the opposite reason: a generator-half profile
    # re-draws the greedy token grid by design (Q8 flips 95.83% of positions),
    # so there is no oracle-parity grid for a gate to compare it against. What
    # they do inherit unconditionally is the CODEC half's guarantee -- their
    # codec tensors are bit-identical to F32's, so the cloning path's RVQ
    # encode stays byte-exact -- which is what makes them shippable and the
    # two codec-half profiles above not.

    # Plan 4 Task 11's CUDA sweep, same manifest and sentinel as the CPU gate
    # above -- only the requested backend and the tolerance cell it checks
    # against change. --accelerate moves the codec's decode graph onto the
    # primary CUDA device (Task 9's accelerator twin); as of Plan 5 Task 1 it
    # ALSO moves the generator (Task 1's own twin) -- docs/backends.md's
    # discrete-outputs rule no longer holds this family's generator on CPU
    # unconditionally, per the narrow, measured exception recorded there.
    # The validator's own placement check fails the gate if either half
    # lands wrong (both must now be fully off the CPU under --accelerate).
    #
    # --tokens-informational: exact-token identity does NOT survive the
    # generator's move (Plan 5 Task 1/2 measured 3 of 17 greedy grids exact,
    # the rest 0.2%-98.3% content-flipped against the CPU oracle) and this
    # flag stops that from failing the gate. What still fails the gate
    # unconditionally, flag or not: every greedy grid's SIZE against the
    # oracle's own (RuleDurationEstimator fixes the canvas length before the
    # first generator forward runs, so a size mismatch here is a real
    # regression, not the expected content drift), the cloning path's
    # ref.tokens grids (codec-only, untouched by the generator's move, still
    # byte-exact), and the tolerance cell's probes (re-measured against this
    # same generator-on-CUDA configuration -- see
    # tests/tolerances/omnivoice.json's own note for the run this cell came
    # from). A gate with all three of those still real is not a gate that
    # asserts nothing; it is the structural half of the old claim plus a
    # calibrated numerical half, with only per-token content identity
    # retired -- deliberately, per jiangzhuo's 2026-08-08 ruling and the
    # Listening Audit backing it.
    #
    # TIMEOUT: measured at approximately 350-360s for the full twenty-case
    # sweep on build/rel-dgx-spark (this task's own re-measurement, quiet
    # host) -- markedly faster than the CPU gate's ~7 minutes now that the
    # generator itself runs on the accelerator rather than dominating wall
    # time on the CPU, though per-case process/model-load overhead (twenty
    # separate runner invocations) keeps it from scaling down by anything
    # close to the generator's own 22x RTF speedup. 14400s is kept as
    # headroom for a slower or busier machine, consistent with the CPU
    # gate's own rationale, rather than retuned to this measurement.
    if(SYNTH_CUDA AND EXISTS "${SYNTH_OMNIVOICE_TEST_MODEL}" AND EXISTS "${_synth_omnivoice_golden_sentinel}")
        add_test(
            NAME synthesize-omnivoice-replay-golden-cuda
            COMMAND ${SYNTH_UV_EXECUTABLE} run
                --project ${CMAKE_SOURCE_DIR}/scripts/envs/omnivoice
                --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-omnivoice-replay.py
                --manifest ${CMAKE_SOURCE_DIR}/tests/golden/omnivoice/omnivoice-0-6b.manifest.json
                --model ${SYNTH_OMNIVOICE_TEST_MODEL}
                --runner $<TARGET_FILE:synthesize-omnivoice-replay-real>
                --accelerate --tokens-informational --check --profile F32 --backend CUDA --stage replay)
        set_tests_properties(synthesize-omnivoice-replay-golden-cuda PROPERTIES
            LABELS "integration;omnivoice;golden;cuda"
            TIMEOUT 14400
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()

    # OmniVoice's public-request validator needs no oracle payload -- it
    # asserts relations between runs of this port's public seam (seed echo,
    # same-seed byte identity, cross-seed artifact_differs), not agreement
    # with the reference -- so it is model-guarded only, unlike the golden
    # gate above which also needs the (uncommitted) oracle sentinel.
    if(EXISTS "${SYNTH_OMNIVOICE_TEST_MODEL}")
        add_test(
            NAME synthesize-omnivoice-public-request
            COMMAND ${SYNTH_UV_EXECUTABLE} run
                --project ${CMAKE_SOURCE_DIR}/scripts/envs/omnivoice
                --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-omnivoice-public.py
                --model ${SYNTH_OMNIVOICE_TEST_MODEL}
                --runner $<TARGET_FILE:synthesize-omnivoice-public-real>
                --profile F32 --backend cpu)
        set_tests_properties(synthesize-omnivoice-public-request PROPERTIES
            LABELS "integration;omnivoice;abi"
            TIMEOUT 1800
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()

    # Plan 4 Task 11: the same public-request relations driven with
    # --backend cuda, now that tests/omnivoice_public_real.c actually forwards
    # it (accumulated requirement 2 -- before this task the flag reached only
    # the report dict). Check 14 (this stage's own positive assertion,
    # accumulated requirement 1) only runs in this CUDA arm.
    if(SYNTH_CUDA AND EXISTS "${SYNTH_OMNIVOICE_TEST_MODEL}")
        add_test(
            NAME synthesize-omnivoice-public-request-cuda
            COMMAND ${SYNTH_UV_EXECUTABLE} run
                --project ${CMAKE_SOURCE_DIR}/scripts/envs/omnivoice
                --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-omnivoice-public.py
                --model ${SYNTH_OMNIVOICE_TEST_MODEL}
                --runner $<TARGET_FILE:synthesize-omnivoice-public-real>
                --profile F32 --backend cuda)
        set_tests_properties(synthesize-omnivoice-public-request-cuda PROPERTIES
            LABELS "integration;omnivoice;abi;cuda"
            TIMEOUT 1800
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()

    # Qwen3-TTS's two validators. Both need the package, which is why they sat
    # unregistered while the family shipped -- and being unregistered is how the
    # replay stage stopped being something CI could run at all.
    #
    # The replay validator runs with --check, so it is a gate against the
    # committed tolerances rather than a measurement: without it the file is a
    # record nothing enforces. The golden sentinel is the oracle payload the
    # replay needs, which is deliberately not committed.
    set(_synth_qwen3_tts_golden_sentinel
        "${CMAKE_SOURCE_DIR}/build/goldens/qwen3-tts/qwen3-minimal/audio/pcm.f32")
    if(EXISTS "${SYNTH_QWEN3_TTS_TEST_MODEL}")
        if(EXISTS "${_synth_qwen3_tts_golden_sentinel}")
            add_test(
                NAME synthesize-qwen3-tts-replay-golden
                COMMAND ${SYNTH_UV_EXECUTABLE} run
                    --project ${CMAKE_SOURCE_DIR}/scripts/envs/qwen3-tts
                    --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-qwen3-tts-replay.py
                    --manifest ${CMAKE_SOURCE_DIR}/tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice.manifest.json
                    --model ${SYNTH_QWEN3_TTS_TEST_MODEL}
                    --runner $<TARGET_FILE:synthesize-qwen3-tts-replay-real>
                    --check --profile BF16 --backend CPU --stage replay)
            set_tests_properties(synthesize-qwen3-tts-replay-golden PROPERTIES
                LABELS "integration;qwen3-tts;golden"
                TIMEOUT 7200
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        endif()
        # The public-request phase needs no oracle payload: it asserts relations
        # between runs of this port, not agreement with the reference.
        add_test(
            NAME synthesize-qwen3-tts-public-request
            COMMAND ${SYNTH_UV_EXECUTABLE} run
                --project ${CMAKE_SOURCE_DIR}/scripts/envs/qwen3-tts
                --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-qwen3-tts-public.py
                --model ${SYNTH_QWEN3_TTS_TEST_MODEL}
                --runner $<TARGET_FILE:synthesize-qwen3-tts-public-real>
                --profile BF16 --backend cpu)
        set_tests_properties(synthesize-qwen3-tts-public-request PROPERTIES
            LABELS "integration;qwen3-tts;abi"
            TIMEOUT 1800
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()

    # Kokoro's Golden validators drive one runner selected by stage name; see
    # tests/kokoro_stages_real.cpp for why the family has one adapter and not
    # seven.
    set(_synth_kokoro_golden_sentinel
        "${CMAKE_SOURCE_DIR}/build/goldens/kokoro/kokoro-upstream-default/audio/pcm.f32")
    if(EXISTS "${SYNTH_KOKORO_TEST_MODEL}" AND EXISTS "${_synth_kokoro_golden_sentinel}")
        foreach(_synth_kokoro_stage plbert duration prosody text-encoder source decoder waveform)
            add_test(
                NAME synthesize-kokoro-${_synth_kokoro_stage}-golden
                COMMAND ${SYNTH_UV_EXECUTABLE} run
                    --project ${CMAKE_SOURCE_DIR}/scripts/envs/kokoro
                    --locked python
                        ${CMAKE_SOURCE_DIR}/scripts/validate-kokoro-${_synth_kokoro_stage}.py
                    --manifest ${CMAKE_SOURCE_DIR}/tests/golden/kokoro/kokoro-v1-0.manifest.json
                    --model ${SYNTH_KOKORO_TEST_MODEL}
                    --runner $<TARGET_FILE:synthesize-kokoro-stages>
                    --threads 1)
            set_tests_properties(synthesize-kokoro-${_synth_kokoro_stage}-golden PROPERTIES
                LABELS "integration;kokoro;golden"
                TIMEOUT 3600
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        endforeach()
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
    USES_TERMINAL)
get_property(_synth_unit_targets GLOBAL PROPERTY SYNTH_UNIT_TEST_TARGETS)
if(_synth_unit_targets)
    add_dependencies(synthesize-check-unit ${_synth_unit_targets})
endif()
if(SYNTH_BUILD_TOOLS)
    add_dependencies(synthesize-check-unit synthesize-quantize)
endif()
if(SYNTH_BUILD_CLI)
    add_dependencies(synthesize-check-unit synthesize-cli)
endif()

if(SYNTH_BUILD_INTEGRATION_TESTS)
    add_custom_target(synthesize-check-integration
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L integration
        USES_TERMINAL)
    # The CLI is not a test and not a runner, but the integration gate drives
    # it, so it has to be registered like anything else the gate needs.
    if(SYNTH_BUILD_CLI)
        synth_register_integration_target(synthesize-cli)
    endif()
    get_property(_synth_integration_targets GLOBAL PROPERTY SYNTH_INTEGRATION_TARGETS)
    if(_synth_integration_targets)
        add_dependencies(synthesize-check-integration ${_synth_integration_targets})
    endif()
endif()
