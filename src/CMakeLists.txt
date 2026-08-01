add_library(synthesize
    audio-delivery.cpp
    gguf-metadata.cpp
    backend-device.cpp
    backend-module.cpp
    backend-plan.cpp
    cpu-parallelism.cpp
    random-stream.cpp
    synthesize.cpp
    synthesis-request.cpp
    text-frontend.cpp
    bpe-frontend.cpp
    voice-profile.cpp
    arch/qwen3-tts/bpe.cpp
    arch/qwen3-tts/catalog.cpp
    arch/qwen3-tts/codec.cpp
    arch/qwen3-tts/code-predictor.cpp
    arch/qwen3-tts/code-predictor-host.cpp
    arch/qwen3-tts/model.cpp
    arch/qwen3-tts/operations.cpp
    arch/qwen3-tts/talker.cpp
    arch/qwen3-tts/talker-host.cpp
    arch/qwen3-tts/weights.cpp
    arch/omnivoice/catalog.cpp
    arch/omnivoice/codec.cpp
    arch/omnivoice/codec-host.cpp
    arch/omnivoice/frontend-host.cpp
    arch/omnivoice/generator.cpp
    arch/omnivoice/generator-host.cpp
    arch/omnivoice/model.cpp
    arch/omnivoice/weights.cpp
    arch/kokoro/catalog.cpp
    arch/kokoro/decoder.cpp
    arch/kokoro/decoder-host.cpp
    arch/kokoro/duration.cpp
    arch/kokoro/duration-host.cpp
    arch/kokoro/generator.cpp
    arch/kokoro/model.cpp
    arch/kokoro/operations.cpp
    arch/kokoro/plbert.cpp
    arch/kokoro/prosody.cpp
    arch/kokoro/quantization.cpp
    arch/kokoro/source.cpp
    arch/kokoro/text-encoder.cpp
    arch/kokoro/weights.cpp
    arch/vits/acoustic-flow.cpp
    arch/vits/acoustic-flow-host.cpp
    arch/vits/duration-path.cpp
    arch/vits/duration-predictor.cpp
    arch/vits/duration-predictor-host.cpp
    arch/vits/latent-sampling.cpp
    arch/vits/latent-sampling-host.cpp
    arch/vits/model.cpp
    arch/vits/operations.cpp
    arch/vits/prior-expansion.cpp
    arch/vits/prior-expansion-host.cpp
    arch/vits/text-encoder.cpp
    arch/vits/text-encoder-host.cpp
    arch/vits/voice-conditioning.cpp
    arch/vits/waveform-decoder-host.cpp
    arch/vits/waveform-decoder.cpp
    arch/vits/weights.cpp
)
add_library(synthesize::synthesize ALIAS synthesize)

target_include_directories(synthesize
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(synthesize PRIVATE $<BUILD_INTERFACE:ggml>)
target_compile_definitions(synthesize PRIVATE SYNTH_BUILDING_LIBRARY)

if(SYNTH_GGML_BACKEND_DL)
    target_compile_definitions(synthesize PRIVATE SYNTH_GGML_BACKEND_DL)
    if(CMAKE_DL_LIBS)
        target_link_libraries(synthesize PRIVATE ${CMAKE_DL_LIBS})
    endif()
endif()

if(SYNTH_BUILD_SHARED)
    target_compile_definitions(synthesize PUBLIC SYNTH_SHARED)
    if(UNIX AND NOT APPLE)
        target_link_options(synthesize PRIVATE
            "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/synthesize.map")
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(synthesize PRIVATE
        -Wall
        -Wextra
        -Wshadow
        -Wpedantic
        -Wno-unused-parameter
    )
endif()

set_target_properties(synthesize PROPERTIES
    OUTPUT_NAME synthesize
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION "${SYNTH_ABI_VERSION}"
)

if(WIN32 AND SYNTH_BUILD_SHARED)
    set_target_properties(synthesize PROPERTIES
        OUTPUT_NAME "synthesize-${SYNTH_ABI_VERSION}")
endif()
