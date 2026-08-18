cmake_minimum_required(VERSION 3.24)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/synthesize.h" _synth_header)
string(REGEX MATCH "define +SYNTH_VERSION_MAJOR +([0-9]+)" _ "${_synth_header}")
set(SYNTH_VERSION_MAJOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "define +SYNTH_VERSION_MINOR +([0-9]+)" _ "${_synth_header}")
set(SYNTH_VERSION_MINOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "define +SYNTH_VERSION_PATCH +([0-9]+)" _ "${_synth_header}")
set(SYNTH_VERSION_PATCH "${CMAKE_MATCH_1}")
string(REGEX MATCH
    "define +SYNTH_ABI_VERSION +\\(\\(uint32_t\\) +([0-9]+)u\\)"
    _ "${_synth_header}")
set(SYNTH_ABI_VERSION "${CMAKE_MATCH_1}")

foreach(_synth_version_component MAJOR MINOR PATCH)
    if(NOT SYNTH_VERSION_${_synth_version_component} MATCHES "^[0-9]+$")
        message(FATAL_ERROR
            "Could not parse SYNTH_VERSION_${_synth_version_component} from include/synthesize.h")
    endif()
endforeach()
if(NOT SYNTH_ABI_VERSION MATCHES "^[0-9]+$")
    message(FATAL_ERROR
        "Could not parse SYNTH_ABI_VERSION from include/synthesize.h")
endif()

project(synthesize
    VERSION ${SYNTH_VERSION_MAJOR}.${SYNTH_VERSION_MINOR}.${SYNTH_VERSION_PATCH}
    LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(_synth_metal_default ON)
else()
    set(_synth_metal_default OFF)
endif()

option(SYNTH_BUILD_SHARED "Build synthesize and ggml as shared libraries" OFF)
if(SYNTH_BUILD_SHARED AND UNIX AND NOT APPLE)
    set(CMAKE_INSTALL_RPATH "\$ORIGIN")
elseif(SYNTH_BUILD_SHARED AND APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
endif()
option(SYNTH_GGML_BACKEND_DL "Build GGML execution backends as loadable modules" OFF)
if(SYNTH_GGML_BACKEND_DL)
    if(NOT SYNTH_BUILD_SHARED)
        message(FATAL_ERROR
            "SYNTH_GGML_BACKEND_DL requires SYNTH_BUILD_SHARED=ON")
    endif()
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    if(WIN32)
        set(_synth_backend_module_install_dir
            "${CMAKE_INSTALL_BINDIR}/synthesize/backends")
    else()
        set(_synth_backend_module_install_dir
            "${CMAKE_INSTALL_LIBDIR}/synthesize/backends")
    endif()
    set(GGML_BACKEND_DIR "${_synth_backend_module_install_dir}"
        CACHE PATH "" FORCE)
else()
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DIR "" CACHE PATH "" FORCE)
endif()
set(_synth_cpu_runtime_dispatch_default OFF)
if(SYNTH_GGML_BACKEND_DL AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(_synth_cpu_runtime_dispatch_default ON)
endif()
option(SYNTH_CPU_RUNTIME_DISPATCH
    "Build runtime-selected x86 CPU execution modules"
    ${_synth_cpu_runtime_dispatch_default})
if(SYNTH_CPU_RUNTIME_DISPATCH)
    if(NOT SYNTH_GGML_BACKEND_DL)
        message(FATAL_ERROR
            "SYNTH_CPU_RUNTIME_DISPATCH requires SYNTH_GGML_BACKEND_DL=ON")
    endif()
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        message(FATAL_ERROR
            "SYNTH_CPU_RUNTIME_DISPATCH is supported only on x86-64")
    endif()
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
else()
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
endif()
option(SYNTH_BUILD_CLI "Build the reference command-line Adapter" ON)
option(SYNTH_BUILD_TOOLS "Build model inspection and quantization tools" OFF)
option(SYNTH_BUILD_TESTS "Build synthesize tests and validation runners" ON)
option(SYNTH_BUILD_PYTHON_TESTS "Register locked Python converter and oracle unit tests" ON)
option(SYNTH_BUILD_INTEGRATION_TESTS "Register tests requiring local model and Golden artifacts" OFF)
option(SYNTH_BUILD_PYTHON_PROVIDER
    "Install only the Python Native Provider wheel component" OFF)
set(SYNTH_PYTHON_PROVIDER_ID "default" CACHE STRING
    "Native Provider identity stamped into Python packaging artifacts")
set_property(CACHE SYNTH_PYTHON_PROVIDER_ID PROPERTY STRINGS default cu13)
option(SYNTH_SANITIZE "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(SYNTH_CUDA "Enable the GGML CUDA backend" OFF)
option(SYNTH_METAL "Enable the GGML Metal backend" ${_synth_metal_default})
option(SYNTH_VULKAN "Enable the GGML Vulkan backend" OFF)
option(SYNTH_USE_OPENMP "Use OpenMP instead of GGML's native CPU thread pool" OFF)
set(SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM "" CACHE STRING
    "Internal release-preset CUDA platform policy selector")
mark_as_advanced(SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM)
set(SYNTH_VITS_TEST_MODEL "${CMAKE_SOURCE_DIR}/models/vits-ljspeech/vits-ljspeech-F32.gguf"
    CACHE FILEPATH "Local VITS F32 GGUF used by integration tests")
set(SYNTH_VITS_VCTK_TEST_MODEL "${CMAKE_SOURCE_DIR}/models/vits-vctk/vits-vctk-F32.gguf"
    CACHE FILEPATH "Local multi-speaker VITS VCTK F32 GGUF used by integration tests")
set(SYNTH_KOKORO_TEST_MODEL "${CMAKE_SOURCE_DIR}/models/kokoro-v1-0/kokoro-v1-0-F32.gguf"
    CACHE FILEPATH "Local Kokoro F32 GGUF used by integration tests")
# The source profile, because this family's is BF16 rather than F32: the talker
# checkpoint stores bfloat16 and widening it would describe a different model.
set(SYNTH_QWEN3_TTS_TEST_MODEL
    "${CMAKE_SOURCE_DIR}/models/qwen3-tts-12hz-0-6b-customvoice/qwen3-tts-12hz-0-6b-customvoice-BF16.gguf"
    CACHE FILEPATH "Local Qwen3-TTS BF16 GGUF used by integration tests")
# Stage 2's Reference Model Variant: the Base checkpoint, profile-sources
# voice mode, no Preset Voice Catalog. Same BF16 source-profile reasoning as
# the CustomVoice variable above -- this family's talker checkpoint is
# bfloat16 regardless of which variant it is.
set(SYNTH_QWEN3_TTS_BASE_TEST_MODEL
    "${CMAKE_SOURCE_DIR}/models/qwen3-tts-12hz-0-6b-base/qwen3-tts-12hz-0-6b-base-BF16.gguf"
    CACHE FILEPATH "Local Qwen3-TTS Base BF16 GGUF used by integration tests")
# Stage 3's Reference Model Variant: the VoiceDesign checkpoint, profile-sources
# voice mode, Description Text rather than Reference Audio, no Preset Voice
# Catalog. Same BF16 source-profile reasoning as the two variables above.
# Added in Task 7 Fix Round 2, once synthesize-qwen3-tts-voicedesign-prefill-real
# gained an add_test that needs a real model path at configure time -- the
# same reason SYNTH_QWEN3_TTS_BASE_TEST_MODEL exists rather than a literal.
set(SYNTH_QWEN3_TTS_VOICEDESIGN_TEST_MODEL
    "${CMAKE_SOURCE_DIR}/models/qwen3-tts-12hz-1-7b-voicedesign/qwen3-tts-12hz-1-7b-voicedesign-BF16.gguf"
    CACHE FILEPATH "Local Qwen3-TTS VoiceDesign BF16 GGUF used by integration tests")
# The Base variant's clone path needs a second locally-materialized artifact
# besides the package: the Reference Audio it enrolls from. It is a cache
# variable for the same reason every model path above is -- `models/` is
# gitignored, the file is fetched rather than committed, and a checkout that
# keeps it somewhere else must be able to say so instead of editing
# tests/CMakeLists.txt. It is the same clip
# scripts/dump_reference_qwen3_tts_speaker.py caches, keyed by the Golden
# Manifest's own reference locator.
set(SYNTH_QWEN3_TTS_REFERENCE_WAV
    "${CMAKE_SOURCE_DIR}/models/qwen3-tts-reference-audio/clone.wav"
    CACHE FILEPATH "Local 24 kHz mono Reference Audio clip used by Qwen3-TTS Base clone tests")
# F32 is this family's source profile: the checkpoint stores F32 in both the
# generator and the codec halves, so it is the package a port is checked
# against. Plan 4 Task 3 measured both Q8_CODEC_MIXED and F16_CODEC codec-only
# Quantization Profiles against the exact-token gate and both failed the
# cloning path's RVQ encode grid (docs/porting/families/omnivoice.md's
# "Quantization Profile Shape"). Amended 2026-08-10: this used to conclude
# "so this family ships F32-only", which stopped being true on 2026-08-09 when
# the generator-half F16 and Q8 profiles were published; the identically
# worded comment in tests/CMakeLists.txt was amended then and this one was
# missed. What still holds is the narrower claim -- no per-profile test-model
# variable or golden gate exists for the two CODEC-half profiles, on
# purpose: a permanently failing registered gate is not an acceptable branch
# state, and the measured evidence for both is recorded in that doc and in
# reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md instead.
set(SYNTH_OMNIVOICE_TEST_MODEL "${CMAKE_SOURCE_DIR}/models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf"
    CACHE FILEPATH "Local OmniVoice F32 GGUF used by integration tests")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SynthesizeReleaseCudaPolicy.cmake)
synth_validate_release_cuda_targets()
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SynthesizeCudaToolchainPolicy.cmake)
if(SYNTH_CUDA)
    find_package(CUDAToolkit 13.3 EXACT REQUIRED)
    synth_validate_cuda_toolkit()
endif()

if(SYNTH_SANITIZE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(FATAL_ERROR "SYNTH_SANITIZE requires Clang or GCC")
    endif()
    add_compile_options(
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -fno-sanitize-recover=undefined
        -fsanitize-recover=pointer-overflow)
    add_link_options(
        -fsanitize=address,undefined
        -fno-sanitize-recover=undefined
        -fsanitize-recover=pointer-overflow)
endif()

set(GGML_CUDA ${SYNTH_CUDA} CACHE BOOL "" FORCE)
set(GGML_METAL ${SYNTH_METAL} CACHE BOOL "" FORCE)
set(GGML_VULKAN ${SYNTH_VULKAN} CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
set(GGML_OPENMP ${SYNTH_USE_OPENMP} CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ${SYNTH_BUILD_SHARED} CACHE BOOL "" FORCE)
if(SYNTH_BUILD_PYTHON_PROVIDER AND SYNTH_CUDA)
    set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
endif()
if(SYNTH_METAL)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

# ggml is a git submodule, not a vendored copy. An uninitialized one leaves an
# empty directory here, and add_subdirectory's own error names a missing
# CMakeLists.txt without saying why it is missing or how to fix it.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "ggml/ is empty: it is a git submodule and has not been initialized.\n"
        "  git submodule update --init --recursive\n"
        "or clone with --recurse-submodules. See ggml-patches/README.md.")
endif()

add_subdirectory(ggml)

if(SYNTH_GGML_BACKEND_DL)
    foreach(_synth_backend_target IN LISTS GGML_AVAILABLE_BACKENDS)
        if(APPLE)
            set_property(TARGET ${_synth_backend_target}
                PROPERTY INSTALL_RPATH "@loader_path/../..")
        elseif(UNIX)
            set_property(TARGET ${_synth_backend_target}
                PROPERTY INSTALL_RPATH "\$ORIGIN/../..")
        endif()
    endforeach()
endif()

if(NOT SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM STREQUAL "")
    find_program(SYNTH_CUOBJDUMP
        NAMES cuobjdump
        HINTS ${CUDAToolkit_BIN_DIR}
        NO_DEFAULT_PATH
        REQUIRED)
    mark_as_advanced(SYNTH_CUOBJDUMP)

    if(SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM STREQUAL "linux-aarch64")
        set(_synth_release_cubins "sm_121a")
    elseif(SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM STREQUAL "linux-x86_64")
        set(_synth_release_cubins
            "sm_75,sm_80,sm_86,sm_89,sm_90,sm_100,sm_120a")
    endif()

    add_custom_target(synthesize-check-release-cubins
        COMMAND ${CMAKE_COMMAND}
            -DSYNTH_CUOBJDUMP=${SYNTH_CUOBJDUMP}
            -DSYNTH_CUDA_LIBRARY=$<TARGET_FILE:ggml-cuda>
            -DSYNTH_EXPECTED_CUBINS=${_synth_release_cubins}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyCudaCubins.cmake
        DEPENDS ggml-cuda
        COMMENT "Verifying exact native CUDA cubin set and absence of PTX"
        VERBATIM)
endif()

# third_party/libsamplerate: libsamplerate 0.2.2 vendored verbatim (ADR 0009:
# docs/adr/0009-pin-reference-audio-normalization.md), statically compiled
# into the private Audio Normalizer Module (src/audio-normalizer.cpp). Never
# edit the vendored sources; configure them only through the compile
# definitions below. Provenance and the exact kept file subset are recorded in
# third_party/libsamplerate/VENDORED.md.
add_library(synth-libsamplerate OBJECT
    third_party/libsamplerate/src/samplerate.c
    third_party/libsamplerate/src/src_sinc.c
    third_party/libsamplerate/src/src_linear.c
    third_party/libsamplerate/src/src_zoh.c
)
target_include_directories(synth-libsamplerate
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libsamplerate/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libsamplerate/src
)
target_compile_definitions(synth-libsamplerate PRIVATE
    # ADR 0009 pins SRC_SINC_BEST_QUALITY only; Medium and Fast stay disabled.
    # Zero-order-hold and Linear are compiled in regardless, because
    # samplerate.c's converter dispatch calls their constructors
    # unconditionally rather than behind an ENABLE_* guard.
    ENABLE_SINC_BEST_CONVERTER
    # This tree is never run through its own ./configure or CMake, so no
    # generated config.h ever exists. HAVE_CONFIG_H is intentionally left
    # undefined, which means the vendored sources' own
    # "#ifdef HAVE_CONFIG_H / #include config.h" guards never fire; these are
    # the handful of macros the sources otherwise read unconditionally.
    HAVE_STDBOOL_H
    PACKAGE="libsamplerate"
    VERSION="0.2.2"
)
set_target_properties(synth-libsamplerate PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    # Vendored code; our own warning flags are not its contract to satisfy.
    # Scoped to this target only, never applied globally.
    target_compile_options(synth-libsamplerate PRIVATE -w)
endif()
find_library(SYNTH_LIBM_LIBRARY m)
if(SYNTH_LIBM_LIBRARY)
    target_link_libraries(synth-libsamplerate PUBLIC ${SYNTH_LIBM_LIBRARY})
endif()

add_subdirectory(src)

if(SYNTH_BUILD_PYTHON_PROVIDER)
    include(cmake/python-provider-install.cmake)
endif()

if(SYNTH_BUILD_CLI)
    add_subdirectory(examples)
endif()

if(SYNTH_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(SYNTH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(TARGETS synthesize
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/synthesize.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# GGML's own rules are `install(TARGETS ggml LIBRARY PUBLIC_HEADER)`. Naming
# specific artifact kinds excludes every kind not named, so that installs no
# static archive anywhere, and on Windows no DLL and no import library either --
# the Provider wheels and the installed SDK would ship without the library they
# exist to carry. Rather than patch the vendored tree, install the artifacts by
# file, which does not depend on how upstream classifies them.
# Only Windows needs this. Verified on Linux: upstream's rules install the static
# archives and the versioned shared objects with their symlinks correctly for both
# BUILD_SHARED_LIBS settings, so adding file-level rules there is redundant. On
# Windows a shared library's artifacts are RUNTIME (the DLL) and ARCHIVE (the
# import library), neither of which `LIBRARY` covers, so nothing would be
# installed at all. This has not been executed on Windows; synthesize-installed-sdk-test
# is the gate that proves it there.
if(WIN32 AND SYNTH_BUILD_SHARED)
    foreach(_synth_ggml_target ggml ggml-base)
        if(TARGET ${_synth_ggml_target})
            # The DLL belongs next to the executables that load it; the import
            # library belongs with the link-time artifacts.
            install(FILES $<TARGET_FILE:${_synth_ggml_target}>
                DESTINATION ${CMAKE_INSTALL_BINDIR})
            install(FILES $<TARGET_LINKER_FILE:${_synth_ggml_target}>
                DESTINATION ${CMAKE_INSTALL_LIBDIR})
        endif()
    endforeach()
endif()

if(TARGET synthesize-cli)
    install(TARGETS synthesize-cli
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

set(_synth_cmake_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/synthesize")
configure_package_config_file(
    cmake/synthesize-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/synthesize-config.cmake
    INSTALL_DESTINATION ${_synth_cmake_install_dir})
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/synthesize-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
if(WIN32 AND SYNTH_BUILD_SHARED)
    set(SYNTH_INSTALLED_LIBRARY_NAME "synthesize-${SYNTH_ABI_VERSION}")
else()
    set(SYNTH_INSTALLED_LIBRARY_NAME "synthesize")
endif()
configure_file(
    cmake/synthesize-targets.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/synthesize-targets.cmake @ONLY)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/synthesize-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/synthesize-config-version.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/synthesize-targets.cmake
    DESTINATION ${_synth_cmake_install_dir})

set(SYNTH_PKGCONFIG_PRIVATE_LIBS "")
set(SYNTH_PKGCONFIG_DEFINES "")
if(SYNTH_BUILD_SHARED)
    set(SYNTH_PKGCONFIG_DEFINES "-DSYNTH_SHARED")
else()
    string(APPEND SYNTH_PKGCONFIG_PRIVATE_LIBS
        " -lggml -lggml-cpu -lggml-base")
    if(SYNTH_CUDA)
        string(APPEND SYNTH_PKGCONFIG_PRIVATE_LIBS
            " -lggml-cuda -lcudart -lcublas -lcublasLt -lcuda")
    endif()
    if(SYNTH_VULKAN)
        string(APPEND SYNTH_PKGCONFIG_PRIVATE_LIBS
            " -lggml-vulkan -lvulkan")
    endif()
    if(APPLE)
        string(APPEND SYNTH_PKGCONFIG_PRIVATE_LIBS " -lc++")
        if(SYNTH_METAL)
            string(APPEND SYNTH_PKGCONFIG_PRIVATE_LIBS
                " -framework Foundation -framework Metal -framework MetalKit")
        endif()
    elseif(UNIX)
        string(APPEND SYNTH_PKGCONFIG_PRIVATE_LIBS
            " -lstdc++ -lm -ldl -lpthread")
    endif()
    string(STRIP "${SYNTH_PKGCONFIG_PRIVATE_LIBS}"
        SYNTH_PKGCONFIG_PRIVATE_LIBS)
endif()
configure_file(cmake/synthesize.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/synthesize.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/synthesize.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
