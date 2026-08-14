# Runs the committed codec-encoder validator, and regenerates the port's half of
# its input first so it can never compare a stale dump.
#
# WHY THIS EXISTS. scripts/validate-qwen3-tts-codec_encoder.py computes all
# three of the thresholds Task 12 committed --
# `codec.chain`, `codec.rvq_reconstruction.semantic` and
# `codec.rvq_reconstruction.acoustic` -- and until 2026-08-14 NOTHING INVOKED
# IT. It was the only unregistered validator in the tree (every
# validate-vits-*, validate-omnivoice-* and the other two validate-qwen3-tts-*
# scripts are CTest tests), so all three numbers were inert: published in
# tests/tolerances/qwen3-tts.json and enforced by nobody. The one thing that DID
# run, `synthesize-tolerance-coverage`, is a metadata check -- it asserts that a
# cell and a same-named validator script both exist and never compares a number,
# so it was structurally incapable of noticing. `codec.chain` in particular --
# which that file calls the primary gate, and the only one of the three
# independent of the two float32 codebook tables being bit-identical by
# construction -- had no consumer of any kind. This registration is what makes
# them enforceable, and it is the only place `codec.chain` is checked at all.
#
# WHY THE DRIVER RUNS HERE RATHER THAN BEING PRE-MATERIALIZED. The oracle and
# the upstream-float32 dumps are captured artifacts and are guarded on
# existence; the PORT's dumps are this build's own output, and pointing the
# validator at a directory somebody generated once would gate a past revision of
# the port. So the driver is re-run into the build tree on every invocation. The
# oracle root supplies the driver's input as well as the comparison's reference:
# the driver takes `waveform.f32` and NOT a WAV, deliberately -- upstream casts
# the clip to bfloat16 before its first convolution, so feeding the WAV puts an
# input rounding through eleven convolutions before the comparison starts. That
# is the same distinction tests/qwen3_tts_icl_real.cpp records from the other
# side, where feeding the WAV is the point.
#
# It is a CMake -P script rather than a bare add_test because the run is two
# stages -- N driver invocations, then one validator -- the same shape
# tests/check-cli-real.cmake and tests/check-python-api-wheel.cmake already use.

foreach(_synth_required_variable
        SYNTH_SOURCE_DIR SYNTH_MODEL SYNTH_DRIVER SYNTH_ORACLE_ROOT
        SYNTH_UPSTREAM_F32_ROOT SYNTH_OUTPUT_ROOT SYNTH_UV_EXECUTABLE SYNTH_CASES)
    if(NOT DEFINED ${_synth_required_variable} OR "${${_synth_required_variable}}" STREQUAL "")
        message(FATAL_ERROR "check-qwen3-tts-codec-encoder requires ${_synth_required_variable}")
    endif()
endforeach()

# `SYNTH_CASES` arrives as one CMake list; the registration passes it escaped.
string(REPLACE "," ";" _synth_case_list "${SYNTH_CASES}")

set(_synth_validator_args "")
foreach(_synth_case IN LISTS _synth_case_list)
    set(_synth_waveform "${SYNTH_ORACLE_ROOT}/${_synth_case}/codec_encoder/waveform.f32")
    if(NOT EXISTS "${_synth_waveform}")
        message(FATAL_ERROR
            "case ${_synth_case}: the oracle dump has no codec_encoder/waveform.f32 at ${_synth_waveform}. "
            "This test is registered only when that artifact exists, so reaching here means the dump was "
            "removed after configure -- re-run scripts/dump_reference_qwen3_tts_codec_encoder.py and "
            "re-configure.")
    endif()
    set(_synth_case_out "${SYNTH_OUTPUT_ROOT}/${_synth_case}")
    file(REMOVE_RECURSE "${_synth_case_out}")
    file(MAKE_DIRECTORY "${_synth_case_out}")
    execute_process(
        COMMAND "${SYNTH_DRIVER}" "${SYNTH_MODEL}" "${_synth_waveform}" "${_synth_case_out}"
        RESULT_VARIABLE _synth_driver_status
        OUTPUT_VARIABLE _synth_driver_output
        ERROR_VARIABLE _synth_driver_error)
    if(NOT _synth_driver_status EQUAL 0)
        message(FATAL_ERROR
            "codec-encoder driver failed on ${_synth_case}:\n${_synth_driver_output}\n${_synth_driver_error}")
    endif()
    list(APPEND _synth_validator_args "--case" "${_synth_case}")
endforeach()

execute_process(
    COMMAND "${SYNTH_UV_EXECUTABLE}" run
        --project "${SYNTH_SOURCE_DIR}/scripts/envs/qwen3-tts"
        --locked python "${SYNTH_SOURCE_DIR}/scripts/validate-qwen3-tts-codec_encoder.py"
        --port "${SYNTH_OUTPUT_ROOT}"
        --upstream-f32 "${SYNTH_UPSTREAM_F32_ROOT}"
        --oracle "${SYNTH_ORACLE_ROOT}"
        # Absolute, not left to the script's own relative default: that default
        # resolves against the working directory, and a CTest run has no reason
        # to be in the source root. Passing it also means the cell this gate
        # reads is the committed one by construction rather than by cwd.
        --tolerances "${SYNTH_SOURCE_DIR}/tests/tolerances/qwen3-tts.json"
        ${_synth_validator_args}
    RESULT_VARIABLE _synth_validate_status
    OUTPUT_VARIABLE _synth_validate_output
    ERROR_VARIABLE _synth_validate_error)
message(STATUS "${_synth_validate_output}")
if(NOT _synth_validate_status EQUAL 0)
    message(FATAL_ERROR
        "validate-qwen3-tts-codec_encoder.py failed:\n${_synth_validate_output}\n${_synth_validate_error}")
endif()
