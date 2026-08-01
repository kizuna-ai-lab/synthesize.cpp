# One check for every Model Family. What differs between them is the
# linguistic input a package accepts, whether it needs a Voice named, and
# which input kind its package rejects, so those are inputs rather than a
# second copy of this file.
foreach(variable SYNTH_CLI SYNTH_MODEL SYNTH_OUTPUT_DIR)
    if(NOT DEFINED ${variable})
        message(FATAL_ERROR "${variable} is required")
    endif()
endforeach()
if(NOT DEFINED SYNTH_TOKEN_IDS AND NOT DEFINED SYNTH_TEXT)
    message(FATAL_ERROR "one of SYNTH_TOKEN_IDS or SYNTH_TEXT is required")
endif()

set(voice_arguments "")
if(DEFINED SYNTH_VOICE AND NOT SYNTH_VOICE STREQUAL "")
    set(voice_arguments --voice "${SYNTH_VOICE}")
endif()

# VITS and Kokoro only accept token ids (their frontend is a symbol map).
# OmniVoice's package declares text_utf8 as its only supported input kind, so
# it drives the CLI's --text path instead of --token-ids.
set(input_arguments "")
if(DEFINED SYNTH_TEXT)
    set(input_arguments --text "${SYNTH_TEXT}")
else()
    set(input_arguments --token-ids "${SYNTH_TOKEN_IDS}")
endif()

# The unsupported-input arm below exercises whatever linguistic input kind
# this package's input_flags does NOT declare: text for VITS/Kokoro (which
# declare phonemes + token-ids only), phonemes for a text-only package like
# OmniVoice. Defaults to the flag every family used before this became a
# parameter, so a call that does not set it is unchanged.
if(NOT DEFINED SYNTH_UNSUPPORTED_FLAG)
    set(SYNTH_UNSUPPORTED_FLAG "--text")
endif()
if(NOT DEFINED SYNTH_UNSUPPORTED_VALUE)
    set(SYNTH_UNSUPPORTED_VALUE "Hello")
endif()

file(MAKE_DIRECTORY "${SYNTH_OUTPUT_DIR}")
set(first "${SYNTH_OUTPUT_DIR}/cli-seed-42-a.wav")
set(repeat "${SYNTH_OUTPUT_DIR}/cli-seed-42-b.wav")
set(different "${SYNTH_OUTPUT_DIR}/cli-seed-43.wav")
set(unsupported "${SYNTH_OUTPUT_DIR}/cli-unsupported.wav")
file(REMOVE "${first}" "${repeat}" "${different}" "${unsupported}")

function(run_synthesis output seed)
    execute_process(
        COMMAND "${SYNTH_CLI}"
            --model "${SYNTH_MODEL}"
            --output "${output}"
            ${input_arguments}
            ${voice_arguments}
            --seed "${seed}"
            --backend cpu
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "CLI synthesis failed (${result}): ${stdout}${stderr}")
    endif()
    if(NOT EXISTS "${output}")
        message(FATAL_ERROR "CLI reported success without writing ${output}")
    endif()
    file(READ "${output}" header_hex HEX LIMIT 12)
    string(LENGTH "${header_hex}" header_length)
    string(SUBSTRING "${header_hex}" 0 8 riff_signature)
    string(SUBSTRING "${header_hex}" 16 8 wave_signature)
    if(NOT header_length EQUAL 24 OR
       NOT riff_signature STREQUAL "52494646" OR
       NOT wave_signature STREQUAL "57415645")
        message(FATAL_ERROR "invalid WAV header: ${header_hex}")
    endif()
endfunction()

run_synthesis("${first}" 42)
run_synthesis("${repeat}" 42)
run_synthesis("${different}" 43)
file(SHA256 "${first}" first_sha256)
file(SHA256 "${repeat}" repeat_sha256)
file(SHA256 "${different}" different_sha256)
if(NOT first_sha256 STREQUAL repeat_sha256)
    message(FATAL_ERROR "same-seed CLI WAV files differ")
endif()
if(first_sha256 STREQUAL different_sha256)
    message(FATAL_ERROR "different-seed CLI WAV files are identical")
endif()

execute_process(
    COMMAND "${SYNTH_CLI}"
        --model "${SYNTH_MODEL}"
        --output "${unsupported}"
        "${SYNTH_UNSUPPORTED_FLAG}" "${SYNTH_UNSUPPORTED_VALUE}"
        ${voice_arguments}
        --backend cpu
    RESULT_VARIABLE unsupported_result
    OUTPUT_VARIABLE unsupported_stdout
    ERROR_VARIABLE unsupported_stderr)
if(unsupported_result EQUAL 0)
    message(FATAL_ERROR "unsupported input unexpectedly succeeded")
endif()
if(EXISTS "${unsupported}")
    message(FATAL_ERROR "failed CLI synthesis left an output file")
endif()

file(REMOVE "${first}" "${repeat}" "${different}" "${unsupported}")
