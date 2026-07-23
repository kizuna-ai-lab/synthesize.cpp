foreach(_synth_required_variable
        SYNTH_CUOBJDUMP SYNTH_CUDA_LIBRARY SYNTH_EXPECTED_CUBINS)
    if(NOT DEFINED ${_synth_required_variable} OR
       "${${_synth_required_variable}}" STREQUAL "")
        message(FATAL_ERROR
            "VerifyCudaCubins requires ${_synth_required_variable}")
    endif()
endforeach()

if(NOT EXISTS "${SYNTH_CUOBJDUMP}")
    message(FATAL_ERROR "cuobjdump does not exist: ${SYNTH_CUOBJDUMP}")
endif()
if(NOT EXISTS "${SYNTH_CUDA_LIBRARY}")
    message(FATAL_ERROR "CUDA library does not exist: ${SYNTH_CUDA_LIBRARY}")
endif()

execute_process(
    COMMAND "${SYNTH_CUOBJDUMP}" --list-elf "${SYNTH_CUDA_LIBRARY}"
    RESULT_VARIABLE _synth_elf_status
    OUTPUT_VARIABLE _synth_elf_output
    ERROR_VARIABLE _synth_elf_error)
if(NOT _synth_elf_status EQUAL 0)
    message(FATAL_ERROR
        "cuobjdump --list-elf failed (${_synth_elf_status}): "
        "${_synth_elf_error}")
endif()

string(REGEX MATCHALL "sm_[0-9]+[a-z]*"
    _synth_actual_cubins "${_synth_elf_output}")
list(REMOVE_DUPLICATES _synth_actual_cubins)
list(SORT _synth_actual_cubins)

string(REPLACE "," ";" _synth_expected_cubins "${SYNTH_EXPECTED_CUBINS}")
list(REMOVE_DUPLICATES _synth_expected_cubins)
list(SORT _synth_expected_cubins)

if(NOT "${_synth_actual_cubins}" STREQUAL "${_synth_expected_cubins}")
    message(FATAL_ERROR
        "CUDA release library must contain the exact native cubin set "
        "'${_synth_expected_cubins}', got '${_synth_actual_cubins}'")
endif()

execute_process(
    COMMAND "${SYNTH_CUOBJDUMP}" --list-ptx "${SYNTH_CUDA_LIBRARY}"
    RESULT_VARIABLE _synth_ptx_status
    OUTPUT_VARIABLE _synth_ptx_output
    ERROR_VARIABLE _synth_ptx_error)
if(NOT _synth_ptx_status EQUAL 0)
    message(FATAL_ERROR
        "cuobjdump --list-ptx failed (${_synth_ptx_status}): "
        "${_synth_ptx_error}")
endif()
if(_synth_ptx_output MATCHES "PTX file[ \t]+[0-9]+:")
    message(FATAL_ERROR
        "CUDA release library must not contain PTX; native cubins are required")
endif()

message(STATUS
    "Verified CUDA cubins: ${_synth_actual_cubins}; embedded PTX: none")
