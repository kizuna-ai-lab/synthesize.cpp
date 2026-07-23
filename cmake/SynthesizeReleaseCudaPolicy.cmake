function(synth_validate_release_cuda_targets)
    if(NOT DEFINED SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM OR
       SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM STREQUAL "")
        return()
    endif()

    if(NOT SYNTH_CUDA)
        message(FATAL_ERROR
            "A CUDA release preset requires SYNTH_CUDA=ON")
    endif()
    if(NOT DEFINED GGML_NATIVE OR GGML_NATIVE)
        message(FATAL_ERROR
            "A CUDA release preset requires GGML_NATIVE=OFF")
    endif()

    if(SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM STREQUAL "linux-aarch64")
        set(_synth_expected_cuda_architectures "121a-real")
    elseif(SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM STREQUAL "linux-x86_64")
        set(_synth_expected_cuda_architectures
            "75-real;80-real;86-real;89-real;90-real;100-real;120a-real")
    else()
        message(FATAL_ERROR
            "unknown release CUDA platform: ${SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM}")
    endif()

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR
       NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL
           "${_synth_expected_cuda_architectures}")
        message(FATAL_ERROR
            "Release ${SYNTH_INTERNAL_RELEASE_CUDA_PLATFORM} requires the exact "
            "native cubin target set '${_synth_expected_cuda_architectures}', got "
            "'${CMAKE_CUDA_ARCHITECTURES}'")
    endif()
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    synth_validate_release_cuda_targets()
endif()
