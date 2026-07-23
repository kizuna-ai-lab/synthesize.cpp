function(synth_validate_cuda_toolkit)
    if(NOT DEFINED CUDAToolkit_VERSION OR
       NOT CUDAToolkit_VERSION MATCHES "^13\\.3($|\\.)")
        message(FATAL_ERROR
            "synthesize.cpp CUDA builds require CUDA Toolkit 13.3; got "
            "'${CUDAToolkit_VERSION}'")
    endif()
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    synth_validate_cuda_toolkit()
endif()
