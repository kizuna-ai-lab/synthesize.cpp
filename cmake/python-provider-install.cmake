# Component-filtered install rules for the platform-native Python Provider.
# scikit-build-core packages only COMPONENT wheel, keeping the ordinary C SDK
# headers, CMake package, pkg-config metadata, and import libraries out of the
# Provider wheel.

if(NOT SYNTH_BUILD_SHARED)
    message(FATAL_ERROR
        "SYNTH_BUILD_PYTHON_PROVIDER requires SYNTH_BUILD_SHARED=ON")
endif()
if(NOT SYNTH_GGML_BACKEND_DL)
    message(FATAL_ERROR
        "SYNTH_BUILD_PYTHON_PROVIDER requires SYNTH_GGML_BACKEND_DL=ON")
endif()
if(NOT SYNTH_PYTHON_PROVIDER_ID STREQUAL "default" AND
   NOT SYNTH_PYTHON_PROVIDER_ID STREQUAL "cu13")
    message(FATAL_ERROR
        "Unknown SYNTH_PYTHON_PROVIDER_ID: ${SYNTH_PYTHON_PROVIDER_ID}")
endif()
if(SYNTH_PYTHON_PROVIDER_ID STREQUAL "default" AND SYNTH_CUDA)
    message(FATAL_ERROR "The default Provider cannot contain CUDA")
endif()
if(SYNTH_PYTHON_PROVIDER_ID STREQUAL "cu13" AND NOT SYNTH_CUDA)
    message(FATAL_ERROR "The cu13 Provider requires SYNTH_CUDA=ON")
endif()

set(_synth_provider_core_targets synthesize ggml ggml-base)
set(_synth_provider_backend_targets "")
set(_synth_provider_backends "")
foreach(_synth_accelerator IN ITEMS cuda metal vulkan)
    if(TARGET ggml-${_synth_accelerator})
        list(APPEND _synth_provider_backend_targets
            ggml-${_synth_accelerator})
        list(APPEND _synth_provider_backends ${_synth_accelerator})
    endif()
endforeach()
set(_synth_provider_cpu_targets "")
foreach(_synth_backend_target IN LISTS GGML_AVAILABLE_BACKENDS)
    if(TARGET ${_synth_backend_target} AND
       _synth_backend_target MATCHES "^ggml-cpu($|-)")
        list(APPEND _synth_provider_cpu_targets ${_synth_backend_target})
    endif()
endforeach()
if(_synth_provider_cpu_targets)
    list(APPEND _synth_provider_backend_targets
        ${_synth_provider_cpu_targets})
    list(APPEND _synth_provider_backends cpu)
endif()
list(REMOVE_DUPLICATES _synth_provider_core_targets)
list(REMOVE_DUPLICATES _synth_provider_backend_targets)
list(REMOVE_DUPLICATES _synth_provider_backends)
set(_synth_provider_targets
    ${_synth_provider_core_targets}
    ${_synth_provider_backend_targets})

foreach(_synth_provider_target IN LISTS _synth_provider_targets)
    # Wheel zip files cannot represent the versioned symlink chain emitted by
    # VERSION/SOVERSION. A Provider validates its exact release/header contract
    # before dlopen, so wheel-local libraries use one undecorated physical name.
    set_property(TARGET ${_synth_provider_target} PROPERTY VERSION)
    set_property(TARGET ${_synth_provider_target} PROPERTY SOVERSION)
    if(_synth_provider_target IN_LIST _synth_provider_backend_targets)
        set(_synth_provider_destination "_native/synthesize/backends")
        if(APPLE)
            set_property(TARGET ${_synth_provider_target}
                PROPERTY INSTALL_RPATH "@loader_path/../..")
        elseif(UNIX)
            set_property(TARGET ${_synth_provider_target}
                PROPERTY INSTALL_RPATH "\$ORIGIN/../..")
        endif()
    else()
        set(_synth_provider_destination "_native")
        if(APPLE)
            set_property(TARGET ${_synth_provider_target}
                PROPERTY INSTALL_RPATH "@loader_path")
        elseif(UNIX)
            set_property(TARGET ${_synth_provider_target}
                PROPERTY INSTALL_RPATH "\$ORIGIN")
        endif()
    endif()
    install(TARGETS ${_synth_provider_target}
        LIBRARY DESTINATION ${_synth_provider_destination} COMPONENT wheel
        RUNTIME DESTINATION ${_synth_provider_destination} COMPONENT wheel
        ARCHIVE DESTINATION _native-dev COMPONENT wheel-dev
        PUBLIC_HEADER DESTINATION _native-dev COMPONENT wheel-dev)
endforeach()

file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/include/synthesize.h"
    SYNTH_PROVIDER_PUBLIC_HEADER_HASH)
list(JOIN _synth_provider_backends "\", \""
    _synth_provider_backends_joined)
set(SYNTH_PROVIDER_BACKENDS_PY "\"${_synth_provider_backends_joined}\"")
if(SYNTH_PYTHON_PROVIDER_ID STREQUAL "cu13")
    set(SYNTH_PROVIDER_DISTRIBUTION "synthesize-cpp-native-cu13")
else()
    set(SYNTH_PROVIDER_DISTRIBUTION "synthesize-cpp-native")
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/python-native/_contract.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/python-provider/_contract.py"
    @ONLY)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/python-provider/_contract.py"
    DESTINATION . COMPONENT wheel)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/python-provider/contract.json"
"{
  \"provider_id\": \"${SYNTH_PYTHON_PROVIDER_ID}\",
  \"distribution\": \"${SYNTH_PROVIDER_DISTRIBUTION}\",
  \"version\": \"${PROJECT_VERSION}\",
  \"base_release\": \"${PROJECT_VERSION}\",
  \"header_hash\": \"${SYNTH_PROVIDER_PUBLIC_HEADER_HASH}\",
  \"backends\": [\"${_synth_provider_backends_joined}\"]
}
")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/python-provider/contract.json"
    DESTINATION _native COMPONENT wheel)

message(STATUS
    "Python Provider ${SYNTH_PYTHON_PROVIDER_ID}: "
    "${SYNTH_PROVIDER_DISTRIBUTION} ${PROJECT_VERSION}; "
    "backends=${_synth_provider_backends}")
