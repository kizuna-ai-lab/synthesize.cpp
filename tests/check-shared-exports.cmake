if(NOT DEFINED SYNTH_NM OR NOT DEFINED SYNTH_LIBRARY)
    message(FATAL_ERROR "SYNTH_NM and SYNTH_LIBRARY are required")
endif()

execute_process(
    COMMAND "${SYNTH_NM}" -D --defined-only "${SYNTH_LIBRARY}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "nm failed: ${error}")
endif()

string(REPLACE "\n" ";" lines "${output}")
foreach(line IN LISTS lines)
    if(line MATCHES "[ \t]([^ \t]+)$")
        set(symbol "${CMAKE_MATCH_1}")
        string(REGEX REPLACE "@.*$" "" symbol "${symbol}")
        if(NOT symbol MATCHES "^synth_" AND
           NOT symbol MATCHES "^SYNTHESIZE_" AND
           NOT symbol STREQUAL "_init" AND
           NOT symbol STREQUAL "_fini")
            message(FATAL_ERROR "unexpected public symbol: ${symbol}")
        endif()
    endif()
endforeach()
