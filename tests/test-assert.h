#pragma once

#include <iostream>

#define SYNTH_TEST_CHECK(expression)                                                               \
    do {                                                                                           \
        if (!(expression)) {                                                                       \
            std::cerr << __FILE__ << ':' << __LINE__ << ": check failed: " << #expression << '\n'; \
            return 1;                                                                              \
        }                                                                                          \
    } while (false)
