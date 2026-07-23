#include <stdint.h>

#if defined(_WIN32)
#    define TEST_EXPORT __declspec(dllexport)
#else
#    define TEST_EXPORT __attribute__((visibility("default")))
#endif

TEST_EXPORT uint32_t synth_abi_version(void) {
    return 1u;
}
