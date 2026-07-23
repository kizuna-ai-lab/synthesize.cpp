#include "synthesize.h"

#include <stdint.h>
#include <string.h>

#define CHECK(condition)       \
    do {                       \
        if (!(condition)) {    \
            return __LINE__;   \
        }                      \
    } while (0)

static int known_kind(const char * kind) {
    return strcmp(kind, "cpu") == 0 || strcmp(kind, "accel") == 0 ||
           strcmp(kind, "cuda") == 0 || strcmp(kind, "metal") == 0 ||
           strcmp(kind, "vulkan") == 0 || strcmp(kind, "sycl") == 0 ||
           strcmp(kind, "gpu") == 0 || strcmp(kind, "unknown") == 0;
}

int main(void) {
    CHECK(SYNTH_DEVICE_TYPE_CPU == 0);
    CHECK(SYNTH_DEVICE_TYPE_GPU == 1);
    CHECK(SYNTH_DEVICE_TYPE_IGPU == 2);
    CHECK(SYNTH_DEVICE_TYPE_ACCEL == 3);
    CHECK(SYNTH_DEVICE_MEMORY_INFO_VALID == 1);
    CHECK(SYNTH_DEVICE_MEMORY_SHARED == 2);
    CHECK(SYNTH_DEVICE_MEMORY_INFO_APPROXIMATE == 4);

    synth_backend_device_t initialized;
    memset(&initialized, 0xa5, sizeof(initialized));
    synth_backend_device_init(&initialized, sizeof(initialized));
    CHECK(initialized.struct_size == sizeof(initialized));
    CHECK(initialized.name == NULL && initialized.description == NULL);
    CHECK(initialized.kind == NULL && initialized.device_id == NULL);
    CHECK(initialized.memory_total == 0 && initialized.memory_free == 0);
    CHECK(initialized.device_type == 0 && initialized.flags == 0);
    synth_backend_device_init(NULL, sizeof(initialized));
    synth_backend_device_init(&initialized, 0);

    synth_backend_device_t undersized;
    memset(&undersized, 0xa5, sizeof(undersized));
    undersized.struct_size = sizeof(uint32_t);
    CHECK(synth_backend_device_get(0, &undersized) == SYNTH_ERR_BAD_STRUCT_SIZE);

    const uint32_t count = synth_backend_device_count();
    CHECK(count > 0);

    int found_cpu = 0;
    int found_cuda = 0;
    int found_metal = 0;
    int found_vulkan = 0;
    for (uint32_t i = 0; i < count; ++i) {
        synth_backend_device_t device;
        synth_backend_device_init(&device, sizeof(device));
        CHECK(synth_backend_device_get(i, &device) == SYNTH_OK);
        CHECK(device.name != NULL && device.name[0] != '\0');
        CHECK(device.description != NULL);
        CHECK(device.kind != NULL && known_kind(device.kind));
        CHECK(device.device_type <= SYNTH_DEVICE_TYPE_ACCEL);
        CHECK((device.flags & ~(SYNTH_DEVICE_MEMORY_INFO_VALID |
                                SYNTH_DEVICE_MEMORY_SHARED |
                                SYNTH_DEVICE_MEMORY_INFO_APPROXIMATE)) == 0);
        if ((device.flags & SYNTH_DEVICE_MEMORY_INFO_VALID) != 0) {
            CHECK(device.memory_total > 0);
        } else {
            CHECK(device.memory_total == 0 && device.memory_free == 0);
        }
        CHECK((device.flags & (SYNTH_DEVICE_MEMORY_SHARED |
                               SYNTH_DEVICE_MEMORY_INFO_APPROXIMATE)) == 0 ||
              (device.flags & SYNTH_DEVICE_MEMORY_INFO_VALID) != 0);
        found_cpu |= strcmp(device.kind, "cpu") == 0;
        found_cuda |= strcmp(device.kind, "cuda") == 0;
        found_metal |= strcmp(device.kind, "metal") == 0;
        found_vulkan |= strcmp(device.kind, "vulkan") == 0;
    }

    CHECK(found_cpu);
    CHECK(synth_backend_available(SYNTH_BACKEND_AUTO) == SYNTH_TRUE);
    CHECK(synth_backend_available(SYNTH_BACKEND_CPU) == SYNTH_TRUE);
    CHECK(synth_backend_available(SYNTH_BACKEND_CPU_ACCEL) == SYNTH_TRUE);
    CHECK((synth_backend_available(SYNTH_BACKEND_CUDA) == SYNTH_TRUE) == found_cuda);
    CHECK((synth_backend_available(SYNTH_BACKEND_METAL) == SYNTH_TRUE) == found_metal);
    CHECK((synth_backend_available(SYNTH_BACKEND_VULKAN) == SYNTH_TRUE) == found_vulkan);
    CHECK(synth_backend_available((synth_backend_request_t) 999u) == SYNTH_FALSE);

    synth_backend_device_t device;
    synth_backend_device_init(&device, sizeof(device));
    CHECK(synth_backend_device_get(count, &device) == SYNTH_ERR_INVALID_ARG);
    CHECK(synth_backend_device_get(UINT32_MAX, &device) == SYNTH_ERR_INVALID_ARG);
    CHECK(synth_backend_device_get(0, NULL) == SYNTH_ERR_INVALID_ARG);
    CHECK(synth_model_get_device(NULL, &device) == SYNTH_ERR_INVALID_ARG);
    CHECK(synth_model_get_device(NULL, NULL) == SYNTH_ERR_INVALID_ARG);
    return 0;
}
