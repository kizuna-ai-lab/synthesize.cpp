#include <synthesize.h>

int main(void) {
    synth_version_t version;
    synth_version_init(&version, sizeof(version));
    if (synth_get_version(&version) != SYNTH_OK) {
        return 1;
    }
    if (synth_abi_version() != SYNTH_ABI_VERSION || version.abi_version != SYNTH_ABI_VERSION ||
        version.version_major != SYNTH_VERSION_MAJOR || version.version_minor != SYNTH_VERSION_MINOR ||
        version.version_patch != SYNTH_VERSION_PATCH) {
        return 2;
    }
    return 0;
}
