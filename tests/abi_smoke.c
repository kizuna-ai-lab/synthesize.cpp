#include "synthesize.h"

#include <stddef.h>
#include <string.h>

int main(void) {
    static const struct {
        synth_status_t status;
        const char *   text;
    } statuses[] = {
        { SYNTH_OK,                       "ok"                        },
        { SYNTH_ERR_INVALID_ARG,          "invalid argument"          },
        { SYNTH_ERR_BAD_STRUCT_SIZE,      "bad structure size"        },
        { SYNTH_ERR_FILE_NOT_FOUND,       "file not found"            },
        { SYNTH_ERR_IO,                   "I/O error"                 },
        { SYNTH_ERR_GGUF,                 "invalid GGUF"              },
        { SYNTH_ERR_UNSUPPORTED_ARCH,     "unsupported architecture"  },
        { SYNTH_ERR_UNSUPPORTED_VARIANT,  "unsupported model variant" },
        { SYNTH_ERR_UNSUPPORTED_INPUT,    "unsupported input"         },
        { SYNTH_ERR_UNSUPPORTED_LANGUAGE, "unsupported language"      },
        { SYNTH_ERR_UNSUPPORTED_VOICE,    "unsupported voice"         },
        { SYNTH_ERR_UNSUPPORTED_CONTROL,  "unsupported control"       },
        { SYNTH_ERR_MISSING_RESOURCE,     "missing resource"          },
        { SYNTH_ERR_TEXT_FRONTEND,        "text frontend error"       },
        { SYNTH_ERR_INPUT_TOO_LONG,       "input too long"            },
        { SYNTH_ERR_OUTPUT_LIMIT,         "output limit reached"      },
        { SYNTH_ERR_OOM,                  "out of memory"             },
        { SYNTH_ERR_BACKEND,              "backend error"             },
        { SYNTH_ERR_CANCELLED,            "cancelled"                 },
        { SYNTH_ERR_SINK,                 "audio sink error"          },
        { SYNTH_ERR_INTERNAL,             "internal error"            },
    };

    for (size_t i = 0; i < sizeof(statuses) / sizeof(statuses[0]); ++i) {
        if (strcmp(synth_status_string(statuses[i].status), statuses[i].text) != 0) {
            return 1;
        }
    }
    if (strcmp(synth_status_string((synth_status_t) -1), "unknown status") != 0 ||
        strcmp(synth_status_string((synth_status_t) 999), "unknown status") != 0) {
        return 2;
    }
    if (synth_get_version(NULL) != SYNTH_ERR_INVALID_ARG) {
        return 3;
    }
    synth_version_init(NULL, sizeof(synth_version_t));

    synth_version_t untouched;
    memset(&untouched, 0xa5, sizeof(untouched));
    synth_version_init(&untouched, 0);
    for (size_t i = 0; i < sizeof(untouched); ++i) {
        if (((const unsigned char *) &untouched)[i] != 0xa5) {
            return 4;
        }
    }

    synth_version_t undersized;
    memset(&undersized, 0xa5, sizeof(undersized));
    undersized.struct_size = sizeof(uint64_t) - 1;
    if (synth_get_version(&undersized) != SYNTH_ERR_BAD_STRUCT_SIZE) {
        return 5;
    }

    for (uint64_t size = 1; size < sizeof(uint64_t); ++size) {
        synth_version_t partial;
        memset(&partial, 0xa5, sizeof(partial));
        synth_version_init(&partial, size);
        const unsigned char * bytes = (const unsigned char *) &partial;
        for (size_t i = 0; i < sizeof(partial); ++i) {
            const unsigned char expected = i < size ? 0x00 : 0xa5;
            if (bytes[i] != expected) {
                return 6;
            }
        }
    }

    synth_version_t header_only;
    memset(&header_only, 0xa5, sizeof(header_only));
    synth_version_init(&header_only, sizeof(uint64_t));
    if (header_only.struct_size != sizeof(uint64_t) || synth_get_version(&header_only) != SYNTH_OK) {
        return 7;
    }
    const unsigned char * header_bytes = (const unsigned char *) &header_only;
    for (size_t i = sizeof(uint64_t); i < sizeof(header_only); ++i) {
        if (header_bytes[i] != 0xa5) {
            return 8;
        }
    }

    synth_version_t version;
    synth_version_init(&version, sizeof(version));
    if (synth_abi_version() != SYNTH_ABI_VERSION || synth_get_version(&version) != SYNTH_OK) {
        return 9;
    }
    if (version.abi_version != SYNTH_ABI_VERSION || version.version_major != SYNTH_VERSION_MAJOR ||
        version.version_minor != SYNTH_VERSION_MINOR || version.version_patch != SYNTH_VERSION_PATCH) {
        return 10;
    }

    struct guarded_version {
        synth_version_t value;
        unsigned char   guard[16];
    } guarded;

    memset(&guarded, 0xa5, sizeof(guarded));
    synth_version_init(&guarded.value, sizeof(guarded));
    for (size_t i = 0; i < sizeof(guarded.guard); ++i) {
        if (guarded.guard[i] != 0xa5) {
            return 11;
        }
    }
    if (guarded.value.struct_size != sizeof(guarded) || synth_get_version(&guarded.value) != SYNTH_OK) {
        return 12;
    }
    for (size_t i = 0; i < sizeof(guarded.guard); ++i) {
        if (guarded.guard[i] != 0xa5) {
            return 13;
        }
    }
    return 0;
}
