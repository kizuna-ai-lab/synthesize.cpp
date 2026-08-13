#include "arch/qwen3-tts/profile.h"

#include "arch/qwen3-tts/speaker-encoder-host.h"
#include "arch/qwen3-tts/weights.h"
#include "ggml.h"
#include "gguf-metadata.h"
#include "gguf.h"
#include "sha256.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace synth::qwen3tts {

synth_status_t create_x_vector_profile(const HParams &                         hparams,
                                       const SpeakerEncoderWeights &           speaker_encoder,
                                       const std::vector<float> &              pcm_24k,
                                       const std::string &                     transcript,
                                       const std::string &                     language_tag,
                                       int                                     threads,
                                       std::shared_ptr<const XVectorProfile> & output,
                                       const char *&                           out_diagnostic_code,
                                       const char *&                           out_diagnostic_message) {
    output.reset();
    out_diagnostic_code    = nullptr;
    out_diagnostic_message = nullptr;

    // D4: the clone mode is fixed when the Profile is created. Plan 2
    // implements exactly one mode, so a transcript names a mode with no
    // implementation behind it -- accepting it and silently building the
    // x-vector Profile anyway would hand back the weaker clone the caller
    // did not ask for. Checked first, before the encode chain below ever
    // runs: there is no reason to pay for a graph pass over reference audio
    // for a request this function is about to refuse anyway. A
    // whitespace-only transcript is "present" by this check and is refused
    // the same way -- Plan 3 is what gives "empty" and "whitespace" distinct
    // meanings (the spec's section 9 error table), and until it lands both
    // are simply "a transcript was supplied".
    if (!transcript.empty()) {
        out_diagnostic_code = "voice_profile.transcript_unsupported";
        out_diagnostic_message =
            "this package's Plan 2 runtime implements x-vector cloning only; a reference transcript names the "
            "transcript-assisted mode, which has no implementation yet";
        return SYNTH_ERR_INVALID_ARG;
    }

    // Task 8 finding: this family's own writer (serialize_x_vector_profile)
    // must never be able to produce an envelope its own reader
    // (load_profile_from_memory's prescan_buffer, tied to this exact bound
    // via kMaxLanguageTagLength/kPrescanMaxStringLength -- see that
    // constant's own header comment, profile.h) refuses. Checked here,
    // cheaply, before the expensive encode chain, the same reasoning the
    // transcript check above already uses.
    if (language_tag.size() > kMaxLanguageTagLength) {
        out_diagnostic_code    = "voice_profile.language_tag_too_long";
        out_diagnostic_message = "the reference language tag exceeds this package's maximum supported length";
        return SYNTH_ERR_INVALID_ARG;
    }

    // Model::prepare_x_vector's own guard, reproduced here rather than
    // reached through it: a CustomVoice package resolves no
    // SpeakerEncoderWeights at all (build_model_weights leaves
    // weights.speaker_encoder default-constructed for it), so this refuses
    // before encode_speaker_reference ever sees a weights struct with every
    // pointer null.
    if (!hparams.has_speaker_encoder) {
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }

    XVectorEncoding      encoding;
    const synth_status_t encode_status = encode_speaker_reference(hparams, speaker_encoder, pcm_24k, threads, encoding,
                                                                  out_diagnostic_code, out_diagnostic_message);
    if (encode_status != SYNTH_OK) {
        // encode_speaker_reference's own refusals -- including the
        // "voice_profile.reference_silent" digitally-silent-reference
        // rejection -- already set the diagnostic out-params; propagate
        // both unchanged rather than re-deriving them here.
        return encode_status;
    }

    auto profile          = std::make_shared<XVectorProfile>();
    profile->mode         = CloneMode::XVector;
    profile->x_vector     = std::move(encoding.x_vector);
    profile->ref_rms      = encoding.ref_rms;
    profile->language_tag = language_tag;
    output                = std::move(profile);
    return SYNTH_OK;
}

// ---------------------------------------------------------------------------
// Task 8: the v1 Serialized Voice Profile envelope. See profile.h's own
// header comment on this section for the schema/kind split and the status
// mapping load_profile_from_memory below implements. Structure duplicated
// verbatim from arch/omnivoice/profile.cpp's own envelope writer/reader --
// this project's own house rule (CLAUDE.md: "family internals are private")
// leaves the framing, the prescan, and the writer unfactored between
// families on purpose, so a second family repeats them rather than sharing
// them.
// ---------------------------------------------------------------------------

namespace {

constexpr const char * kEnvelopeArchitecture  = "synthprofile";
constexpr uint32_t     kEnvelopeFormatVersion = 1;
constexpr const char * kEnvelopeModelFamily   = "qwen3-tts";
// The SAME string the package's own ProfileContract requires
// (weights.cpp's read_profile_contract, "qwen3-tts-voice-clone") -- see
// profile.h's header comment on why this is one schema for both clone modes
// rather than two schemas.
constexpr const char * kEnvelopeSchema        = "qwen3-tts-voice-clone";
constexpr uint32_t     kEnvelopeSchemaVersion = 1;
constexpr const char * kKindXVector           = "x-vector";
constexpr const char * kKeyCompatibilityId    = "synthesize.voice_profile.compatibility_id";
constexpr const char * kKeyContentSha256      = "synthesize.voice_profile.content_sha256";
constexpr const char * kTensorXVector         = "profile.x_vector";

struct GgufContextDeleter {
    void operator()(gguf_context * context) const {
        if (context != nullptr) {
            gguf_free(context);
        }
    }
};

using OwnedGgufContext = std::unique_ptr<gguf_context, GgufContextDeleter>;

// ---------------------------------------------------------------------------
// Little-endian byte-buffer writers -- GGUF's own encoding (gguf.h's header
// comment), and this project's blanket little-endian-host assumption.
// Verbatim copies of arch/omnivoice/profile.cpp's own helpers of the same
// name; see this section's own header comment for why they are duplicated
// rather than shared.
// ---------------------------------------------------------------------------

void put_bytes(std::vector<uint8_t> & out, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    out.insert(out.end(), bytes, bytes + size);
}

template <typename T> void put(std::vector<uint8_t> & out, T value) {
    put_bytes(out, &value, sizeof(value));
}

void put_gguf_string(std::vector<uint8_t> & out, const std::string & value) {
    put<uint64_t>(out, uint64_t(value.size()));
    put_bytes(out, value.data(), value.size());
}

void pad_to_alignment(std::vector<uint8_t> & out, size_t alignment) {
    while (out.size() % alignment != 0) {
        out.push_back(0);
    }
}

// Encodes every KV pair `ctx` holds, in insertion order, exactly mirroring
// gguf.cpp's own gguf_write_out for the metadata portion of a GGUF file --
// see write_envelope below for why the header and tensor sections are
// hand-written instead of going through gguf's own (file-only) writer.
// Handles only the value shapes this schema ever produces (uint32, float32,
// and string scalars; one 32-element uint8 array shape, used for both
// 32-byte ID fields) -- returning false for anything else, since nothing
// here ever encodes untrusted input and reaching that arm is this writer's
// own defect.
//
// Records the byte offset of `kKeyContentSha256`'s 32-byte VALUE into
// `out_sha_offset` as it is written, so write_envelope can patch the real
// digest in afterward without a second pass over the buffer.
bool encode_metadata_kv(const gguf_context * ctx, std::vector<uint8_t> & out, size_t & out_sha_offset) {
    bool          found_sha = false;
    const int64_t n_kv      = gguf_get_n_kv(ctx);
    for (int64_t index = 0; index < n_kv; ++index) {
        const std::string key  = gguf_get_key(ctx, index);
        const gguf_type   type = gguf_get_kv_type(ctx, index);
        put_gguf_string(out, key);
        if (type == GGUF_TYPE_ARRAY) {
            const gguf_type element_type = gguf_get_arr_type(ctx, index);
            const size_t    count        = gguf_get_arr_n(ctx, index);
            put<int32_t>(out, int32_t(GGUF_TYPE_ARRAY));
            put<int32_t>(out, int32_t(element_type));
            put<uint64_t>(out, uint64_t(count));
            if (element_type != GGUF_TYPE_UINT8) {
                return false;
            }
            const auto * data = static_cast<const uint8_t *>(gguf_get_arr_data(ctx, index));
            if (data == nullptr && count > 0) {
                return false;
            }
            if (key == kKeyContentSha256) {
                out_sha_offset = out.size();
                found_sha      = true;
            }
            put_bytes(out, data, count);
            continue;
        }
        put<int32_t>(out, int32_t(type));
        switch (type) {
            case GGUF_TYPE_UINT32:
                put<uint32_t>(out, gguf_get_val_u32(ctx, index));
                break;
            case GGUF_TYPE_FLOAT32:
                put<float>(out, gguf_get_val_f32(ctx, index));
                break;
            case GGUF_TYPE_STRING:
                put_gguf_string(out, gguf_get_val_str(ctx, index));
                break;
            default:
                return false;
        }
    }
    return found_sha;
}

// Appends the common v1 envelope header metadata every kind shares.
// `compatibility_id` is copied verbatim. `content_sha256` is seeded at
// zero -- docs/c-interface.md's exact rule: write_envelope hashes the
// assembled buffer with this placeholder still in place, then patches the
// real digest into the same 32 bytes afterward.
void set_common_metadata(gguf_context * ctx, const char * kind, const uint8_t (&compatibility_id)[32]) {
    gguf_set_val_str(ctx, "general.architecture", kEnvelopeArchitecture);
    gguf_set_val_u32(ctx, "synthesize.voice_profile.format_version", kEnvelopeFormatVersion);
    gguf_set_val_str(ctx, "synthesize.voice_profile.model_family", kEnvelopeModelFamily);
    gguf_set_val_str(ctx, "synthesize.voice_profile.schema", kEnvelopeSchema);
    gguf_set_val_u32(ctx, "synthesize.voice_profile.schema_version", kEnvelopeSchemaVersion);
    gguf_set_arr_data(ctx, kKeyCompatibilityId, GGUF_TYPE_UINT8, compatibility_id, 32);
    static constexpr uint8_t kZeroDigest[32] = {};
    gguf_set_arr_data(ctx, kKeyContentSha256, GGUF_TYPE_UINT8, kZeroDigest, 32);
    gguf_set_val_str(ctx, "synthesize.voice_profile.kind", kind);
}

// Hand-writes the header, tensor-info section (always exactly one entry --
// Plan 2 has one kind, and it always carries an x-vector tensor), alignment
// padding, and tensor data that gguf.h's own writer API cannot produce into
// a memory buffer: see arch/omnivoice/profile.cpp's own write_envelope for
// why (the function that can, gguf_write_to_buf, lives in ggml-impl.h, out
// of reach across the submodule boundary). The metadata KV section above
// genuinely goes through gguf's own setter/getter API; only the
// header/tensor-info/tensor-data framing below is this project's own.
//
// `ctx` must already hold `content_sha256` seeded at 32 zero bytes
// (set_common_metadata's job).
synth_status_t write_envelope(const gguf_context *       ctx,
                              const std::vector<float> & x_vector,
                              std::vector<uint8_t> &     out_bytes) {
    std::vector<uint8_t> bytes;

    put_bytes(bytes, GGUF_MAGIC, 4);
    put<uint32_t>(bytes, uint32_t(GGUF_VERSION));
    put<int64_t>(bytes, int64_t(1));  // n_tensors: always exactly one, the x-vector
    put<int64_t>(bytes, gguf_get_n_kv(ctx));

    size_t sha_offset = 0;
    if (!encode_metadata_kv(ctx, bytes, sha_offset)) {
        return SYNTH_ERR_INTERNAL;
    }

    put_gguf_string(bytes, kTensorXVector);
    put<uint32_t>(bytes, uint32_t(1));              // n_dims
    put<int64_t>(bytes, int64_t(x_vector.size()));  // ne[0] = enc_dim
    put<int32_t>(bytes, int32_t(GGML_TYPE_F32));
    put<uint64_t>(bytes, uint64_t(0));              // the only tensor starts at offset 0
    pad_to_alignment(bytes, GGUF_DEFAULT_ALIGNMENT);

    put_bytes(bytes, x_vector.data(), x_vector.size() * sizeof(float));
    pad_to_alignment(bytes, GGUF_DEFAULT_ALIGNMENT);

    // The digest step docs/c-interface.md prescribes: hash the buffer with
    // content_sha256 already zero (set_common_metadata seeded it that way,
    // untouched above), then overwrite just those 32 bytes with the result.
    uint8_t digest[32];
    synth::sha256(bytes.data(), bytes.size(), digest);
    std::memcpy(bytes.data() + sha_offset, digest, sizeof(digest));

    out_bytes = std::move(bytes);
    return SYNTH_OK;
}

// Reads a required, exactly-32-element uint8 array metadata value. `false`
// covers "missing", "wrong GGUF type", and "wrong element count" alike --
// the caller maps all three to the same malformed outcome.
bool read_u8_32_array(const gguf_context * ctx, const char * key, uint8_t (&out)[32]) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, id) != GGUF_TYPE_UINT8 ||
        gguf_get_arr_n(ctx, id) != 32) {
        return false;
    }
    const void * data = gguf_get_arr_data(ctx, id);
    if (data == nullptr) {
        return false;
    }
    std::memcpy(out, data, 32);
    return true;
}

// Locates the byte OFFSET of `key`'s 32-element uint8 array VALUE within a
// raw, untrusted GGUF byte buffer, by searching for that KV entry's own
// on-disk encoding PREFIX. Independent of the parsed gguf_context's own
// bookkeeping: gguf_get_arr_data only ever hands back a pointer into the
// PARSED context's own internal copy, never an offset into the caller's
// original bytes, and the digest check below needs the latter to hash "the
// complete input with that one value treated as zero" rather than a
// reconstruction built from parsed fields. See
// arch/omnivoice/profile.cpp's own find_u8_32_value_offset for the full
// rationale, including why `search_size` bounds the search to the METADATA
// region rather than the whole buffer.
bool find_u8_32_value_offset(const uint8_t * data, size_t search_size, const std::string & key, size_t & out_offset) {
    std::vector<uint8_t> needle;
    put<uint64_t>(needle, uint64_t(key.size()));
    put_bytes(needle, key.data(), key.size());
    put<int32_t>(needle, int32_t(GGUF_TYPE_ARRAY));
    put<int32_t>(needle, int32_t(GGUF_TYPE_UINT8));
    put<uint64_t>(needle, uint64_t(32));

    const uint8_t * begin = data;
    const uint8_t * end   = data + search_size;
    const uint8_t * found = std::search(begin, end, needle.begin(), needle.end());
    if (found == end) {
        return false;
    }
    const size_t value_offset = size_t(found - begin) + needle.size();
    if (value_offset + 32 > search_size) {
        return false;
    }
    out_offset = value_offset;
    return true;
}

// ---------------------------------------------------------------------------
// Untrusted-buffer pre-scan: a defect in ggml's own parser that this loader
// cannot fix downstream of calling it, only guard against beforehand. See
// arch/omnivoice/profile.cpp's own prescan_buffer header comment for the
// full history (two separate ggml eager-read asserts that abort the whole
// process, found by fuzzing) and why the fix is POSITIVE validation -- this
// loader accepts ONLY what its own writer (write_envelope,
// set_common_metadata, serialize_x_vector_profile, all in this file) ever
// emits, before gguf_init_from_buffer ever runs, rather than trying to
// enumerate ggml's internal invariants one crash at a time.
//
// PrescanKeySpec, kPrescanKnownKeys, kPrescanKnownKeyCount, and
// kPrescanKvCountXVector -- the exact, closed set of metadata keys this
// family's writer ever emits, and the exact per-kind KV count -- live in
// profile.h, not here: see that header's own comment on why
// (tests/qwen3_tts_profile_test.cpp needs the SAME table this file's own
// prescan_buffer validates against, not a hand-transcription of it).
// ---------------------------------------------------------------------------

constexpr uint64_t kPrescanMaxKeyLength    = 256;
// Tied to kMaxLanguageTagLength (profile.h) rather than a second,
// independent literal: create_x_vector_profile's own creation-time cap and
// serialize_x_vector_profile's own defensive re-check must never drift from
// this load-time ceiling, or a language_tag creation accepts could stop
// round-tripping through this exact loader -- precisely the defect a
// reviewer measured here before this tie existed (a 1 MiB + 8 byte
// language_tag serialized fine and then failed to load with a bare
// INVALID_ARG: our own writer's output, rejected by our own reader). This
// family's own Serialized Profile carries no free-form transcript in Plan 2
// (a real BCP-47 language_tag never approaches this bound), so the ceiling
// exists purely so a hostile string length claim cannot itself be used to
// justify an oversized skip -- the same defensive role
// omnivoice::kPrescanMaxStringLength plays for its own transcript field, at
// the same magnitude (both are `1u << 20`, not "much larger" as an earlier
// draft of this comment claimed).
constexpr uint64_t kPrescanMaxStringLength = kMaxLanguageTagLength;

bool prescan_has_remaining(size_t offset, size_t size, size_t need) {
    return offset <= size && need <= size - offset;
}

bool prescan_skip(size_t & offset, size_t size, size_t amount) {
    if (!prescan_has_remaining(offset, size, amount)) {
        return false;
    }
    offset += amount;
    return true;
}

bool prescan_read_bytes(const uint8_t * data, size_t size, size_t & offset, void * out, size_t amount) {
    if (!prescan_has_remaining(offset, size, amount)) {
        return false;
    }
    std::memcpy(out, data + offset, amount);
    offset += amount;
    return true;
}

template <typename T> bool prescan_read(const uint8_t * data, size_t size, size_t & offset, T & out) {
    return prescan_read_bytes(data, size, offset, &out, sizeof(out));
}

// The fixed byte width of every scalar GGUF type this walk can skip without
// decoding it (everything except STRING, which is itself length-prefixed
// and handled separately in prescan_skip_value).
bool prescan_fixed_type_size(gguf_type type, size_t & out_size) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            out_size = 1;
            return true;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            out_size = 2;
            return true;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            out_size = 4;
            return true;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            out_size = 8;
            return true;
        default:
            return false;
    }
}

// Skips over one value's bytes (a single scalar, or `count` elements of an
// array) without decoding it. `count` is 1 for a scalar, matching
// gguf_init_from_reader's own convention.
bool prescan_skip_value(const uint8_t * data, size_t size, size_t & offset, gguf_type type, uint64_t count) {
    if (type == GGUF_TYPE_STRING) {
        for (uint64_t index = 0; index < count; ++index) {
            uint64_t length = 0;
            if (!prescan_read(data, size, offset, length) || length > kPrescanMaxStringLength) {
                return false;
            }
            if (!prescan_skip(offset, size, size_t(length))) {
                return false;
            }
        }
        return true;
    }
    size_t element_size = 0;
    if (!prescan_fixed_type_size(type, element_size)) {
        return false;  // GGUF_TYPE_ARRAY-of-ARRAY or any other unrecognized type tag
    }
    if (count > SIZE_MAX / element_size) {
        return false;  // overflow guard on count * element_size
    }
    return prescan_skip(offset, size, size_t(count) * element_size);
}

// The full walk: positive validation against the exact format this family's
// own writer produces. Returns false for anything outside that format --
// whether or not gguf_init_from_buffer would also refuse it -- and the
// caller maps `false` to SYNTH_ERR_INVALID_ARG without ever calling
// gguf_init_from_buffer on these bytes.
//
// Scoping note (reviewer finding, minor): the per-entry lookup below matches
// each buffer key against kPrescanKnownKeys by NAME, not by POSITION -- "our
// own format" is currently enforced as the key SET (every key present is
// whitelisted, no key repeats, and the count matches exactly) plus each
// key's own declared type/shape, but NOT the writer's own emission order. A
// hand-built buffer with `set_common_metadata`'s 8 keys followed by
// `ref_rms`/`language_tag` in fully REVERSED order, with a digest
// recomputed over that reversed layout, loads successfully. This has no
// security impact -- content_sha256 is a corruption check, never a
// signature (sha256.h's own header comment) -- and is inherited verbatim
// from omnivoice::prescan_buffer, which has the identical property for its
// own two kinds. Left unpinned rather than fixed: pinning order here would
// mean checking `key == kPrescanKnownKeys[index].key` positionally for
// indices 0-7 (fine, `kind`'s own value is not yet known at that point) but
// would need to BRANCH on the `kind` value read at index 7 to know which
// kind-specific sequence to expect from index 8 onward once a second kind
// (Plan 3's "icl") exists -- a real design decision for whoever adds that
// second kind, not a one-line fix to make now for a kind that does not
// exist yet.
bool prescan_buffer(const uint8_t * data, size_t size) {
    size_t offset = 0;

    char magic[4];
    if (!prescan_read_bytes(data, size, offset, magic, sizeof(magic)) ||
        std::memcmp(magic, GGUF_MAGIC, sizeof(magic)) != 0) {
        return false;
    }

    uint32_t version = 0;
    if (!prescan_read(data, size, offset, version) || version != GGUF_VERSION) {
        // This project only ever WRITES GGUF_VERSION (3); a different
        // version is either an old writer this loader never claimed to
        // support or corrupt data -- both malformed for a v1 Serialized
        // Profile's own contract.
        return false;
    }

    int64_t n_tensors = 0;
    int64_t n_kv      = 0;
    // Exactly 1 tensor -- the x-vector. Plan 2 has one kind and it always
    // carries this one tensor; a Plan 3 "icl" kind that needs a different
    // tensor count widens this check the way
    // omnivoice::prescan_buffer's own `(n_tensors != 0 && n_tensors != 1)`
    // already does for its two kinds.
    if (!prescan_read(data, size, offset, n_tensors) || n_tensors != 1) {
        return false;
    }
    // Exactly kPrescanKvCountXVector metadata entries -- see that
    // constant's own comment (profile.h).
    if (!prescan_read(data, size, offset, n_kv) || n_kv != kPrescanKvCountXVector) {
        return false;
    }

    bool seen[kPrescanKnownKeyCount] = {};

    for (int64_t index = 0; index < n_kv; ++index) {
        uint64_t key_length = 0;
        if (!prescan_read(data, size, offset, key_length) || key_length > kPrescanMaxKeyLength) {
            return false;
        }
        if (!prescan_has_remaining(offset, size, size_t(key_length))) {
            return false;
        }
        const std::string key(reinterpret_cast<const char *>(data + offset), size_t(key_length));
        offset += size_t(key_length);

        int32_t type_raw = 0;
        if (!prescan_read(data, size, offset, type_raw)) {
            return false;
        }
        gguf_type type     = gguf_type(type_raw);
        bool      is_array = false;
        uint64_t  count    = 1;
        if (type == GGUF_TYPE_ARRAY) {
            is_array                 = true;
            int32_t element_type_raw = 0;
            if (!prescan_read(data, size, offset, element_type_raw)) {
                return false;
            }
            type = gguf_type(element_type_raw);
            if (!prescan_read(data, size, offset, count)) {
                return false;
            }
        }

        // Positive validation (this section's own header comment): `key`
        // must be one of the exact keys this writer ever emits, it must not
        // repeat, and its declared shape must match that key's own exact
        // type/array-ness/count.
        size_t spec_index = kPrescanKnownKeyCount;
        for (size_t candidate = 0; candidate < kPrescanKnownKeyCount; ++candidate) {
            if (key == kPrescanKnownKeys[candidate].key) {
                spec_index = candidate;
                break;
            }
        }
        if (spec_index == kPrescanKnownKeyCount) {
            return false;  // unknown key -- includes "general.alignment", which this
                           // writer never emits at all.
        }
        if (seen[spec_index]) {
            return false;  // duplicate key
        }
        seen[spec_index] = true;

        const PrescanKeySpec & spec = kPrescanKnownKeys[spec_index];
        if (type != spec.type || is_array != spec.is_array || (is_array && count != spec.count)) {
            return false;
        }

        if (!prescan_skip_value(data, size, offset, type, count)) {
            return false;
        }
    }

    // The tensor section: exactly one entry named "profile.x_vector" with
    // this writer's own exact shape (write_envelope: n_dims=1, type
    // GGML_TYPE_F32, offset 0, since it is the only tensor). ne[0] (the
    // x-vector's own width) is the one field that legitimately varies with
    // the package's own declared enc_dim, so it is only checked for
    // positivity here; the exact-width check against enc_dim still happens
    // later, against the real parsed tensor size, before any allocation
    // sized from it.
    uint64_t name_length = 0;
    if (!prescan_read(data, size, offset, name_length) || name_length > kPrescanMaxKeyLength) {
        return false;
    }
    if (!prescan_has_remaining(offset, size, size_t(name_length))) {
        return false;
    }
    const std::string tensor_name(reinterpret_cast<const char *>(data + offset), size_t(name_length));
    offset += size_t(name_length);
    if (tensor_name != kTensorXVector) {
        return false;
    }

    uint32_t n_dims = 0;
    if (!prescan_read(data, size, offset, n_dims) || n_dims != 1) {
        return false;
    }

    int64_t ne0 = 0;
    if (!prescan_read(data, size, offset, ne0) || ne0 <= 0) {
        return false;
    }

    int32_t tensor_type_raw = 0;
    if (!prescan_read(data, size, offset, tensor_type_raw) || tensor_type_raw != int32_t(GGML_TYPE_F32)) {
        return false;
    }

    uint64_t tensor_offset = 0;
    if (!prescan_read(data, size, offset, tensor_offset) || tensor_offset != 0) {
        return false;
    }

    return true;
}

}  // namespace

synth_status_t serialize_x_vector_profile(const XVectorProfile & profile,
                                          const uint8_t (&compatibility_id)[32],
                                          std::vector<uint8_t> & out_bytes) {
    if (profile.x_vector.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // Defensive, independent of whatever create_x_vector_profile already
    // checked (a reviewer finding on this task): this function has no way
    // to know whether `profile` reached it through that path or was built
    // by hand, so it re-asserts the writer's own invariant itself rather
    // than trusting a caller-specific history -- see kMaxLanguageTagLength's
    // own header comment (profile.h) for the exact defect this closes.
    if (profile.language_tag.size() > kMaxLanguageTagLength) {
        return SYNTH_ERR_INVALID_ARG;
    }
    OwnedGgufContext ctx(gguf_init_empty());
    if (ctx == nullptr) {
        return SYNTH_ERR_OOM;
    }
    set_common_metadata(ctx.get(), kKindXVector, compatibility_id);
    gguf_set_val_f32(ctx.get(), "synthesize.voice_profile.ref_rms", profile.ref_rms);
    // Known gap (reviewer finding, minor, left unfixed): gguf_set_val_str
    // takes a null-terminated `const char *` -- ggml/include/gguf.h has no
    // length-aware string setter/getter pair at all -- so an EMBEDDED NUL in
    // `language_tag` silently truncates here (a 5-byte "en\0US" round-trips
    // as a 2-byte "en", not as itself). Fixing this for real would mean
    // bypassing gguf's own KV setter/getter API for this one field
    // specifically, which is exactly the "genuinely goes through gguf's own
    // setter/getter API" property write_envelope's own header comment relies
    // on to keep the metadata section a thin wrapper rather than a second
    // hand-rolled encoder alongside the header/tensor-info framing that
    // already IS hand-rolled. A BCP-47 tag (this family's own declared shape
    // for this field, Task 9's job to enforce) is ASCII letters, digits, and
    // hyphens only, so a real caller's input can never contain a NUL to
    // begin with; only a caller that already bypassed that shape validation
    // could reach this gap.
    gguf_set_val_str(ctx.get(), "synthesize.voice_profile.language_tag", profile.language_tag.c_str());
    return write_envelope(ctx.get(), profile.x_vector, out_bytes);
}

synth_status_t load_profile_from_memory(uint32_t        enc_dim,
                                        const uint8_t * data,
                                        size_t          data_size,
                                        const uint8_t (&compatibility_id)[32],
                                        synth::ProfileFamilyTag &     out_family_tag,
                                        std::shared_ptr<const void> & out_payload,
                                        const char *&                 out_diagnostic_code,
                                        const char *&                 out_diagnostic_message) {
    out_payload.reset();
    out_family_tag         = synth::ProfileFamilyTag::None;
    out_diagnostic_code    = nullptr;
    out_diagnostic_message = nullptr;

    if (data == nullptr || data_size == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Independent raw-byte hardening BEFORE gguf_init_from_buffer ever
    // touches these bytes -- see prescan_buffer's own header comment for
    // why: ggml's parser aborts the process on a reserved key (at minimum
    // "general.alignment") declared with a type its own eager read does not
    // accept, and that call happens inside gguf_init_from_buffer itself,
    // before any line below this one runs.
    if (!prescan_buffer(data, data_size)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Structural parse only: `ctx == nullptr` means gguf_init_from_buffer
    // never reads (or allocates for) the tensor DATA blob, no matter how
    // large the file's own tensor-info section claims it is -- only the
    // bounded tensor-info section (name/shape/type/offset) is parsed here.
    // The declared element count is checked against `enc_dim` below, using
    // only that bounded info, before this function ever allocates anything
    // sized from it.
    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = nullptr;
    OwnedGgufContext ctx(gguf_init_from_buffer(data, data_size, init_params));
    if (ctx == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    gguf_context * g = ctx.get();

    if (gguf_get_alignment(g) != GGUF_DEFAULT_ALIGNMENT) {
        return SYNTH_ERR_INVALID_ARG;
    }

    GgufMetadata meta(g, "qwen3-tts");
    if (!meta.require_string("general.architecture", kEnvelopeArchitecture)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    uint32_t format_version = 0;
    if (!meta.u32("synthesize.voice_profile.format_version", format_version) ||
        format_version != kEnvelopeFormatVersion) {
        return SYNTH_ERR_INVALID_ARG;
    }
    std::string model_family;
    if (!meta.string("synthesize.voice_profile.model_family", model_family)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (model_family != kEnvelopeModelFamily) {
        // A structurally sound envelope for a DIFFERENT family, not corrupt
        // data -- this loader understands the shape, just not for itself.
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }
    std::string schema;
    uint32_t    schema_version = 0;
    if (!meta.string("synthesize.voice_profile.schema", schema) ||
        !meta.u32("synthesize.voice_profile.schema_version", schema_version)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (schema != kEnvelopeSchema || schema_version != kEnvelopeSchemaVersion) {
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }
    std::string kind;
    if (!meta.string("synthesize.voice_profile.kind", kind)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (kind != kKindXVector) {
        // Plan 2 recognizes exactly one kind. A value outside it means the
        // payload structure the rest of the bytes describe cannot be
        // interpreted by this build at all, which this project treats as
        // malformed rather than merely unsupported (profile.h's own header
        // comment on this function, including the forward-compatibility
        // cost this trade carries). This branch fires for a HAND-BUILT
        // buffer that keeps Plan 2's own key set/count and swaps out just
        // this string -- a REAL Plan 3 "icl" envelope, carrying its own
        // additional keys, never reaches this line: the prescan whitelist/
        // `n_kv` gate above already rejected it first.
        return SYNTH_ERR_INVALID_ARG;
    }
    uint8_t file_compatibility_id[32];
    if (!read_u8_32_array(g, kKeyCompatibilityId, file_compatibility_id)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (std::memcmp(file_compatibility_id, compatibility_id, sizeof(file_compatibility_id)) != 0) {
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }
    uint8_t stored_digest[32];
    if (!read_u8_32_array(g, kKeyContentSha256, stored_digest)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    if (gguf_get_n_tensors(g) != 1) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Size arithmetic BEFORE allocation: the declared element count is
    // compared against `enc_dim` using only cheap tensor-info getters (no
    // tensor payload byte has been read yet) before anything sized from it
    // is ever allocated below -- neither the digest scratch copy (sized
    // from `data_size`, which the caller already materialized, not from
    // anything this file's own bytes claim) nor, later, the x-vector itself.
    const int64_t tensor_id = gguf_find_tensor(g, kTensorXVector);
    if (tensor_id < 0 || gguf_get_tensor_type(g, tensor_id) != GGML_TYPE_F32) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const size_t tensor_bytes = gguf_get_tensor_size(g, tensor_id);
    if (tensor_bytes == 0 || tensor_bytes % sizeof(float) != 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const uint64_t element_count = tensor_bytes / sizeof(float);
    if (element_count != enc_dim) {
        // Not a ceiling, an exact match: a Profile of the wrong width builds
        // a wrong-shaped prompt (profile.h's own header comment).
        return SYNTH_ERR_INVALID_ARG;
    }

    // Digest verification (docs/c-interface.md's exact rule): hash the
    // complete input with content_sha256's own 32 bytes treated as zero, and
    // compare against the value stored there. The search is bounded to the
    // METADATA region, exactly as arch/omnivoice/profile.cpp's own loader
    // bounds it (PR #6 triage FIX 6): `content_sha256` is always a metadata
    // KV, so its on-disk encoding can only ever legitimately start before
    // `gguf_get_data_offset`, never at or past it.
    const size_t metadata_size = std::min(data_size, gguf_get_data_offset(g));
    size_t       sha_offset    = 0;
    if (!find_u8_32_value_offset(data, metadata_size, kKeyContentSha256, sha_offset)) {
        // The parse above already confirmed this key exists as a 32-element
        // uint8 array; failing to relocate it by its own on-disk encoding is
        // this loader's own defect, not a caller mistake.
        return SYNTH_ERR_INTERNAL;
    }
    std::vector<uint8_t> scratch(data, data + data_size);
    std::memset(scratch.data() + sha_offset, 0, 32);
    uint8_t computed_digest[32];
    synth::sha256(scratch.data(), scratch.size(), computed_digest);
    if (std::memcmp(computed_digest, stored_digest, sizeof(computed_digest)) != 0) {
        return SYNTH_ERR_INVALID_ARG;
    }

    float       ref_rms = 0.0f;
    std::string language_tag;
    if (!meta.f32("synthesize.voice_profile.ref_rms", ref_rms) ||
        !meta.string("synthesize.voice_profile.language_tag", language_tag)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // Payload-value parity (reviewer finding): load_profile_from_memory
    // validated structure exhaustively but, until now, values not at all --
    // a hand-edited envelope could carry a ref_rms state
    // encode_speaker_reference's own "voice_profile.reference_silent"
    // rejection (speaker-encoder-host.cpp) refuses by name at creation
    // (EXACTLY ref_rms == 0.0f), or one reference_rms() -- a sqrt of a sum
    // of squares over real PCM -- could never itself produce (negative,
    // infinite, or NaN), straight into Task 11's consumer. One check rejects
    // all four measured states: `!isfinite` catches NaN and +/-inf,
    // `!(ref_rms > 0.0f)` catches zero and negative (and NaN again, since
    // any comparison with NaN is false) -- bare INVALID_ARG, no diagnostic
    // code, the same style omnivoice::load_profile_from_memory's own
    // ref_rms>0 parity check uses.
    if (!std::isfinite(ref_rms) || !(ref_rms > 0.0f)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Truncation guard: the DECLARED tensor byte range must actually fit
    // inside the SUPPLIED buffer. gguf_init_from_buffer was called with
    // `ctx == nullptr` specifically so it never performed this read (or the
    // allocation it implies) itself -- see this function's own comment
    // above the size-arithmetic block.
    const size_t data_offset   = gguf_get_data_offset(g);
    const size_t tensor_offset = gguf_get_tensor_offset(g, tensor_id);
    if (data_offset > data_size) {
        return SYNTH_ERR_INVALID_ARG;
    }
    size_t remaining = data_size - data_offset;
    if (tensor_offset > remaining) {
        return SYNTH_ERR_INVALID_ARG;
    }
    remaining -= tensor_offset;
    if (tensor_bytes > remaining) {
        return SYNTH_ERR_INVALID_ARG;
    }

    std::vector<float> x_vector(size_t{ element_count });
    std::memcpy(x_vector.data(), data + data_offset + tensor_offset, tensor_bytes);

    // Payload-value parity, extended to the x-vector itself:
    // encode_speaker_reference's own graph-level check (speaker-encoder-host.cpp)
    // already refuses a non-finite element with SYNTH_ERR_INTERNAL before a
    // created XVectorProfile can ever hold one -- a serialized envelope
    // claiming one is the same malformed state, mapped to INVALID_ARG
    // instead (the untrusted-bytes counterpart of that internal check, the
    // same status split this function already draws elsewhere between "our
    // own defect" and "the bytes are bad"). An identically-all-zero
    // x-vector is refused alongside it: this network's own bias terms make
    // an exactly-all-zero output indistinguishable from a corrupted or
    // hand-forged payload rather than a real embedding of any reference
    // audio, silent or not -- a reviewer measured both states loading with
    // SYNTH_OK before this check existed.
    bool any_nonzero = false;
    for (float value : x_vector) {
        if (!std::isfinite(value)) {
            return SYNTH_ERR_INVALID_ARG;
        }
        any_nonzero = any_nonzero || (value != 0.0f);
    }
    if (!any_nonzero) {
        return SYNTH_ERR_INVALID_ARG;
    }

    auto profile          = std::make_shared<XVectorProfile>();
    profile->mode         = CloneMode::XVector;
    profile->x_vector     = std::move(x_vector);
    profile->ref_rms      = ref_rms;
    profile->language_tag = std::move(language_tag);

    out_payload    = std::move(profile);
    out_family_tag = synth::ProfileFamilyTag::Qwen3TtsClone;
    return SYNTH_OK;
}

}  // namespace synth::qwen3tts
