#include "arch/qwen3-tts/profile.h"

#include "arch/qwen3-tts/bpe.h"
#include "arch/qwen3-tts/codec-encoder-host.h"
#include "arch/qwen3-tts/speaker-encoder-host.h"
#include "arch/qwen3-tts/weights.h"
#include "ggml.h"
#include "gguf-metadata.h"
#include "gguf.h"
#include "sha256.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
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

    // D4: the clone mode is fixed when the Profile is created. This function
    // prepares the x-vector mode, so a transcript names the OTHER mode --
    // accepting it and silently building the x-vector Profile anyway would
    // hand back the weaker clone the caller did not ask for. Checked first,
    // before the encode chain below ever runs: there is no reason to pay for
    // a graph pass over reference audio for a request this function is about
    // to refuse anyway. A whitespace-only transcript is "present" by this
    // check and is refused the same way; create_icl_profile is where "empty"
    // and "whitespace-only" acquire their own named refusal (the design's
    // section 9 error table), because that is the function for which they are
    // a MALFORMED input rather than simply the wrong mode.
    if (!transcript.empty()) {
        out_diagnostic_code = "voice_profile.transcript_unsupported";
        out_diagnostic_message =
            "this function prepares x-vector Voice Profiles only; a reference transcript names the "
            "transcript-assisted mode, which create_icl_profile prepares";
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

synth_status_t create_icl_profile(const HParams &                     hparams,
                                  const SpeakerEncoderWeights &       speaker_encoder,
                                  const CodecEncoderWeights &         codec_encoder,
                                  const std::vector<float> &          pcm_24k,
                                  const std::string &                 transcript,
                                  const std::vector<int32_t> &        reference_text_ids,
                                  const std::string &                 language_tag,
                                  int                                 threads,
                                  std::shared_ptr<const IclProfile> & output,
                                  const char *&                       out_diagnostic_code,
                                  const char *&                       out_diagnostic_message) {
    output.reset();
    out_diagnostic_code    = nullptr;
    out_diagnostic_message = nullptr;

    // D4, from this side: the transcript is what MAKES a Profile an ICL
    // Profile, so a blank one names the x-vector mode instead and is refused
    // rather than silently downgraded to it -- the exact mirror of
    // create_x_vector_profile's own refusal above. "Blank" is bpe.h's own
    // qwen_transcript_is_blank, the same predicate qwen_reference_transcript_ids
    // applies before tokenizing, so the two can never disagree; it covers the
    // empty string and the whitespace-only one alike, which is one row and one
    // status in the design's section 9 error table.
    if (qwen_transcript_is_blank(transcript)) {
        out_diagnostic_code = "voice_profile.transcript_blank";
        out_diagnostic_message =
            "a transcript-assisted Voice Profile needs a reference transcript that names speech; "
            "this one is empty or whitespace-only";
        return SYNTH_ERR_INVALID_ARG;
    }

    // All-or-nothing, not trusted from the caller. The ids are produced one
    // layer up (Model::tokenize_reference_transcript at the dispatch site, the
    // only place with the BPE tables), so this function cannot re-derive them
    // -- but it can refuse the half-present state, and it must: an ICL prompt
    // assembled from reference codes with no reference text is a
    // plausible-sounding wrong prompt, not a failure anything downstream would
    // report. Task 11 enforces the same rule again at the synthesis seam.
    if (reference_text_ids.empty()) {
        out_diagnostic_code    = "voice_profile.reference_text_ids_missing";
        out_diagnostic_message = "the reference transcript produced no token ids for the transcript-assisted prompt";
        return SYNTH_ERR_INVALID_ARG;
    }

    // And every id has to be one this package's talker can actually look up.
    // Task 9 review, closing the last asymmetry in the "validate positively
    // against what our own writer emits" rule: serialize_icl_profile and
    // load_profile_from_memory both bound this stream to
    // `[0, talker.text_vocab_size)`, and without this check the CREATOR could
    // hand back an in-process Profile our own writer would then refuse -- the
    // mirror image of the `language_tag` defect a reviewer measured on Task 8,
    // and with nothing between such a Profile and Task 11's prompt builder,
    // where an out-of-range id is a `ggml_get_rows` abort rather than a status.
    // Checked here, before the encode chains, for the same reason the
    // transcript and language_tag checks above are.
    // The `<= 0` half is defence in depth and cannot fire for any package this
    // port loads: weights.cpp:347-352 refuses a package whose
    // `talker.text_vocab_size` is zero (`bound.limit == 0`), and read_tokens
    // sits unconditionally in read_hparams' own `&&` chain. It is kept because
    // this function takes `HParams` by reference rather than a loaded Model,
    // so a caller can hand it a synthetic struct -- and without it the
    // comparison below would admit every id against a zero bound. Labelled
    // rather than dropped, the same way the `groups`/`frames == 0` clauses in
    // load_profile_from_memory are.
    const int64_t text_vocab_size = int64_t(hparams.talker.text_vocab_size);
    for (int32_t id : reference_text_ids) {
        if (text_vocab_size <= 0 || id < 0 || int64_t(id) >= text_vocab_size) {
            out_diagnostic_code    = "voice_profile.reference_text_ids_out_of_range";
            out_diagnostic_message = "a reference transcript token id falls outside this package's own text vocabulary";
            return SYNTH_ERR_INVALID_ARG;
        }
    }

    // The tie kMaxLanguageTagLength's own header comment (profile.h) records
    // applies to both kinds or it applies to neither: this family's writer
    // must never be able to emit an envelope its own reader refuses,
    // whichever kind it wrote. Checked here, cheaply, before either encode
    // chain, exactly as create_x_vector_profile checks it.
    if (language_tag.size() > kMaxLanguageTagLength) {
        out_diagnostic_code    = "voice_profile.language_tag_too_long";
        out_diagnostic_message = "the reference language tag exceeds this package's maximum supported length";
        return SYNTH_ERR_INVALID_ARG;
    }

    // One flag, both encoders. Model::prepare_x_vector and
    // Model::prepare_codec_reference (model.cpp) each guard on this same
    // `has_speaker_encoder`, because the catalog resolves `speaker_encoder.*`
    // and `codec.encoder.*` under it alike -- a CustomVoice package leaves
    // BOTH weights structs default-constructed, every pointer null.
    if (!hparams.has_speaker_encoder) {
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }

    // The speaker half first, so an ICL Profile and an x-vector Profile
    // prepared from the same clip carry the same embedding and refuse
    // identically -- and so the cheaper of the two graph passes is the one
    // that reports a silent reference.
    XVectorEncoding      speaker_encoding;
    const synth_status_t speaker_status = encode_speaker_reference(
        hparams, speaker_encoder, pcm_24k, threads, speaker_encoding, out_diagnostic_code, out_diagnostic_message);
    if (speaker_status != SYNTH_OK) {
        return speaker_status;
    }

    CodecEncoding        codec_encoding;
    const synth_status_t codec_status = encode_codec_reference(hparams, codec_encoder, pcm_24k, threads, codec_encoding,
                                                               out_diagnostic_code, out_diagnostic_message);
    if (codec_status != SYNTH_OK) {
        // Which of its refusals carry a diagnostic, stated exactly, because
        // the sentence that stood here until 2026-08-15 did not. It claimed
        // that a sub-frame clip, a non-finite sample and the silent-reference
        // rejection "already set the diagnostic out-params"; only the silent
        // one did.
        //
        // Named: silence as "voice_profile.reference_silent" (raised
        // identically to the speaker path above), an empty clip as
        // "voice_profile.reference_too_short" (the core layer's own code,
        // reused rather than duplicated), and a non-finite sample as
        // "voice_profile.reference_not_finite". The non-finite one is the arm
        // that matters: it is the only one of the three a caller can reach
        // through the public seam, because nothing before it inspects
        // reference sample values at the target rate.
        //
        // Bare, on purpose: a geometry failure. encode_codec_reference cannot
        // tell a sub-frame clip from a package whose shapes disagree, and the
        // sub-frame case is already refused with a name by the core layer's
        // min_frames_per_clip preflight. Its own comment says why.
        return codec_status;
    }

    // Defensive, and unreachable through any caller: encode_codec_reference's
    // own postcondition is `groups * frames` codes with both positive, and
    // codec_encoder_check_waveform has already refused a clip too short to
    // fill one frame. Restated here because Task 9's writer sizes a tensor
    // from these two numbers, where a disagreement would stop being a wrong
    // answer and start being a wrong allocation.
    if (codec_encoding.groups == 0 || codec_encoding.frames == 0 ||
        codec_encoding.codes.size() != size_t(codec_encoding.groups) * size_t(codec_encoding.frames)) {
        return SYNTH_ERR_INTERNAL;
    }

    auto profile                  = std::make_shared<IclProfile>();
    profile->speaker.mode         = CloneMode::Icl;
    profile->speaker.x_vector     = std::move(speaker_encoding.x_vector);
    profile->speaker.ref_rms      = speaker_encoding.ref_rms;
    profile->speaker.language_tag = language_tag;
    // Moved verbatim, in the encoder's own GROUP-FASTEST order. Any transpose
    // added here is a defect (codec-encoder-host.h's own header comment).
    profile->codes                = std::move(codec_encoding.codes);
    profile->groups               = codec_encoding.groups;
    profile->frames               = codec_encoding.frames;
    profile->reference_text_ids   = reference_text_ids;
    output                        = std::move(profile);
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

constexpr const char * kEnvelopeArchitecture   = "synthprofile";
constexpr uint32_t     kEnvelopeFormatVersion  = 1;
constexpr const char * kEnvelopeModelFamily    = "qwen3-tts";
// The SAME string the package's own ProfileContract requires
// (weights.cpp's read_profile_contract, "qwen3-tts-voice-clone") -- see
// profile.h's header comment on why this is one schema for both clone modes
// rather than two schemas.
constexpr const char * kEnvelopeSchema         = "qwen3-tts-voice-clone";
constexpr uint32_t     kEnvelopeSchemaVersion  = 1;
constexpr const char * kKindXVector            = "x-vector";
// Plan 3 Task 9's whole addition to the envelope: a second value of the SAME
// key, in the SAME schema at the SAME version.
constexpr const char * kKindIcl                = "icl";
constexpr const char * kKeyCompatibilityId     = "synthesize.voice_profile.compatibility_id";
constexpr const char * kKeyContentSha256       = "synthesize.voice_profile.content_sha256";
constexpr const char * kKeyKind                = "synthesize.voice_profile.kind";
constexpr const char * kKeyRefRms              = "synthesize.voice_profile.ref_rms";
constexpr const char * kKeyLanguageTag         = "synthesize.voice_profile.language_tag";
constexpr const char * kKeyCodeGroups          = "synthesize.voice_profile.code_groups";
constexpr const char * kKeyReferenceFrames     = "synthesize.voice_profile.reference_frames";
constexpr const char * kTensorXVector          = "profile.x_vector";
constexpr const char * kTensorCodes            = "profile.codes";
constexpr const char * kTensorReferenceTextIds = "profile.reference_text_ids";

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

// Appends the common v1 envelope header metadata every kind shares -- the
// ten kCommon entries of kPrescanKnownKeys (profile.h), in this function's
// own order, which is the order that table lists them in and the order both
// writers emit. `ref_rms` and `language_tag` moved in here from
// serialize_x_vector_profile when the ICL kind turned out to carry both;
// keeping the emission in ONE function is what keeps "every kind emits every
// kCommon key" true by construction rather than by two writers agreeing.
//
// `compatibility_id` is copied verbatim. `content_sha256` is seeded at
// zero -- docs/c-interface.md's exact rule: write_envelope hashes the
// assembled buffer with this placeholder still in place, then patches the
// real digest into the same 32 bytes afterward.
void set_common_metadata(gguf_context * ctx,
                         const char *   kind,
                         const uint8_t (&compatibility_id)[32],
                         float               ref_rms,
                         const std::string & language_tag) {
    gguf_set_val_str(ctx, "general.architecture", kEnvelopeArchitecture);
    gguf_set_val_u32(ctx, "synthesize.voice_profile.format_version", kEnvelopeFormatVersion);
    gguf_set_val_str(ctx, "synthesize.voice_profile.model_family", kEnvelopeModelFamily);
    gguf_set_val_str(ctx, "synthesize.voice_profile.schema", kEnvelopeSchema);
    gguf_set_val_u32(ctx, "synthesize.voice_profile.schema_version", kEnvelopeSchemaVersion);
    gguf_set_arr_data(ctx, kKeyCompatibilityId, GGUF_TYPE_UINT8, compatibility_id, 32);
    static constexpr uint8_t kZeroDigest[32] = {};
    gguf_set_arr_data(ctx, kKeyContentSha256, GGUF_TYPE_UINT8, kZeroDigest, 32);
    gguf_set_val_str(ctx, kKeyKind, kind);
    gguf_set_val_f32(ctx, kKeyRefRms, ref_rms);
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
    // for this field, validated at the dispatch site) is ASCII letters,
    // digits, and hyphens only, so a real caller's input can never contain a
    // NUL to begin with; only a caller that already bypassed that shape
    // validation could reach this gap.
    gguf_set_val_str(ctx, kKeyLanguageTag, language_tag.c_str());
}

// One tensor of an envelope, as write_envelope needs it: gguf's own context
// has no per-dimension shape getter to read a shape back FROM
// (gguf_get_tensor_size only ever returns a flattened byte count), so the
// writer carries the shape locally rather than round-tripping it through
// `ctx`. Every tensor either writer emits is 1-D, which is what lets this
// hold a single extent.
struct EnvelopeTensor {
    const char * name         = nullptr;
    ggml_type    type         = GGML_TYPE_F32;
    int64_t      elements     = 0;
    size_t       element_size = 0;
    const void * data         = nullptr;
};

// Hand-writes the header, tensor-info section, alignment padding, and tensor
// data that gguf.h's own writer API cannot produce into a memory buffer: see
// arch/omnivoice/profile.cpp's own write_envelope for why (the function that
// can, gguf_write_to_buf, lives in ggml-impl.h, out of reach across the
// submodule boundary). The metadata KV section above genuinely goes through
// gguf's own setter/getter API; only the header/tensor-info/tensor-data
// framing below is this project's own.
//
// `tensors` is one entry for the "x-vector" kind and three for "icl", in the
// writer's own fixed order. Each tensor's declared offset is the cumulative
// sum of its predecessors' ALIGNMENT-PADDED sizes, which is not a stylistic
// choice: gguf_init_from_buffer itself refuses any other layout
// (ggml/src/gguf.cpp:766 checks `ti.offset == ctx->size`, accumulated with
// GGML_PAD at :770), so a writer that packed them tightly would emit
// envelopes ggml's own parser rejects.
//
// `ctx` must already hold `content_sha256` seeded at 32 zero bytes
// (set_common_metadata's job).
synth_status_t write_envelope(const gguf_context *                ctx,
                              const std::vector<EnvelopeTensor> & tensors,
                              std::vector<uint8_t> &              out_bytes) {
    std::vector<uint8_t> bytes;

    put_bytes(bytes, GGUF_MAGIC, 4);
    put<uint32_t>(bytes, uint32_t(GGUF_VERSION));
    put<int64_t>(bytes, int64_t(tensors.size()));
    put<int64_t>(bytes, gguf_get_n_kv(ctx));

    size_t sha_offset = 0;
    if (!encode_metadata_kv(ctx, bytes, sha_offset)) {
        return SYNTH_ERR_INTERNAL;
    }

    uint64_t tensor_offset = 0;
    for (const EnvelopeTensor & tensor : tensors) {
        put_gguf_string(bytes, tensor.name);
        put<uint32_t>(bytes, uint32_t(1));  // n_dims: every envelope tensor is 1-D
        put<int64_t>(bytes, tensor.elements);
        put<int32_t>(bytes, int32_t(tensor.type));
        put<uint64_t>(bytes, tensor_offset);
        const uint64_t size = uint64_t(tensor.elements) * uint64_t(tensor.element_size);
        tensor_offset += size;
        while (tensor_offset % GGUF_DEFAULT_ALIGNMENT != 0) {
            ++tensor_offset;
        }
    }
    pad_to_alignment(bytes, GGUF_DEFAULT_ALIGNMENT);

    for (const EnvelopeTensor & tensor : tensors) {
        put_bytes(bytes, tensor.data, size_t(tensor.elements) * tensor.element_size);
        pad_to_alignment(bytes, GGUF_DEFAULT_ALIGNMENT);
    }

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

// Whether `tensor_id`'s DECLARED byte range actually fits inside a buffer of
// `data_size` bytes -- the truncation guard, split out of copy_tensor_bytes
// below so that a caller can run it BEFORE sizing anything from the declared
// count rather than after.
//
// THAT ORDER IS THE WHOLE POINT, and getting it wrong shipped here once. The
// first version of this task sized `std::vector<int32_t>` from a declared
// element count and only then called copy_tensor_bytes; a reviewer measured a
// 1 KiB Serialized Profile driving a ~1,550,000 KiB (1.48 GiB) zero-filled
// resident allocation
// before the load was refused, with the reachable ceiling around 2^62 elements
// for the id stream. `gguf_init_from_buffer` does NOT close this: called with
// `params.ctx == nullptr` it never reads the tensor data at all, and its own
// tensor-info pass only accumulates padded sizes (ggml/src/gguf.cpp:761-782)
// without ever comparing them to the buffer it was handed. Neither does the
// prescan, which walks the tensor-INFO section and stops there. So a declared
// count is attacker-chosen until this function has seen it. This is the same
// defect shape this project fixed in the WAV readers three days earlier: the
// VALUES were bounded correctly and the SIZE was taken from an untrusted field
// before anything had validated it.
bool tensor_range_fits(const gguf_context * g, size_t data_size, int64_t tensor_id, size_t tensor_bytes) {
    const size_t data_offset   = gguf_get_data_offset(g);
    const size_t tensor_offset = gguf_get_tensor_offset(g, tensor_id);
    if (data_offset > data_size) {
        return false;
    }
    size_t remaining = data_size - data_offset;
    if (tensor_offset > remaining) {
        return false;
    }
    remaining -= tensor_offset;
    return tensor_bytes <= remaining;
}

// Copies `tensor_id`'s DECLARED byte range out of the untrusted buffer, and
// only after checking that the range actually fits inside it.
// gguf_init_from_buffer is called with `ctx == nullptr` specifically so it
// never performs this read (or the allocation it implies) itself; see
// load_profile_from_memory's own comment above its size-arithmetic block.
// `out` must already be sized to `tensor_bytes`.
//
// The range check is repeated here rather than assumed from the caller: this
// function performs the memcpy, so it is the one that must not be able to run
// out of bounds no matter which caller reaches it.
bool copy_tensor_bytes(const gguf_context * g,
                       const uint8_t *      data,
                       size_t               data_size,
                       int64_t              tensor_id,
                       size_t               tensor_bytes,
                       void *               out) {
    if (!tensor_range_fits(g, data_size, tensor_id, tensor_bytes)) {
        return false;
    }
    const size_t data_offset   = gguf_get_data_offset(g);
    const size_t tensor_offset = gguf_get_tensor_offset(g, tensor_id);
    if (data_offset > data_size) {
        return false;
    }
    size_t remaining = data_size - data_offset;
    if (tensor_offset > remaining) {
        return false;
    }
    remaining -= tensor_offset;
    if (tensor_bytes > remaining) {
        return false;
    }
    std::memcpy(out, data + data_offset + tensor_offset, tensor_bytes);
    return true;
}

// Locates a required I32 tensor by name and reports its DECLARED element
// count, reading no payload byte -- only the bounded tensor-info section the
// parse above already materialized. `false` covers "absent", "wrong type",
// "zero-length", "a byte count that is not a whole number of int32s", and "a
// declared byte range that does not fit inside `data_size`" alike; every one
// of them is malformed for this schema.
//
// THE RANGE CHECK IS PART OF THIS FUNCTION AND NOT OF ITS CALLER, deliberately
// and as a fix: this is the function that hands a count to code that will size
// an allocation from it, so no count leaves here until the bytes behind it are
// known to exist. Doing it at the call site instead is what shipped the
// ~1,550,000 KiB allocation tensor_range_fits' own header comment records --
// the caller had the check, one line too late.
bool i32_tensor_elements(const gguf_context * g,
                         size_t               data_size,
                         const char *         name,
                         int64_t &            out_id,
                         uint64_t &           out_elements) {
    out_id = gguf_find_tensor(g, name);
    if (out_id < 0 || gguf_get_tensor_type(g, out_id) != GGML_TYPE_I32) {
        return false;
    }
    const size_t bytes = gguf_get_tensor_size(g, out_id);
    if (bytes == 0 || bytes % sizeof(int32_t) != 0) {
        return false;
    }
    if (!tensor_range_fits(g, data_size, out_id, bytes)) {
        return false;
    }
    out_elements = bytes / sizeof(int32_t);
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
// PrescanKeySpec, kPrescanKnownKeys, kPrescanKnownKeyCount,
// kPrescanKvCountXVector and kPrescanKvCountIcl -- the exact, closed set of
// metadata keys this family's writers ever emit, and the exact per-kind KV
// counts -- live in profile.h, not here: see that header's own comment on why
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
        // Defensive: prescan_buffer only ever passes a kPrescanKnownKeys
        // entry's own type here, so every tag reaching this switch is one of
        // the cases above. A caller that skipped that check would land here
        // rather than skipping an unknown width.
        return false;
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
// own format" is enforced as the per-kind key SET (every key present is
// whitelisted, no key repeats, the count matches that kind exactly, and the
// set is exactly that kind's own scoped set) plus each key's own declared
// type/shape, but NOT the writer's own emission order. A hand-built buffer
// with the ten kCommon keys in fully REVERSED order, with a digest
// recomputed over that reversed layout, loads successfully. This has no
// security impact -- content_sha256 is a corruption check, never a
// signature (sha256.h's own header comment) -- and is inherited verbatim
// from omnivoice::prescan_buffer, which has the identical property for its
// own two kinds. Left unpinned rather than fixed: order is not a property
// anything downstream reads, and pinning it would mean this walk could no
// longer resolve `kind` from wherever it happens to sit before deciding
// which sequence to expect.
//
// WHAT THIS WALK DOES ENFORCE THAT OMNIVOICE'S DOES NOT is the per-kind key
// SET. omnivoice::prescan_buffer never reads `scope` at all; this one
// captures the `kind` string during the walk and then requires the seen set
// to equal that kind's scoped set exactly. Without that, an "x-vector"
// envelope carrying BOTH kIclOnly keys would have twelve whitelisted,
// duplicate-free keys and an `n_kv` equal to kPrescanKvCountIcl -- every
// count-and-name gate would pass and the two stray keys would simply be
// ignored downstream. A union whitelist cannot say "that key belongs to the
// other kind", and that is the exact defect PrescanKeyScope was added for.
//
// MEASURED, stated precisely because an earlier draft of this paragraph was
// not: the SET check and the per-kind COUNT check MASK EACH OTHER. Deleting
// either alone leaves the whole test suite passing; only with both gone does
// the twelve-key "x-vector" forgery load with SYNTH_OK. Demonstrating that
// needs a SEALED buffer -- real compatibility_id, finite non-zero x-vector,
// digest recomputed -- because a zero-filled hand-built one is refused earlier
// by the compatibility_id comparison, with a different status. See
// prescan_buffer's own "MUTUAL MASKING" comment for the general rule this is
// the first instance of.
bool prescan_scope_applies(PrescanKeyScope scope, bool is_icl) {
    return scope == PrescanKeyScope::kCommon || (is_icl && scope == PrescanKeyScope::kIclOnly);
}

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
    // Exactly 1 tensor for "x-vector" (the x-vector itself) or exactly 3 for
    // "icl" (that same x-vector, the reference code grid, and the reference
    // text ids). Which of the two this buffer must have is not decidable
    // until `kind` is read below, so the header check is the disjunction and
    // the exact cross-check happens after the KV walk.
    if (!prescan_read(data, size, offset, n_tensors) || (n_tensors != 1 && n_tensors != 3)) {
        return false;
    }
    // Exactly kPrescanKvCountXVector or kPrescanKvCountIcl metadata entries,
    // never a ceiling -- see those constants' own comments (profile.h). Again
    // a disjunction here and an exact per-kind check below, for the same
    // reason: this gate runs before the loop that reads `kind`.
    if (!prescan_read(data, size, offset, n_kv) || (n_kv != kPrescanKvCountXVector && n_kv != kPrescanKvCountIcl)) {
        return false;
    }

    bool        seen[kPrescanKnownKeyCount] = {};
    std::string kind;
    bool        found_kind = false;

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

        // The declared type tag stays an int32_t for the whole walk and is
        // only ever COMPARED against a known enumerator -- it is never
        // converted to `gguf_type`. That enum has no fixed underlying type,
        // so converting a value outside its range (anything negative, or 16
        // and up) is undefined behaviour, and every byte here is
        // caller-supplied. The tensor type tag further down already reads
        // this way (`tensor_type_raw != int32_t(GGML_TYPE_F32)`), so this
        // matches the shape the same function already uses rather than
        // adding a range test against GGUF_TYPE_COUNT -- which would be a
        // blacklist against ggml's own enum extent, exactly the shape this
        // pre-scan deleted (see this section's header comment).
        int32_t type_raw = 0;
        if (!prescan_read(data, size, offset, type_raw)) {
            return false;
        }
        bool     is_array = false;
        uint64_t count    = 1;
        if (type_raw == int32_t(GGUF_TYPE_ARRAY)) {
            is_array                 = true;
            int32_t element_type_raw = 0;
            if (!prescan_read(data, size, offset, element_type_raw)) {
                return false;
            }
            type_raw = element_type_raw;
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
        if (type_raw != int32_t(spec.type) || is_array != spec.is_array || (is_array && count != spec.count)) {
            return false;
        }

        // The one value this walk DECODES rather than skips. `kind` decides
        // which key set, which KV count and which tensor section the rest of
        // this buffer must have, so it cannot be left to the post-parse
        // branch: everything below depends on it.
        if (key == kKeyKind) {
            uint64_t value_length = 0;
            if (!prescan_read(data, size, offset, value_length) || value_length > kPrescanMaxStringLength) {
                return false;
            }
            if (!prescan_has_remaining(offset, size, size_t(value_length))) {
                return false;
            }
            kind.assign(reinterpret_cast<const char *>(data + offset), size_t(value_length));
            offset += size_t(value_length);
            found_kind = true;
            continue;
        }

        // `spec.type` rather than the buffer's own tag: the two are equal by
        // the comparison just above, and the spec's copy is a real
        // enumerator by construction, so no value the enum cannot hold ever
        // reaches the switch in prescan_fixed_type_size.
        if (!prescan_skip_value(data, size, offset, spec.type, count)) {
            return false;
        }
    }

    if (!found_kind) {
        return false;
    }
    const bool is_icl = (kind == kKindIcl);
    if (!is_icl && kind != kKindXVector) {
        // An unrecognized kind cannot name a key set, a KV count or a tensor
        // section, so the walk cannot continue -- and the caller maps this to
        // the same SYNTH_ERR_INVALID_ARG the post-parse `kind` branch would
        // have returned, which is why a hand-built buffer that keeps a real
        // kind's shape and swaps only this string still reports identically.
        return false;
    }

    // The per-kind key SET, the per-kind KV count, and the per-kind tensor
    // count -- all three exact, all three resolved from the `kind` just read.
    //
    // MUTUAL MASKING, AND WHY EVERY ONE OF THESE NEEDS A COMMENT LIKE THIS.
    // Each of the three has a second enforcement somewhere else -- the KV
    // count is backed by the key SET check just below it, the key set is
    // backed by the count, and the tensor count is backed by
    // load_profile_from_memory's own `gguf_get_n_tensors` comparison. A test
    // that deletes ONE of a masked pair sees the suite pass and concludes the
    // rule is untested or redundant; both conclusions are wrong. The rules
    // here are the ones that run BEFORE gguf_init_from_buffer, on bytes ggml's
    // parser has not touched, which is the entire reason this pre-scan exists
    // (see this section's header comment). Anyone ADDING a check to this
    // function should assume it is masked until they have deleted it together
    // with its partner, and should say in a comment which partner that is --
    // single-deletion inversion cannot see this class of gap, and a reviewer
    // found two instances of it here that the original inversion pass missed.
    if (n_kv != (is_icl ? kPrescanKvCountIcl : kPrescanKvCountXVector)) {
        return false;  // masked by the key SET check below
    }
    // (The per-kind TENSOR count is checked below, where the array it has to
    // agree with is in scope -- see `expected_count`.)
    for (size_t candidate = 0; candidate < kPrescanKnownKeyCount; ++candidate) {
        if (seen[candidate] != prescan_scope_applies(kPrescanKnownKeys[candidate].scope, is_icl)) {
            return false;  // masked by the KV count check above
        }
    }

    // The tensor section: this writer's own exact entries, in its own order,
    // each 1-D with the exact type it writes and at the exact cumulative,
    // alignment-padded offset write_envelope computes. Each ne[0] is the one
    // field that legitimately varies (with the package's enc_dim, with the
    // reference length, with the transcript's token count), so it is only
    // checked for positivity here; the exact-width checks against the
    // package's own declared numbers still happen later, against the real
    // parsed tensor sizes, before any allocation sized from them.
    struct ExpectedTensor {
        const char * name;
        int32_t      type;
    };

    const ExpectedTensor x_vector_only[] = {
        { kTensorXVector, int32_t(GGML_TYPE_F32) },
    };
    const ExpectedTensor icl_tensors[] = {
        { kTensorXVector,          int32_t(GGML_TYPE_F32) },
        { kTensorCodes,            int32_t(GGML_TYPE_I32) },
        { kTensorReferenceTextIds, int32_t(GGML_TYPE_I32) },
    };
    const ExpectedTensor * expected = is_icl ? icl_tensors : x_vector_only;
    // THE PER-KIND TENSOR COUNT, AND THE LOOP BOUND, BOTH COME FROM THE ARRAY
    // rather than from the buffer's own `n_tensors` -- and that is a memory
    // safety property, not a tidiness one. The header disjunction alone only
    // narrows `n_tensors` to {1, 3}, so an "x-vector" envelope declaring 3
    // would walk a ONE-element expectation array three times and read
    // `x_vector_only[1]` and `[2]` past its end, on untrusted input.
    //
    // MEASURED (Task 9 review, Important 1), AND THE OUTCOME DEPENDS ON THE
    // BUILD -- which is why the build is named here, the same rule this
    // project already applies to performance numbers. Mutation: this per-kind
    // count removed and the loop below re-bound to the buffer's own
    // `n_tensors`, with load_profile_from_memory's own tensor-count check left
    // in place. Against the three-tensors arm in
    // tests/qwen3_tts_profile_test.cpp:
    //   * Release (2026-08-14, gcc 13.3, x86_64): the test binary SEGFAULTS,
    //     exit 139, inside this walk -- the garbage `const char *` is
    //     dereferenced by the std::string comparison below;
    //   * SYNTH_SANITIZE=ON / RelWithDebInfo: UBSan reports "load of address
    //     ... with insufficient space for an object of type 'const struct
    //     ExpectedTensor'" at the `expected[index]` read, and the process exits
    //     1 without finishing the suite.
    // It is undefined behaviour, so neither symptom is THE outcome; what holds
    // across both is that no status is ever returned to assert on. A status
    // assertion cannot see this. Only the sanitizer gate, or this structural
    // bound, can.
    //
    // (An earlier draft of this comment said the sanitizer build "still
    // returns SYNTH_ERR_INVALID_ARG, because the garbage it reads happens not
    // to match a tensor name". That was measured against a WEAKER version of
    // the test arm, whose buffer carried only one tensor-info entry -- so the
    // walk read padding rather than a valid entry and the names mismatched
    // harmlessly. Once the arm was strengthened to carry three well-formed
    // entries, the claim stopped being true and is corrected here rather than
    // quietly dropped.)
    //
    // So the count and the bound are derived from one `std::size`, and THE
    // PRESCAN'S OWN earlier hand-written `(is_icl ? 3 : 1)` copy of this
    // predicate was deleted rather than left to drift from the array it
    // describes. Its twin in load_profile_from_memory is still hand-written
    // and is NOT derived from these arrays, which are local to this function:
    // if a fourth ICL tensor were ever added, this check would follow the
    // array automatically and that one would not, and the two would disagree
    // in the fail-closed direction (the loader refuses a count it did not
    // expect). Worth knowing before adding one.
    //
    // Masked by load_profile_from_memory's own `gguf_get_n_tensors` comparison
    // (see prescan_buffer's "MUTUAL MASKING" comment) -- but not redundant
    // with it: that one runs after ggml has already parsed the tensor-info
    // section, and this one is what keeps THIS walk in bounds.
    const int64_t expected_count    = is_icl ? int64_t(std::size(icl_tensors)) : int64_t(std::size(x_vector_only));
    if (n_tensors != expected_count) {
        return false;
    }

    uint64_t expected_offset = 0;
    for (int64_t index = 0; index < expected_count; ++index) {
        uint64_t name_length = 0;
        if (!prescan_read(data, size, offset, name_length) || name_length > kPrescanMaxKeyLength) {
            return false;
        }
        if (!prescan_has_remaining(offset, size, size_t(name_length))) {
            return false;
        }
        const std::string tensor_name(reinterpret_cast<const char *>(data + offset), size_t(name_length));
        offset += size_t(name_length);
        if (tensor_name != expected[index].name) {
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
        if (!prescan_read(data, size, offset, tensor_type_raw) || tensor_type_raw != expected[index].type) {
            return false;
        }

        uint64_t tensor_offset = 0;
        if (!prescan_read(data, size, offset, tensor_offset) || tensor_offset != expected_offset) {
            return false;
        }

        // Both element types this schema writes are four bytes wide, so the
        // running offset needs no per-type table. The overflow guard is not
        // decorative: `ne0` is caller-supplied and only known positive.
        constexpr uint64_t kElementSize = 4;
        if (uint64_t(ne0) > (UINT64_MAX - GGUF_DEFAULT_ALIGNMENT - expected_offset) / kElementSize) {
            return false;
        }
        expected_offset += uint64_t(ne0) * kElementSize;
        while (expected_offset % GGUF_DEFAULT_ALIGNMENT != 0) {
            ++expected_offset;
        }
    }

    return true;
}

}  // namespace

synth_status_t serialize_x_vector_profile(const XVectorProfile & profile,
                                          const uint8_t (&compatibility_id)[32],
                                          std::vector<uint8_t> & out_bytes) {
    // THE SILENT DOWNGRADE D4 EXISTS TO PREVENT, refused at the one place
    // that sees the payload's mode and the target kind together. Since Plan
    // 3's Task 8, `XVectorProfile{mode == CloneMode::Icl}` is a constructible
    // object -- it is exactly `IclProfile::speaker`, and that composition is
    // deliberate: it keeps src/synthesize.cpp's type-erased read of `.mode`
    // through an `XVectorProfile *` well-defined for both modes. The cost is
    // that this writer could be handed one and would emit a perfectly valid,
    // digest-correct `kind="x-vector"` envelope with the reference codes and
    // the reference text ids simply absent, so an ICL Profile would round
    // trip back as an x-vector Profile with nothing on either side
    // disagreeing. Checked FIRST, before any other invariant: this is the one
    // failure here that is about what the caller MEANT rather than about
    // whether the bytes can be written.
    if (profile.mode != CloneMode::XVector) {
        return SYNTH_ERR_INVALID_ARG;
    }
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
    set_common_metadata(ctx.get(), kKindXVector, compatibility_id, profile.ref_rms, profile.language_tag);
    const std::vector<EnvelopeTensor> tensors = {
        { kTensorXVector, GGML_TYPE_F32, int64_t(profile.x_vector.size()), sizeof(float), profile.x_vector.data() },
    };
    return write_envelope(ctx.get(), tensors, out_bytes);
}

synth_status_t serialize_icl_profile(const HParams &    hparams,
                                     const IclProfile & profile,
                                     const uint8_t (&compatibility_id)[32],
                                     std::vector<uint8_t> & out_bytes) {
    // The mirror of the x-vector writer's own mode guard above, and the same
    // rule from the other side: this function writes `kind="icl"`, so a
    // payload whose own CloneMode says otherwise would produce an envelope
    // that means something its author did not.
    if (profile.speaker.mode != CloneMode::Icl) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (profile.speaker.x_vector.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (profile.speaker.language_tag.size() > kMaxLanguageTagLength) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // create_icl_profile refuses the half-present state (a transcript with no
    // ids); a hand-built payload can still hold it, and an ICL envelope with
    // no reference text is a plausible-sounding wrong prompt rather than
    // anything downstream would report.
    if (profile.reference_text_ids.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // The grid's own three numbers have to agree with each other, with the
    // package, and with what a uint32 metadata value can carry. `groups` is
    // exact rather than a ceiling for the same reason `enc_dim` is on the
    // read side: a grid of the wrong width builds a wrong-shaped prompt.
    // `quantizer_count` rather than `talker.code_group_count` for the reason
    // load_profile_from_memory's own copy of this check records:
    // weights.cpp:249-255 refuses any package where the two differ, so they
    // are interchangeable, and this is the one the writer's own source
    // (encode_codec_reference) produces.
    if (profile.groups == 0 || profile.frames == 0 ||
        profile.groups != uint64_t(hparams.codec.decoder.quantizer_count) || profile.groups > UINT32_MAX ||
        profile.frames > UINT32_MAX || profile.codes.size() != size_t(profile.groups) * size_t(profile.frames)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // Positive validation, writer side: this family must never emit an
    // envelope its own reader refuses (kMaxLanguageTagLength's header comment
    // records the reviewer-measured defect that rule was written against),
    // and load_profile_from_memory range-checks both discrete streams against
    // exactly these two package widths.
    const int32_t codebook_size = int32_t(hparams.codec.decoder.codebook_size);
    if (codebook_size <= 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    for (int32_t code : profile.codes) {
        if (code < 0 || code >= codebook_size) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }
    // Defence in depth, and it cannot fire for a package this port loads:
    // weights.cpp:347-352 refuses one whose `talker.text_vocab_size` is zero,
    // and read_tokens is unconditional in read_hparams' `&&` chain. Kept
    // because this function takes `HParams` rather than a loaded Model, so a
    // synthetic struct can reach it -- and a zero bound would otherwise admit
    // every id. Same labelling as the `groups`/`frames == 0` clauses nearby.
    const int64_t text_vocab_size = int64_t(hparams.talker.text_vocab_size);
    if (text_vocab_size <= 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    for (int32_t id : profile.reference_text_ids) {
        if (id < 0 || int64_t(id) >= text_vocab_size) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }

    OwnedGgufContext ctx(gguf_init_empty());
    if (ctx == nullptr) {
        return SYNTH_ERR_OOM;
    }
    set_common_metadata(ctx.get(), kKindIcl, compatibility_id, profile.speaker.ref_rms, profile.speaker.language_tag);
    gguf_set_val_u32(ctx.get(), kKeyCodeGroups, uint32_t(profile.groups));
    gguf_set_val_u32(ctx.get(), kKeyReferenceFrames, uint32_t(profile.frames));

    // The codes go out FLAT, in IclProfile's own GROUP-FASTEST order, with no
    // transpose and no reshape -- codec-encoder-host.h's own header comment
    // records why the next reader will want to add one and why it is wrong.
    const std::vector<EnvelopeTensor> tensors = {
        { kTensorXVector,          GGML_TYPE_F32, int64_t(profile.speaker.x_vector.size()),   sizeof(float),
         profile.speaker.x_vector.data()                                                                                            },
        { kTensorCodes,            GGML_TYPE_I32, int64_t(profile.codes.size()),              sizeof(int32_t), profile.codes.data() },
        { kTensorReferenceTextIds, GGML_TYPE_I32, int64_t(profile.reference_text_ids.size()), sizeof(int32_t),
         profile.reference_text_ids.data()                                                                                          },
    };
    return write_envelope(ctx.get(), tensors, out_bytes);
}

synth_status_t load_profile_from_memory(const HParams & hparams,
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
    if (!meta.string(kKeyKind, kind)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const bool is_icl = (kind == kKindIcl);
    if (!is_icl && kind != kKindXVector) {
        // This build recognizes exactly two kinds. A value outside them means
        // the payload structure the rest of the bytes describe cannot be
        // interpreted at all, which this project treats as malformed rather
        // than merely unsupported (profile.h's own header comment on this
        // function, including the forward-compatibility cost that trade
        // carries). Reaching THIS line rather than the prescan's own
        // identical rejection takes a buffer whose key set, KV count and
        // tensor section all match one of the two real kinds while its `kind`
        // string names neither -- which the prescan cannot happen upon, since
        // it resolves the expected shape FROM this same string. Both paths
        // return the same status, so a caller cannot tell them apart, and
        // this one is kept because dropping it would leave the post-parse
        // branch below trusting a value nothing here had checked.
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

    // Masked by prescan_buffer's own per-kind tensor-count check, which runs
    // first and rejects the same buffers -- deleting either one alone leaves
    // the suite passing, and only deleting BOTH makes a wrong tensor count
    // observable. Kept anyway: the two guard different things. The prescan's
    // copy keeps its own tensor walk in bounds; this one keeps the
    // gguf_find_tensor lookups below from operating on a context whose shape
    // this function has not itself agreed to. See prescan_buffer's own
    // "MUTUAL MASKING" comment for why a check here should be assumed masked
    // until proven otherwise.
    if (gguf_get_n_tensors(g) != (is_icl ? 3 : 1)) {
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
    if (element_count != hparams.speaker_encoder.enc_dim) {
        // Not a ceiling, an exact match: a Profile of the wrong width builds
        // a wrong-shaped prompt (profile.h's own header comment).
        return SYNTH_ERR_INVALID_ARG;
    }
    // The x-vector's count is already package-bounded by the line above, so
    // this is not the stream that could drive a hostile allocation. The range
    // check runs here anyway, and BEFORE the vector below rather than beside
    // the copy, so that "no declared count reaches a container until its bytes
    // are known to exist" is a property of every path through this function
    // rather than of two of the three -- which is exactly the asymmetry that
    // let the ICL streams ship unguarded.
    if (!tensor_range_fits(g, data_size, tensor_id, tensor_bytes)) {
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
    if (!meta.f32(kKeyRefRms, ref_rms) || !meta.string(kKeyLanguageTag, language_tag)) {
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

    // Truncation guard, then the copy: the DECLARED tensor byte range must
    // actually fit inside the SUPPLIED buffer before anything is read out of
    // it.
    std::vector<float> x_vector(size_t{ element_count });
    if (!copy_tensor_bytes(g, data, data_size, tensor_id, tensor_bytes, x_vector.data())) {
        return SYNTH_ERR_INVALID_ARG;
    }

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

    if (!is_icl) {
        auto profile          = std::make_shared<XVectorProfile>();
        profile->mode         = CloneMode::XVector;
        profile->x_vector     = std::move(x_vector);
        profile->ref_rms      = ref_rms;
        profile->language_tag = std::move(language_tag);

        out_payload    = std::move(profile);
        out_family_tag = synth::ProfileFamilyTag::Qwen3TtsClone;
        return SYNTH_OK;
    }

    // ----------------------------------------------------------------------
    // The "icl" kind's own two rows. Everything above ran unchanged, which is
    // the point of composing IclProfile around an XVectorProfile: an ICL
    // Profile is an x-vector Profile plus two things, and the loader reads it
    // that way too.
    // ----------------------------------------------------------------------

    uint32_t declared_groups = 0;
    uint32_t declared_frames = 0;
    if (!meta.u32(kKeyCodeGroups, declared_groups) || !meta.u32(kKeyReferenceFrames, declared_frames)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // Exact, not a ceiling, for the same reason `enc_dim` is: a code grid of
    // the wrong width builds a wrong-shaped prompt, and every downstream
    // reader indexes a frame by this stride.
    //
    // `quantizer_count` AND NOT `talker.code_group_count`, which is what
    // talker-host.cpp's own reference_is_well_formed indexes by -- and the two
    // are interchangeable, on the record: weights.cpp:249-255 refuses any
    // package where `codec.decoder.quantizer_count != talker.code_group_count`
    // ("the codec takes %u code groups but the talker emits %u"), and
    // read_codec sits unconditionally in read_hparams' own `&&` chain, so no
    // voice mode or variant can skip it. Cited here because picking one of two
    // numbers with no note sends the next reader to a third file to find out
    // whether the choice is safe. This one is chosen because it is the number
    // the WRITER's source (encode_codec_reference) produces, which is what
    // "validate positively against what our own writer emits" means.
    //
    // The two `== 0` clauses are defence in depth and cannot be the sole cause
    // of a refusal: i32_tensor_elements already rejects a zero-length codes
    // tensor, so a zero factor always breaks the product check below first.
    if (declared_groups == 0 || declared_groups != hparams.codec.decoder.quantizer_count || declared_frames == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // THE PACKAGE'S OWN REFERENCE CEILING, APPLIED TO THE DECLARED FRAME
    // COUNT. Without this the ceiling binds at CREATION and not at LOAD, and
    // the two ends of the same contract disagree by two orders of magnitude:
    // src/voice-profile.cpp refuses a reference longer than
    // `max_frames_per_clip`/`max_total_frames` (720,000 PCM frames = 30 s for
    // the Base package), so this family's own writer cannot emit more than
    // ceil(720000 / 1920) = 375 code frames -- while the reader, before this
    // check existed, accepted anything the buffer could back, which for a
    // ~1 MiB envelope is about 16,300 frames, 43x the ceiling.
    //
    // WHY THAT IS A MEMORY-SAFETY MATTER AND NOT A TIDINESS ONE, and why
    // Task 9's fix does not reach it. That fix made an unchecked count
    // unobtainable -- `tensor_range_fits` runs inside `i32_tensor_elements`,
    // so a declared count is always backed by real bytes. It is: 16,300
    // frames is only ~1 MiB of codes, and the loader's own allocation is
    // therefore small and honest. The amplification is DOWNSTREAM and
    // QUADRATIC: `run_synthesis` builds `prefill = frames + 10` and then a
    // `prefill x prefill` attention mask twice over (model.cpp's
    // ggml_new_tensor_2d and its host-side std::vector), plus KV caches at
    // ~235 kB per position. A ~1 MiB envelope therefore drives gigabytes of
    // resident set, growing as the SQUARE of the declared count, and both the
    // load and the synthesis return SYNTH_OK -- no status anywhere moves, so
    // only a peak-RSS measurement can see it. tests/qwen3_tts_icl_real.cpp's
    // assertion 8 is that measurement.
    //
    // The sibling family already had exactly this check
    // (arch/omnivoice/profile.h's `max_total_frames` parameter, bounding its
    // declared element count before any vector sizes itself); this loader
    // already takes the whole `HParams`, so the bound was in scope and simply
    // unused. It is the same doctrine this file states for the code and id
    // ranges: validate positively against what OUR OWN WRITER can emit.
    //
    // INVALID_ARG and not INPUT_TOO_LONG, deliberately: creation reports a
    // caller's over-long clip, but there is no legitimate caller here -- a
    // Serialized Profile declaring more frames than the package's own encoder
    // could ever have produced is malformed, which is what every other
    // refusal in this loader returns.
    //
    // The two factors are read as uint64 so the ceiling divide cannot wrap,
    // and a package declaring a zero hop or a zero ceiling is refused rather
    // than allowed to compute an unbounded limit -- read_hparams keeps both
    // non-zero for every real package, so this is defence in depth.
    {
        const uint64_t samples_per_frame = hparams.codec.hop_length;
        const uint64_t pcm_ceiling = std::min(hparams.profile.max_frames_per_clip, hparams.profile.max_total_frames);
        if (samples_per_frame == 0 || pcm_ceiling == 0) {
            return SYNTH_ERR_INVALID_ARG;
        }
        const uint64_t max_reference_frames = (pcm_ceiling + samples_per_frame - 1) / samples_per_frame;
        if (uint64_t(declared_frames) > max_reference_frames) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }

    // Both counts below arrive already checked against what `data_size` can
    // actually hold -- i32_tensor_elements refuses a declared range that does
    // not fit -- so the two `std::vector` constructions further down cannot be
    // sized from a number the buffer does not back. See tensor_range_fits'
    // own header comment for the measured defect that ordering closes.
    int64_t  codes_id       = -1;
    uint64_t codes_elements = 0;
    if (!i32_tensor_elements(g, data_size, kTensorCodes, codes_id, codes_elements)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // The cross-invariant the two kIclOnly keys exist to make checkable, and
    // the one create_icl_profile asserts on its own output -- here against
    // untrusted bytes, where a disagreement is a wrong allocation rather than
    // a wrong answer. Computed in uint64 so the product cannot wrap: both
    // factors came out of uint32 metadata.
    if (codes_elements != uint64_t(declared_groups) * uint64_t(declared_frames)) {
        return SYNTH_ERR_INVALID_ARG;
    }

    int64_t  ids_id       = -1;
    uint64_t ids_elements = 0;
    if (!i32_tensor_elements(g, data_size, kTensorReferenceTextIds, ids_id, ids_elements)) {
        return SYNTH_ERR_INVALID_ARG;
    }
    // THE SAME ASYMMETRY AS THE FRAME CEILING ABOVE, ON THE ADJACENT FIELD --
    // the THIRD instance of it on this branch, after Task 9's declared counts
    // and MB1's frame ceiling, which is why it is closed the same way rather
    // than argued about. Creation bounds this stream:
    // Model::tokenize_reference_transcript passes `hparams.max_input_tokens`
    // into qwen_reference_transcript_ids, which forwards it to the text
    // frontend, and bpe-frontend.cpp refuses an output longer than that. The
    // Base package declares 1024, so this family's own writer cannot emit more
    // than 1,024 reference text ids -- while the reader, before this check,
    // accepted whatever the buffer could back, which for a ~1 MiB envelope is
    // about 262,000, roughly 256x.
    //
    // WHAT IT COSTS, AND WHY IT IS SMALLER THAN THE FRAME CEILING'S: this one
    // is LINEAR, not quadratic. The ids never reach `prefill` -- append_icl_block
    // emits exactly `reference_frames + 1` positions whichever alignment arm
    // runs, so the quadratic term stays bounded by the field above. They are
    // copied into `reference_text_tokens`, then into the prompt's text track
    // and, on the truncate arm, into the trailing decode schedule, as
    // `TalkerInputPosition` values at roughly 32 bytes each -- about a 20x
    // linear amplification on top of a decode schedule hundreds of times
    // longer than the package says it accepts. Measured rather than estimated;
    // the figure is in tests/qwen3_tts_icl_real.cpp's assertion 8, which
    // exercises this bound with the same instrument it uses for the frame one.
    //
    // The header's old sentence -- "the two ICL streams have no such fixed
    // width, so the buffer itself is what bounds them" -- was written before
    // that shape was recognised as a defect on the sibling field, and is
    // corrected there too. `max_input_tokens == 0` is refused rather than
    // treated as "no limit", the same defence-in-depth choice the frame
    // ceiling makes.
    if (hparams.max_input_tokens == 0 || ids_elements > hparams.max_input_tokens) {
        return SYNTH_ERR_INVALID_ARG;
    }

    std::vector<int32_t> codes(size_t{ codes_elements });
    if (!copy_tensor_bytes(g, data, data_size, codes_id, codes.size() * sizeof(int32_t), codes.data())) {
        return SYNTH_ERR_INVALID_ARG;
    }
    std::vector<int32_t> reference_text_ids(size_t{ ids_elements });
    if (!copy_tensor_bytes(g, data, data_size, ids_id, reference_text_ids.size() * sizeof(int32_t),
                           reference_text_ids.data())) {
        return SYNTH_ERR_INVALID_ARG;
    }

    // Payload-value parity for the two discrete streams, and the only
    // protection either of them has -- see this function's own header comment
    // (profile.h) for what each one is an index INTO and what happens when it
    // is out of range. Positive validation, against exactly what this
    // family's own writer can emit: encode_codec_reference's own
    // postcondition for a code, and the width of the talker's own text
    // embedding table for an id. NOT an enumeration of what some downstream
    // consumer asserts today -- that direction is a blacklist, and
    // docs/porting/families/omnivoice.md records why this project does not
    // take it for untrusted bytes.
    const int32_t codebook_size = int32_t(hparams.codec.decoder.codebook_size);
    if (codebook_size <= 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    for (int32_t code : codes) {
        if (code < 0 || code >= codebook_size) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }
    // Defence in depth, and it cannot fire for a package this port loads:
    // weights.cpp:347-352 refuses one whose `talker.text_vocab_size` is zero,
    // and read_tokens is unconditional in read_hparams' `&&` chain. Kept
    // because this function takes `HParams` rather than a loaded Model, so a
    // synthetic struct can reach it -- and a zero bound would otherwise admit
    // every id. Same labelling as the `groups`/`frames == 0` clauses nearby.
    const int64_t text_vocab_size = int64_t(hparams.talker.text_vocab_size);
    if (text_vocab_size <= 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    for (int32_t id : reference_text_ids) {
        // The negative half is not symmetry for its own sake: an id is
        // carried into the talker prompt as a `uint32_t`, so -1 arrives there
        // as roughly 4e9 and indexes a table with a few hundred thousand
        // rows. The abort that follows is not a status any caller can map.
        if (id < 0 || int64_t(id) >= text_vocab_size) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }

    auto profile                  = std::make_shared<IclProfile>();
    profile->speaker.mode         = CloneMode::Icl;
    profile->speaker.x_vector     = std::move(x_vector);
    profile->speaker.ref_rms      = ref_rms;
    profile->speaker.language_tag = std::move(language_tag);
    profile->codes                = std::move(codes);
    profile->groups               = declared_groups;
    profile->frames               = declared_frames;
    profile->reference_text_ids   = std::move(reference_text_ids);

    out_payload    = std::move(profile);
    out_family_tag = synth::ProfileFamilyTag::Qwen3TtsClone;
    return SYNTH_OK;
}

}  // namespace synth::qwen3tts
