// create_x_vector_profile (src/arch/qwen3-tts/profile.cpp): the seam that
// turns a normalized reference clip into the in-memory Voice Profile a
// caller synthesizes with -- the object Task 9's public-seam dispatch will
// hand back from synth_voice_profile_create_from_reference once this
// family's capability snapshot stops being all-zero.
//
// Synthetic HParams and in-memory weights drawn from a shared LCG, no GGUF
// file and no loaded Model involved -- tests/omnivoice_reference_encoder_test.cpp's
// own fixture style, and the same shape
// tests/qwen3_tts_speaker_encoder_host_test.cpp (Task 5) already uses for the
// layer immediately underneath this one. profile.h's own header comment
// records why this function takes `HParams`/`SpeakerEncoderWeights` directly
// rather than a `Model &`: `synth::qwen3tts::Model` can only be built from a
// real GGUF (Model::load/load_cpu), and this family has no synthetic-package
// test harness yet.
//
// What this file covers: the two rejections create_x_vector_profile itself
// names (a transcript Plan 2 cannot honour; a package with no speaker
// encoder), that encode_speaker_reference's own silent-reference refusal
// propagates unchanged, and that a successful call produces a Profile fixed
// to CloneMode::XVector carrying the declared enc_dim width and the caller's
// language tag verbatim. encode_speaker_reference's own graph-level
// rejections (a mismatched mel width, a resolver that dropped a pointer, a
// non-finite embedding) are already Task 5's own coverage
// (qwen3_tts_speaker_encoder_host_test.cpp) and are not repeated here.

#include "arch/qwen3-tts/catalog.h"
#include "arch/qwen3-tts/profile.h"
#include "arch/qwen3-tts/weights.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "test-assert.h"
#include "voice-profile-handle.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using synth::qwen3tts::CloneMode;
using synth::qwen3tts::Conv1dWeights;
using synth::qwen3tts::HParams;
using synth::qwen3tts::kPrescanKnownKeyCount;
using synth::qwen3tts::kPrescanKnownKeys;
using synth::qwen3tts::kPrescanKvCountXVector;
using synth::qwen3tts::PrescanKeyScope;
using synth::qwen3tts::SpeakerEncoderBlockWeights;
using synth::qwen3tts::SpeakerEncoderWeights;
using synth::qwen3tts::XVectorProfile;

// Small and distinct from the checkpoint's own widths -- a swapped extent is
// only visible when nothing lines up by coincidence, the same reasoning
// tests/qwen3_tts_speaker_encoder_test.cpp and its host-level sibling give
// for their own fixtures.
constexpr int64_t  kMelBins           = 6;
constexpr int64_t  kChannels          = 16;
constexpr int64_t  kAggregated        = 3 * kChannels;
constexpr int64_t  kRes2NetScale      = 8;
constexpr int64_t  kSeChannels        = 5;
constexpr int64_t  kAttentionChannels = 7;
constexpr int64_t  kEncDim            = 11;
constexpr uint64_t kSeed              = 20260812u;

// The mel front end's own parameters. kSampleRate is this package's real
// declared Reference Audio rate (24 kHz) rather than an arbitrary synthetic
// value, purely so one_second_of_speech() below is truthfully named -- the
// ECAPA graph itself does not care what the sample rate is named, only that
// compute_log_mel and the fixed mel-bin/channel widths below agree with each
// other.
constexpr uint32_t kNFft       = 64;
constexpr uint32_t kHopLength  = 16;
constexpr uint32_t kWinLength  = 64;
constexpr uint32_t kSampleRate = 24000;

class LcgStream {
  public:
    explicit LcgStream(uint64_t seed) : state_(seed) {}

    float next() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        return float(state_ >> 40) / 8388608.0f - 1.0f;
    }

    std::vector<float> fill(size_t count, float scale) {
        std::vector<float> values(count);
        for (float & value : values) {
            value = next() * scale;
        }
        return values;
    }

  private:
    uint64_t state_;
};

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

Context make_context(size_t bytes) {
    ggml_init_params parameters{};
    parameters.mem_size = bytes;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

HParams make_hparams() {
    HParams hparams;
    hparams.has_speaker_encoder         = true;
    hparams.speaker_encoder.enc_dim     = uint32_t(kEncDim);
    hparams.speaker_encoder.sample_rate = kSampleRate;
    hparams.speaker_encoder.mel_bins    = uint32_t(kMelBins);
    hparams.speaker_encoder.n_fft       = kNFft;
    hparams.speaker_encoder.hop_length  = kHopLength;
    hparams.speaker_encoder.win_length  = kWinLength;
    hparams.speaker_encoder.fmin        = 0.0f;
    hparams.speaker_encoder.fmax        = float(kSampleRate) / 2.0f;
    return hparams;
}

// ggml reports a Conv1d kernel stored as [out, in, kernel] in reverse -- the
// same layout tests/qwen3_tts_speaker_encoder_test.cpp's own add_conv uses.
void add_conv(ggml_context *               context,
              Conv1dWeights &              target,
              int64_t                      kernel,
              int64_t                      in,
              int64_t                      out,
              std::vector<ggml_tensor *> & order,
              std::vector<float> &         scale) {
    target.weight = ggml_new_tensor_3d(context, GGML_TYPE_F32, kernel, in, out);
    target.bias   = ggml_new_tensor_1d(context, GGML_TYPE_F32, out);
    order.push_back(target.weight);
    scale.push_back(0.5f);
    order.push_back(target.bias);
    scale.push_back(0.25f);
}

// Owns the backend and buffer the weights live on -- the same shape
// tests/qwen3_tts_speaker_encoder_host_test.cpp's own Fixture uses, for the
// same reason: a separate ggml_backend_t from whatever
// encode_speaker_reference opens internally, matching production's own
// split between Model::load's persistent weights buffer and
// encode_speaker_reference's own per-call BackendPlan.
struct Fixture {
    ggml_backend_t        backend = nullptr;
    Context               persistent;
    ggml_backend_buffer_t buffer = nullptr;
    HParams               hparams;
    SpeakerEncoderWeights weights;

    Fixture()                            = default;
    Fixture(const Fixture &)             = delete;
    Fixture & operator=(const Fixture &) = delete;

    ~Fixture() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

bool build_fixture(Fixture & fixture) {
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (device == nullptr) {
        return false;
    }
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }
    fixture.hparams     = make_hparams();
    fixture.persistent  = make_context(ggml_tensor_overhead() * 256);
    ggml_context * pctx = fixture.persistent.get();
    if (pctx == nullptr) {
        return false;
    }

    std::vector<ggml_tensor *> order;
    std::vector<float>         scale;
    add_conv(pctx, fixture.weights.stem, 5, kMelBins, kChannels, order, scale);
    fixture.weights.blocks.assign(3, SpeakerEncoderBlockWeights{});
    for (SpeakerEncoderBlockWeights & block : fixture.weights.blocks) {
        const int64_t width = kChannels / kRes2NetScale;
        add_conv(pctx, block.tdnn1, 1, kChannels, kChannels, order, scale);
        block.res2net.assign(size_t(kRes2NetScale - 1), Conv1dWeights{});
        for (Conv1dWeights & split : block.res2net) {
            add_conv(pctx, split, 3, width, width, order, scale);
        }
        add_conv(pctx, block.se1, 1, kChannels, kSeChannels, order, scale);
        add_conv(pctx, block.se2, 1, kSeChannels, kChannels, order, scale);
        add_conv(pctx, block.tdnn2, 1, kChannels, kChannels, order, scale);
    }
    add_conv(pctx, fixture.weights.mfa, 1, kAggregated, kAggregated, order, scale);
    add_conv(pctx, fixture.weights.asp_tdnn, 1, 3 * kAggregated, kAttentionChannels, order, scale);
    add_conv(pctx, fixture.weights.asp, 1, kAttentionChannels, kAggregated, order, scale);
    add_conv(pctx, fixture.weights.fc, 1, 2 * kAggregated, kEncDim, order, scale);

    fixture.buffer = ggml_backend_alloc_ctx_tensors(pctx, fixture.backend);
    if (fixture.buffer == nullptr) {
        return false;
    }
    LcgStream stream(kSeed);
    for (size_t index = 0; index < order.size(); ++index) {
        const std::vector<float> values = stream.fill(size_t(ggml_nelements(order[index])), scale[index]);
        ggml_backend_tensor_set(order[index], values.data(), 0, ggml_nbytes(order[index]));
    }
    return true;
}

// Literally one second at this package's own declared 24 kHz Reference Audio
// rate (kSampleRate above) -- far past compute_log_mel's own minimum sample
// count and kSpeakerEncoderDeepestReflection's 4-frame floor, so nothing
// here exercises either domain edge; those are Task 5's own coverage.
std::vector<float> one_second_of_speech() {
    LcgStream stream(kSeed + 100);
    return stream.fill(size_t(kSampleRate), 0.2f);
}

// D4: a transcript names the transcript-assisted mode, which Plan 2 does not
// implement. Accepting it and silently building the x-vector Profile anyway
// would hand back a weaker clone than the caller asked for -- rejected by
// name rather than ignored.
int test_a_transcript_is_rejected_not_ignored() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const synth_status_t                  status = synth::qwen3tts::create_x_vector_profile(
        fixture.hparams, fixture.weights, one_second_of_speech(), /*transcript=*/"hello there",
        /*language_tag=*/"", /*threads=*/1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.transcript_unsupported") == 0);
    SYNTH_TEST_CHECK(message != nullptr);
    return 0;
}

// The spec's section 9 error table: a transcript that is present but
// whitespace-only is still "a transcript was supplied", so it takes the same
// rejection as any other non-empty transcript in Plan 2. The distinction
// between "empty" and "whitespace-only" only starts to matter from Plan 3
// on, once a transcript is actually consumed -- pinned now rather than
// discovered then.
int test_a_whitespace_transcript_is_rejected_too() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const synth_status_t                  status = synth::qwen3tts::create_x_vector_profile(
        fixture.hparams, fixture.weights, one_second_of_speech(), "   ", "", 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.transcript_unsupported") == 0);
    return 0;
}

// jiangzhuo's 2026-08-12 ruling (speaker-encoder-host.cpp's own header
// comment): a digitally silent reference still produces a finite, stable
// x-vector, and that embedding is meaningless, not a cloned voice.
// encode_speaker_reference's own refusal must reach this caller unchanged.
int test_a_silent_reference_is_refused_by_name() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const std::vector<float>              silence(size_t(kSampleRate), 0.0f);
    const synth_status_t status = synth::qwen3tts::create_x_vector_profile(fixture.hparams, fixture.weights, silence,
                                                                           "", "", 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(profile == nullptr);
    SYNTH_TEST_CHECK(code != nullptr && std::strcmp(code, "voice_profile.reference_silent") == 0);
    SYNTH_TEST_CHECK(message != nullptr);
    return 0;
}

// Model::prepare_x_vector's own CustomVoice guard, reproduced here (this
// function does not have a Model to delegate to -- see profile.h's own
// comment): a package with no speaker encoder at all must be refused before
// encode_speaker_reference ever sees a weights struct with every pointer
// null, and before the transcript-less, otherwise-valid request in this test
// would otherwise succeed.
int test_a_package_with_no_speaker_encoder_is_refused() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    fixture.hparams.has_speaker_encoder           = false;
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const synth_status_t                  status = synth::qwen3tts::create_x_vector_profile(
        fixture.hparams, fixture.weights, one_second_of_speech(), "", "", 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(profile == nullptr);
    return 0;
}

// The success path: the Profile is fixed to CloneMode::XVector, carries the
// package's declared enc_dim width, a positive ref_rms, and the caller's
// language tag verbatim.
int test_a_prepared_profile_carries_the_declared_width() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const synth_status_t                  status = synth::qwen3tts::create_x_vector_profile(
        fixture.hparams, fixture.weights, one_second_of_speech(), "", "", 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_OK);
    SYNTH_TEST_CHECK(profile != nullptr);
    SYNTH_TEST_CHECK(code == nullptr);
    SYNTH_TEST_CHECK(message == nullptr);
    SYNTH_TEST_CHECK(profile->x_vector.size() == fixture.hparams.speaker_encoder.enc_dim);
    SYNTH_TEST_CHECK(profile->mode == CloneMode::XVector);
    SYNTH_TEST_CHECK(profile->ref_rms > 0.0f);
    SYNTH_TEST_CHECK(profile->language_tag.empty());
    return 0;
}

// language_tag is stored verbatim, not validated or normalized -- that is
// this function's caller's job (Task 9's dispatch, mirroring how
// voice-profile.cpp validates a reference's language tag before ever
// reaching omnivoice::create_clone_prompt).
int test_a_language_tag_is_stored_verbatim() {
    Fixture fixture;
    SYNTH_TEST_CHECK(build_fixture(fixture));
    const char *                          code    = nullptr;
    const char *                          message = nullptr;
    std::shared_ptr<const XVectorProfile> profile;
    const synth_status_t                  status = synth::qwen3tts::create_x_vector_profile(
        fixture.hparams, fixture.weights, one_second_of_speech(), "", "en-US", 1, profile, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_OK);
    SYNTH_TEST_CHECK(profile != nullptr);
    SYNTH_TEST_CHECK(profile->language_tag == "en-US");
    return 0;
}

// =============================================================================
// Task 8: the v1 Serialized Profile envelope -- round trip and tamper matrix.
//
// Unlike tests/omnivoice_serialize_test.cpp's own tamper matrix, every arm
// below drives arch/qwen3-tts/profile.cpp's own
// serialize_x_vector_profile/load_profile_from_memory DIRECTLY, never a
// public synth_voice_profile_serialize/synth_voice_profile_load_from_memory
// seam: Task 9 is what opens that seam for this family, and this task's own
// brief is explicit that it "does not open the public seam." There is also
// no synthetic-package harness for this family yet (unlike OmniVoice's
// tests/omnivoice_synthetic_package.h), so every payload here is hand-built,
// the same "no real forward pass, no real Model" style
// tests/omnivoice_serialize_test.cpp's own header comment describes for its
// OWN family-level fixtures.
//
// The byte-level surgery helpers below (find_u8_32_value/find_string_value/
// find_u32_value, and the make_header/put_kv_*/common_kv_bytes hand-built-
// buffer builders) are this file's own second, independent transcription of
// arch/qwen3-tts/profile.cpp's own on-disk encoding -- the same tolerance
// tests/omnivoice_serialize_test.cpp's own header comment already accepts
// for a small, stable helper duplicated for a different purpose (constructing
// adversarial buffers, never deciding what the loader does with them: every
// actual PASS/FAIL verdict below still runs through the real
// load_profile_from_memory).
// =============================================================================

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

// Locates the byte OFFSET of a 32-element uint8 array KV's own VALUE bytes,
// by searching for that entry's on-disk encoding prefix -- a test-side
// mirror of profile.cpp's own find_u8_32_value_offset, used here only to
// locate fields to tamper with.
bool find_u8_32_value(const std::vector<uint8_t> & bytes, const std::string & key, size_t & out_offset) {
    std::vector<uint8_t> needle;
    put<uint64_t>(needle, uint64_t(key.size()));
    put_bytes(needle, key.data(), key.size());
    put<int32_t>(needle, int32_t(GGUF_TYPE_ARRAY));
    put<int32_t>(needle, int32_t(GGUF_TYPE_UINT8));
    put<uint64_t>(needle, uint64_t(32));
    const auto found = std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
    if (found == bytes.end()) {
        return false;
    }
    const size_t offset = size_t(found - bytes.begin()) + needle.size();
    if (offset + 32 > bytes.size()) {
        return false;
    }
    out_offset = offset;
    return true;
}

bool find_string_value(const std::vector<uint8_t> & bytes,
                       const std::string &          key,
                       size_t &                     out_offset,
                       size_t &                     out_length) {
    std::vector<uint8_t> needle;
    put<uint64_t>(needle, uint64_t(key.size()));
    put_bytes(needle, key.data(), key.size());
    put<int32_t>(needle, int32_t(GGUF_TYPE_STRING));
    const auto found = std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
    if (found == bytes.end()) {
        return false;
    }
    size_t offset = size_t(found - bytes.begin()) + needle.size();
    if (offset + sizeof(uint64_t) > bytes.size()) {
        return false;
    }
    uint64_t length = 0;
    std::memcpy(&length, bytes.data() + offset, sizeof(length));
    offset += sizeof(length);
    if (offset + length > bytes.size()) {
        return false;
    }
    out_offset = offset;
    out_length = size_t(length);
    return true;
}

bool find_u32_value(const std::vector<uint8_t> & bytes, const std::string & key, size_t & out_offset) {
    std::vector<uint8_t> needle;
    put<uint64_t>(needle, uint64_t(key.size()));
    put_bytes(needle, key.data(), key.size());
    put<int32_t>(needle, int32_t(GGUF_TYPE_UINT32));
    const auto found = std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
    if (found == bytes.end()) {
        return false;
    }
    const size_t offset = size_t(found - bytes.begin()) + needle.size();
    if (offset + sizeof(uint32_t) > bytes.size()) {
        return false;
    }
    out_offset = offset;
    return true;
}

// ---------------------------------------------------------------------------
// Hand-built raw GGUF buffers, for the structural tamper arms (an unknown
// key, a known key with the wrong type, an unrecognized `kind`) that a
// simple byte-flip of a valid envelope cannot produce cleanly: `common_kv_bytes`
// transcribes set_common_metadata's own 8-key order (profile.cpp), NOT
// guessed -- the same discipline tests/omnivoice_serialize_test.cpp's own
// common_kv_bytes holds itself to.
// ---------------------------------------------------------------------------

std::vector<uint8_t> make_header(int64_t n_tensors, int64_t n_kv) {
    std::vector<uint8_t> bytes;
    put_bytes(bytes, GGUF_MAGIC, 4);
    put<uint32_t>(bytes, uint32_t(GGUF_VERSION));
    put<int64_t>(bytes, n_tensors);
    put<int64_t>(bytes, n_kv);
    return bytes;
}

void put_kv_string(std::vector<uint8_t> & out, const std::string & key, const std::string & value) {
    put_gguf_string(out, key);
    put<int32_t>(out, int32_t(GGUF_TYPE_STRING));
    put_gguf_string(out, value);
}

void put_kv_u32(std::vector<uint8_t> & out, const std::string & key, uint32_t value) {
    put_gguf_string(out, key);
    put<int32_t>(out, int32_t(GGUF_TYPE_UINT32));
    put<uint32_t>(out, value);
}

void put_kv_f32(std::vector<uint8_t> & out, const std::string & key, float value) {
    put_gguf_string(out, key);
    put<int32_t>(out, int32_t(GGUF_TYPE_FLOAT32));
    put<float>(out, value);
}

void put_kv_u8_array32(std::vector<uint8_t> & out, const std::string & key) {
    put_gguf_string(out, key);
    put<int32_t>(out, int32_t(GGUF_TYPE_ARRAY));
    put<int32_t>(out, int32_t(GGUF_TYPE_UINT8));
    put<uint64_t>(out, uint64_t(32));
    const std::vector<uint8_t> value(32, 0);
    put_bytes(out, value.data(), value.size());
}

// The 8 metadata keys set_common_metadata (arch/qwen3-tts/profile.cpp)
// writes for every envelope, in that function's own order, ending with
// `kind` itself.
std::vector<uint8_t> common_kv_bytes(const std::string & kind) {
    std::vector<uint8_t> bytes;
    put_kv_string(bytes, "general.architecture", "synthprofile");
    put_kv_u32(bytes, "synthesize.voice_profile.format_version", 1);
    put_kv_string(bytes, "synthesize.voice_profile.model_family", "qwen3-tts");
    put_kv_string(bytes, "synthesize.voice_profile.schema", "qwen3-tts-voice-clone");
    put_kv_u32(bytes, "synthesize.voice_profile.schema_version", 1);
    put_kv_u8_array32(bytes, "synthesize.voice_profile.compatibility_id");
    put_kv_u8_array32(bytes, "synthesize.voice_profile.content_sha256");
    put_kv_string(bytes, "synthesize.voice_profile.kind", kind);
    return bytes;
}

// Appends a well-formed "profile.x_vector" tensor-info entry (the writer's
// own exact shape: n_dims=1, type F32, offset 0) plus zero-filled payload
// bytes and alignment padding, onto a buffer whose header already declares
// n_tensors=1.
void append_x_vector_tensor(std::vector<uint8_t> & bytes, int64_t element_count) {
    put_gguf_string(bytes, "profile.x_vector");
    put<uint32_t>(bytes, uint32_t(1));  // n_dims
    put<int64_t>(bytes, element_count);
    put<int32_t>(bytes, int32_t(GGML_TYPE_F32));
    put<uint64_t>(bytes, uint64_t(0));  // offset 0, the only tensor
    pad_to_alignment(bytes, GGUF_DEFAULT_ALIGNMENT);
    const std::vector<uint8_t> payload(size_t(element_count) * sizeof(float), 0);
    put_bytes(bytes, payload.data(), payload.size());
    pad_to_alignment(bytes, GGUF_DEFAULT_ALIGNMENT);
}

// Assembles a complete, hand-built envelope from a raw KV byte blob (built
// from common_kv_bytes plus whatever else a given arm appends) and the
// x-vector's own element count. The content_sha256 stored inside `kv` is
// whatever put_kv_u8_array32 wrote (32 zero bytes) -- deliberately never a
// real digest, since every structural arm below is rejected before
// load_profile_from_memory's own digest check ever runs.
std::vector<uint8_t> assemble_hand_built(const std::vector<uint8_t> & kv, int64_t n_kv, int64_t element_count) {
    std::vector<uint8_t> bytes = make_header(/*n_tensors=*/1, n_kv);
    put_bytes(bytes, kv.data(), kv.size());
    append_x_vector_tensor(bytes, element_count);
    return bytes;
}

void fill_compatibility_id(uint8_t (&id)[32]) {
    // Arbitrary but non-zero, so a reviewer scanning a hex dump never
    // mistakes this for an all-zero placeholder that was never set.
    for (size_t index = 0; index < sizeof(id); ++index) {
        id[index] = uint8_t(index + 7);
    }
}

std::shared_ptr<XVectorProfile> make_serializable_profile(uint32_t            enc_dim,
                                                          float               ref_rms,
                                                          const std::string & language_tag) {
    auto profile  = std::make_shared<XVectorProfile>();
    profile->mode = CloneMode::XVector;
    LcgStream stream(kSeed + 900);
    profile->x_vector     = stream.fill(enc_dim, 1.0f);
    profile->ref_rms      = ref_rms;
    profile->language_tag = language_tag;
    return profile;
}

// Builds a real, well-formed round-trippable envelope through the REAL
// writer -- the base every byte-flip tamper arm below starts from.
std::vector<uint8_t> build_valid_bytes(uint32_t enc_dim, const uint8_t (&compatibility_id)[32]) {
    const std::shared_ptr<XVectorProfile> profile = make_serializable_profile(enc_dim, 0.5f, "en-US");
    std::vector<uint8_t>                  bytes;
    if (synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes) != SYNTH_OK) {
        return {};
    }
    return bytes;
}

// --- Round trip: a hand-built payload -> the real writer -> the real
// reader -> every field matches.
int test_round_trip_of_a_well_formed_profile() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::shared_ptr<XVectorProfile> profile = make_serializable_profile(kProfileEncDim, 0.5f, "en-US");

    std::vector<uint8_t> bytes;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes) == SYNTH_OK);
    SYNTH_TEST_CHECK(!bytes.empty());

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_OK);
    SYNTH_TEST_CHECK(family_tag == synth::ProfileFamilyTag::Qwen3TtsClone);
    SYNTH_TEST_CHECK(payload != nullptr);
    SYNTH_TEST_CHECK(code == nullptr);
    SYNTH_TEST_CHECK(message == nullptr);

    const auto * reloaded = static_cast<const XVectorProfile *>(payload.get());
    SYNTH_TEST_CHECK(reloaded->mode == CloneMode::XVector);
    SYNTH_TEST_CHECK(reloaded->x_vector == profile->x_vector);
    SYNTH_TEST_CHECK(reloaded->ref_rms == profile->ref_rms);
    SYNTH_TEST_CHECK(reloaded->language_tag == profile->language_tag);
    return 0;
}

// --- Determinism: two serialize calls over the identical payload produce
// byte-identical output.
int test_serialize_is_deterministic() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::shared_ptr<XVectorProfile> profile = make_serializable_profile(kProfileEncDim, 0.5f, "en-US");
    std::vector<uint8_t>                  first, second;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, first) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, second) == SYNTH_OK);
    SYNTH_TEST_CHECK(first == second);
    return 0;
}

// --- `model_family` is "omnivoice" -> UNSUPPORTED_VOICE: another family's
// envelope is "not mine", not "corrupt". "omnivoice" and "qwen3-tts" are
// both exactly 9 bytes, so the field is overwritten with the literal string
// the row names rather than an arbitrary XOR-garbled one.
int test_wrong_model_family_is_unsupported_voice() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());

    size_t            offset = 0, length = 0;
    const std::string replacement = "omnivoice";
    SYNTH_TEST_CHECK(find_string_value(bytes, "synthesize.voice_profile.model_family", offset, length));
    SYNTH_TEST_CHECK(length == replacement.size());
    std::memcpy(bytes.data() + offset, replacement.data(), replacement.size());

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- `schema` is a different string -> UNSUPPORTED_VOICE: a future/other
// schema this build cannot read. A single-byte flip is enough (any value
// other than the exact expected string trips the same check the row names);
// tests/omnivoice_serialize_test.cpp's own schema arm uses the identical
// technique.
int test_wrong_schema_is_unsupported_voice() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());

    size_t offset = 0, length = 0;
    SYNTH_TEST_CHECK(find_string_value(bytes, "synthesize.voice_profile.schema", offset, length));
    SYNTH_TEST_CHECK(length > 0);
    bytes[offset] ^= 0xFF;

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- `schema_version` is 2 -> UNSUPPORTED_VOICE: a future version this
// build cannot read.
int test_wrong_schema_version_is_unsupported_voice() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());

    size_t offset = 0;
    SYNTH_TEST_CHECK(find_u32_value(bytes, "synthesize.voice_profile.schema_version", offset));
    const uint32_t future_version = 2;
    std::memcpy(bytes.data() + offset, &future_version, sizeof(future_version));

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- `kind` is "icl" -> INVALID_ARG, deliberately not UNSUPPORTED_VOICE: an
// unrecognized kind inside a schema this build owns is malformed, the exact
// case Plan 3 turns into a success. "icl" (3 bytes) is shorter than
// "x-vector" (8 bytes), so an in-place byte patch of a valid envelope cannot
// produce it without breaking every offset downstream -- this envelope is
// hand-built instead. Everything past the `kind` check (compatibility_id,
// content_sha256, the tensor payload) is checked LATER than `kind` in
// load_profile_from_memory's own order, so it never needs to be genuinely
// valid for this arm: reaching INVALID_ARG here has to be the `kind` check,
// not anything downstream of it.
int test_unrecognized_kind_is_invalid_arg() {
    constexpr int64_t    kProfileEncDim = 11;
    std::vector<uint8_t> kv             = common_kv_bytes("icl");
    put_kv_f32(kv, "synthesize.voice_profile.ref_rms", 0.5f);
    put_kv_string(kv, "synthesize.voice_profile.language_tag", "en");
    const std::vector<uint8_t> bytes = assemble_hand_built(kv, kPrescanKvCountXVector, kProfileEncDim);

    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        uint32_t(kProfileEncDim), bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- compatibility id one byte off -> UNSUPPORTED_VOICE: prepared for a
// different package.
int test_wrong_compatibility_id_is_unsupported_voice() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());

    size_t offset = 0;
    SYNTH_TEST_CHECK(find_u8_32_value(bytes, "synthesize.voice_profile.compatibility_id", offset));
    bytes[offset] ^= 0xFF;

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    // The expected compatibility_id passed here is the ORIGINAL, unchanged
    // one -- only the envelope's own embedded copy was tampered with.
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_UNSUPPORTED_VOICE);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- content_sha256 one byte off -> INVALID_ARG: the bytes were altered.
int test_tampered_content_sha256_is_invalid_arg() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());

    size_t offset = 0;
    SYNTH_TEST_CHECK(find_u8_32_value(bytes, "synthesize.voice_profile.content_sha256", offset));
    bytes[offset] ^= 0xFF;

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- Truncated buffer -> INVALID_ARG.
int test_truncated_buffer_is_invalid_arg() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());
    const std::vector<uint8_t> truncated(bytes.begin(), bytes.begin() + long(bytes.size() / 2));

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, truncated.data(), truncated.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- Alignment not GGUF_DEFAULT_ALIGNMENT -> INVALID_ARG.
//
// The mechanism here is NOT load_profile_from_memory's own post-parse
// `gguf_get_alignment(g) != GGUF_DEFAULT_ALIGNMENT` comparison: ggml only
// ever sets a non-default alignment from a "general.alignment" metadata KV
// (ggml/src/gguf.cpp), and that key is never a member of kPrescanKnownKeys
// (this writer never emits it), so prescan_buffer's own unknown-key
// whitelist rejects any buffer that declares it before gguf_init_from_buffer
// -- and therefore that post-parse comparison -- ever runs. This arm still
// proves the row's own black-box contract (a declared non-default alignment
// is refused with INVALID_ARG); it is caught one layer earlier than the
// specific comparison that would otherwise enforce it, exactly the "no
// special case needed" property profile.cpp's own prescan_buffer header
// comment claims for the identical "general.alignment" hole in ggml's own
// eager reader.
int test_non_default_alignment_is_invalid_arg() {
    constexpr int64_t    kProfileEncDim = 11;
    std::vector<uint8_t> kv             = common_kv_bytes("x-vector");
    put_kv_f32(kv, "synthesize.voice_profile.ref_rms", 0.5f);
    put_kv_u32(kv, "general.alignment", 64);  // in place of language_tag, keeping n_kv == 10
    const std::vector<uint8_t> bytes = assemble_hand_built(kv, kPrescanKvCountXVector, kProfileEncDim);

    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        uint32_t(kProfileEncDim), bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- Tensor count 0 -> INVALID_ARG: exactly one x-vector tensor. Patches
// the header's own n_tensors field (byte offset 8: magic[4] + version[4]) on
// a real, otherwise-valid envelope.
int test_tensor_count_zero_is_invalid_arg() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(bytes.size() > 16);
    const int64_t zero_tensors = 0;
    std::memcpy(bytes.data() + 8, &zero_tensors, sizeof(zero_tensors));

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- Tensor count 2 -> INVALID_ARG: same field, the other direction.
int test_tensor_count_two_is_invalid_arg() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(bytes.size() > 16);
    const int64_t two_tensors = 2;
    std::memcpy(bytes.data() + 8, &two_tensors, sizeof(two_tensors));

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- x-vector length != enc_dim -> INVALID_ARG: a Profile of the wrong
// width builds a wrong-shaped prompt. The envelope itself is genuinely
// valid (built by the real writer for width 11); only the `enc_dim` the
// CALLER declares differs, exactly the way a caller would present a
// serialized Profile from one package to a Model expecting a different
// speaker-encoder width.
int test_x_vector_length_mismatch_is_invalid_arg() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(!bytes.empty());

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim + 1, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- A key outside the prescan whitelist -> INVALID_ARG: refused before
// gguf_init_from_buffer runs. "language_tag" is replaced with a bogus key,
// keeping n_kv == 10.
int test_key_outside_whitelist_is_invalid_arg() {
    constexpr int64_t    kProfileEncDim = 11;
    std::vector<uint8_t> kv             = common_kv_bytes("x-vector");
    put_kv_f32(kv, "synthesize.voice_profile.ref_rms", 0.5f);
    put_kv_string(kv, "synthesize.voice_profile.bogus", "z");
    const std::vector<uint8_t> bytes = assemble_hand_built(kv, kPrescanKvCountXVector, kProfileEncDim);

    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        uint32_t(kProfileEncDim), bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- A whitelisted key with the wrong gguf type -> INVALID_ARG: `ref_rms`
// (FLOAT32 in kPrescanKnownKeys) declared as UINT32 instead.
int test_whitelisted_key_wrong_type_is_invalid_arg() {
    constexpr int64_t    kProfileEncDim = 11;
    std::vector<uint8_t> kv             = common_kv_bytes("x-vector");
    put_kv_u32(kv, "synthesize.voice_profile.ref_rms", 42);  // wrong type: should be FLOAT32
    put_kv_string(kv, "synthesize.voice_profile.language_tag", "en");
    const std::vector<uint8_t> bytes = assemble_hand_built(kv, kPrescanKvCountXVector, kProfileEncDim);

    uint8_t compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        uint32_t(kProfileEncDim), bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// --- n_kv other than the pinned count -> INVALID_ARG. Patches the header's
// own n_kv field (byte offset 16: magic[4] + version[4] + n_tensors[8]) on a
// real, otherwise-valid envelope to one less than kPrescanKvCountXVector. As
// written, prescan_buffer's own KV-count check rejects this immediately,
// before a single KV entry is read. Mutation-tested by temporarily
// disabling just that comparison: the walk still ends up rejected even
// then, because trusting a too-small n_kv stops the KV loop one entry
// early and misreads the real "language_tag" KV bytes that follow as the
// start of the tensor-info section, tripping the tensor-name check instead
// (this exact buffer's own KV entries are otherwise perfectly well-formed,
// so nothing else fires first). The two checks overlap on THIS buffer;
// they do not overlap in general -- a buffer whose entries are still
// well-formed one-for-one with a legitimately different `kind` shape,
// which this format cannot describe until Plan 3 adds a second n_kv value,
// would only be caught by the count check itself.
int test_wrong_n_kv_count_is_invalid_arg() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    std::vector<uint8_t> bytes = build_valid_bytes(kProfileEncDim, compatibility_id);
    SYNTH_TEST_CHECK(bytes.size() > 24);
    const int64_t wrong_n_kv = kPrescanKvCountXVector - 1;
    std::memcpy(bytes.data() + 16, &wrong_n_kv, sizeof(wrong_n_kv));

    synth::ProfileFamilyTag     family_tag = synth::ProfileFamilyTag::None;
    std::shared_ptr<const void> payload;
    const char *                code    = nullptr;
    const char *                message = nullptr;
    const synth_status_t        status  = synth::qwen3tts::load_profile_from_memory(
        kProfileEncDim, bytes.data(), bytes.size(), compatibility_id, family_tag, payload, code, message);
    SYNTH_TEST_CHECK(status == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(payload == nullptr);
    return 0;
}

// =============================================================================
// Writer-agreement test (the shape of
// tests/omnivoice_serialize_writer_agreement_test.cpp): pins the REAL
// serialize_x_vector_profile's emitted key set/count against the REAL
// kPrescanKnownKeys whitelist (profile.h), independent of profile.cpp's own
// hand-rolled prescan_buffer walk -- parsed here with ggml's own
// gguf_init_from_buffer instead. A key REMOVED from the writer while left in
// kPrescanKnownKeys is a silently too-permissive whitelist that nothing else
// in this file would notice: every tamper arm above starts from either a
// real writer round trip or a hand-built buffer that already targets a
// SPECIFIC known key, so none of them would catch a whitelist entry the
// writer stopped emitting.
// =============================================================================

int test_writer_emits_exactly_the_whitelisted_keys() {
    constexpr uint32_t kProfileEncDim = 11;
    uint8_t            compatibility_id[32];
    fill_compatibility_id(compatibility_id);
    const std::shared_ptr<XVectorProfile> profile = make_serializable_profile(kProfileEncDim, 0.5f, "en-US");
    std::vector<uint8_t>                  bytes;
    SYNTH_TEST_CHECK(synth::qwen3tts::serialize_x_vector_profile(*profile, compatibility_id, bytes) == SYNTH_OK);
    SYNTH_TEST_CHECK(!bytes.empty());

    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = nullptr;
    gguf_context * ctx   = gguf_init_from_buffer(bytes.data(), bytes.size(), init_params);
    SYNTH_TEST_CHECK(ctx != nullptr);
    std::set<std::string> emitted_keys;
    const int64_t         n_kv = gguf_get_n_kv(ctx);
    for (int64_t index = 0; index < n_kv; ++index) {
        emitted_keys.insert(gguf_get_key(ctx, index));
    }
    gguf_free(ctx);

    // Every kPrescanKnownKeys entry applies to the "x-vector" kind (Plan 2
    // has no other kind yet), filtered by `scope` rather than re-typed as
    // fresh string literals here -- the same discipline
    // tests/omnivoice_serialize_writer_agreement_test.cpp's own
    // expected_keys_for holds itself to.
    std::set<std::string> expected_keys;
    for (size_t index = 0; index < kPrescanKnownKeyCount; ++index) {
        const auto & spec = kPrescanKnownKeys[index];
        if (spec.scope == PrescanKeyScope::kCommon || spec.scope == PrescanKeyScope::kXVectorOnly) {
            expected_keys.insert(spec.key);
        }
    }

    SYNTH_TEST_CHECK(emitted_keys == expected_keys);
    SYNTH_TEST_CHECK(int64_t(emitted_keys.size()) == kPrescanKvCountXVector);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(test_a_transcript_is_rejected_not_ignored() == 0);
    SYNTH_TEST_CHECK(test_a_whitespace_transcript_is_rejected_too() == 0);
    SYNTH_TEST_CHECK(test_a_silent_reference_is_refused_by_name() == 0);
    SYNTH_TEST_CHECK(test_a_package_with_no_speaker_encoder_is_refused() == 0);
    SYNTH_TEST_CHECK(test_a_prepared_profile_carries_the_declared_width() == 0);
    SYNTH_TEST_CHECK(test_a_language_tag_is_stored_verbatim() == 0);

    SYNTH_TEST_CHECK(test_round_trip_of_a_well_formed_profile() == 0);
    SYNTH_TEST_CHECK(test_serialize_is_deterministic() == 0);
    SYNTH_TEST_CHECK(test_wrong_model_family_is_unsupported_voice() == 0);
    SYNTH_TEST_CHECK(test_wrong_schema_is_unsupported_voice() == 0);
    SYNTH_TEST_CHECK(test_wrong_schema_version_is_unsupported_voice() == 0);
    SYNTH_TEST_CHECK(test_unrecognized_kind_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_wrong_compatibility_id_is_unsupported_voice() == 0);
    SYNTH_TEST_CHECK(test_tampered_content_sha256_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_truncated_buffer_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_non_default_alignment_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_tensor_count_zero_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_tensor_count_two_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_x_vector_length_mismatch_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_key_outside_whitelist_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_whitelisted_key_wrong_type_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_wrong_n_kv_count_is_invalid_arg() == 0);
    SYNTH_TEST_CHECK(test_writer_emits_exactly_the_whitelisted_keys() == 0);

    std::printf("qwen3-tts-profile: all checks passed\n");
    return 0;
}
