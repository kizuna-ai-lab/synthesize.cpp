// Serialized Voice Profiles (Plan 3's Task 16): the v1 GGUF envelope
// round-trip and tamper matrix, against the synthetic OmniVoice package. A
// real forward pass buys nothing for this file's own scope -- the envelope
// writer/reader in arch/omnivoice/profile.cpp never touches a tensor beyond
// copying `reference_tokens` verbatim -- so every ClonePrompt here is
// hand-built rather than produced by a real Model::encode_reference pass:
// "a synthetic ClonePrompt on the synthetic package" per this task's own
// brief. The model-guarded round trip against the REAL package (byte-
// identical PCM from synthesizing off the original and the reloaded
// profile) lives in tests/omnivoice_profile_test.cpp instead, where a real
// forward pass is exactly the point.
//
// Every check below drives the PUBLIC seam
// (synth_voice_profile_serialize/synth_voice_profile_load_from_memory), not
// arch/omnivoice/profile.cpp's own serialize_clone_prompt/
// load_profile_from_memory directly -- a hand-built `synth_voice_profile`
// (this file's own make_profile) is what lets a ClonePrompt payload reach
// the public serializer without a real encode chain.

#include "arch/omnivoice/omnivoice.h"
#include "arch/omnivoice/profile.h"
#include "gguf.h"
#include "model-handle.h"
#include "omnivoice_synthetic_package.h"
#include "sha256.h"
#include "synthesize.h"
#include "test-assert.h"
#include "voice-profile-handle.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using synth::omnivoice::ClonePrompt;
using synth::omnivoice::DesignInstruct;

void put_bytes(std::vector<uint8_t> & out, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    out.insert(out.end(), bytes, bytes + size);
}

template <typename T> void put(std::vector<uint8_t> & out, T value) {
    put_bytes(out, &value, sizeof(value));
}

// Test-side mirrors of arch/omnivoice/profile.cpp's own
// find_u8_32_value_offset: a second small copy (this project's own
// tolerance for one, per profile.cpp's header comment on
// is_cjk_ideograph), used only to locate specific fields to tamper with
// below -- every actual DECISION about the resulting bytes still runs
// through the public synth_voice_profile_load_from_memory seam, never
// through this file's own understanding of the format.
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

void put_gguf_string(std::vector<uint8_t> & out, const std::string & value) {
    put<uint64_t>(out, uint64_t(value.size()));
    put_bytes(out, value.data(), value.size());
}

// A minimal, hand-built raw GGUF buffer declaring exactly one metadata key,
// "general.alignment", with an arbitrary caller-chosen type tag and raw
// value bytes -- everything arch/omnivoice/profile.cpp's prescan_buffer
// exists to catch before gguf_init_from_buffer ever sees it (ggml's own
// gguf_get_val_u32, called eagerly during construction to resolve
// alignment, aborts the process if this key's stored type is not exactly
// UINT32 with one element -- gguf.cpp:610, asserted at gguf.cpp:194/:1102).
// Deliberately NOT built through synth::omnivoice::serialize_clone_prompt:
// this project's own writer never emits "general.alignment" at all, so the
// crash reproduction has to be hand-assembled the same way an adversarial
// buffer would be.
std::vector<uint8_t> make_alignment_buffer(int32_t type, const std::vector<uint8_t> & value_bytes) {
    std::vector<uint8_t> bytes;
    put_bytes(bytes, GGUF_MAGIC, 4);
    put<uint32_t>(bytes, uint32_t(GGUF_VERSION));
    put<int64_t>(bytes, int64_t(0));  // n_tensors
    put<int64_t>(bytes, int64_t(1));  // n_kv
    put_gguf_string(bytes, "general.alignment");
    put<int32_t>(bytes, type);
    put_bytes(bytes, value_bytes.data(), value_bytes.size());
    return bytes;
}

std::vector<uint8_t> u32_value_bytes(uint32_t value) {
    std::vector<uint8_t> bytes;
    put<uint32_t>(bytes, value);
    return bytes;
}

std::vector<uint8_t> string_value_bytes(const std::string & value) {
    std::vector<uint8_t> bytes;
    put_gguf_string(bytes, value);
    return bytes;
}

// ---------------------------------------------------------------------------
// Fix round 2 (reviewer's positive-validation whitelist): generic raw KV
// entry builders, for constructing buffers with a SPECIFIC key set/count/
// duplicate that doesn't correspond to any real profile -- these test
// arch/omnivoice/profile.cpp's prescan_buffer whitelist directly, through
// the public seam, the same way round 1's arms tested its predecessor.
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

void put_kv_u8_array32(std::vector<uint8_t> & out, const std::string & key) {
    put_gguf_string(out, key);
    put<int32_t>(out, int32_t(GGUF_TYPE_ARRAY));
    put<int32_t>(out, int32_t(GGUF_TYPE_UINT8));
    put<uint64_t>(out, uint64_t(32));
    const std::vector<uint8_t> value(32, 0);
    put_bytes(out, value.data(), value.size());
}

// The 8 metadata keys set_common_metadata (arch/omnivoice/profile.cpp)
// writes for EVERY kind, in that function's own order, ending with `kind`
// itself -- transcribed here, not guessed, the same standard fix round 2's
// own prescan_buffer whitelist holds itself to. A buffer built from exactly
// this plus one more (kind-appropriate) entry has the valid n_kv=9 shape
// (DesignInstruct); a valid ClonePrompt needs three more on top (n_kv=11).
std::vector<uint8_t> common_kv_bytes(const std::string & kind) {
    std::vector<uint8_t> bytes;
    put_kv_string(bytes, "general.architecture", "synthprofile");
    put_kv_u32(bytes, "synthesize.voice_profile.format_version", 1);
    put_kv_string(bytes, "synthesize.voice_profile.model_family", "omnivoice");
    put_kv_string(bytes, "synthesize.voice_profile.schema", "omnivoice-clone-prompt");
    put_kv_u32(bytes, "synthesize.voice_profile.schema_version", 1);
    put_kv_u8_array32(bytes, "synthesize.voice_profile.compatibility_id");
    put_kv_u8_array32(bytes, "synthesize.voice_profile.content_sha256");
    put_kv_string(bytes, "synthesize.voice_profile.kind", kind);
    return bytes;
}

std::shared_ptr<ClonePrompt> make_clone_prompt(uint64_t frames, int32_t fill_token, const std::string & transcript) {
    auto prompt = std::make_shared<ClonePrompt>();
    prompt->reference_tokens.assign(size_t(frames) * 8, fill_token);
    prompt->transcript_text = transcript;
    prompt->ref_rms         = 0.5f;
    prompt->language_tag    = "en";
    return prompt;
}

// A `synth_voice_profile` wrapping a hand-built payload, for driving the
// public serialize seam without a real Model::encode_reference forward
// pass. Stack-lifetime only: never pass the result to
// synth_voice_profile_free (that deletes a heap object; this is not one).
synth_voice_profile make_profile(const synth_model_t *       model,
                                 std::shared_ptr<const void> payload,
                                 synth::ProfileFamilyTag     tag) {
    synth_voice_profile profile;
    profile.model      = model;
    profile.family_tag = tag;
    profile.payload    = std::move(payload);
    return profile;
}

synth_status_t serialize_profile(const synth_voice_profile & profile, std::vector<uint8_t> & out_bytes) {
    synth_voice_profile_serialize_params_t params;
    synth_voice_profile_serialize_params_init(&params, sizeof(params));
    synth_byte_buffer_t * buffer = nullptr;
    const synth_status_t  status = synth_voice_profile_serialize(&profile, &params, &buffer);
    if (status != SYNTH_OK) {
        return status;
    }
    out_bytes.assign(buffer->data, buffer->data + buffer->data_size);
    synth_byte_buffer_free(buffer);
    return SYNTH_OK;
}

synth_status_t load_bytes(const synth_model_t *        model,
                          const std::vector<uint8_t> & bytes,
                          synth_voice_profile_t **     out_profile) {
    synth_voice_profile_load_params_t params;
    synth_voice_profile_load_params_init(&params, sizeof(params));
    params.data      = bytes.data();
    params.data_size = bytes.size();
    return synth_voice_profile_load_from_memory(model, &params, out_profile);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <scratch-dir>\n", argv[0]);
        return 2;
    }
    const std::string scratch_dir = argv[1];

    const std::string                                  synthetic_path = scratch_dir + "/omnivoice-serialize.gguf";
    synth::omnivoice::testing::SyntheticPackageOptions options;
    options.ascii_text_vocab = true;  // enough to tokenize "a" below
    SYNTH_TEST_CHECK(synth::omnivoice::testing::write_synthetic_package(synthetic_path, options));

    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof(load_params));
    load_params.backend   = SYNTH_BACKEND_CPU;
    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(synthetic_path.c_str(), &load_params, &model) == SYNTH_OK);

    // Serialized Profile is now claimed alongside REFERENCE_AUDIO/
    // DESCRIPTION_TEXT (docs/c-interface.md: "Any Loaded Model that creates
    // a Profile from Reference Audio, Description Text, or Random Seed also
    // sets SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE") -- this is the
    // "completes the staged flags" half of this task's own brief.
    {
        synth_voice_profile_capabilities_t capabilities;
        synth_voice_profile_capabilities_init(&capabilities, sizeof(capabilities));
        SYNTH_TEST_CHECK(synth_model_get_voice_profile_capabilities(model, &capabilities) == SYNTH_OK);
        SYNTH_TEST_CHECK((capabilities.source_flags & SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE) != 0);
        SYNTH_TEST_CHECK(capabilities.profile_schema != nullptr);
        SYNTH_TEST_CHECK(std::string(capabilities.profile_schema, size_t(capabilities.profile_schema_size)) ==
                         "omnivoice-clone-prompt");
        SYNTH_TEST_CHECK(capabilities.profile_schema_version == 1);
        bool any_nonzero = false;
        for (uint8_t byte : capabilities.profile_compatibility_id) {
            any_nonzero = any_nonzero || (byte != 0);
        }
        SYNTH_TEST_CHECK(any_nonzero);
    }

    const std::string kTranscript = "a";  // ascii_text_vocab's own index-4 mapping

    // --- Round trip (ClonePrompt): a hand-built payload -> public
    // serialize -> public load_from_memory -> the reloaded payload's own
    // fields match, including a transcript_ids re-tokenization that agrees
    // with tokenizing the same text directly through the family's own
    // frontend.
    std::vector<uint8_t> base_bytes;
    {
        std::shared_ptr<ClonePrompt> prompt  = make_clone_prompt(2, 1, kTranscript);
        const synth_voice_profile    profile = make_profile(model, prompt, synth::ProfileFamilyTag::OmnivoiceClone);
        SYNTH_TEST_CHECK(serialize_profile(profile, base_bytes) == SYNTH_OK);
        SYNTH_TEST_CHECK(!base_bytes.empty());

        synth_voice_profile_t * loaded = nullptr;
        SYNTH_TEST_CHECK(load_bytes(model, base_bytes, &loaded) == SYNTH_OK);
        SYNTH_TEST_CHECK(loaded != nullptr);
        SYNTH_TEST_CHECK(loaded->model == model);
        SYNTH_TEST_CHECK(loaded->family_tag == synth::ProfileFamilyTag::OmnivoiceClone);
        const auto * reloaded = static_cast<const ClonePrompt *>(loaded->payload.get());
        SYNTH_TEST_CHECK(reloaded->reference_tokens == prompt->reference_tokens);
        SYNTH_TEST_CHECK(reloaded->transcript_text == kTranscript);
        SYNTH_TEST_CHECK(reloaded->ref_rms == prompt->ref_rms);
        SYNTH_TEST_CHECK(reloaded->language_tag == "en");
        SYNTH_TEST_CHECK(!reloaded->transcript_ids.empty());

        std::unique_ptr<synth::omnivoice::Model> family_model;
        SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(synthetic_path, family_model) == SYNTH_OK);
        std::vector<int32_t> expected_ids;
        SYNTH_TEST_CHECK(family_model->text_frontend()->prepare(SYNTH_INPUT_TEXT_UTF8, kTranscript.data(),
                                                                kTranscript.size(), 0, expected_ids) == SYNTH_OK);
        SYNTH_TEST_CHECK(reloaded->transcript_ids == expected_ids);

        synth_voice_profile_free(loaded);
    }

    // --- Determinism: two serialize calls over the identical payload
    // produce byte-identical output (sha256-compared too, per this task's
    // own brief wording).
    {
        std::shared_ptr<ClonePrompt> prompt  = make_clone_prompt(2, 1, kTranscript);
        const synth_voice_profile    profile = make_profile(model, prompt, synth::ProfileFamilyTag::OmnivoiceClone);
        std::vector<uint8_t>         first, second;
        SYNTH_TEST_CHECK(serialize_profile(profile, first) == SYNTH_OK);
        SYNTH_TEST_CHECK(serialize_profile(profile, second) == SYNTH_OK);
        SYNTH_TEST_CHECK(first == second);
        uint8_t digest_a[32];
        uint8_t digest_b[32];
        synth::sha256(first.data(), first.size(), digest_a);
        synth::sha256(second.data(), second.size(), digest_b);
        SYNTH_TEST_CHECK(std::memcmp(digest_a, digest_b, sizeof(digest_a)) == 0);
    }

    // --- Round trip (DesignInstruct): the real public creation path this
    // time (nothing here needs the neural encoder either way, so there is
    // no reason to hand-build this one), through the same serialize/
    // load_from_memory pair.
    {
        synth_voice_description_params_t params;
        synth_voice_description_params_init(&params, sizeof(params));
        const char * description = "male";
        params.description       = description;
        params.description_size  = std::strlen(description);

        synth_voice_profile_t * created = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(model, &params, &created) == SYNTH_OK);

        std::vector<uint8_t> design_bytes;
        SYNTH_TEST_CHECK(serialize_profile(*created, design_bytes) == SYNTH_OK);

        synth_voice_profile_t * loaded = nullptr;
        SYNTH_TEST_CHECK(load_bytes(model, design_bytes, &loaded) == SYNTH_OK);
        SYNTH_TEST_CHECK(loaded->family_tag == synth::ProfileFamilyTag::OmnivoiceDesign);
        const auto * reloaded = static_cast<const DesignInstruct *>(loaded->payload.get());
        const auto * original = static_cast<const DesignInstruct *>(created->payload.get());
        SYNTH_TEST_CHECK(reloaded->instruct == original->instruct);

        synth_voice_profile_free(created);
        synth_voice_profile_free(loaded);
    }

    // =========================================================================
    // Tamper matrix: every arm mutates one COPY of `base_bytes` (the valid
    // ClonePrompt round trip's own output above), then drives it through
    // the PUBLIC synth_voice_profile_load_from_memory -- never the family
    // loader directly, so every arm proves what a real caller observes.
    // =========================================================================

    // --- Flip one payload (tensor) byte -> INVALID_ARG (digest mismatch).
    // base_bytes' tensor data is exactly 64 bytes (16 tokens), already a
    // multiple of the 32-byte alignment, so the file's very last byte is
    // genuine token data, not padding.
    {
        std::vector<uint8_t> tampered = base_bytes;
        tampered.back() ^= 0xFF;
        synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, tampered, &profile) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
    }

    // --- Wrong compatibility_id -> UNSUPPORTED_VOICE: a structurally sound
    // envelope this loader understands, just not for this Model.
    {
        std::vector<uint8_t> tampered = base_bytes;
        size_t               offset   = 0;
        SYNTH_TEST_CHECK(find_u8_32_value(tampered, "synthesize.voice_profile.compatibility_id", offset));
        tampered[offset] ^= 0xFF;
        synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, tampered, &profile) == SYNTH_ERR_UNSUPPORTED_VOICE);
        SYNTH_TEST_CHECK(profile == nullptr);
    }

    // --- Wrong schema -> UNSUPPORTED_VOICE.
    {
        std::vector<uint8_t> tampered = base_bytes;
        size_t               offset = 0, length = 0;
        SYNTH_TEST_CHECK(find_string_value(tampered, "synthesize.voice_profile.schema", offset, length));
        SYNTH_TEST_CHECK(length > 0);
        tampered[offset] ^= 0xFF;
        synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, tampered, &profile) == SYNTH_ERR_UNSUPPORTED_VOICE);
        SYNTH_TEST_CHECK(profile == nullptr);
    }

    // --- Wrong `kind` -> INVALID_ARG: unlike model_family/schema/
    // compatibility_id above, `kind` is this ONE schema's own internal tag
    // (arch/omnivoice/profile.h's own header comment on
    // load_profile_from_memory justifies this split) -- a value outside its
    // two-member enum means the payload structure the rest of the bytes
    // describe cannot be interpreted at all, which this project treats as
    // malformed rather than a different, merely-unsupported thing.
    {
        std::vector<uint8_t> tampered = base_bytes;
        size_t               offset = 0, length = 0;
        SYNTH_TEST_CHECK(find_string_value(tampered, "synthesize.voice_profile.kind", offset, length));
        SYNTH_TEST_CHECK(length > 0);
        tampered[offset] ^= 0xFF;
        synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, tampered, &profile) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
    }

    // --- Truncated buffer -> INVALID_ARG.
    {
        std::vector<uint8_t>    tampered(base_bytes.begin(), base_bytes.begin() + base_bytes.size() / 2);
        synth_voice_profile_t * profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, tampered, &profile) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
    }

    // --- Token outside [0, mask_id) -> INVALID_ARG. The synthetic package
    // declares audio.vocab_size=5, mask_id=4 (omnivoice_small_layout.h); a
    // digitally valid, correctly-digested envelope whose one token equals
    // the mask id is still refused, caught by the token-range check rather
    // than the digest (this envelope's digest IS valid for its own,
    // otherwise-malformed content).
    {
        std::shared_ptr<ClonePrompt> prompt  = make_clone_prompt(1, /*fill_token=*/4, kTranscript);
        const synth_voice_profile    profile = make_profile(model, prompt, synth::ProfileFamilyTag::OmnivoiceClone);
        std::vector<uint8_t>         bytes;
        SYNTH_TEST_CHECK(serialize_profile(profile, bytes) == SYNTH_OK);
        synth_voice_profile_t * loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }

    // --- Token count beyond max_total_frames -> INVALID_ARG, decided from
    // the DECLARED tensor size alone: arch/omnivoice/profile.cpp's own
    // load_profile_from_memory checks `frames > max_total_frames` using
    // only gguf_get_tensor_size (no tensor byte read) before ever building
    // the destination std::vector<int32_t> -- this envelope's digest is
    // otherwise entirely valid, so reaching INVALID_ARG here has to be that
    // check, not a corruption one.
    {
        constexpr uint64_t           kMaxTotalFrames = 3000;  // synthesize.reference.max_total_frames, small_hparams()
        std::shared_ptr<ClonePrompt> prompt          = make_clone_prompt(kMaxTotalFrames + 1, 0, kTranscript);
        const synth_voice_profile    profile = make_profile(model, prompt, synth::ProfileFamilyTag::OmnivoiceClone);
        std::vector<uint8_t>         bytes;
        SYNTH_TEST_CHECK(serialize_profile(profile, bytes) == SYNTH_OK);
        synth_voice_profile_t * loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }

    // --- Cross-family guard: a structurally valid omnivoice envelope
    // loaded against a Loaded Model of a DIFFERENT family dispatches to the
    // generic "unsupported" fallback every other family already takes for
    // create_from_reference/create_from_description, without this loader's
    // own byte parsing ever running. `fake_model` is a plain, stack-local,
    // never-through-synth_model_load `synth_model` (model-handle.h) with
    // every family pointer left null -- safe because the fallback path
    // touches only `info.family`, never `omnivoice`/`vits`/etc.
    {
        synth_model fake_model;
        fake_model.info.family         = synth::ModelFamily::Vits;
        synth_voice_profile_t * loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(&fake_model, base_bytes, &loaded) == SYNTH_ERR_UNSUPPORTED_VOICE);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }

    // =========================================================================
    // Untrusted-buffer pre-scan (fix round 1, reviewer FINDING 1): a raw
    // GGUF buffer that declares GGML's own reserved key "general.alignment"
    // with the wrong type used to abort the WHOLE PROCESS inside
    // gguf_init_from_buffer (ggml/src/gguf.cpp:610's eager
    // gguf_get_val_u32, asserted at :194/:1102) -- before
    // load_profile_from_memory's own validation ever ran.
    // arch/omnivoice/profile.cpp's prescan_buffer now catches this, and
    // every arm below, independently of gguf_init_from_buffer, and reaching
    // the SYNTH_TEST_CHECK after each `load_bytes` call is itself part of
    // the proof: a still-aborting guard would take the whole test process
    // down before ever reaching it, not just fail an assertion.
    // =========================================================================

    // --- (a) The exact wrong-typed-alignment buffer that aborted the
    // process before this fix: "general.alignment" declared as a STRING
    // instead of a UINT32.
    {
        const std::vector<uint8_t> bytes  = make_alignment_buffer(GGUF_TYPE_STRING, string_value_bytes("x"));
        synth_voice_profile_t *    loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
        // Reaching this line at all -- rather than the process having
        // already died to SIGABRT -- IS the "process survives" proof.
    }

    // --- (b) Wrong magic.
    {
        std::vector<uint8_t> bytes     = make_alignment_buffer(GGUF_TYPE_UINT32, u32_value_bytes(32));
        bytes[0]                       = uint8_t('X');
        synth_voice_profile_t * loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }

    // --- (c) Wrong version (this project only ever writes GGUF_VERSION==3;
    // an old v1/v2 writer or corrupt data is refused the same way).
    {
        std::vector<uint8_t> bytes         = make_alignment_buffer(GGUF_TYPE_UINT32, u32_value_bytes(32));
        const uint32_t       wrong_version = 2;
        std::memcpy(bytes.data() + 4, &wrong_version, sizeof(wrong_version));
        synth_voice_profile_t * loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }

    // --- (d) Truncated header: fewer than the 24 bytes
    // (magic+version+n_tensors+n_kv) prescan_buffer needs before it can
    // even look at a single KV entry.
    {
        const std::vector<uint8_t> bytes  = { 'G', 'G', 'U', 'F', 3, 0, 0, 0, 1, 2, 3 };
        synth_voice_profile_t *    loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }

    // --- (e) A KV walk that runs past the buffer end: a well-formed header
    // declaring one KV entry, truncated partway through that entry's key
    // bytes.
    {
        std::vector<uint8_t> bytes = make_alignment_buffer(GGUF_TYPE_UINT32, u32_value_bytes(32));
        // Header (24 bytes) + the key's own 8-byte length prefix + a few
        // key bytes, then stop -- well short of "general.alignment"'s full
        // 18 bytes, let alone the type tag and value that would follow.
        bytes.resize(24 + 8 + 4);
        synth_voice_profile_t * loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }

    // --- (f) Absurd kv_count: declares far more KV entries than either of
    // prescan_buffer's own two exact-count values (9 or 11), with no actual
    // entries following -- rejected from the header alone, before the walk
    // ever starts.
    {
        std::vector<uint8_t> bytes;
        put_bytes(bytes, GGUF_MAGIC, 4);
        put<uint32_t>(bytes, uint32_t(GGUF_VERSION));
        put<int64_t>(bytes, int64_t(0));        // n_tensors
        put<int64_t>(bytes, int64_t(1) << 40);  // n_kv: absurd
        synth_voice_profile_t * loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }

    // =========================================================================
    // Fix round 2 (reviewer's positive-validation whitelist): the fuzz
    // campaign that found the empty-key crash, and the arms proving
    // prescan_buffer's whitelist -- not merely "not obviously hostile" --
    // now governs acceptance. Every arm again drives the PUBLIC
    // synth_voice_profile_load_from_memory only.
    // =========================================================================

    // --- (g) Empty key: the reviewer's own minimal repro (40 bytes --
    // "GGUF" + version 3 + n_tensors 0 + n_kv 1 + key_length 0 + type
    // UINT32 + 4 value bytes) that reached gguf_kv's constructors and hit
    // GGML_ASSERT(!key.empty()) (gguf.cpp:143/151/161/167) through round
    // 1's guard, which never special-cased an EMPTY key the way it did
    // "general.alignment". Fix round 2's whitelist rejects it because "" is
    // not a member of the known key set -- no special case needed.
    {
        std::vector<uint8_t> bytes = make_header(/*n_tensors=*/0, /*n_kv=*/1);
        put<uint64_t>(bytes, uint64_t(0));   // key_length = 0: the empty key itself
        put<int32_t>(bytes, int32_t(GGUF_TYPE_UINT32));
        put<uint32_t>(bytes, uint32_t(42));  // 4 arbitrary value bytes
        SYNTH_TEST_CHECK(bytes.size() == 40);
        synth_voice_profile_t * loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
        // Reaching this line at all, rather than the process having already
        // died to SIGABRT, is itself the "process survives" proof -- same
        // as round 1's arm (a).
    }

    // --- (h) Unknown key: a well-formed DesignInstruct-shaped buffer
    // (n_kv=9, the exact count this writer's own DesignInstruct produces)
    // whose 9th entry is a key outside the known set entirely, in place of
    // "instruct".
    {
        std::vector<uint8_t> kv = common_kv_bytes("design-instruct");
        put_kv_string(kv, "synthesize.voice_profile.bogus", "x");
        const std::vector<uint8_t> bytes = [&] {
            std::vector<uint8_t> out = make_header(0, 9);
            put_bytes(out, kv.data(), kv.size());
            return out;
        }();
        synth_voice_profile_t * loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }

    // --- (i) Duplicate key: the same 9-entry shape, but the 9th entry
    // repeats "kind" (already the 8th, from common_kv_bytes) instead of
    // contributing "instruct".
    {
        std::vector<uint8_t> kv = common_kv_bytes("design-instruct");
        put_kv_string(kv, "synthesize.voice_profile.kind", "design-instruct");
        const std::vector<uint8_t> bytes = [&] {
            std::vector<uint8_t> out = make_header(0, 9);
            put_bytes(out, kv.data(), kv.size());
            return out;
        }();
        synth_voice_profile_t * loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }

    // --- (j) Wrong type for a known key: "format_version" (UINT32 in every
    // real envelope) declared as a STRING instead, in an otherwise
    // 9-entry DesignInstruct shape.
    {
        std::vector<uint8_t> kv;
        put_kv_string(kv, "general.architecture", "synthprofile");
        put_kv_string(kv, "synthesize.voice_profile.format_version", "1");  // wrong type: STRING, not UINT32
        put_kv_string(kv, "synthesize.voice_profile.model_family", "omnivoice");
        put_kv_string(kv, "synthesize.voice_profile.schema", "omnivoice-clone-prompt");
        put_kv_u32(kv, "synthesize.voice_profile.schema_version", 1);
        put_kv_u8_array32(kv, "synthesize.voice_profile.compatibility_id");
        put_kv_u8_array32(kv, "synthesize.voice_profile.content_sha256");
        put_kv_string(kv, "synthesize.voice_profile.kind", "design-instruct");
        put_kv_string(kv, "synthesize.voice_profile.instruct", "male");
        const std::vector<uint8_t> bytes = [&] {
            std::vector<uint8_t> out = make_header(0, 9);
            put_bytes(out, kv.data(), kv.size());
            return out;
        }();
        synth_voice_profile_t * loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }

    // --- (k) Bogus tensor name: an 11-entry ClonePrompt-shaped KV section
    // (common 8 plus transcript_text/ref_rms/language_tag) with one
    // declared tensor named something other than "profile.reference_tokens".
    {
        std::vector<uint8_t> kv = common_kv_bytes("clone-prompt");
        put_kv_string(kv, "synthesize.voice_profile.transcript_text", "a");
        put_kv_u32(kv, "synthesize.voice_profile.ref_rms", 0x3f000000u);  // 0.5f, bit pattern
        put_kv_string(kv, "synthesize.voice_profile.language_tag", "en");

        std::vector<uint8_t> bytes = make_header(/*n_tensors=*/1, /*n_kv=*/11);
        put_bytes(bytes, kv.data(), kv.size());
        put_gguf_string(bytes, "profile.wrong_tensor_name");
        put<uint32_t>(bytes, uint32_t(2));  // n_dims
        put<int64_t>(bytes, int64_t(1));    // ne[0]
        put<int64_t>(bytes, int64_t(8));    // ne[1]
        put<int32_t>(bytes, int32_t(GGML_TYPE_I32));
        put<uint64_t>(bytes, uint64_t(0));  // offset

        synth_voice_profile_t * loaded = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(load_bytes(model, bytes, &loaded) == SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(loaded == nullptr);
    }

    synth_model_free(model);
    return 0;
}
