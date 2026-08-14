// The speaker-source rules a SynthesisRequest has to satisfy before the
// talker prompt is built from it -- and in particular the one this task adds:
// the transcript-assisted (ICL) fields are all-present or all-absent, never
// half.
//
// WHY THIS IS A UNIT TEST AT ALL. These rules used to sit inline in
// Model::run_synthesis, where nothing but a loaded 2.5 GB package could reach
// them (tests/qwen3_tts_xvector_length_test.cpp says so at the top of its own
// file, and pays the integration cost for the two rules it covers). The
// half-present ICL states are worse than expensive to reach that way: they
// are UNREACHABLE. src/synthesize.cpp's dispatch assigns the ICL fields as a
// group from one IclProfile, so no caller that goes through the public seam
// can construct a request carrying codes without ids, or ids without frames.
// A check no test can reach is a check that cannot fail, and this repository
// has now found fifteen of those. Hoisting the rules into
// validate_speaker_sources (qwen3-tts.h) is what makes them reachable from a
// synthetic HParams and a handful of vectors.
//
// What this file does NOT claim: that run_synthesis calls the function. That
// is tests/qwen3_tts_xvector_length_test.cpp's two cases, against the real
// package, and they still go through run_synthesis.

#include "arch/qwen3-tts/qwen3-tts.h"
#include "arch/qwen3-tts/weights.h"
#include "test-assert.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

// The production geometry's own numbers, so a mistake about which of the two
// widths a field is measured in shows up as a mismatch rather than as a
// coincidence between two equal small integers.
constexpr uint32_t kHiddenSize = 1024;
constexpr uint32_t kGroups     = 16;

synth::qwen3tts::HParams base_hparams() {
    synth::qwen3tts::HParams h;
    h.talker.hidden_size      = kHiddenSize;
    h.talker.code_group_count = kGroups;
    return h;
}

std::vector<float> x_vector_of(size_t length) {
    return std::vector<float>(length, 0.25f);
}

// `frames * kGroups` codes, GROUP-FASTEST. The VALUES are never inspected by
// validate_speaker_sources -- the per-code table bounds belong to
// talker-host.cpp's reference_is_well_formed, deliberately not duplicated
// here -- so any in-range filler does.
std::vector<int32_t> reference_grid(uint64_t frames) {
    return std::vector<int32_t>(size_t(frames) * kGroups, 1);
}

std::vector<int32_t> reference_ids(size_t count) {
    std::vector<int32_t> ids(count);
    for (size_t index = 0; index < count; ++index) {
        ids[index] = int32_t(10 + index);
    }
    return ids;
}

// THE COMPLETE ICL REQUEST every case below perturbs exactly one dimension
// of. Built once so that a case's own diff from this baseline is the whole of
// what it is testing: if the baseline itself stopped being accepted, every
// negative case below would still "pass" for the wrong reason, which is why
// the accepted case is asserted first in main().
struct IclFixture {
    std::vector<float>                x_vector = x_vector_of(kHiddenSize);
    std::vector<int32_t>              codes    = reference_grid(13);
    std::vector<int32_t>              ids      = reference_ids(30);
    synth::qwen3tts::SynthesisRequest request;

    IclFixture() {
        request.x_vector           = &x_vector;
        request.reference_codes    = &codes;
        request.reference_text_ids = &ids;
        request.reference_frames   = 13;
    }
};

}  // namespace

int main() {
    const synth::qwen3tts::HParams hparams = base_hparams();

    // --- The baseline: a complete ICL request is accepted. Asserted first,
    // because every negative case below is a one-field perturbation of it and
    // a rejected baseline would make all of them vacuous.
    {
        IclFixture fixture;
        SYNTH_TEST_CHECK(synth::qwen3tts::validate_speaker_sources(hparams, fixture.request) == SYNTH_OK);
    }

    // --- Dimension: reference_codes. Present in the baseline, absent here,
    // with the ids and the frame count left claiming a reference that has no
    // audio behind it. This is the half that would prefix the reference
    // transcript to the target text with nothing to align it against.
    {
        IclFixture fixture;
        fixture.request.reference_codes = nullptr;
        SYNTH_TEST_CHECK(synth::qwen3tts::validate_speaker_sources(hparams, fixture.request) == SYNTH_ERR_INVALID_ARG);
    }

    // --- Dimension: reference_text_ids, null. The half that would leave the
    // codec track holding the reference's frames while the text track carries
    // only the target text -- the two-track alignment moved by the whole
    // length of the missing transcript.
    {
        IclFixture fixture;
        fixture.request.reference_text_ids = nullptr;
        SYNTH_TEST_CHECK(synth::qwen3tts::validate_speaker_sources(hparams, fixture.request) == SYNTH_ERR_INVALID_ARG);
    }

    // --- Dimension: reference_text_ids, non-null but EMPTY. A separate case
    // from the null one above, and not symmetry for its own sake: a non-null
    // empty vector passes a pointer check and then produces exactly the same
    // shifted alignment the null case would. Nothing downstream refuses it --
    // build_talker_prompt requires TARGET text and never requires reference
    // text -- so if this line is the only one that can catch it, it has to be
    // its own case.
    {
        IclFixture           fixture;
        std::vector<int32_t> empty;
        fixture.request.reference_text_ids = &empty;
        SYNTH_TEST_CHECK(synth::qwen3tts::validate_speaker_sources(hparams, fixture.request) == SYNTH_ERR_INVALID_ARG);
    }

    // --- Dimension: reference_frames. Zero is the "absent" value for a
    // count, so a request carrying both grids and a zero frame count is half
    // again -- and this one is not caught by the size relation either, since
    // validate_speaker_sources deliberately leaves that to
    // reference_is_well_formed.
    {
        IclFixture fixture;
        fixture.request.reference_frames = 0;
        SYNTH_TEST_CHECK(synth::qwen3tts::validate_speaker_sources(hparams, fixture.request) == SYNTH_ERR_INVALID_ARG);
    }

    // --- The other direction of "half": fields appearing WITHOUT the codes,
    // on a request that is otherwise an ordinary x-vector one. An x-vector
    // Profile that somehow reached the seam with a frame count attached must
    // not synthesize as if the reference were merely absent.
    {
        IclFixture fixture;
        fixture.request.reference_codes    = nullptr;
        fixture.request.reference_text_ids = nullptr;
        SYNTH_TEST_CHECK(synth::qwen3tts::validate_speaker_sources(hparams, fixture.request) == SYNTH_ERR_INVALID_ARG);
    }
    {
        IclFixture fixture;
        fixture.request.reference_codes  = nullptr;
        fixture.request.reference_frames = 0;
        SYNTH_TEST_CHECK(synth::qwen3tts::validate_speaker_sources(hparams, fixture.request) == SYNTH_ERR_INVALID_ARG);
    }

    // --- Dimension: x_vector, on an otherwise complete ICL request. D5 marks
    // the speaker embedding `yes` in BOTH mode columns, so a reference with
    // no embedding beside it is not a cheaper ICL request: it is one whose
    // speaker slot still reads the inert codec_pad row the prompt writes as a
    // placeholder.
    {
        IclFixture fixture;
        fixture.request.x_vector = nullptr;
        SYNTH_TEST_CHECK(synth::qwen3tts::validate_speaker_sources(hparams, fixture.request) == SYNTH_ERR_INVALID_ARG);
    }

    // --- The two shapes that carry no reference at all, both accepted: an
    // ordinary preset-Voice request and an ordinary x-vector one. Without
    // these the whole file could pass with a function that refused
    // everything.
    {
        synth::qwen3tts::SynthesisRequest preset;
        preset.voice_id = "aiden";
        SYNTH_TEST_CHECK(synth::qwen3tts::validate_speaker_sources(hparams, preset) == SYNTH_OK);
    }
    {
        const std::vector<float>          x_vector = x_vector_of(kHiddenSize);
        synth::qwen3tts::SynthesisRequest external;
        external.x_vector = &x_vector;
        SYNTH_TEST_CHECK(synth::qwen3tts::validate_speaker_sources(hparams, external) == SYNTH_OK);
    }

    // --- The two rules this function inherited from run_synthesis when it
    // was hoisted out of it. Covered here so the hoist itself is falsifiable
    // without a 2.5 GB package; tests/qwen3_tts_xvector_length_test.cpp keeps
    // its own cases, which are what prove run_synthesis still CALLS this.
    {
        const std::vector<float>          x_vector = x_vector_of(kHiddenSize);
        synth::qwen3tts::SynthesisRequest both;
        both.voice_id = "aiden";
        both.x_vector = &x_vector;
        SYNTH_TEST_CHECK(synth::qwen3tts::validate_speaker_sources(hparams, both) == SYNTH_ERR_INVALID_ARG);
    }
    for (size_t length : { size_t(kHiddenSize) - 1, size_t(kHiddenSize) + 1 }) {
        const std::vector<float>          x_vector = x_vector_of(length);
        synth::qwen3tts::SynthesisRequest external;
        external.x_vector = &x_vector;
        SYNTH_TEST_CHECK(synth::qwen3tts::validate_speaker_sources(hparams, external) == SYNTH_ERR_INVALID_ARG);
    }

    return 0;
}
