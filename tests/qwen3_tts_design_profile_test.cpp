// The Description Text Voice Profile payload.
//
// It holds the instruct string and nothing else, mirroring
// src/arch/omnivoice/profile.h's struct of the same name -- and deliberately
// NOT its validation. OmniVoice can reject an instruct because upstream defines
// a closed attribute vocabulary; this family's is free-form and upstream accepts
// any string, so a vocabulary invented here would reject input upstream accepts.
// Design D2.

#include "arch/qwen3-tts/profile.h"
#include "arch/qwen3-tts/weights.h"
#include "test-assert.h"

#include <string>

using synth::qwen3tts::create_design_profile;
using synth::qwen3tts::DesignInstruct;
using synth::qwen3tts::HParams;
using synth::qwen3tts::kMaxDesignInstructBytes;

namespace {

HParams voice_design_hparams() {
    HParams hparams;
    hparams.model_variant   = "qwen3-tts-12hz-1-7b-voicedesign";
    hparams.profile_sources = SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT;
    return hparams;
}

}  // namespace

int test_design_profile_holds_the_string_verbatim() {
    const HParams     hparams = voice_design_hparams();
    DesignInstruct    payload;
    const std::string instruct = "A warm, low voice, unhurried, with a slight rasp.";
    SYNTH_TEST_CHECK(create_design_profile(hparams, instruct, payload) == SYNTH_OK);
    // VERBATIM: no canonicalisation, no trimming, no reordering. Upstream
    // tokenizes whatever it is given, so anything done here would be a
    // difference from upstream that no tolerance would show.
    SYNTH_TEST_CHECK(payload.instruct == instruct);
    return 0;
}

int test_an_empty_instruct_is_accepted() {
    // Design D3: upstream's `instruct_ids.append(None)` path. An empty
    // description is a legal request for an unconditioned voice, not an error.
    const HParams  hparams = voice_design_hparams();
    DesignInstruct payload;
    SYNTH_TEST_CHECK(create_design_profile(hparams, "", payload) == SYNTH_OK);
    SYNTH_TEST_CHECK(payload.instruct.empty());
    return 0;
}

int test_invalid_utf8_is_refused() {
    const HParams  hparams = voice_design_hparams();
    DesignInstruct payload;
    // A lone continuation byte, and a truncated three-byte sequence.
    for (const std::string bad : { std::string("\x80"), std::string("\xE2\x82") }) {
        payload.instruct = "untouched";
        SYNTH_TEST_CHECK(create_design_profile(hparams, bad, payload) == SYNTH_ERR_INVALID_ARG);
        // On refusal the output is left alone rather than half-written.
        SYNTH_TEST_CHECK(payload.instruct == "untouched");
    }
    // And valid multi-byte UTF-8 is NOT refused -- without this the check could
    // pass by rejecting everything non-ASCII, which would reject the Chinese and
    // Japanese this package declares.
    payload.instruct.clear();
    SYNTH_TEST_CHECK(create_design_profile(hparams, "温かく低い声、少しかすれた", payload) == SYNTH_OK);
    SYNTH_TEST_CHECK(!payload.instruct.empty());
    return 0;
}

int test_an_over_long_instruct_is_refused_at_the_boundary() {
    const HParams  hparams = voice_design_hparams();
    DesignInstruct payload;
    // Exactly at the bound is accepted; one byte over is refused. A test that
    // only checked a wildly long string would pass on an off-by-one.
    SYNTH_TEST_CHECK(create_design_profile(hparams, std::string(kMaxDesignInstructBytes, 'a'), payload) == SYNTH_OK);
    SYNTH_TEST_CHECK(create_design_profile(hparams, std::string(kMaxDesignInstructBytes + 1, 'a'), payload) ==
                     SYNTH_ERR_INVALID_ARG);
    return 0;
}

int main() {
    SYNTH_TEST_CHECK(test_design_profile_holds_the_string_verbatim() == 0);
    SYNTH_TEST_CHECK(test_an_empty_instruct_is_accepted() == 0);
    SYNTH_TEST_CHECK(test_invalid_utf8_is_refused() == 0);
    SYNTH_TEST_CHECK(test_an_over_long_instruct_is_refused_at_the_boundary() == 0);
    return 0;
}
