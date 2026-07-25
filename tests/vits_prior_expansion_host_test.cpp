#include "arch/vits/prior-expansion-host.h"
#include "arch/vits/vits.h"
#include "test-assert.h"

#include <limits>
#include <vector>

int main() {
    synth::vits::TextEncoderOutput text;
    text.channels    = 2;
    text.token_count = 3;
    text.m_p         = { 1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f };
    text.logs_p      = { -1.0f, -2.0f, -3.0f, -10.0f, -20.0f, -30.0f };
    text.mask        = { 1.0f, 1.0f, 1.0f };

    synth::vits::DurationOutput duration;
    duration.token_count = 3;
    duration.frame_count = 4;
    duration.attention   = {
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };

    synth::vits::PreparedPriorExpansionInput prepared;
    SYNTH_TEST_CHECK(synth::vits::prepare_prior_expansion_input(text, duration, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.channels == 2 && prepared.token_count == 3 && prepared.frame_count == 4);
    SYNTH_TEST_CHECK(prepared.m_p_channel_fastest == std::vector<float>({ 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f }));
    SYNTH_TEST_CHECK(prepared.logs_p_channel_fastest ==
                     std::vector<float>({ -1.0f, -10.0f, -2.0f, -20.0f, -3.0f, -30.0f }));

    synth::vits::PriorExpansionOutput output;
    SYNTH_TEST_CHECK(synth::vits::finalize_prior_expansion_output(
                         2, 4, { 1.0f, 10.0f, 2.0f, 20.0f, 2.0f, 20.0f, 3.0f, 30.0f },
                         { -1.0f, -10.0f, -2.0f, -20.0f, -2.0f, -20.0f, -3.0f, -30.0f }, output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.channels == 2 && output.frame_count == 4);
    SYNTH_TEST_CHECK(output.m_p == std::vector<float>({ 1.0f, 2.0f, 2.0f, 3.0f, 10.0f, 20.0f, 20.0f, 30.0f }));
    SYNTH_TEST_CHECK(output.logs_p ==
                     std::vector<float>({ -1.0f, -2.0f, -2.0f, -3.0f, -10.0f, -20.0f, -20.0f, -30.0f }));

    synth::vits::TextEncoderOutput invalid_text = text;
    invalid_text.token_count                    = 4;
    SYNTH_TEST_CHECK(synth::vits::prepare_prior_expansion_input(invalid_text, duration, prepared) ==
                     SYNTH_ERR_INVALID_ARG);
    synth::vits::DurationOutput invalid_duration = duration;
    invalid_duration.attention.pop_back();
    SYNTH_TEST_CHECK(synth::vits::prepare_prior_expansion_input(text, invalid_duration, prepared) ==
                     SYNTH_ERR_INVALID_ARG);
    invalid_text        = text;
    invalid_text.m_p[2] = std::numeric_limits<float>::infinity();
    SYNTH_TEST_CHECK(synth::vits::prepare_prior_expansion_input(invalid_text, duration, prepared) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::finalize_prior_expansion_output(0, 4, {}, {}, output) == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(synth::vits::finalize_prior_expansion_output(2, 4, { 1.0f }, { 1.0f }, output) ==
                     SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(output.m_p.empty() && output.logs_p.empty());
    return 0;
}
