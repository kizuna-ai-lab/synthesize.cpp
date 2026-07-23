#include "arch/vits/text-encoder-host.h"
#include "arch/vits/vits.h"
#include "arch/vits/weights.h"
#include "test-assert.h"

#include <cstdint>
#include <limits>
#include <vector>

int main() {
    synth::vits::HParams hparams;
    hparams.vocab_size            = 5;
    hparams.inter_channels        = 2;
    hparams.text_attention_window = 1;
    hparams.max_input_tokens      = 4;

    synth::vits::PreparedTextEncoderInput prepared;
    SYNTH_TEST_CHECK(synth::vits::prepare_text_encoder_input(hparams, { 0, 4 }, 1, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.token_count == 2);
    SYNTH_TEST_CHECK(prepared.relative_indices == std::vector<int32_t>({ 1, 2, 0, 1 }));

    SYNTH_TEST_CHECK(synth::vits::prepare_text_encoder_input(hparams, { 0, 1, 2 }, 2, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.relative_indices == std::vector<int32_t>({ 1, 2, 3, 0, 1, 2, 3, 0, 1 }));

    SYNTH_TEST_CHECK(synth::vits::prepare_text_encoder_input(hparams, {}, 1, prepared) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(prepared.token_count == 0 && prepared.relative_indices.empty());
    SYNTH_TEST_CHECK(synth::vits::prepare_text_encoder_input(hparams, { 0 }, 0, prepared) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::prepare_text_encoder_input(hparams, { 0 }, -1, prepared) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::prepare_text_encoder_input(hparams, { -1 }, 1, prepared) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::prepare_text_encoder_input(hparams, { 5 }, 1, prepared) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::prepare_text_encoder_input(hparams, { 0, 0, 0, 0, 0 }, 1, prepared) ==
                     SYNTH_ERR_INPUT_TOO_LONG);

    synth::vits::HParams invalid = hparams;
    invalid.vocab_size           = 0;
    SYNTH_TEST_CHECK(synth::vits::prepare_text_encoder_input(invalid, { 0 }, 1, prepared) == SYNTH_ERR_INVALID_ARG);
    invalid                       = hparams;
    invalid.text_attention_window = 0;
    SYNTH_TEST_CHECK(synth::vits::prepare_text_encoder_input(invalid, { 0 }, 1, prepared) == SYNTH_ERR_INVALID_ARG);
    invalid.text_attention_window = static_cast<uint32_t>((std::numeric_limits<int32_t>::max() - 1) / 2) + 1U;
    SYNTH_TEST_CHECK(synth::vits::prepare_text_encoder_input(invalid, { 0 }, 1, prepared) == SYNTH_ERR_INVALID_ARG);

    const std::vector<float>       m_p_channel_major  = { 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f };
    const std::vector<float>       logs_channel_major = { -1.0f, -10.0f, -2.0f, -20.0f, -3.0f, -30.0f };
    synth::vits::TextEncoderOutput output;
    SYNTH_TEST_CHECK(synth::vits::finalize_text_encoder_output(hparams, 3, m_p_channel_major, logs_channel_major,
                                                               output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.channels == 2 && output.token_count == 3);
    SYNTH_TEST_CHECK(output.m_p == std::vector<float>({ 1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f }));
    SYNTH_TEST_CHECK(output.logs_p == std::vector<float>({ -1.0f, -2.0f, -3.0f, -10.0f, -20.0f, -30.0f }));
    SYNTH_TEST_CHECK(output.mask == std::vector<float>({ 1.0f, 1.0f, 1.0f }));

    SYNTH_TEST_CHECK(synth::vits::finalize_text_encoder_output(hparams, 0, {}, {}, output) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(output.m_p.empty() && output.logs_p.empty() && output.mask.empty());
    SYNTH_TEST_CHECK(synth::vits::finalize_text_encoder_output(hparams, 3, { 1.0f }, logs_channel_major, output) ==
                     SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(output.m_p.empty() && output.logs_p.empty() && output.mask.empty());
    return 0;
}
