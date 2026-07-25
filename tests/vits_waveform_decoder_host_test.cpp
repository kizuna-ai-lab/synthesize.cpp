#include "arch/vits/vits.h"
#include "arch/vits/waveform-decoder-host.h"
#include "test-assert.h"

#include <limits>
#include <vector>

int main() {
    synth::vits::AcousticFlowOutput flow;
    flow.channels    = 2;
    flow.frame_count = 3;
    flow.z           = { 1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f };

    synth::vits::PreparedWaveformDecoderInput prepared;
    SYNTH_TEST_CHECK(synth::vits::prepare_waveform_decoder_input(flow, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.channels == 2 && prepared.frame_count == 3);
    SYNTH_TEST_CHECK(prepared.z_channel_fastest == std::vector<float>({ 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f }));

    synth::vits::WaveformDecoderOutput output;
    SYNTH_TEST_CHECK(synth::vits::finalize_waveform_decoder_output(4, { -1.0f, -0.5f, 0.5f, 1.0f }, output) ==
                     SYNTH_OK);
    SYNTH_TEST_CHECK(output.sample_count == 4);
    SYNTH_TEST_CHECK(output.pcm == std::vector<float>({ -1.0f, -0.5f, 0.5f, 1.0f }));

    synth::vits::AcousticFlowOutput invalid = flow;
    invalid.z.pop_back();
    SYNTH_TEST_CHECK(synth::vits::prepare_waveform_decoder_input(invalid, prepared) == SYNTH_ERR_INVALID_ARG);
    invalid      = flow;
    invalid.z[0] = std::numeric_limits<float>::infinity();
    SYNTH_TEST_CHECK(synth::vits::prepare_waveform_decoder_input(invalid, prepared) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::finalize_waveform_decoder_output(0, {}, output) == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(synth::vits::finalize_waveform_decoder_output(2, { 0.0f }, output) == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(synth::vits::finalize_waveform_decoder_output(1, { std::numeric_limits<float>::quiet_NaN() },
                                                                   output) == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(output.sample_count == 0 && output.pcm.empty());
    return 0;
}
