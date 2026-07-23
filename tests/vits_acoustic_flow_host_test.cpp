#include "arch/vits/acoustic-flow-host.h"
#include "arch/vits/vits.h"
#include "test-assert.h"

#include <limits>
#include <vector>

int main() {
    synth::vits::LatentSamplingOutput latent;
    latent.channels    = 2;
    latent.frame_count = 3;
    latent.z_p         = { 1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f };

    synth::vits::PreparedAcousticFlowInput prepared;
    SYNTH_TEST_CHECK(synth::vits::prepare_acoustic_flow_input(latent, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.channels == 2 && prepared.frame_count == 3);
    SYNTH_TEST_CHECK(prepared.z_p_channel_fastest == std::vector<float>({ 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f }));

    synth::vits::AcousticFlowOutput output;
    SYNTH_TEST_CHECK(synth::vits::finalize_acoustic_flow_output(2, 3, { -9.0f, 9.0f, -8.0f, 19.0f, -7.0f, 29.0f },
                                                                output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.channels == 2 && output.frame_count == 3);
    SYNTH_TEST_CHECK(output.z == std::vector<float>({ -9.0f, -8.0f, -7.0f, 9.0f, 19.0f, 29.0f }));

    synth::vits::LatentSamplingOutput invalid = latent;
    invalid.z_p.pop_back();
    SYNTH_TEST_CHECK(synth::vits::prepare_acoustic_flow_input(invalid, prepared) == SYNTH_ERR_INVALID_ARG);
    invalid        = latent;
    invalid.z_p[1] = std::numeric_limits<float>::infinity();
    SYNTH_TEST_CHECK(synth::vits::prepare_acoustic_flow_input(invalid, prepared) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::finalize_acoustic_flow_output(0, 3, {}, output) == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(synth::vits::finalize_acoustic_flow_output(2, 3, { 1.0f }, output) == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(output.z.empty());
    return 0;
}
