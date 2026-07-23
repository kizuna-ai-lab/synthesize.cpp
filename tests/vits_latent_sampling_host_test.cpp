#include "arch/vits/latent-sampling-host.h"
#include "arch/vits/vits.h"
#include "test-assert.h"

#include <limits>
#include <vector>

int main() {
    synth::vits::PriorExpansionOutput prior;
    prior.channels                 = 2;
    prior.frame_count              = 3;
    prior.m_p                      = { 1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f };
    prior.logs_p                   = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    const std::vector<float> noise = { 2.0f, 4.0f, 6.0f, -2.0f, -4.0f, -6.0f };

    synth::vits::PreparedLatentSamplingInput prepared;
    SYNTH_TEST_CHECK(synth::vits::prepare_latent_sampling_input(prior, noise, 0.5f, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.channels == 2 && prepared.frame_count == 3 && prepared.noise_scale == 0.5f);
    SYNTH_TEST_CHECK(prepared.m_p_channel_fastest == std::vector<float>({ 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f }));
    SYNTH_TEST_CHECK(prepared.logs_p_channel_fastest == std::vector<float>({ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }));
    SYNTH_TEST_CHECK(prepared.noise_channel_fastest == std::vector<float>({ 2.0f, -2.0f, 4.0f, -4.0f, 6.0f, -6.0f }));

    synth::vits::LatentSamplingOutput output;
    SYNTH_TEST_CHECK(synth::vits::finalize_latent_sampling_output(2, 3, { 2.0f, 9.0f, 4.0f, 18.0f, 6.0f, 27.0f },
                                                                  output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.channels == 2 && output.frame_count == 3);
    SYNTH_TEST_CHECK(output.z_p == std::vector<float>({ 2.0f, 4.0f, 6.0f, 9.0f, 18.0f, 27.0f }));

    synth::vits::PriorExpansionOutput invalid_prior = prior;
    invalid_prior.m_p.pop_back();
    SYNTH_TEST_CHECK(synth::vits::prepare_latent_sampling_input(invalid_prior, noise, 0.5f, prepared) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::prepare_latent_sampling_input(prior, { 1.0f }, 0.5f, prepared) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::prepare_latent_sampling_input(prior, noise, std::numeric_limits<float>::infinity(),
                                                                prepared) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::prepare_latent_sampling_input(prior, noise, -0.1f, prepared) ==
                     SYNTH_ERR_INVALID_ARG);
    invalid_prior           = prior;
    invalid_prior.logs_p[1] = std::numeric_limits<float>::quiet_NaN();
    SYNTH_TEST_CHECK(synth::vits::prepare_latent_sampling_input(invalid_prior, noise, 0.5f, prepared) ==
                     SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::finalize_latent_sampling_output(0, 3, {}, output) == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(synth::vits::finalize_latent_sampling_output(2, 3, { 1.0f }, output) == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(output.z_p.empty());
    return 0;
}
