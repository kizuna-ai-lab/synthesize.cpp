#include "arch/vits/duration-predictor-host.h"
#include "arch/vits/vits.h"
#include "test-assert.h"

#include <limits>
#include <vector>

int main() {
    synth::vits::PreparedDurationInput prepared;
    const std::vector<float>           source = { 1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f };
    SYNTH_TEST_CHECK(synth::vits::prepare_duration_input(3, source, 0.8f, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(prepared.noise_channel_fastest == std::vector<float>({ 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f }));

    SYNTH_TEST_CHECK(synth::vits::prepare_duration_input(3, source, 0.0f, prepared) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::vits::prepare_duration_input(0, {}, 0.8f, prepared) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::prepare_duration_input(3, { 1.0f }, 0.8f, prepared) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::prepare_duration_input(3, source, -0.1f, prepared) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::prepare_duration_input(3, source, std::numeric_limits<float>::quiet_NaN(),
                                                         prepared) == SYNTH_ERR_INVALID_ARG);
    std::vector<float> nonfinite = source;
    nonfinite[4]                 = std::numeric_limits<float>::infinity();
    SYNTH_TEST_CHECK(synth::vits::prepare_duration_input(3, nonfinite, 0.8f, prepared) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(prepared.noise_channel_fastest.empty());

    synth::vits::DurationPredictorOutput output;
    SYNTH_TEST_CHECK(synth::vits::finalize_duration_output(3, { -1.0f, 0.0f, 1.0f }, output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.token_count == 3);
    SYNTH_TEST_CHECK(output.logw == std::vector<float>({ -1.0f, 0.0f, 1.0f }));
    SYNTH_TEST_CHECK(synth::vits::finalize_duration_output(0, {}, output) == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(synth::vits::finalize_duration_output(3, { 1.0f }, output) == SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(synth::vits::finalize_duration_output(1, { std::numeric_limits<float>::infinity() }, output) ==
                     SYNTH_ERR_INTERNAL);
    SYNTH_TEST_CHECK(output.logw.empty());
    return 0;
}
