#include "arch/vits/duration-path.h"
#include "test-assert.h"
#include "vits-test-fixture.h"

#include <cmath>
#include <limits>
#include <vector>

int main() {
    synth::vits::HParams     hparams = synth::test::small_vits_hparams();
    const std::vector<float> logw    = { 0.0f, std::log(2.0f), std::log(0.5f) };

    synth::vits::ResolvedDurationPath output;
    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, logw, 1.0f, output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.frame_count == 4);
    SYNTH_TEST_CHECK(output.w_ceil == std::vector<float>({ 1.0f, 2.0f, 1.0f }));
    SYNTH_TEST_CHECK(output.attention == std::vector<float>({
                                             1.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             1.0f,
                                             0.0f,
                                             0.0f,
                                             1.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             1.0f,
                                         }));

    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, logw, 2.0f, output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.frame_count == 3);
    SYNTH_TEST_CHECK(output.w_ceil == std::vector<float>({ 1.0f, 1.0f, 1.0f }));

    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, logw, 0.5f, output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.frame_count == 7);
    SYNTH_TEST_CHECK(output.w_ceil == std::vector<float>({ 2.0f, 4.0f, 1.0f }));

    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, { -1000.0f }, 1.0f, output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.frame_count == 1);
    SYNTH_TEST_CHECK(output.w_ceil == std::vector<float>({ 0.0f }));
    SYNTH_TEST_CHECK(output.attention == std::vector<float>({ 0.0f }));

    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, {}, 1.0f, output) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, { 0.0f }, 0.49f, output) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, { 0.0f }, 2.01f, output) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, { 0.0f }, std::numeric_limits<float>::quiet_NaN(),
                                                        output) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, { std::numeric_limits<float>::infinity() }, 1.0f,
                                                        output) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, { 10.0f }, 1.0f, output) == SYNTH_ERR_OUTPUT_LIMIT);
    SYNTH_TEST_CHECK(output.w_ceil.empty() && output.attention.empty());
    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, { std::log(20.0f), std::log(10.0f) }, 1.0f, output) ==
                     SYNTH_ERR_OUTPUT_LIMIT);
    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, std::vector<float>(9, 0.0f), 1.0f, output) ==
                     SYNTH_ERR_INVALID_ARG);

    synth::vits::HParams huge_limit = hparams;
    huge_limit.max_output_frames    = std::numeric_limits<uint64_t>::max();
    const float above_uint64_logw   = 64.0f * std::log(2.0f) + 0.01f;
    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(huge_limit, { above_uint64_logw }, 1.0f, output) ==
                     SYNTH_ERR_OUTPUT_LIMIT);

    hparams.hop_length = 0;
    SYNTH_TEST_CHECK(synth::vits::resolve_duration_path(hparams, { 0.0f }, 1.0f, output) == SYNTH_ERR_INVALID_ARG);
    return 0;
}
