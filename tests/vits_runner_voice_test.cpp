#include "test-assert.h"
#include "vits-runner-voice.h"

#include <cstdint>

int main() {
    uint32_t speaker_index = UINT32_MAX;
    SYNTH_TEST_CHECK(parse_vits_runner_speaker_index("0", speaker_index) && speaker_index == 0);
    SYNTH_TEST_CHECK(parse_vits_runner_speaker_index("108", speaker_index) && speaker_index == 108);
    SYNTH_TEST_CHECK(parse_vits_runner_speaker_index("4294967295", speaker_index) && speaker_index == UINT32_MAX);
    SYNTH_TEST_CHECK(!parse_vits_runner_speaker_index("", speaker_index));
    SYNTH_TEST_CHECK(!parse_vits_runner_speaker_index("-1", speaker_index));
    SYNTH_TEST_CHECK(!parse_vits_runner_speaker_index("4x", speaker_index));
    SYNTH_TEST_CHECK(!parse_vits_runner_speaker_index("4294967296", speaker_index));
    return 0;
}
