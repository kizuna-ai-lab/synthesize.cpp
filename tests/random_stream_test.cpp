#include "random-stream.h"
#include "synthesize.h"
#include "test-assert.h"

#include <cmath>
#include <vector>

int main() {
    synth::NormalRandomStream first(42);
    synth::NormalRandomStream second(42);
    synth::NormalRandomStream different(43);
    std::vector<float>        a(257);
    std::vector<float>        b(257);
    std::vector<float>        c(257);
    first.fill(a.data(), a.size());
    second.fill(b.data(), b.size());
    different.fill(c.data(), c.size());
    SYNTH_TEST_CHECK(a == b);
    SYNTH_TEST_CHECK(a != c);
    for (float value : a) {
        SYNTH_TEST_CHECK(std::isfinite(value));
    }

    synth::NormalRandomStream split(7);
    synth::NormalRandomStream whole(7);
    std::vector<float>        split_values(5);
    std::vector<float>        whole_values(5);
    split.fill(split_values.data(), 3);
    split.fill(split_values.data() + 3, 2);
    whole.fill(whole_values.data(), whole_values.size());
    SYNTH_TEST_CHECK(split_values == whole_values);

    synth::NormalRandomStream zero(0);
    SYNTH_TEST_CHECK(std::isfinite(zero.next()));

    const uint64_t random_seed = synth::nondeterministic_seed();
    SYNTH_TEST_CHECK(random_seed != SYNTH_SEED_RANDOM);
    return 0;
}
