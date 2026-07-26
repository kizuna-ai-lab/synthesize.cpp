#include "cpu-parallelism.h"
#include "test-assert.h"

#include <thread>

#ifdef __linux__
#    include <sched.h>
#endif

// The clamp exists because GGML's barrier spins on a pause hint without yielding
// the core: asking for more threads than the process can actually run lets waiters
// starve the worker that has not arrived, and ggml_barrier deadlocks. That was the
// reason a local patch to the vendored barrier once existed. Clamping the request
// removes the cause, so this test guards the clamp rather than the barrier.
int main() {
    const int available = synth::available_cpu_parallelism();

    // Never zero: a caller uses this directly as a thread count.
    SYNTH_TEST_CHECK(available >= 1);

    // Never more than the machine has, when the machine reports anything.
    const unsigned hardware = std::thread::hardware_concurrency();
    if (hardware > 0) {
        SYNTH_TEST_CHECK(available <= static_cast<int>(hardware));
    }

    // Stable across calls; nothing here caches, so a differing second answer
    // would mean the probe itself is reading something volatile.
    SYNTH_TEST_CHECK(synth::available_cpu_parallelism() == available);

#ifdef __linux__
    // The load-bearing property: narrowing the affinity mask must narrow the
    // answer. hardware_concurrency does not do this, which is exactly the bug
    // that produced oversubscription inside containers.
    cpu_set_t original;
    CPU_ZERO(&original);
    if (sched_getaffinity(0, sizeof(original), &original) == 0 && CPU_COUNT(&original) > 1) {
        // Pin to the single lowest CPU in the current mask, so the test works
        // even when it already runs under a restricted mask.
        int first = -1;
        for (int cpu = 0; cpu < CPU_SETSIZE && first < 0; ++cpu) {
            if (CPU_ISSET(cpu, &original)) {
                first = cpu;
            }
        }
        SYNTH_TEST_CHECK(first >= 0);

        cpu_set_t single;
        CPU_ZERO(&single);
        CPU_SET(first, &single);
        SYNTH_TEST_CHECK(sched_setaffinity(0, sizeof(single), &single) == 0);
        const int pinned = synth::available_cpu_parallelism();
        SYNTH_TEST_CHECK(sched_setaffinity(0, sizeof(original), &original) == 0);

        SYNTH_TEST_CHECK(pinned == 1);
        // And the restoration worked, so later tests in the same binary are not
        // left pinned.
        SYNTH_TEST_CHECK(synth::available_cpu_parallelism() == available);
    }
#endif
    return 0;
}
