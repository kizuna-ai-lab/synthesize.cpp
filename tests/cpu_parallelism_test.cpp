#include "cpu-parallelism.h"
#include "test-assert.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef __linux__
#    include <sched.h>
#endif

// parallel_for's promise is a COVER: every index in [0, count) is handed to the
// body exactly once, in some contiguous sub-range, however many threads run.
// Everything the OmniVoice decode loop relies on rests on that plus the caller's
// own discipline of writing only per-index storage, so this is what to hold.
//
// The counts below deliberately straddle the internal chunk size (32) -- a
// count under one chunk must take the inline path, a count just over it must
// not drop the short final chunk -- without the test naming that constant,
// which is an implementation detail it has no business pinning.
int check_parallel_for_covers_every_index() {
    const size_t counts[]  = { 1, 2, 31, 32, 33, 64, 65, 1000, 4097 };
    const int    workers[] = { 1, 2, 3, 8, 64 };

    for (size_t count : counts) {
        for (int worker_count : workers) {
            std::vector<int> visits(count, 0);
            std::vector<int> range_ok(count, 0);
            synth::parallel_for(count, worker_count, [&](size_t begin, size_t end) {
                // A range must be non-empty, ordered and inside the loop.
                const bool sane = begin < end && end <= count;
                for (size_t index = begin; index < end; ++index) {
                    // Distinct elements of a vector<int> are distinct memory
                    // locations, so unsynchronized writes here are exactly the
                    // access pattern the real caller uses -- if the cover is
                    // broken the count lands at 0 or 2 rather than racing.
                    visits[index] += 1;
                    range_ok[index] = sane ? 1 : 0;
                }
            });
            for (size_t index = 0; index < count; ++index) {
                SYNTH_TEST_CHECK(visits[index] == 1);
                SYNTH_TEST_CHECK(range_ok[index] == 1);
            }
        }
    }

    // Zero work calls the body not at all -- not once with an empty range.
    int calls = 0;
    synth::parallel_for(0, 8, [&](size_t, size_t) { calls += 1; });
    SYNTH_TEST_CHECK(calls == 0);

    // A single worker runs inline on the calling thread: one call, whole range,
    // no thread created. The id comparison is the part worth having -- it is
    // what lets a caller reason that workers == 1 is the serial loop itself.
    const std::thread::id        self = std::this_thread::get_id();
    std::vector<size_t>          bounds;
    std::vector<std::thread::id> ids;
    synth::parallel_for(1000, 1, [&](size_t begin, size_t end) {
        bounds.push_back(begin);
        bounds.push_back(end);
        ids.push_back(std::this_thread::get_id());
    });
    SYNTH_TEST_CHECK(bounds.size() == 2);
    SYNTH_TEST_CHECK(bounds[0] == 0 && bounds[1] == 1000);
    SYNTH_TEST_CHECK(ids.size() == 1 && ids[0] == self);

    // A non-positive worker count is a caller's clamp gone wrong, not a reason
    // to do nothing: it degrades to the serial loop rather than skipping work.
    std::vector<int> zero_visits(50, 0);
    synth::parallel_for(50, 0, [&](size_t begin, size_t end) {
        for (size_t index = begin; index < end; ++index) {
            zero_visits[index] += 1;
        }
    });
    for (int seen : zero_visits) {
        SYNTH_TEST_CHECK(seen == 1);
    }
    return 0;
}

// The property the OmniVoice scan actually needs: a body that is a pure
// function of its index produces byte-identical output whatever the worker
// count, so the parallel path can never be a different answer from the serial
// one -- only a faster one. Bytes, not values: a float comparison would accept
// a differently-rounded result, which is the failure this is here to catch.
int check_parallel_for_is_bit_identical_to_serial() {
    constexpr size_t kCount = 5000;

    // Deliberately float arithmetic with a per-index reduction, the shape of
    // the real scoring body: a reduction that leaked across indices would land
    // differently under a different split.
    const auto compute = [](size_t index) {
        double accumulator = 0.0;
        for (uint32_t step = 0; step < 64; ++step) {
            accumulator += 1.0 / double(index + step + 1);
        }
        return float(accumulator);
    };

    std::vector<float> serial(kCount, 0.0f);
    synth::parallel_for(kCount, 1, [&](size_t begin, size_t end) {
        for (size_t index = begin; index < end; ++index) {
            serial[index] = compute(index);
        }
    });

    for (int worker_count : { 2, 3, 4, 8, 20 }) {
        std::vector<float> parallel(kCount, 0.0f);
        synth::parallel_for(kCount, worker_count, [&](size_t begin, size_t end) {
            for (size_t index = begin; index < end; ++index) {
                parallel[index] = compute(index);
            }
        });
        SYNTH_TEST_CHECK(std::memcmp(serial.data(), parallel.data(), kCount * sizeof(float)) == 0);
    }
    return 0;
}

// An exception thrown inside a worker must unwind into the caller, the way the
// serial loop it replaces would. The alternative is std::terminate, which is
// what an uncaught throw on a std::thread does -- a crash in place of an error
// return, and one that only appears under allocation failure.
int check_parallel_for_propagates_an_exception() {
    std::atomic<int> entered{ 0 };
    bool             caught = false;
    try {
        synth::parallel_for(4096, 8, [&](size_t begin, size_t end) {
            entered.fetch_add(1, std::memory_order_relaxed);
            if (begin == 0) {
                throw std::runtime_error("worker failure");
            }
            (void) end;
        });
    } catch (const std::runtime_error &) {
        caught = true;
    }
    SYNTH_TEST_CHECK(caught);
    SYNTH_TEST_CHECK(entered.load() >= 1);

    // And the loop still works afterwards: nothing about the failure is sticky.
    std::vector<int> visits(100, 0);
    synth::parallel_for(100, 4, [&](size_t begin, size_t end) {
        for (size_t index = begin; index < end; ++index) {
            visits[index] += 1;
        }
    });
    for (int seen : visits) {
        SYNTH_TEST_CHECK(seen == 1);
    }
    return 0;
}

// First the two POLICY answers, then the parallel_for checks above.
//
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

    // The default a context takes is deliberately below the CPU count. Asking
    // for exactly the CPU count measured ten times slower than asking for half,
    // because GGML's barrier spins: once the workers fill every CPU, a thread
    // the scheduler moves aside stalls all the others.
    const int threads = synth::default_synthesis_threads();
    SYNTH_TEST_CHECK(threads >= 1);
    SYNTH_TEST_CHECK(threads <= available);
    if (available > 2) {
        SYNTH_TEST_CHECK(threads < available);
        SYNTH_TEST_CHECK(threads == available / 2);
    } else {
        SYNTH_TEST_CHECK(threads == 1);
    }
    SYNTH_TEST_CHECK(synth::default_synthesis_threads() == threads);

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

    if (check_parallel_for_covers_every_index() != 0) {
        return 1;
    }
    if (check_parallel_for_is_bit_identical_to_serial() != 0) {
        return 1;
    }
    if (check_parallel_for_propagates_an_exception() != 0) {
        return 1;
    }
    return 0;
}
