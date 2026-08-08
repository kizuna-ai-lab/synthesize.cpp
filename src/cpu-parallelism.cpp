#include "cpu-parallelism.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#ifdef __linux__
#    include <sched.h>

#    include <cstdio>
#endif

namespace synth {

namespace {

#ifdef __linux__

// cgroup v2 writes "max 100000" for unlimited or "200000 100000" for two CPUs.
// cgroup v1 splits the same pair across two files. A quota below one whole CPU
// still permits one thread, so the result is never rounded down to zero.
int cgroup_cpu_limit() {
    long long quota  = -1;
    long long period = 0;
    if (std::FILE * file = std::fopen("/sys/fs/cgroup/cpu.max", "re")) {
        char text[64] = { 0 };
        if (std::fscanf(file, "%63s %lld", text, &period) == 2 && text[0] != 'm') {
            quota = std::strtoll(text, nullptr, 10);
        }
        std::fclose(file);
    }
    if (quota <= 0 || period <= 0) {
        quota  = -1;
        period = 0;
        if (std::FILE * file = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "re")) {
            if (std::fscanf(file, "%lld", &quota) != 1) {
                quota = -1;
            }
            std::fclose(file);
        }
        if (std::FILE * file = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "re")) {
            if (std::fscanf(file, "%lld", &period) != 1) {
                period = 0;
            }
            std::fclose(file);
        }
    }
    if (quota <= 0 || period <= 0) {
        return 0;  // no quota configured
    }
    const long long cpus = (quota + period - 1) / period;
    return cpus <= 0 ? 1 : static_cast<int>(std::min<long long>(cpus, 1024));
}

int affinity_cpu_count() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return 0;
    }
    const int count = CPU_COUNT(&set);
    return count > 0 ? count : 0;
}

#endif  // __linux__

// How many indices one worker claims from the cursor at a time. Small enough
// that the slowest core's last chunk is a short tail, large enough that the
// atomic fetch_add is nowhere near the cost of the work it hands out (the
// OmniVoice candidate scan this was measured on spends ~7 us per index, so a
// chunk is ~220 us of work behind one atomic increment).
constexpr size_t kChunk = 32;

}  // namespace

int available_cpu_parallelism() {
    const unsigned hardware = std::thread::hardware_concurrency();
    int            limit    = hardware == 0 ? 1 : static_cast<int>(std::min<unsigned>(hardware, 1024));
#ifdef __linux__
    if (const int affinity = affinity_cpu_count(); affinity > 0) {
        limit = std::min(limit, affinity);
    }
    if (const int quota = cgroup_cpu_limit(); quota > 0) {
        limit = std::min(limit, quota);
    }
#endif
    return std::max(1, limit);
}

int default_synthesis_threads() {
    const int available = available_cpu_parallelism();
    if (available <= 2) {
        return 1;
    }
    return std::min(available / 2, available - 1);
}

void parallel_for(size_t count, int workers, const std::function<void(size_t begin, size_t end)> & body) {
    if (count == 0) {
        return;
    }
    // Never more threads than there are chunks to hand out: a thread that would
    // find the cursor already past the end costs a spawn and a join to do
    // nothing. This also makes the serial fallback below cover every range
    // shorter than a single chunk without a separate threshold to tune.
    const size_t chunks       = (count + kChunk - 1) / kChunk;
    const size_t thread_count = std::min<size_t>(workers < 1 ? 1 : size_t(workers), chunks);
    if (thread_count <= 1) {
        body(0, count);
        return;
    }

    std::atomic<size_t> cursor{ 0 };
    std::mutex          failure_lock;
    std::exception_ptr  failure;

    const auto drain = [&]() {
        for (;;) {
            const size_t begin = cursor.fetch_add(kChunk, std::memory_order_relaxed);
            if (begin >= count) {
                return;
            }
            try {
                body(begin, std::min(begin + kChunk, count));
            } catch (...) {
                // Keep the first failure and stop claiming work. Whoever else
                // is still draining finishes its own chunks; the caller sees
                // the exception once every thread has been joined, which is the
                // only point at which touching `failure` is safe again.
                const std::lock_guard<std::mutex> guard(failure_lock);
                if (!failure) {
                    failure = std::current_exception();
                }
                return;
            }
        }
    };

    std::vector<std::thread> helpers;
    helpers.reserve(thread_count - 1);
    for (size_t index = 1; index < thread_count; ++index) {
        try {
            helpers.emplace_back(drain);
        } catch (...) {
            // Out of threads. The calling thread drains the cursor by itself
            // below, so fewer helpers is slower and still correct -- whereas
            // letting this escape would destroy `helpers` with live threads
            // still attached, which is std::terminate.
            break;
        }
    }
    drain();
    for (std::thread & helper : helpers) {
        helper.join();
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

}  // namespace synth
