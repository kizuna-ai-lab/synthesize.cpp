#include "cpu-parallelism.h"

#include <algorithm>
#include <thread>

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

}  // namespace synth
