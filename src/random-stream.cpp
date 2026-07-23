#include "random-stream.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <random>

namespace synth {

NormalRandomStream::NormalRandomStream(uint64_t seed) : state_(seed) {}

uint64_t NormalRandomStream::next_u64() {
    state_ += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t value = state_;
    value          = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value          = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

double NormalRandomStream::uniform_open() {
    constexpr double inverse = 1.0 / 9007199254740992.0;
    return (static_cast<double>(next_u64() >> 11) + 0.5) * inverse;
}

float NormalRandomStream::next() {
    if (has_spare_) {
        has_spare_ = false;
        return spare_;
    }
    double first  = 0.0;
    double second = 0.0;
    double radius = 0.0;
    do {
        first  = 2.0 * uniform_open() - 1.0;
        second = 2.0 * uniform_open() - 1.0;
        radius = first * first + second * second;
    } while (radius >= 1.0 || radius == 0.0);
    const double scale = std::sqrt(-2.0 * std::log(radius) / radius);
    spare_             = static_cast<float>(second * scale);
    has_spare_         = true;
    return static_cast<float>(first * scale);
}

void NormalRandomStream::fill(float * output, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        output[index] = next();
    }
}

uint64_t nondeterministic_seed() {
    static std::atomic<uint64_t> counter{ 0 };
    uint64_t seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    seed ^= counter.fetch_add(UINT64_C(0x9e3779b97f4a7c15), std::memory_order_relaxed);
    try {
        std::random_device device;
        seed ^= static_cast<uint64_t>(device()) << 32;
        seed ^= static_cast<uint64_t>(device());
    } catch (...) {
        // Time plus the process-local counter remains a usable fallback.
    }
    return seed == UINT64_MAX ? UINT64_MAX - 1 : seed;
}

}  // namespace synth
