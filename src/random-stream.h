#pragma once

#include <cstddef>
#include <cstdint>

namespace synth {

class NormalRandomStream {
  public:
    explicit NormalRandomStream(uint64_t seed);

    float next();
    void  fill(float * output, size_t count);

  private:
    uint64_t state_     = 0;
    float    spare_     = 0.0f;
    bool     has_spare_ = false;

    uint64_t next_u64();
    double   uniform_open();
};

uint64_t nondeterministic_seed();

}  // namespace synth
