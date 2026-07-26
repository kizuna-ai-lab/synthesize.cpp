#pragma once

namespace synth {

// The number of CPUs this process may actually run on, which is not the same as
// the number the machine has.
//
// std::thread::hardware_concurrency reports the host's CPU count and ignores both
// the scheduler affinity mask and the cgroup CPU quota, so inside a two-vCPU
// container on a large host it can be an order of magnitude too high. Asking GGML
// for more threads than the process can run oversubscribes them, and GGML's
// barrier spins on a pause hint without ever yielding the core: waiters then
// starve the worker that has not arrived and ggml_barrier deadlocks. That was the
// original reason for a local patch to the vendored barrier; clamping the request
// removes the cause instead.
//
// Returns at least 1.
int available_cpu_parallelism();

}  // namespace synth
