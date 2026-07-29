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

// How many threads a new Synthesis Context asks for, which is deliberately fewer
// than the CPUs it may use.
//
// Asking for exactly the CPU count is catastrophic here, and not by a little.
// GGML's barrier spins rather than yielding, so once the workers fill every CPU
// any thread the scheduler moves aside stalls all the others -- and synthesis is
// batch-one autoregressive decoding, which crosses that barrier roughly a
// hundred times per output frame. Measured on a twenty-CPU machine, one
// sentence, real-time factor against thread count:
//
//     4 -> 1.50    10 -> 1.18    14 -> 1.17    18 -> 1.99
//     8 -> 1.25    12 -> 1.13    16 -> 1.36    20 -> 10.2
//
// The degradation starts well before the cliff, so the headroom has to be more
// than one CPU. Half is the conservative reading of that curve and lands within
// five percent of the best count measured; an embedder who knows its machine can
// say otherwise through synth_context_set_threads.
//
// Returns at least 1, and never more than available_cpu_parallelism() - 1 unless
// there is only one CPU to begin with.
int default_synthesis_threads();

}  // namespace synth
