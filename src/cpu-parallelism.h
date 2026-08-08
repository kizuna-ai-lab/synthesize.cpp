#pragma once

#include <cstddef>
#include <functional>

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

// Runs `body` over contiguous sub-ranges covering [0, count) exactly once each,
// on at most `workers` threads -- one of which is always the calling thread, so
// `workers == 1` runs the whole range inline with no thread created at all.
// Returns once every sub-range has completed.
//
// THE CONTRACT THIS PLACES ON `body`, and the reason it is stated first: the
// caller of this function is responsible for the RESULT being independent of
// how the range happens to be split. `body` must read only immutable state and
// write only to storage indexed by the loop variable; anything it consumes in
// sequence -- a random stream above all -- must be drawn before the call and
// indexed here, never drawn inside. Splitting is dynamic (see below), so the
// split is not even reproducible from one call to the next on the same input.
// This function makes an already-independent loop faster; it cannot make a
// dependent one correct.
//
// Chunking is dynamic -- an atomic cursor handing out fixed-size chunks -- and
// that is a measured choice, not a stylistic one. This project's reference host
// is heterogeneous (ten Cortex-X925 plus ten Cortex-A725), and on the OmniVoice
// candidate scan static equal blocks reached only 3.9x at ten workers where the
// cursor reached 6.4-6.7x: with equal blocks every fast core finishes early and
// waits for a slow one. A chunk is large enough that the atomic is not the
// bottleneck and small enough that the tail is short.
//
// Exceptions escaping `body` on a worker thread are caught, the first one is
// re-thrown on the calling thread after every thread has been joined, and the
// remaining chunks of the failing worker are abandoned. Without that a throwing
// body would call std::terminate instead of unwinding into the caller the way
// the serial loop it replaces would.
//
// Not a thread pool: the threads are created and joined per call. That is
// deliberate here -- a persistent pool that spins would reintroduce exactly the
// oversubscription hazard described above, and the measurement above already
// includes the per-call spawn and join cost.
void parallel_for(size_t count, int workers, const std::function<void(size_t begin, size_t end)> & body);

}  // namespace synth
