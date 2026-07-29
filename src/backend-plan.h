#pragma once

#include "ggml-backend.h"
#include "synthesize.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace synth {

struct BackendPlacement {
    uint64_t node_count              = 0;
    uint64_t view_node_count         = 0;
    uint64_t primary_node_count      = 0;
    uint64_t cpu_fallback_node_count = 0;
    uint64_t accelerator_node_count  = 0;
    uint64_t other_node_count        = 0;
    uint64_t unassigned_node_count   = 0;
    uint64_t split_count             = 0;
    // Nodes whose backend is not a CPU device, counted by device type and
    // independently of which backend is primary.
    //
    // The four counters above classify against the primary, so a node on the
    // primary is never reached by the type test: when the primary is the
    // accelerator its nodes land in primary_node_count and accelerator_node_count
    // stays zero. That is the right shape for asking "did anything fall back",
    // and the wrong shape for asking "did this run off the CPU", which is what
    // docs/backends.md's discrete-output rule needs to be able to check.
    uint64_t off_cpu_node_count      = 0;
};

// Owns every initialized backend participating in one loaded model. The
// primary backend owns weights; scheduler_backends_ is ordered by execution
// priority with CPU last, as required by GGML's fallback scheduler.
class BackendPlan {
  public:
    static synth_status_t create(ggml_backend_dev_t             primary_device,
                                 bool                           include_accelerators,
                                 std::unique_ptr<BackendPlan> & output);

    ~BackendPlan();
    BackendPlan(const BackendPlan &)             = delete;
    BackendPlan & operator=(const BackendPlan &) = delete;
    BackendPlan(BackendPlan &&)                  = delete;
    BackendPlan & operator=(BackendPlan &&)      = delete;

    ggml_backend_t     primary() const;
    ggml_backend_dev_t primary_device() const;
    size_t             scheduler_backend_count() const;
    ggml_backend_dev_t scheduler_device(size_t index) const;

    ggml_backend_sched_t create_scheduler(size_t graph_size) const;
    void                 set_threads(int threads) const;
    bool                 assign_to_primary(ggml_backend_sched_t scheduler, ggml_tensor * tensor) const;

    // The CPU backend this plan schedules onto. Present whether or not CPU is the
    // primary: GGML's fallback scheduler always carries one.
    ggml_backend_t cpu_backend() const;

    // A scheduler over the CPU backend alone, for a stage held on CPU on purpose
    // because its output is discrete. Mixing backends inside one graph is not an
    // alternative: forcing a large graph's nodes onto CPU while its weights stay
    // in the primary buffer was measured at 2,372 scheduler splits on Kokoro's
    // duration stage, five times slower end to end than leaving it on the GPU.
    // A single-backend scheduler over CPU-resident weights is one split.
    ggml_backend_sched_t create_cpu_scheduler(size_t graph_size) const;

    BackendPlacement inspect_placement(ggml_backend_sched_t scheduler, const ggml_cgraph * graph) const;
    void log_placement_if_enabled(const char * stage, ggml_backend_sched_t scheduler, const ggml_cgraph * graph) const;

  private:
    BackendPlan() = default;

    ggml_backend_t              primary_ = nullptr;
    std::vector<ggml_backend_t> owned_backends_;
    std::vector<ggml_backend_t> scheduler_backends_;
};

}  // namespace synth
