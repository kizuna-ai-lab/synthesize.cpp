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
    BackendPlacement     inspect_placement(ggml_backend_sched_t scheduler, const ggml_cgraph * graph) const;
    void log_placement_if_enabled(const char * stage, ggml_backend_sched_t scheduler, const ggml_cgraph * graph) const;

  private:
    BackendPlan() = default;

    ggml_backend_t              primary_ = nullptr;
    std::vector<ggml_backend_t> owned_backends_;
    std::vector<ggml_backend_t> scheduler_backends_;
};

}  // namespace synth
