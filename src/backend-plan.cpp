#include "backend-plan.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

namespace synth {

namespace {

ggml_backend_t initialize_backend(ggml_backend_dev_t device) {
    if (device == nullptr) {
        return nullptr;
    }
    try {
        return ggml_backend_dev_init(device, nullptr);
    } catch (...) {
        return nullptr;
    }
}

void append_accelerators(std::vector<ggml_backend_t> & owned, std::vector<ggml_backend_t> & scheduled) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_t backend = initialize_backend(device);
        if (backend != nullptr) {
            owned.push_back(backend);
            scheduled.push_back(backend);
        }
    }
}

}  // namespace

synth_status_t BackendPlan::create(ggml_backend_dev_t             primary_device,
                                   bool                           include_accelerators,
                                   std::unique_ptr<BackendPlan> & output) {
    output.reset();
    if (primary_device == nullptr) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const auto primary_type = ggml_backend_dev_type(primary_device);
    if (primary_type != GGML_BACKEND_DEVICE_TYPE_CPU && primary_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
        primary_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return SYNTH_ERR_BACKEND;
    }

    try {
        auto         plan           = std::unique_ptr<BackendPlan>(new BackendPlan());
        const size_t registry_count = ggml_backend_dev_count();
        plan->owned_backends_.reserve(registry_count + 1);
        plan->scheduler_backends_.reserve(registry_count + 1);
        plan->primary_ = initialize_backend(primary_device);
        if (plan->primary_ == nullptr) {
            return SYNTH_ERR_BACKEND;
        }
        plan->owned_backends_.push_back(plan->primary_);

        if (primary_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            if (include_accelerators) {
                append_accelerators(plan->owned_backends_, plan->scheduler_backends_);
            }
            plan->scheduler_backends_.push_back(plan->primary_);
        } else {
            plan->scheduler_backends_.push_back(plan->primary_);
            if (include_accelerators) {
                append_accelerators(plan->owned_backends_, plan->scheduler_backends_);
            }
            ggml_backend_dev_t cpu_device  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            ggml_backend_t     cpu_backend = initialize_backend(cpu_device);
            if (cpu_backend == nullptr) {
                return SYNTH_ERR_BACKEND;
            }
            plan->owned_backends_.push_back(cpu_backend);
            plan->scheduler_backends_.push_back(cpu_backend);
        }

        output = std::move(plan);
        return SYNTH_OK;
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    } catch (...) {
        return SYNTH_ERR_INTERNAL;
    }
}

BackendPlan::~BackendPlan() {
    for (auto it = owned_backends_.rbegin(); it != owned_backends_.rend(); ++it) {
        try {
            ggml_backend_free(*it);
        } catch (...) {
            // Destructors may not unwind across the public C ABI.
        }
    }
}

ggml_backend_t BackendPlan::primary() const {
    return primary_;
}

ggml_backend_dev_t BackendPlan::primary_device() const {
    return primary_ == nullptr ? nullptr : ggml_backend_get_device(primary_);
}

size_t BackendPlan::scheduler_backend_count() const {
    return scheduler_backends_.size();
}

ggml_backend_dev_t BackendPlan::scheduler_device(size_t index) const {
    return index >= scheduler_backends_.size() ? nullptr : ggml_backend_get_device(scheduler_backends_[index]);
}

ggml_backend_sched_t BackendPlan::create_scheduler(size_t graph_size) const {
    if (graph_size == 0 || scheduler_backends_.empty() ||
        scheduler_backends_.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return nullptr;
    }
    try {
        return ggml_backend_sched_new(const_cast<ggml_backend_t *>(scheduler_backends_.data()), nullptr,
                                      static_cast<int>(scheduler_backends_.size()), graph_size, false, true);
    } catch (...) {
        return nullptr;
    }
}

void BackendPlan::set_threads(int threads) const {
    const int resolved_threads = std::max(1, threads);
    for (ggml_backend_t backend : scheduler_backends_) {
        ggml_backend_dev_t device   = ggml_backend_get_device(backend);
        ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
        if (registry == nullptr) {
            continue;
        }
        auto set_n_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_n_threads"));
        if (set_n_threads != nullptr) {
            set_n_threads(backend, resolved_threads);
        }
    }
}

bool BackendPlan::assign_to_primary(ggml_backend_sched_t scheduler, ggml_tensor * tensor) const {
    if (scheduler == nullptr || tensor == nullptr || primary_ == nullptr ||
        !ggml_backend_supports_op(primary_, tensor)) {
        return false;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, tensor, primary_);
    return true;
}

BackendPlacement BackendPlan::inspect_placement(ggml_backend_sched_t scheduler, const ggml_cgraph * graph) const {
    BackendPlacement placement;
    if (scheduler == nullptr || graph == nullptr) {
        return placement;
    }
    auto *    mutable_graph = const_cast<ggml_cgraph *>(graph);
    const int node_count    = ggml_graph_n_nodes(mutable_graph);
    placement.node_count    = static_cast<uint64_t>(node_count);
    placement.split_count   = static_cast<uint64_t>(std::max(0, ggml_backend_sched_get_n_splits(scheduler)));
    for (int node_index = 0; node_index < node_count; ++node_index) {
        ggml_tensor * node = ggml_graph_node(mutable_graph, node_index);
        if (ggml_is_view(node)) {
            ++placement.view_node_count;
            continue;
        }
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(scheduler, node);
        if (backend == nullptr) {
            ++placement.unassigned_node_count;
            continue;
        }
        if (backend == primary_) {
            ++placement.primary_node_count;
            continue;
        }
        const auto type = ggml_backend_dev_type(ggml_backend_get_device(backend));
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            ++placement.cpu_fallback_node_count;
        } else if (type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            ++placement.accelerator_node_count;
        } else {
            ++placement.other_node_count;
        }
    }
    return placement;
}

void BackendPlan::log_placement_if_enabled(const char *         stage,
                                           ggml_backend_sched_t scheduler,
                                           const ggml_cgraph *  graph) const {
    const char * enabled = std::getenv("SYNTH_DEBUG_BACKEND_PLACEMENT");
    if (enabled == nullptr || enabled[0] == '\0') {
        return;
    }
    const BackendPlacement placement = inspect_placement(scheduler, graph);
    const char * primary_name = primary_device() == nullptr ? "unknown" : ggml_backend_dev_name(primary_device());
    std::fprintf(stderr,
                 "synth_backend_plan: stage=%s primary=%s nodes=%llu view_nodes=%llu primary_nodes=%llu "
                 "cpu_fallback_nodes=%llu accelerator_nodes=%llu other_nodes=%llu unassigned_nodes=%llu "
                 "splits=%llu\n",
                 stage == nullptr ? "unknown" : stage, primary_name == nullptr ? "unknown" : primary_name,
                 static_cast<unsigned long long>(placement.node_count),
                 static_cast<unsigned long long>(placement.view_node_count),
                 static_cast<unsigned long long>(placement.primary_node_count),
                 static_cast<unsigned long long>(placement.cpu_fallback_node_count),
                 static_cast<unsigned long long>(placement.accelerator_node_count),
                 static_cast<unsigned long long>(placement.other_node_count),
                 static_cast<unsigned long long>(placement.unassigned_node_count),
                 static_cast<unsigned long long>(placement.split_count));

    auto *    mutable_graph = const_cast<ggml_cgraph *>(graph);
    const int node_count    = ggml_graph_n_nodes(mutable_graph);
    for (int node_index = 0; node_index < node_count; ++node_index) {
        ggml_tensor * node = ggml_graph_node(mutable_graph, node_index);
        if (ggml_is_view(node)) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(scheduler, node);
        if (backend == primary_) {
            continue;
        }
        const char * backend_name = backend == nullptr ? "unassigned" : ggml_backend_name(backend);
        const char * tensor_name  = ggml_get_name(node);
        const char * operation    = ggml_op_desc(node);
        std::fprintf(stderr, "synth_backend_plan: stage=%s fallback_node=%d backend=%s op=%s tensor=%s\n",
                     stage == nullptr ? "unknown" : stage, node_index,
                     backend_name == nullptr ? "unknown" : backend_name, operation == nullptr ? "unknown" : operation,
                     tensor_name == nullptr || tensor_name[0] == '\0' ? "(unnamed)" : tensor_name);
    }
}

}  // namespace synth
