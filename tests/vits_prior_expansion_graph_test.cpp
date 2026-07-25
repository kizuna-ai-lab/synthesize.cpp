#include "arch/vits/prior-expansion.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "test-assert.h"
#include "vits-test-fixture.h"

#include <cstring>
#include <vector>

int main() {
    synth::test::GgmlContext context = synth::test::make_ggml_context(4 * 1024 * 1024);
    SYNTH_TEST_CHECK(context != nullptr);
    synth::vits::PriorExpansionGraph graph = synth::vits::build_prior_expansion_graph(context.get(), 2, 3, 4);
    SYNTH_TEST_CHECK(graph.graph != nullptr);
    SYNTH_TEST_CHECK(graph.m_p != nullptr && graph.m_p->ne[0] == 2 && graph.m_p->ne[1] == 3);
    SYNTH_TEST_CHECK(graph.logs_p != nullptr && graph.logs_p->ne[0] == 2 && graph.logs_p->ne[1] == 3);
    SYNTH_TEST_CHECK(graph.attention != nullptr && graph.attention->ne[0] == 3 && graph.attention->ne[1] == 4);
    SYNTH_TEST_CHECK(graph.m_p_expanded != nullptr && graph.m_p_expanded->ne[0] == 2 && graph.m_p_expanded->ne[1] == 4);
    SYNTH_TEST_CHECK(graph.logs_p_expanded != nullptr && graph.logs_p_expanded->ne[0] == 2 &&
                     graph.logs_p_expanded->ne[1] == 4);
    SYNTH_TEST_CHECK(std::strcmp(graph.m_p_expanded->name, "prior.m_p_expanded") == 0);
    SYNTH_TEST_CHECK(std::strcmp(graph.logs_p_expanded->name, "prior.logs_p_expanded") == 0);

    ggml_backend_t backend = ggml_backend_cpu_init();
    SYNTH_TEST_CHECK(backend != nullptr);
    ggml_backend_t       backends[] = { backend };
    ggml_backend_sched_t scheduler  = ggml_backend_sched_new(backends, nullptr, 1, 256, false, true);
    SYNTH_TEST_CHECK(scheduler != nullptr);
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(scheduler, graph.graph));
    const std::vector<float> m_p       = { 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f };
    const std::vector<float> logs_p    = { -1.0f, -10.0f, -2.0f, -20.0f, -3.0f, -30.0f };
    const std::vector<float> attention = {
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    ggml_backend_tensor_set(graph.m_p, m_p.data(), 0, m_p.size() * sizeof(float));
    ggml_backend_tensor_set(graph.logs_p, logs_p.data(), 0, logs_p.size() * sizeof(float));
    ggml_backend_tensor_set(graph.attention, attention.data(), 0, attention.size() * sizeof(float));
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(scheduler, graph.graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual_m(8);
    std::vector<float> actual_logs(8);
    ggml_backend_tensor_get(graph.m_p_expanded, actual_m.data(), 0, actual_m.size() * sizeof(float));
    ggml_backend_tensor_get(graph.logs_p_expanded, actual_logs.data(), 0, actual_logs.size() * sizeof(float));
    SYNTH_TEST_CHECK(actual_m == std::vector<float>({ 1.0f, 10.0f, 2.0f, 20.0f, 2.0f, 20.0f, 3.0f, 30.0f }));
    SYNTH_TEST_CHECK(actual_logs == std::vector<float>({ -1.0f, -10.0f, -2.0f, -20.0f, -2.0f, -20.0f, -3.0f, -30.0f }));
    ggml_backend_sched_free(scheduler);
    ggml_backend_free(backend);

    SYNTH_TEST_CHECK(synth::vits::build_prior_expansion_graph(nullptr, 2, 3, 4).graph == nullptr);
    SYNTH_TEST_CHECK(synth::vits::build_prior_expansion_graph(context.get(), 0, 3, 4).graph == nullptr);
    SYNTH_TEST_CHECK(synth::vits::build_prior_expansion_graph(context.get(), 2, 0, 4).graph == nullptr);
    SYNTH_TEST_CHECK(synth::vits::build_prior_expansion_graph(context.get(), 2, 3, 0).graph == nullptr);
    return 0;
}
