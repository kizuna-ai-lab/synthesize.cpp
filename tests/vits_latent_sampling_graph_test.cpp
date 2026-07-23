#include "arch/vits/latent-sampling.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "test-assert.h"
#include "vits-test-fixture.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

int main() {
    synth::test::GgmlContext context = synth::test::make_ggml_context(2 * 1024 * 1024);
    SYNTH_TEST_CHECK(context != nullptr);
    synth::vits::LatentSamplingGraph graph = synth::vits::build_latent_sampling_graph(context.get(), 2, 3, 0.5f);
    SYNTH_TEST_CHECK(graph.graph != nullptr);
    SYNTH_TEST_CHECK(graph.m_p != nullptr && graph.m_p->ne[0] == 2 && graph.m_p->ne[1] == 3);
    SYNTH_TEST_CHECK(graph.logs_p != nullptr && graph.logs_p->ne[0] == 2 && graph.logs_p->ne[1] == 3);
    SYNTH_TEST_CHECK(graph.latent_noise != nullptr && graph.latent_noise->ne[0] == 2 && graph.latent_noise->ne[1] == 3);
    SYNTH_TEST_CHECK(graph.z_p != nullptr && graph.z_p->ne[0] == 2 && graph.z_p->ne[1] == 3);
    SYNTH_TEST_CHECK(std::strcmp(graph.z_p->name, "latent.z_p") == 0);

    ggml_backend_t backend = ggml_backend_cpu_init();
    SYNTH_TEST_CHECK(backend != nullptr);
    ggml_backend_t       backends[] = { backend };
    ggml_backend_sched_t scheduler  = ggml_backend_sched_new(backends, nullptr, 1, 64, false, true);
    SYNTH_TEST_CHECK(scheduler != nullptr);
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(scheduler, graph.graph));
    const std::vector<float> m_p     = { 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f };
    const float              log_two = std::log(2.0f);
    const std::vector<float> logs    = { log_two, log_two, log_two, log_two, log_two, log_two };
    const std::vector<float> noise   = { 2.0f, -2.0f, 4.0f, -4.0f, 6.0f, -6.0f };
    ggml_backend_tensor_set(graph.m_p, m_p.data(), 0, m_p.size() * sizeof(float));
    ggml_backend_tensor_set(graph.logs_p, logs.data(), 0, logs.size() * sizeof(float));
    ggml_backend_tensor_set(graph.latent_noise, noise.data(), 0, noise.size() * sizeof(float));
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(scheduler, graph.graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual(6);
    ggml_backend_tensor_get(graph.z_p, actual.data(), 0, actual.size() * sizeof(float));
    const std::vector<float> expected = { 3.0f, 8.0f, 6.0f, 16.0f, 9.0f, 24.0f };
    for (size_t index = 0; index < actual.size(); ++index) {
        SYNTH_TEST_CHECK(std::fabs(actual[index] - expected[index]) <= 1.0e-5f);
    }
    ggml_backend_sched_free(scheduler);
    ggml_backend_free(backend);

    SYNTH_TEST_CHECK(synth::vits::build_latent_sampling_graph(nullptr, 2, 3, 0.5f).graph == nullptr);
    SYNTH_TEST_CHECK(synth::vits::build_latent_sampling_graph(context.get(), 0, 3, 0.5f).graph == nullptr);
    SYNTH_TEST_CHECK(synth::vits::build_latent_sampling_graph(context.get(), 2, 0, 0.5f).graph == nullptr);
    SYNTH_TEST_CHECK(synth::vits::build_latent_sampling_graph(context.get(), 2, 3, -0.1f).graph == nullptr);
    SYNTH_TEST_CHECK(
        synth::vits::build_latent_sampling_graph(context.get(), 2, 3, std::numeric_limits<float>::infinity()).graph ==
        nullptr);
    return 0;
}
