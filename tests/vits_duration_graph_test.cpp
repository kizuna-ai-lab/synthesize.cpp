#include "arch/vits/duration-predictor.h"
#include "arch/vits/weights.h"
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
    const synth::vits::HParams hparams         = synth::test::small_vits_hparams();
    synth::test::GgmlContext   weights_context = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(weights_context != nullptr);
    synth::test::populate_duration_weight_tensors(weights_context.get(), hparams);
    synth::vits::DurationWeights weights;
    SYNTH_TEST_CHECK(synth::vits::build_duration_weights(weights_context.get(), hparams, weights) == SYNTH_OK);

    synth::test::GgmlContext graph_context = synth::test::make_ggml_context(64 * 1024 * 1024);
    SYNTH_TEST_CHECK(graph_context != nullptr);
    ggml_tensor * encoded = ggml_new_tensor_2d(graph_context.get(), GGML_TYPE_F32, hparams.hidden_channels, 3);
    ggml_set_input(encoded);
    synth::vits::DurationPredictorGraph graph = synth::vits::build_duration_predictor_graph(
        graph_context.get(), encoded, weights, hparams, 3, hparams.duration_noise_scale_w);
    SYNTH_TEST_CHECK(graph.graph != nullptr);
    SYNTH_TEST_CHECK(graph.duration_noise != nullptr && graph.duration_noise->type == GGML_TYPE_F32 &&
                     graph.duration_noise->ne[0] == 2 && graph.duration_noise->ne[1] == 3);
    SYNTH_TEST_CHECK(graph.logw != nullptr && graph.logw->type == GGML_TYPE_F32 && graph.logw->ne[0] == 1 &&
                     graph.logw->ne[1] == 3);
    SYNTH_TEST_CHECK(std::strcmp(graph.duration_noise->name, "random.duration_noise") == 0);
    SYNTH_TEST_CHECK(std::strcmp(graph.logw->name, "duration.logw") == 0);

    ggml_backend_t backend = ggml_backend_cpu_init();
    SYNTH_TEST_CHECK(backend != nullptr);
    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(weights_context.get(), backend);
    SYNTH_TEST_CHECK(weights_buffer != nullptr);
    for (ggml_tensor * tensor = ggml_get_first_tensor(weights_context.get()); tensor != nullptr;
         tensor               = ggml_get_next_tensor(weights_context.get(), tensor)) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    }
    ggml_backend_t       backends[] = { backend };
    ggml_backend_sched_t scheduler  = ggml_backend_sched_new(backends, nullptr, 1, 16384, false, true);
    SYNTH_TEST_CHECK(scheduler != nullptr);
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(scheduler, graph.graph));
    const std::vector<float> encoded_values(12, 0.0f);
    const std::vector<float> noise_values = { 0.1f, -0.1f, 0.2f, -0.2f, 0.3f, -0.3f };
    ggml_backend_tensor_set(encoded, encoded_values.data(), 0, encoded_values.size() * sizeof(float));
    ggml_backend_tensor_set(graph.duration_noise, noise_values.data(), 0, noise_values.size() * sizeof(float));
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(scheduler, graph.graph) == GGML_STATUS_SUCCESS);
    std::vector<float> logw(3);
    ggml_backend_tensor_get(graph.logw, logw.data(), 0, logw.size() * sizeof(float));
    SYNTH_TEST_CHECK(std::isfinite(logw[0]) && std::isfinite(logw[1]) && std::isfinite(logw[2]));
    ggml_backend_sched_free(scheduler);
    ggml_backend_buffer_free(weights_buffer);
    ggml_backend_free(backend);

    SYNTH_TEST_CHECK(synth::vits::build_duration_predictor_graph(nullptr, encoded, weights, hparams, 3, 0.8f).graph ==
                     nullptr);
    SYNTH_TEST_CHECK(
        synth::vits::build_duration_predictor_graph(graph_context.get(), nullptr, weights, hparams, 3, 0.8f).graph ==
        nullptr);
    SYNTH_TEST_CHECK(
        synth::vits::build_duration_predictor_graph(graph_context.get(), encoded, weights, hparams, 0, 0.8f).graph ==
        nullptr);
    SYNTH_TEST_CHECK(
        synth::vits::build_duration_predictor_graph(graph_context.get(), encoded, weights, hparams, 3, -0.1f).graph ==
        nullptr);
    SYNTH_TEST_CHECK(synth::vits::build_duration_predictor_graph(graph_context.get(), encoded, weights, hparams, 3,
                                                                 std::numeric_limits<float>::quiet_NaN())
                         .graph == nullptr);
    synth::vits::DurationWeights incomplete = weights;
    incomplete.flows.pop_back();
    SYNTH_TEST_CHECK(
        synth::vits::build_duration_predictor_graph(graph_context.get(), encoded, incomplete, hparams, 3, 0.8f).graph ==
        nullptr);

    const synth::vits::HParams conditioned                 = synth::test::small_conditioned_vits_hparams();
    synth::test::GgmlContext   conditioned_weights_context = synth::test::make_ggml_context();
    synth::test::populate_voice_weight_tensors(conditioned_weights_context.get(), conditioned);
    synth::test::populate_duration_weight_tensors(conditioned_weights_context.get(), conditioned);
    synth::vits::VoiceWeights    conditioned_voice;
    synth::vits::DurationWeights conditioned_duration;
    SYNTH_TEST_CHECK(synth::vits::build_voice_weights(conditioned_weights_context.get(), conditioned,
                                                      conditioned_voice) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::vits::build_duration_weights(conditioned_weights_context.get(), conditioned,
                                                         conditioned_duration) == SYNTH_OK);
    synth::test::GgmlContext conditioned_graph_context = synth::test::make_ggml_context(64 * 1024 * 1024);
    ggml_tensor *            conditioned_encoded =
        ggml_new_tensor_2d(conditioned_graph_context.get(), GGML_TYPE_F32, conditioned.hidden_channels, 3);
    ggml_set_input(conditioned_encoded);
    synth::vits::DurationPredictorGraph conditioned_graph = synth::vits::build_duration_predictor_graph(
        conditioned_graph_context.get(), conditioned_encoded, conditioned_duration, conditioned_voice, conditioned, 1,
        3, conditioned.duration_noise_scale_w);
    SYNTH_TEST_CHECK(conditioned_graph.graph != nullptr && conditioned_graph.speaker_index != nullptr);
    SYNTH_TEST_CHECK(conditioned_graph.speaker_index->type == GGML_TYPE_I32 &&
                     conditioned_graph.speaker_index->ne[0] == 1);
    ggml_backend_t        conditioned_backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t conditioned_weights_buffer =
        ggml_backend_alloc_ctx_tensors(conditioned_weights_context.get(), conditioned_backend);
    SYNTH_TEST_CHECK(conditioned_weights_buffer != nullptr);
    for (ggml_tensor * tensor = ggml_get_first_tensor(conditioned_weights_context.get()); tensor != nullptr;
         tensor               = ggml_get_next_tensor(conditioned_weights_context.get(), tensor)) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    }
    ggml_backend_t       conditioned_backends[] = { conditioned_backend };
    ggml_backend_sched_t conditioned_scheduler =
        ggml_backend_sched_new(conditioned_backends, nullptr, 1, 16384, false, true);
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(conditioned_scheduler, conditioned_graph.graph));
    const int32_t            conditioned_speaker = 1;
    const std::vector<float> conditioned_encoded_values(12, 0.0f);
    const std::vector<float> conditioned_noise_values(6, 0.0f);
    ggml_backend_tensor_set(conditioned_graph.speaker_index, &conditioned_speaker, 0, sizeof(conditioned_speaker));
    ggml_backend_tensor_set(conditioned_encoded, conditioned_encoded_values.data(), 0,
                            conditioned_encoded_values.size() * sizeof(float));
    ggml_backend_tensor_set(conditioned_graph.duration_noise, conditioned_noise_values.data(), 0,
                            conditioned_noise_values.size() * sizeof(float));
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(conditioned_scheduler, conditioned_graph.graph) ==
                     GGML_STATUS_SUCCESS);
    std::vector<float> conditioned_logw(3);
    ggml_backend_tensor_get(conditioned_graph.logw, conditioned_logw.data(), 0,
                            conditioned_logw.size() * sizeof(float));
    SYNTH_TEST_CHECK(std::isfinite(conditioned_logw[0]) && std::isfinite(conditioned_logw[1]) &&
                     std::isfinite(conditioned_logw[2]));
    ggml_backend_sched_free(conditioned_scheduler);
    ggml_backend_buffer_free(conditioned_weights_buffer);
    ggml_backend_free(conditioned_backend);
    SYNTH_TEST_CHECK(synth::vits::build_duration_predictor_graph(
                         conditioned_graph_context.get(), conditioned_encoded, conditioned_duration, conditioned_voice,
                         conditioned, conditioned.speaker_count, 3, conditioned.duration_noise_scale_w)
                         .graph == nullptr);
    return 0;
}
