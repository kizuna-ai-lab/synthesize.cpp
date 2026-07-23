#include "arch/vits/acoustic-flow.h"
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
    synth::vits::HParams hparams             = synth::test::small_vits_hparams();
    hparams.inter_channels                   = 4;
    synth::test::GgmlContext weights_context = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(weights_context != nullptr);
    synth::test::populate_flow_weight_tensors(weights_context.get(), hparams);
    synth::vits::FlowWeights weights;
    SYNTH_TEST_CHECK(synth::vits::build_flow_weights(weights_context.get(), hparams, weights) == SYNTH_OK);

    synth::test::GgmlContext graph_context = synth::test::make_ggml_context(64 * 1024 * 1024);
    SYNTH_TEST_CHECK(graph_context != nullptr);
    synth::vits::AcousticFlowGraph graph =
        synth::vits::build_acoustic_flow_graph(graph_context.get(), weights, hparams, 3);
    SYNTH_TEST_CHECK(graph.graph != nullptr);
    SYNTH_TEST_CHECK(graph.z_p != nullptr && graph.z_p->ne[0] == 4 && graph.z_p->ne[1] == 3);
    SYNTH_TEST_CHECK(graph.channel_indices != nullptr && graph.channel_indices->type == GGML_TYPE_I32 &&
                     graph.channel_indices->ne[0] == 4);
    SYNTH_TEST_CHECK(graph.z != nullptr && graph.z->ne[0] == 4 && graph.z->ne[1] == 3);
    SYNTH_TEST_CHECK(std::strcmp(graph.z_p->name, "latent.z_p.input") == 0);
    SYNTH_TEST_CHECK(std::strcmp(graph.z->name, "flow.z") == 0);

    ggml_backend_t backend = ggml_backend_cpu_init();
    SYNTH_TEST_CHECK(backend != nullptr);
    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(weights_context.get(), backend);
    SYNTH_TEST_CHECK(weights_buffer != nullptr);
    for (ggml_tensor * tensor = ggml_get_first_tensor(weights_context.get()); tensor != nullptr;
         tensor               = ggml_get_next_tensor(weights_context.get(), tensor)) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    }
    const std::vector<float> block_zero_bias = { 1.0f, 2.0f };
    const std::vector<float> block_one_bias  = { 10.0f, 20.0f };
    ggml_backend_tensor_set(weights.blocks[0].projection.bias, block_zero_bias.data(), 0,
                            block_zero_bias.size() * sizeof(float));
    ggml_backend_tensor_set(weights.blocks[1].projection.bias, block_one_bias.data(), 0,
                            block_one_bias.size() * sizeof(float));

    ggml_backend_t       backends[] = { backend };
    ggml_backend_sched_t scheduler  = ggml_backend_sched_new(backends, nullptr, 1, 2048, false, true);
    SYNTH_TEST_CHECK(scheduler != nullptr);
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(scheduler, graph.graph));
    const std::vector<int32_t> channel_indices = { 3, 2, 1, 0 };
    const std::vector<float>   input           = {
        1.0f, 10.0f, 100.0f, 1000.0f, 2.0f, 20.0f, 200.0f, 2000.0f, 3.0f, 30.0f, 300.0f, 3000.0f,
    };
    ggml_backend_tensor_set(graph.z_p, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(graph.channel_indices, channel_indices.data(), 0, channel_indices.size() * sizeof(int32_t));
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(scheduler, graph.graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual(12);
    ggml_backend_tensor_get(graph.z, actual.data(), 0, actual.size() * sizeof(float));
    SYNTH_TEST_CHECK(actual == std::vector<float>({
                                   -19.0f,
                                   0.0f,
                                   99.0f,
                                   998.0f,
                                   -18.0f,
                                   10.0f,
                                   199.0f,
                                   1998.0f,
                                   -17.0f,
                                   20.0f,
                                   299.0f,
                                   2998.0f,
                               }));
    ggml_backend_sched_free(scheduler);
    ggml_backend_buffer_free(weights_buffer);
    ggml_backend_free(backend);

    SYNTH_TEST_CHECK(synth::vits::build_acoustic_flow_graph(nullptr, weights, hparams, 3).graph == nullptr);
    SYNTH_TEST_CHECK(synth::vits::build_acoustic_flow_graph(graph_context.get(), weights, hparams, 0).graph == nullptr);
    synth::vits::FlowWeights incomplete = weights;
    incomplete.blocks.pop_back();
    SYNTH_TEST_CHECK(synth::vits::build_acoustic_flow_graph(graph_context.get(), incomplete, hparams, 3).graph ==
                     nullptr);
    synth::vits::HParams overflowing = hparams;
    overflowing.flow_dilation_rate   = std::numeric_limits<uint32_t>::max();
    SYNTH_TEST_CHECK(synth::vits::build_acoustic_flow_graph(graph_context.get(), weights, overflowing, 3).graph ==
                     nullptr);

    synth::vits::HParams conditioned                     = synth::test::small_conditioned_vits_hparams();
    conditioned.inter_channels                           = 4;
    synth::test::GgmlContext conditioned_weights_context = synth::test::make_ggml_context();
    synth::test::populate_voice_weight_tensors(conditioned_weights_context.get(), conditioned);
    synth::test::populate_flow_weight_tensors(conditioned_weights_context.get(), conditioned);
    synth::vits::VoiceWeights conditioned_voice;
    synth::vits::FlowWeights  conditioned_flow;
    SYNTH_TEST_CHECK(synth::vits::build_voice_weights(conditioned_weights_context.get(), conditioned,
                                                      conditioned_voice) == SYNTH_OK);
    SYNTH_TEST_CHECK(
        synth::vits::build_flow_weights(conditioned_weights_context.get(), conditioned, conditioned_flow) == SYNTH_OK);
    synth::test::GgmlContext       conditioned_graph_context = synth::test::make_ggml_context(64 * 1024 * 1024);
    synth::vits::AcousticFlowGraph conditioned_graph         = synth::vits::build_acoustic_flow_graph(
        conditioned_graph_context.get(), conditioned_flow, conditioned_voice, conditioned, 2, 3);
    SYNTH_TEST_CHECK(conditioned_graph.graph != nullptr && conditioned_graph.speaker_index != nullptr);
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
        ggml_backend_sched_new(conditioned_backends, nullptr, 1, 8192, false, true);
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(conditioned_scheduler, conditioned_graph.graph));
    const int32_t              conditioned_speaker = 2;
    const std::vector<float>   conditioned_input(12, 0.0f);
    const std::vector<int32_t> conditioned_indices = { 3, 2, 1, 0 };
    ggml_backend_tensor_set(conditioned_graph.speaker_index, &conditioned_speaker, 0, sizeof(conditioned_speaker));
    ggml_backend_tensor_set(conditioned_graph.z_p, conditioned_input.data(), 0,
                            conditioned_input.size() * sizeof(float));
    ggml_backend_tensor_set(conditioned_graph.channel_indices, conditioned_indices.data(), 0,
                            conditioned_indices.size() * sizeof(int32_t));
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(conditioned_scheduler, conditioned_graph.graph) ==
                     GGML_STATUS_SUCCESS);
    std::vector<float> conditioned_output(12);
    ggml_backend_tensor_get(conditioned_graph.z, conditioned_output.data(), 0,
                            conditioned_output.size() * sizeof(float));
    for (float value : conditioned_output) {
        SYNTH_TEST_CHECK(std::isfinite(value));
    }
    ggml_backend_sched_free(conditioned_scheduler);
    ggml_backend_buffer_free(conditioned_weights_buffer);
    ggml_backend_free(conditioned_backend);
    SYNTH_TEST_CHECK(synth::vits::build_acoustic_flow_graph(conditioned_graph_context.get(), conditioned_flow,
                                                            conditioned_voice, conditioned, conditioned.speaker_count,
                                                            3)
                         .graph == nullptr);
    return 0;
}
