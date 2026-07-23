#include "arch/vits/waveform-decoder.h"
#include "arch/vits/weights.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "test-assert.h"
#include "vits-test-fixture.h"

#include <cmath>
#include <cstring>
#include <vector>

int main() {
    const synth::vits::HParams hparams         = synth::test::small_vits_hparams();
    synth::test::GgmlContext   weights_context = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(weights_context != nullptr);
    synth::test::populate_decoder_weight_tensors(weights_context.get(), hparams);
    synth::vits::DecoderWeights weights;
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(weights_context.get(), hparams, weights) == SYNTH_OK);

    synth::test::GgmlContext graph_context = synth::test::make_ggml_context(32 * 1024 * 1024);
    SYNTH_TEST_CHECK(graph_context != nullptr);
    synth::vits::WaveformDecoderGraph graph =
        synth::vits::build_waveform_decoder_graph(graph_context.get(), weights, hparams, 3);
    SYNTH_TEST_CHECK(graph.graph != nullptr);
    SYNTH_TEST_CHECK(graph.z != nullptr && graph.z->ne[0] == 2 && graph.z->ne[1] == 3);
    SYNTH_TEST_CHECK(graph.pcm != nullptr && graph.pcm->ne[0] == 1 && graph.pcm->ne[1] == 12);
    SYNTH_TEST_CHECK(std::strcmp(graph.z->name, "flow.z.input") == 0);
    SYNTH_TEST_CHECK(std::strcmp(graph.pcm->name, "audio.pcm") == 0);

    ggml_backend_t backend = ggml_backend_cpu_init();
    SYNTH_TEST_CHECK(backend != nullptr);
    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors(weights_context.get(), backend);
    SYNTH_TEST_CHECK(weights_buffer != nullptr);
    for (ggml_tensor * tensor = ggml_get_first_tensor(weights_context.get()); tensor != nullptr;
         tensor               = ggml_get_next_tensor(weights_context.get(), tensor)) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    }

    ggml_backend_t       backends[] = { backend };
    ggml_backend_sched_t scheduler  = ggml_backend_sched_new(backends, nullptr, 1, 4096, false, true);
    SYNTH_TEST_CHECK(scheduler != nullptr);
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(scheduler, graph.graph));
    const std::vector<float> input = { 1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f };
    ggml_backend_tensor_set(graph.z, input.data(), 0, input.size() * sizeof(float));
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(scheduler, graph.graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual(12, 1.0f);
    ggml_backend_tensor_get(graph.pcm, actual.data(), 0, actual.size() * sizeof(float));
    SYNTH_TEST_CHECK(actual == std::vector<float>(12, 0.0f));

    ggml_backend_sched_free(scheduler);
    ggml_backend_buffer_free(weights_buffer);
    ggml_backend_free(backend);

    SYNTH_TEST_CHECK(synth::vits::build_waveform_decoder_graph(nullptr, weights, hparams, 3).graph == nullptr);
    SYNTH_TEST_CHECK(synth::vits::build_waveform_decoder_graph(graph_context.get(), weights, hparams, 0).graph ==
                     nullptr);
    synth::vits::DecoderWeights incomplete = weights;
    incomplete.stages.clear();
    SYNTH_TEST_CHECK(synth::vits::build_waveform_decoder_graph(graph_context.get(), incomplete, hparams, 3).graph ==
                     nullptr);

    const synth::vits::HParams conditioned                 = synth::test::small_conditioned_vits_hparams();
    synth::test::GgmlContext   conditioned_weights_context = synth::test::make_ggml_context();
    synth::test::populate_voice_weight_tensors(conditioned_weights_context.get(), conditioned);
    synth::test::populate_decoder_weight_tensors(conditioned_weights_context.get(), conditioned);
    synth::vits::VoiceWeights   conditioned_voice;
    synth::vits::DecoderWeights conditioned_decoder;
    SYNTH_TEST_CHECK(synth::vits::build_voice_weights(conditioned_weights_context.get(), conditioned,
                                                      conditioned_voice) == SYNTH_OK);
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(conditioned_weights_context.get(), conditioned,
                                                        conditioned_decoder) == SYNTH_OK);
    synth::test::GgmlContext          conditioned_graph_context = synth::test::make_ggml_context(32 * 1024 * 1024);
    synth::vits::WaveformDecoderGraph conditioned_graph         = synth::vits::build_waveform_decoder_graph(
        conditioned_graph_context.get(), conditioned_decoder, conditioned_voice, conditioned, 0, 3);
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
        ggml_backend_sched_new(conditioned_backends, nullptr, 1, 32768, false, true);
    SYNTH_TEST_CHECK(ggml_backend_sched_alloc_graph(conditioned_scheduler, conditioned_graph.graph));
    const int32_t            conditioned_speaker = 0;
    const std::vector<float> conditioned_input(6, 0.0f);
    ggml_backend_tensor_set(conditioned_graph.speaker_index, &conditioned_speaker, 0, sizeof(conditioned_speaker));
    ggml_backend_tensor_set(conditioned_graph.z, conditioned_input.data(), 0, conditioned_input.size() * sizeof(float));
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(conditioned_scheduler, conditioned_graph.graph) ==
                     GGML_STATUS_SUCCESS);
    std::vector<float> conditioned_pcm(12);
    ggml_backend_tensor_get(conditioned_graph.pcm, conditioned_pcm.data(), 0, conditioned_pcm.size() * sizeof(float));
    for (float value : conditioned_pcm) {
        SYNTH_TEST_CHECK(std::isfinite(value));
    }
    ggml_backend_sched_free(conditioned_scheduler);
    ggml_backend_buffer_free(conditioned_weights_buffer);
    ggml_backend_free(conditioned_backend);
    SYNTH_TEST_CHECK(synth::vits::build_waveform_decoder_graph(conditioned_graph_context.get(), conditioned_decoder,
                                                               conditioned_voice, conditioned,
                                                               conditioned.speaker_count, 3)
                         .graph == nullptr);

    synth::vits::HParams q8_hparams             = synth::test::small_vits_hparams();
    q8_hparams.quantization_profile             = synth::vits::QuantizationProfile::Q8Mixed;
    q8_hparams.inter_channels                   = 32;
    q8_hparams.decoder_initial_channels         = 64;
    synth::test::GgmlContext q8_weights_context = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(q8_weights_context != nullptr);
    synth::test::populate_decoder_weight_tensors(q8_weights_context.get(), q8_hparams, {}, {}, {}, GGML_TYPE_I32,
                                                 GGML_TYPE_Q8_0, GGML_TYPE_F16, true);
    synth::vits::DecoderWeights q8_weights;
    SYNTH_TEST_CHECK(synth::vits::build_decoder_weights(q8_weights_context.get(), q8_hparams, q8_weights) == SYNTH_OK);
    synth::test::GgmlContext          q8_graph_context = synth::test::make_ggml_context(32 * 1024 * 1024);
    synth::vits::WaveformDecoderGraph q8_graph =
        synth::vits::build_waveform_decoder_graph(q8_graph_context.get(), q8_weights, q8_hparams, 2);
    SYNTH_TEST_CHECK(q8_graph.graph != nullptr && q8_graph.pcm != nullptr && q8_graph.pcm->ne[1] == 8);
    ggml_backend_t q8_backend = ggml_backend_cpu_init();
    SYNTH_TEST_CHECK(q8_backend != nullptr);
    ggml_backend_buffer_t q8_weights_buffer = ggml_backend_alloc_ctx_tensors(q8_weights_context.get(), q8_backend);
    SYNTH_TEST_CHECK(q8_weights_buffer != nullptr);
    for (ggml_tensor * tensor = ggml_get_first_tensor(q8_weights_context.get()); tensor != nullptr;
         tensor               = ggml_get_next_tensor(q8_weights_context.get(), tensor)) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    }
    ggml_backend_t       q8_backends[] = { q8_backend };
    ggml_backend_sched_t q8_scheduler  = ggml_backend_sched_new(q8_backends, nullptr, 1, 32768, false, true);
    SYNTH_TEST_CHECK(q8_scheduler != nullptr && ggml_backend_sched_alloc_graph(q8_scheduler, q8_graph.graph));
    const std::vector<float> q8_input(64, 0.0f);
    ggml_backend_tensor_set(q8_graph.z, q8_input.data(), 0, q8_input.size() * sizeof(float));
    SYNTH_TEST_CHECK(ggml_backend_sched_graph_compute(q8_scheduler, q8_graph.graph) == GGML_STATUS_SUCCESS);
    std::vector<float> q8_pcm(8, 1.0f);
    ggml_backend_tensor_get(q8_graph.pcm, q8_pcm.data(), 0, q8_pcm.size() * sizeof(float));
    SYNTH_TEST_CHECK(q8_pcm == std::vector<float>(8, 0.0f));
    ggml_backend_sched_free(q8_scheduler);
    ggml_backend_buffer_free(q8_weights_buffer);
    ggml_backend_free(q8_backend);
    return 0;
}
