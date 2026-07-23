#include "arch/vits/text-encoder.h"
#include "arch/vits/weights.h"
#include "ggml.h"
#include "test-assert.h"
#include "vits-test-fixture.h"

#include <cstring>

int main() {
    const synth::vits::HParams hparams         = synth::test::small_vits_hparams();
    synth::test::GgmlContext   weights_context = synth::test::make_ggml_context();
    SYNTH_TEST_CHECK(weights_context != nullptr);
    synth::test::populate_text_weight_tensors(weights_context.get(), hparams);
    synth::vits::TextWeights weights;
    SYNTH_TEST_CHECK(synth::vits::build_text_weights(weights_context.get(), hparams, weights) == SYNTH_OK);

    synth::test::GgmlContext graph_context = synth::test::make_ggml_context(32 * 1024 * 1024);
    SYNTH_TEST_CHECK(graph_context != nullptr);
    synth::vits::TextEncoderGraph graph =
        synth::vits::build_text_encoder_graph(graph_context.get(), weights, hparams, 3);
    SYNTH_TEST_CHECK(graph.graph != nullptr);
    SYNTH_TEST_CHECK(graph.token_ids != nullptr && graph.token_ids->type == GGML_TYPE_I32 &&
                     graph.token_ids->ne[0] == 3);
    SYNTH_TEST_CHECK(graph.relative_indices != nullptr && graph.relative_indices->type == GGML_TYPE_I32 &&
                     graph.relative_indices->ne[0] == 9);
    SYNTH_TEST_CHECK(graph.encoded != nullptr && graph.encoded->ne[0] == 4 && graph.encoded->ne[1] == 3);
    SYNTH_TEST_CHECK(graph.m_p != nullptr && graph.m_p->ne[0] == 2 && graph.m_p->ne[1] == 3);
    SYNTH_TEST_CHECK(graph.logs_p != nullptr && graph.logs_p->ne[0] == 2 && graph.logs_p->ne[1] == 3);
    SYNTH_TEST_CHECK(std::strcmp(graph.m_p->name, "text.m_p") == 0);
    SYNTH_TEST_CHECK(std::strcmp(graph.logs_p->name, "text.logs_p") == 0);
    SYNTH_TEST_CHECK(std::strcmp(graph.encoded->name, "text.encoded") == 0);

    SYNTH_TEST_CHECK(synth::vits::build_text_encoder_graph(nullptr, weights, hparams, 3).graph == nullptr);
    SYNTH_TEST_CHECK(synth::vits::build_text_encoder_graph(graph_context.get(), weights, hparams, 0).graph == nullptr);
    SYNTH_TEST_CHECK(synth::vits::build_text_encoder_graph(graph_context.get(), weights, hparams, 9).graph == nullptr);
    synth::vits::TextWeights incomplete;
    SYNTH_TEST_CHECK(synth::vits::build_text_encoder_graph(graph_context.get(), incomplete, hparams, 3).graph ==
                     nullptr);
    return 0;
}
