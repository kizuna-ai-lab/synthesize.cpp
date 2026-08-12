// Task 10 review, IMPORTANT 1: a SynthesisRequest::x_vector whose length is
// not the talker's hidden size must be refused before it becomes a heap
// over-read, not merely documented by an assert() that NDEBUG compiles out
// in both trees this project ships.
//
// Model::run_synthesis constructs its speaker-embedding input tensor at
// hparams.talker.hidden_size regardless of request.x_vector's actual length,
// then copies that many bytes out of the vector via
// ggml_backend_tensor_set(..., ggml_nbytes(t_speaker_embedding)) -- so a
// vector one element short is read one float past its own allocation unless
// something refuses first. build_talker_prefill_input's own width check
// (talker.cpp) can never catch this: it compares the *tensor's* width, which
// is always built correct, never the caller-supplied vector's.
//
// This needs the real Base package: run_synthesis requires a loaded Model
// (private constructor, reachable only through load()/load_cpu(), and this
// family has no synthetic-package harness for the whole Model yet -- see
// qwen3-tts.h's own comment on why), and the vector's one correct length is
// hparams.talker.hidden_size, which only a real package's own metadata
// supplies honestly rather than a fixture's guess.
//
// Same review, Minor 1: the mutual-exclusivity refusal (a request naming both
// a preset Voice and an external embedding) is family-layer logic with the
// same problem -- no synthetic-package harness for the whole Model -- so it
// piggybacks on this file's already-loaded real Base Model rather than
// staying untested, now that the harness exists for Important 1 above.

#include "arch/qwen3-tts/qwen3-tts.h"
#include "arch/qwen3-tts/weights.h"
#include "synthesize.h"
#include "test-assert.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

int check_mismatched_x_vector_is_refused(const std::string & model_path) {
    std::unique_ptr<synth::qwen3tts::Model> model;
    SYNTH_TEST_CHECK(synth::qwen3tts::Model::load_cpu(model_path, model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);

    std::vector<int32_t> token_ids;
    SYNTH_TEST_CHECK(model->tokenize_request("Hi.", token_ids) == SYNTH_OK);

    const uint32_t hidden_size = model->hparams().talker.hidden_size;
    SYNTH_TEST_CHECK(hidden_size > 1);

    synth::qwen3tts::SynthesisRequest request;
    request.token_ids  = token_ids;
    request.sample     = false;
    request.max_frames = 1;

    synth::qwen3tts::SynthesisOutput output;

    // One element short of hidden_size -- exactly what would otherwise read
    // one float past this vector's own allocation.
    const std::vector<float> too_short(size_t(hidden_size) - 1, 0.25f);
    request.x_vector = &too_short;
    SYNTH_TEST_CHECK(model->run_synthesis(request, output) == SYNTH_ERR_INVALID_ARG);

    // The other side of the same rule: one element long is refused too, not
    // silently truncated.
    const std::vector<float> too_long(size_t(hidden_size) + 1, 0.25f);
    request.x_vector = &too_long;
    SYNTH_TEST_CHECK(model->run_synthesis(request, output) == SYNTH_ERR_INVALID_ARG);

    // The exact width is accepted at this gate -- it may still fail later for
    // unrelated reasons (there is no real speaker Profile behind this vector,
    // just filler values), but not for its length. sample=false and
    // max_frames=1 keep this cheap: the point is that run_synthesis gets past
    // the length check and into the graph, not that it produces plausible
    // audio.
    const std::vector<float> exact_width(size_t(hidden_size), 0.25f);
    request.x_vector = &exact_width;
    SYNTH_TEST_CHECK(model->run_synthesis(request, output) != SYNTH_ERR_INVALID_ARG);

    return 0;
}

// A request may carry a preset Voice or an external embedding, never both --
// this package's Preset Voice Catalog is empty (Base is profile-sources), so
// `voice_id` cannot even name a real entry, but the refusal fires before
// either speaker source is looked up, which is exactly why an unresolvable
// name still proves it: resolve_voice/resolve_language_only are never
// reached from this call.
int check_naming_both_speaker_sources_is_refused(const std::string & model_path) {
    std::unique_ptr<synth::qwen3tts::Model> model;
    SYNTH_TEST_CHECK(synth::qwen3tts::Model::load_cpu(model_path, model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);

    std::vector<int32_t> token_ids;
    SYNTH_TEST_CHECK(model->tokenize_request("Hi.", token_ids) == SYNTH_OK);

    const uint32_t hidden_size = model->hparams().talker.hidden_size;
    SYNTH_TEST_CHECK(hidden_size > 1);
    const std::vector<float> x_vector(size_t(hidden_size), 0.25f);

    synth::qwen3tts::SynthesisRequest request;
    request.token_ids  = token_ids;
    request.sample     = false;
    request.max_frames = 1;
    request.voice_id   = "aiden";  // this package has no catalog at all; the
                                   // name is never looked up.
    request.x_vector   = &x_vector;

    synth::qwen3tts::SynthesisOutput output;
    SYNTH_TEST_CHECK(model->run_synthesis(request, output) == SYNTH_ERR_INVALID_ARG);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        return 1;
    }
    SYNTH_TEST_CHECK(check_mismatched_x_vector_is_refused(argv[1]) == 0);
    SYNTH_TEST_CHECK(check_naming_both_speaker_sources_is_refused(argv[1]) == 0);
    return 0;
}
