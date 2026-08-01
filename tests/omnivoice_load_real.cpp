// The real omnivoice package, loaded twice: once through the family and once
// through the public C interface.
//
// This is the closing check of the foundation slice. Everything before it —
// converter, metadata reader, tensor catalog, frontend — was tested against
// synthetic packages, which prove the rules agree with themselves. Only the real
// package proves they agree with the checkpoint: a shape the catalog derives
// wrongly, a key the converter spells differently, or a vocabulary the frontend
// cannot build all surface here and nowhere earlier.
//
// The public half also proves the family is reachable and *bounded*: an
// invalid request must be refused by THIS family's own validation, because a
// family that fell through to another family's branch would produce audio
// from the wrong model rather than an error. It deliberately does not pay
// for a real synthesis (that is Plan 3's own public driver and Task 6's
// registered gate) -- an empty Linguistic Input is enough to prove the
// public seam reaches this family's code and refuses there, for free.

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/omnivoice.h"
#include "arch/omnivoice/weights.h"
#include "gguf.h"
#include "synthesize.h"
#include "test-assert.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

// The package's own geometry, from the family contract rather than from
// whatever the loader happened to read.
constexpr uint32_t kSamplesPerFrame = 960;
constexpr uint32_t kTextVocabSize   = 151676;
constexpr uint32_t kSampleRate      = 24000;
constexpr float    kMinSpeakingRate = 0.5f;
constexpr float    kMaxSpeakingRate = 2.0f;
// What the converter emitted, from its own report. The catalog derives the same
// number from the package's hyper-parameters; the two agreeing here is the only
// place where the converter's idea of the package and the loader's are compared
// against the package itself.
constexpr uint64_t kTensorCount     = 798;

// The three requestable languages. "auto" is not among them: it is the absence
// of a language, which the prompt's `None` slot carries.
const char * const kLanguageTags[] = { "en", "zh", "ja" };

std::string json_string(const std::string & value) {
    std::string escaped = "\"";
    for (char character : value) {
        if (character == '"' || character == '\\') {
            escaped.push_back('\\');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

void print_info_line(const synth::omnivoice::ModelInfo & info) {
    std::string line = "{";
    line += "\"family\":" + json_string(info.family);
    line += ",\"variant\":" + json_string(info.variant);
    line += ",\"quantization_profile\":" + json_string(info.quantization_profile);
    line += ",\"architecture_version\":" + std::to_string(info.architecture_version);
    line += ",\"input_flags\":" + std::to_string(info.input_flags);
    line += ",\"capability_flags\":" + std::to_string(info.capability_flags);
    line += ",\"output_sample_rate\":" + std::to_string(info.output_sample_rate);
    line += ",\"output_channel_count\":" + std::to_string(info.output_channel_count);
    line += ",\"max_input_tokens\":" + std::to_string(info.max_input_tokens);
    line += ",\"max_output_frames\":" + std::to_string(info.max_output_frames);
    line += ",\"min_speaking_rate\":" + std::to_string(info.min_speaking_rate);
    line += ",\"max_speaking_rate\":" + std::to_string(info.max_speaking_rate);
    line += ",\"has_package_default\":";
    line += info.has_package_default ? "true" : "false";
    line += ",\"language_tags\":[";
    for (size_t index = 0; index < info.language_tags.size(); ++index) {
        line += (index == 0 ? "" : ",") + json_string(info.language_tags[index]);
    }
    line += "]";
    line += ",\"frontend_present\":";
    line += info.frontend_present ? "true" : "false";
    line += ",\"frontend_provider\":" + json_string(info.frontend_provider);
    line += "}";
    std::cout << line << '\n';
}

// A sink that must never be written to: the request below names no
// Linguistic Input, so a chunk arriving here would mean either the new
// empty-input guard failed to fire or some other family's branch answered.
synth_sink_result_t SYNTH_CALL refuse_audio(void * user_data, const synth_audio_chunk_t * chunk) {
    (void) chunk;
    *static_cast<bool *>(user_data) = true;
    return SYNTH_SINK_CONTINUE;
}

struct SeenDiagnostic {
    std::string code;
    std::string message;
    bool        seen = false;
};

void SYNTH_CALL record_diagnostic(void * user_data, const synth_diagnostic_t * diagnostic) {
    auto * seen = static_cast<SeenDiagnostic *>(user_data);
    seen->code.assign(diagnostic->code, static_cast<size_t>(diagnostic->code_size));
    seen->message.assign(diagnostic->message, static_cast<size_t>(diagnostic->message_size));
    seen->seen = true;
}

int check_public_seam(const char * model_path) {
    synth_model_load_params_t params;
    synth_model_load_params_init(&params, sizeof(params));
    params.backend = SYNTH_BACKEND_CPU;

    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(model_path, &params, &model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);

    synth_model_capabilities_t capabilities;
    synth_model_capabilities_init(&capabilities, sizeof(capabilities));
    SYNTH_TEST_CHECK(synth_model_get_capabilities(model, &capabilities) == SYNTH_OK);
    SYNTH_TEST_CHECK(capabilities.output_sample_rate == kSampleRate);
    SYNTH_TEST_CHECK(capabilities.output_channel_count == 1);
    SYNTH_TEST_CHECK((capabilities.input_flags & SYNTH_INPUT_SUPPORT_TEXT_UTF8) != 0);
    SYNTH_TEST_CHECK(capabilities.min_speaking_rate == kMinSpeakingRate);
    SYNTH_TEST_CHECK(capabilities.max_speaking_rate == kMaxSpeakingRate);

    // A loaded model reports where it lives. Plan 1 runs no graph, but the
    // query is documented and answering it with a backend error would be a
    // public-seam defect rather than an unimplemented stage.
    synth_backend_device_t device;
    synth_backend_device_init(&device, sizeof(device));
    SYNTH_TEST_CHECK(synth_model_get_device(model, &device) == SYNTH_OK);
    SYNTH_TEST_CHECK(device.kind != nullptr);
    SYNTH_TEST_CHECK(std::string(device.kind) == "cpu");

    // The Preset Voice Catalog is empty by design; identity arrives through
    // Voice Profiles, and the package default is the unnamed auto-voice.
    uint64_t preset_count = 1;
    SYNTH_TEST_CHECK(synth_model_get_preset_voice_count(model, &preset_count) == SYNTH_OK);
    SYNTH_TEST_CHECK(preset_count == 0);

    uint64_t language_count = 0;
    SYNTH_TEST_CHECK(synth_model_get_language_count(model, &language_count) == SYNTH_OK);
    SYNTH_TEST_CHECK(language_count == 3);
    for (uint64_t index = 0; index < language_count; ++index) {
        synth_language_capability_t language;
        synth_language_capability_init(&language, sizeof(language));
        SYNTH_TEST_CHECK(synth_model_get_language(model, index, &language) == SYNTH_OK);
        const std::string tag(language.tag, static_cast<size_t>(language.tag_size));
        SYNTH_TEST_CHECK(tag == kLanguageTags[index]);
        // A region may fall back to the primary subtag, and no language is the
        // default: a request naming none gets the language-agnostic prompt.
        SYNTH_TEST_CHECK((language.flags & SYNTH_LANGUAGE_REGIONAL_FALLBACK) != 0);
        SYNTH_TEST_CHECK((language.flags & SYNTH_LANGUAGE_DEFAULT) == 0);
    }

    synth_context_t * context = nullptr;
    SYNTH_TEST_CHECK(synth_context_create(model, &context) == SYNTH_OK);

    // A request naming no Voice, which only an empty catalog with a package
    // default accepts — the public consequence of has_package_default. The
    // text itself is whitespace-only rather than truly empty: an
    // `input_count` of zero is refused by prepare_synthesis_request's own
    // generic guard before any family branch ever runs (see
    // src/synthesis-request.cpp), which would prove nothing about this
    // family's own code. Three spaces has a nonzero byte count, so the
    // request reaches this family's branch in src/synthesize.cpp and
    // Model::synthesize's own "no Linguistic Input" guard (Plan 3's Task 5)
    // -- cheap and graph-free, unlike a real synthesis, but still proof that
    // this family's own code is what answers.
    const char *   text = "   ";
    SeenDiagnostic diagnostic;
    bool           wrote_audio = false;

    synth_diagnostic_sink_t diagnostics;
    synth_diagnostic_sink_init(&diagnostics, sizeof(diagnostics));
    diagnostics.emit      = record_diagnostic;
    diagnostics.user_data = &diagnostic;

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind  = SYNTH_INPUT_TEXT_UTF8;
    request.input_data  = text;
    request.input_count = std::strlen(text);
    request.diagnostics = &diagnostics;

    synth_audio_sink_t sink;
    synth_audio_sink_init(&sink, sizeof(sink));
    sink.write     = refuse_audio;
    sink.user_data = &wrote_audio;

    synth_result_t result;
    synth_result_init(&result, sizeof(result));
    // A request naming no Linguistic Input at all is refused by this
    // family's own validation (Model::synthesize's empty/whitespace guard,
    // which runs before any estimate or graph work) -- not audio, and not
    // another family's answer. src/synthesize.cpp's omnivoice branch maps
    // any non-OK, non-OUTPUT_LIMIT status through the generic
    // "synthesis.graph_failed" diagnostic, the same path every other
    // family's graph failure takes.
    SYNTH_TEST_CHECK(synth_synthesize(context, &request, &sink, &result) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(!wrote_audio);
    SYNTH_TEST_CHECK(diagnostic.seen);
    SYNTH_TEST_CHECK(diagnostic.code == "synthesis.graph_failed");

    synth_context_free(context);
    synth_model_free(model);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: synthesize-omnivoice-load-real <model.gguf>\n";
        return 2;
    }
    const std::string model_path = argv[1];

    // Emitted against expected, before anything is loaded. The load path below
    // proves every tensor resolves and none is outside the catalog; this proves
    // the two counts are the same number, which a catalog that resolved one
    // name twice would otherwise hide.
    {
        gguf_init_params parameters{};
        parameters.no_alloc = true;
        parameters.ctx      = nullptr;
        gguf_context * gguf = gguf_init_from_file(model_path.c_str(), parameters);
        SYNTH_TEST_CHECK(gguf != nullptr);
        synth::omnivoice::HParams hparams;
        const synth_status_t      status  = synth::omnivoice::read_hparams(gguf, hparams);
        const uint64_t            emitted = uint64_t(gguf_get_n_tensors(gguf));
        gguf_free(gguf);
        SYNTH_TEST_CHECK(status == SYNTH_OK);
        SYNTH_TEST_CHECK(emitted == kTensorCount);
        SYNTH_TEST_CHECK(synth::omnivoice::expected_tensor_count(hparams) == emitted);
    }

    std::unique_ptr<synth::omnivoice::Model> model;
    SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(model_path, model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);

    synth::omnivoice::ModelInfo info;
    SYNTH_TEST_CHECK(model->get_info(info) == SYNTH_OK);
    print_info_line(info);

    SYNTH_TEST_CHECK(info.family == "omnivoice");
    SYNTH_TEST_CHECK(info.variant == "omnivoice-0-6b");
    SYNTH_TEST_CHECK(info.quantization_profile == "F32");
    SYNTH_TEST_CHECK(info.architecture_version == 1);
    SYNTH_TEST_CHECK(info.output_sample_rate == kSampleRate);
    SYNTH_TEST_CHECK(info.output_channel_count == 1);
    SYNTH_TEST_CHECK(info.has_package_default);
    SYNTH_TEST_CHECK(info.min_speaking_rate == kMinSpeakingRate);
    SYNTH_TEST_CHECK(info.max_speaking_rate == kMaxSpeakingRate);
    SYNTH_TEST_CHECK(info.language_tags.size() == 3);
    for (size_t index = 0; index < info.language_tags.size(); ++index) {
        SYNTH_TEST_CHECK(info.language_tags[index] == kLanguageTags[index]);
    }
    SYNTH_TEST_CHECK(info.frontend_present);
    SYNTH_TEST_CHECK(info.frontend_provider == "synthesize.qwen_bpe");

    SYNTH_TEST_CHECK(model->samples_per_frame() == kSamplesPerFrame);
    SYNTH_TEST_CHECK(model->text_vocab_size() == kTextVocabSize);

    // The frontend is built from the package's own vocabulary and merges, so a
    // package that declares one and carries no arrays never reaches here.
    const std::shared_ptr<const synth::TextFrontend> frontend = model->text_frontend();
    SYNTH_TEST_CHECK(frontend != nullptr);
    std::vector<int32_t> ids;
    SYNTH_TEST_CHECK(frontend->prepare(SYNTH_INPUT_TEXT_UTF8, "Hello.", 6, info.max_input_tokens, ids) == SYNTH_OK);
    SYNTH_TEST_CHECK(!ids.empty());
    for (int32_t id : ids) {
        SYNTH_TEST_CHECK(id >= 0 && uint32_t(id) < kTextVocabSize);
    }

    model.reset();
    return check_public_seam(model_path.c_str());
}
