#include "cli.h"

#include <cstdio>
#include <memory>

namespace {

void SYNTH_CALL print_diagnostic(void *, const synth_diagnostic_t * diagnostic) {
    if (diagnostic == nullptr) {
        return;
    }
    std::fputs("synthesize-cli: ", stderr);
    if (diagnostic->code != nullptr && diagnostic->code_size != 0) {
        std::fwrite(diagnostic->code, 1, static_cast<size_t>(diagnostic->code_size), stderr);
        std::fputs(": ", stderr);
    }
    if (diagnostic->message != nullptr && diagnostic->message_size != 0) {
        std::fwrite(diagnostic->message, 1, static_cast<size_t>(diagnostic->message_size), stderr);
    }
    std::fputc('\n', stderr);
}

struct ModelDeleter {
    void operator()(synth_model_t * model) const { synth_model_free(model); }
};

struct ContextDeleter {
    void operator()(synth_context_t * context) const { synth_context_free(context); }
};

struct AudioDeleter {
    void operator()(synth_audio_buffer_t * audio) const { synth_audio_buffer_free(audio); }
};

}  // namespace

int main(int argc, char ** argv) {
    synth_cli::Options options;
    std::string        error;
    if (!synth_cli::parse_arguments(argc, argv, options, error)) {
        std::fprintf(stderr, "synthesize-cli: %s\n\n%s", error.c_str(), synth_cli::usage_text());
        return 2;
    }
    if (options.show_help) {
        std::fputs(synth_cli::usage_text(), stdout);
        return 0;
    }

    synth_diagnostic_sink_t diagnostics;
    synth_diagnostic_sink_init(&diagnostics, sizeof(diagnostics));
    diagnostics.emit = print_diagnostic;

    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof(load_params));
    load_params.backend      = options.backend;
    load_params.device_index = options.device_index;
    load_params.diagnostics  = &diagnostics;

    synth_model_t * raw_model = nullptr;
    synth_status_t  status = synth_model_load(options.model_path.c_str(), &load_params, &raw_model);
    std::unique_ptr<synth_model_t, ModelDeleter> model(raw_model);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "synthesize-cli: model load failed: %s\n", synth_status_string(status));
        return 1;
    }

    synth_context_t * raw_context = nullptr;
    status = synth_context_create(model.get(), &raw_context);
    std::unique_ptr<synth_context_t, ContextDeleter> context(raw_context);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "synthesize-cli: context creation failed: %s\n", synth_status_string(status));
        return 1;
    }

    synth_request_t request;
    synth_request_init(&request, sizeof(request));
    request.input_kind = options.input_kind;
    if (options.input_kind == SYNTH_INPUT_TOKEN_IDS) {
        request.input_data  = options.token_ids.data();
        request.input_count = options.token_ids.size();
    } else {
        request.input_data  = options.linguistic_input.data();
        request.input_count = options.linguistic_input.size();
    }
    if (!options.language_tag.empty()) {
        request.language_tag      = options.language_tag.data();
        request.language_tag_size = options.language_tag.size();
    }
    if (!options.voice_id.empty()) {
        request.voice_id      = options.voice_id.data();
        request.voice_id_size = options.voice_id.size();
    }
    request.seed              = options.seed;
    request.speaking_rate     = options.speaking_rate;
    request.diagnostics       = &diagnostics;
    request.max_output_frames = options.max_output_frames;

    synth_result_t result;
    synth_result_init(&result, sizeof(result));
    synth_audio_buffer_t * raw_audio = nullptr;
    status = synth_synthesize_to_buffer(context.get(), &request, &raw_audio, &result);
    std::unique_ptr<synth_audio_buffer_t, AudioDeleter> audio(raw_audio);
    if (audio == nullptr) {
        std::fprintf(stderr, "synthesize-cli: synthesis failed: %s\n", synth_status_string(status));
        return 1;
    }

    if (!synth_cli::write_f32_wav(options.output_path, audio->samples, audio->frame_count, audio->sample_rate,
                                  audio->channel_count, error)) {
        std::fprintf(stderr, "synthesize-cli: %s\n", error.c_str());
        return 1;
    }

    std::fprintf(stdout, "wrote %llu frames at %u Hz to %s",
                 static_cast<unsigned long long>(audio->frame_count), audio->sample_rate, options.output_path.c_str());
    if ((result.flags & SYNTH_RESULT_SEED_USED) != 0) {
        std::fprintf(stdout, " (seed %llu)", static_cast<unsigned long long>(result.actual_seed));
    }
    std::fputc('\n', stdout);

    if (status != SYNTH_OK) {
        std::fprintf(stderr, "synthesize-cli: partial output: %s\n", synth_status_string(status));
        return 1;
    }
    return 0;
}
