// The OmniVoice model object: what a package becomes once it is loaded.
//
// Plan 1 loads and reports; the diffusion loop, the codec and the cloning path
// are Plan 2's. That split is deliberate rather than partial work — the load
// path is the only place where the converter, the metadata reader, the tensor
// catalog and the frontend meet, and it is worth closing on the real package
// before any graph exists to blame a wrong number on.
//
// Placement: everything is CPU. This family's canvas is a table of sampled
// codes, so docs/backends.md's discrete-output rule holds the generator and its
// whole input path on the CPU; the codec could move, but there is no codec graph
// yet and no measurement to move it on. The weights therefore live in one CPU
// buffer with no accelerator twin — the seam qwen3-tts carries for its codec
// half arrives with the stage that can use it.

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/omnivoice.h"
#include "arch/omnivoice/weights.h"
#include "backend-plan.h"
#include "bpe-frontend.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-metadata.h"
#include "gguf.h"

#include <fstream>
#include <new>
#include <utility>

namespace synth::omnivoice {

struct Model::Impl {
    gguf_context *                      gguf            = nullptr;
    ggml_context *                      weights_context = nullptr;
    std::unique_ptr<BackendPlan>        backend_plan;
    ggml_backend_buffer_t               weights_buffer = nullptr;
    HParams                             hparams;
    std::shared_ptr<const TextFrontend> frontend;
    ModelWeights                        weights;

    ~Impl() {
        if (weights_buffer != nullptr) {
            ggml_backend_buffer_free(weights_buffer);
        }
        if (weights_context != nullptr) {
            ggml_free(weights_context);
        }
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
    }
};

Model::Model(std::unique_ptr<Impl> implementation) : implementation_(std::move(implementation)) {}

Model::~Model() = default;

synth_status_t Model::get_info(ModelInfo & output) const {
    output                  = ModelInfo{};
    const HParams & hparams = implementation_->hparams;

    output.variant              = hparams.model_variant;
    output.architecture_version = hparams.architecture_version;
    switch (hparams.quantization_profile) {
        case QuantizationProfile::F32:
            output.quantization_profile = "F32";
            break;
    }
    output.input_flags          = hparams.input_flags;
    output.capability_flags     = hparams.capability_flags;
    output.output_sample_rate   = hparams.output_sample_rate;
    output.output_channel_count = hparams.output_channel_count;
    output.max_input_tokens     = hparams.max_input_tokens;
    output.max_output_frames    = hparams.max_output_frames;
    output.min_speaking_rate    = hparams.min_speaking_rate;
    output.max_speaking_rate    = hparams.max_speaking_rate;
    // Auto-voice is the package default and the Preset Voice Catalog is empty;
    // the metadata reader refuses a package that says otherwise, so there is no
    // per-package value to carry here.
    output.has_package_default  = true;
    // No name bridge: this family's tags are already BCP-47, and the same string
    // is what the prompt's language slot carries.
    output.language_tags        = hparams.language_tags;
    output.frontend_present     = hparams.frontend_present;
    output.frontend_provider    = hparams.frontend_provider;
    return SYNTH_OK;
}

ggml_backend_device * Model::primary_device() const {
    return implementation_->backend_plan->primary_device();
}

std::shared_ptr<const TextFrontend> Model::text_frontend() const {
    return implementation_->frontend;
}

uint32_t Model::samples_per_frame() const {
    return implementation_->hparams.codec.hop_length;
}

uint32_t Model::text_vocab_size() const {
    return implementation_->hparams.generator.text_vocab_size;
}

synth_status_t Model::load_cpu(const std::string & path, std::unique_ptr<Model> & output) {
    return load(path, ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), false, output);
}

synth_status_t Model::load(const std::string &      path,
                           ggml_backend_device *    primary_device,
                           bool                     include_accelerators,
                           std::unique_ptr<Model> & output) {
    output.reset();
    if (path.empty()) {
        return SYNTH_ERR_INVALID_ARG;
    }
    std::ifstream probe(path, std::ios::binary);
    if (!probe) {
        return SYNTH_ERR_FILE_NOT_FOUND;
    }
    probe.close();

    try {
        auto           implementation = std::make_unique<Impl>();
        synth_status_t status = BackendPlan::create(primary_device, include_accelerators, implementation->backend_plan);
        if (status != SYNTH_OK) {
            return status;
        }

        gguf_init_params parameters{};
        parameters.no_alloc  = true;
        parameters.ctx       = &implementation->weights_context;
        implementation->gguf = gguf_init_from_file(path.c_str(), parameters);
        if (implementation->gguf == nullptr || implementation->weights_context == nullptr) {
            return SYNTH_ERR_GGUF;
        }
        status = read_hparams(implementation->gguf, implementation->hparams);
        if (status != SYNTH_OK) {
            return status;
        }
        status = build_model_weights(implementation->weights_context, implementation->hparams, implementation->weights);
        if (status != SYNTH_OK) {
            return status;
        }

        if (implementation->hparams.frontend_present) {
            GgufMetadata      meta(implementation->gguf, "omnivoice");
            BpeFrontendConfig config;
            config.provider_id      = implementation->hparams.frontend_provider;
            config.contract_version = implementation->hparams.frontend_contract_version;
            if (!meta.string_array("synthesize.omnivoice.frontend.vocab", config.vocab) ||
                !meta.string_array("synthesize.omnivoice.frontend.merges", config.merges)) {
                return SYNTH_ERR_GGUF;
            }
            // The prompt markers are added tokens past the vocabulary; they
            // carry their ids. No prefix/suffix: this family's prompt is
            // assembled by the synthesis path, not wrapped at tokenize time.
            const SpecialTokens & t = implementation->hparams.tokens;
            config.special_tokens   = {
                { "<|denoise|>",        int32_t(t.denoise)        },
                { "<|lang_start|>",     int32_t(t.lang_start)     },
                { "<|lang_end|>",       int32_t(t.lang_end)       },
                { "<|instruct_start|>", int32_t(t.instruct_start) },
                { "<|instruct_end|>",   int32_t(t.instruct_end)   },
                { "<|text_start|>",     int32_t(t.text_start)     },
                { "<|text_end|>",       int32_t(t.text_end)       },
            };
            std::unique_ptr<TextFrontend> frontend;
            status = make_bpe_frontend(config, frontend);
            if (status != SYNTH_OK) {
                return status;
            }
            implementation->frontend = std::shared_ptr<const TextFrontend>(std::move(frontend));
        }

        // One CPU buffer, no twin; see the placement note at the top of this
        // file.
        implementation->weights_buffer = ggml_backend_alloc_ctx_tensors(implementation->weights_context,
                                                                        implementation->backend_plan->cpu_backend());
        if (implementation->weights_buffer == nullptr) {
            return SYNTH_ERR_OOM;
        }
        ggml_backend_buffer_set_usage(implementation->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        status = stream_tensor_data(path, implementation->gguf, implementation->weights_context, "omnivoice");
        if (status != SYNTH_OK) {
            return status;
        }

        output = std::unique_ptr<Model>(new Model(std::move(implementation)));
        return SYNTH_OK;
    } catch (const std::bad_alloc &) {
        return SYNTH_ERR_OOM;
    } catch (...) {
        return SYNTH_ERR_INTERNAL;
    }
}

}  // namespace synth::omnivoice
