#pragma once

#include "arch/kokoro/kokoro.h"
#include "arch/omnivoice/omnivoice.h"
#include "arch/qwen3-tts/qwen3-tts.h"
#include "arch/vits/vits.h"
#include "model-info.h"

#include <memory>

// A Loaded Model is one family's model behind the family-independent info the
// core runtime reads. Only one implementation pointer is ever set, and `info`
// says which.
//
// Originally synthesize.cpp's own file-local type (opaque everywhere else via
// synthesize.h's forward declaration of `synth_model_t`); moved out to this
// shared header once a second translation unit needed the same complete
// type: src/voice-profile.cpp's Voice Profile dispatch reads `info.family`
// and calls into a family's own Model through the matching pointer, exactly
// the way synthesize.cpp's synthesis dispatch already does.
struct synth_model {
    synth::ModelInfo                         info;
    std::unique_ptr<synth::vits::Model>      vits;
    std::unique_ptr<synth::kokoro::Model>    kokoro;
    std::unique_ptr<synth::qwen3tts::Model>  qwen3_tts;
    std::unique_ptr<synth::omnivoice::Model> omnivoice;
    // What only the VITS synthesis path reads; Kokoro's equivalents live
    // behind its own seeded entry point.
    synth::vits::ModelInfo                   vits_extras;
};
