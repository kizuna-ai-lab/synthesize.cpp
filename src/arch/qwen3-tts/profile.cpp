#include "arch/qwen3-tts/profile.h"

#include "arch/qwen3-tts/speaker-encoder-host.h"
#include "arch/qwen3-tts/weights.h"

#include <utility>

namespace synth::qwen3tts {

synth_status_t create_x_vector_profile(const HParams &                         hparams,
                                       const SpeakerEncoderWeights &           speaker_encoder,
                                       const std::vector<float> &              pcm_24k,
                                       const std::string &                     transcript,
                                       const std::string &                     language_tag,
                                       int                                     threads,
                                       std::shared_ptr<const XVectorProfile> & output,
                                       const char *&                           out_diagnostic_code,
                                       const char *&                           out_diagnostic_message) {
    output.reset();
    out_diagnostic_code    = nullptr;
    out_diagnostic_message = nullptr;

    // D4: the clone mode is fixed when the Profile is created. Plan 2
    // implements exactly one mode, so a transcript names a mode with no
    // implementation behind it -- accepting it and silently building the
    // x-vector Profile anyway would hand back the weaker clone the caller
    // did not ask for. Checked first, before the encode chain below ever
    // runs: there is no reason to pay for a graph pass over reference audio
    // for a request this function is about to refuse anyway. A
    // whitespace-only transcript is "present" by this check and is refused
    // the same way -- Plan 3 is what gives "empty" and "whitespace" distinct
    // meanings (the spec's section 9 error table), and until it lands both
    // are simply "a transcript was supplied".
    if (!transcript.empty()) {
        out_diagnostic_code = "voice_profile.transcript_unsupported";
        out_diagnostic_message =
            "this package's Plan 2 runtime implements x-vector cloning only; a reference transcript names the "
            "transcript-assisted mode, which has no implementation yet";
        return SYNTH_ERR_INVALID_ARG;
    }

    // Model::prepare_x_vector's own guard, reproduced here rather than
    // reached through it: a CustomVoice package resolves no
    // SpeakerEncoderWeights at all (build_model_weights leaves
    // weights.speaker_encoder default-constructed for it), so this refuses
    // before encode_speaker_reference ever sees a weights struct with every
    // pointer null.
    if (!hparams.has_speaker_encoder) {
        return SYNTH_ERR_UNSUPPORTED_VOICE;
    }

    XVectorEncoding      encoding;
    const synth_status_t encode_status = encode_speaker_reference(hparams, speaker_encoder, pcm_24k, threads, encoding,
                                                                  out_diagnostic_code, out_diagnostic_message);
    if (encode_status != SYNTH_OK) {
        // encode_speaker_reference's own refusals -- including the
        // "voice_profile.reference_silent" digitally-silent-reference
        // rejection -- already set the diagnostic out-params; propagate
        // both unchanged rather than re-deriving them here.
        return encode_status;
    }

    auto profile          = std::make_shared<XVectorProfile>();
    profile->mode         = CloneMode::XVector;
    profile->x_vector     = std::move(encoding.x_vector);
    profile->ref_rms      = encoding.ref_rms;
    profile->language_tag = language_tag;
    output                = std::move(profile);
    return SYNTH_OK;
}

}  // namespace synth::qwen3tts
