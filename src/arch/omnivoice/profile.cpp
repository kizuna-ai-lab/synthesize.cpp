#include "arch/omnivoice/profile.h"

#include "arch/omnivoice/frontend-host.h"
#include "arch/omnivoice/omnivoice.h"
#include "text-frontend.h"

#include <utility>

namespace synth::omnivoice {

synth_status_t create_clone_prompt(Model &                              model,
                                   const std::vector<float> &           pcm_24k,
                                   const std::string &                  transcript,
                                   const std::string &                  language_tag,
                                   int                                  threads,
                                   std::shared_ptr<const ClonePrompt> & output,
                                   const char *&                        out_diagnostic_code,
                                   const char *&                        out_diagnostic_message) {
    output.reset();
    out_diagnostic_code    = nullptr;
    out_diagnostic_message = nullptr;

    ReferenceEncoding    encoding;
    const synth_status_t encode_status = model.encode_reference(pcm_24k, threads, encoding);
    if (encode_status != SYNTH_OK) {
        return encode_status;
    }

    // jiangzhuo's ruling, 2026-08-01 (docs/porting/families/omnivoice.md's
    // "Silent-Reference Rejection" note): upstream's own `_post_process_audio`
    // (omnivoice/models/omnivoice.py:898-899) multiplies a digitally silent
    // reference's synthesized output by zero rather than refusing the
    // request, so a silent clip still "succeeds" and hands back silence. This
    // port refuses instead -- silence is not a cloned voice, and shipping one
    // as if it were would be indistinguishable, from the caller's side, from
    // a real defect swallowing the whole reference. Deliberate divergence,
    // not an oversight.
    if (encoding.ref_rms == 0.0f) {
        out_diagnostic_code    = "voice_profile.reference_silent";
        out_diagnostic_message = "the reference audio is digitally silent; nothing can be cloned from it";
        return SYNTH_ERR_INVALID_ARG;
    }

    const std::string                         canonical_transcript = add_punctuation(transcript);
    std::vector<int32_t>                      transcript_ids;
    const std::shared_ptr<const TextFrontend> frontend = model.text_frontend();
    if (frontend == nullptr || frontend->prepare(SYNTH_INPUT_TEXT_UTF8, canonical_transcript.data(),
                                                 canonical_transcript.size(), 0, transcript_ids) != SYNTH_OK) {
        out_diagnostic_code    = "voice_profile.transcript_unencodable";
        out_diagnostic_message = "the reference transcript contains a byte this package's frontend has no token for";
        return SYNTH_ERR_INVALID_ARG;
    }

    auto prompt              = std::make_shared<ClonePrompt>();
    prompt->reference_tokens = std::move(encoding.tokens);
    prompt->transcript_ids   = std::move(transcript_ids);
    prompt->transcript_text  = canonical_transcript;
    prompt->ref_rms          = encoding.ref_rms;
    prompt->language_tag     = language_tag;
    output                   = std::move(prompt);
    return SYNTH_OK;
}

}  // namespace synth::omnivoice
