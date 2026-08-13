#pragma once

#include "codec-encoder-host.h"
#include "model-info.h"
#include "speaker-encoder-host.h"
#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend_device;

namespace synth::qwen3tts {

// SpeakerEncoderWeights and CodecEncoderWeights are already forward-declared by
// speaker-encoder-host.h and codec-encoder-host.h above; HParams needs its own
// declaration for Model::hparams() below. Declared rather than included:
// catalog.h pulls in ggml types this header keeps out of its callers.
struct HParams;

struct ModelInfo {
    std::string family = "qwen3-tts";
    std::string variant;
    std::string quantization_profile;
    uint32_t    architecture_version = 0;

    synth_input_flags_t input_flags          = 0;
    uint32_t            capability_flags     = 0;
    uint32_t            output_sample_rate   = 0;
    uint32_t            output_channel_count = 0;
    uint64_t            max_input_tokens     = 0;
    uint64_t            max_output_frames    = 0;
    float               min_speaking_rate    = 0.0f;
    float               max_speaking_rate    = 0.0f;

    bool                     has_package_default = false;
    std::vector<std::string> preset_voice_ids;
    std::vector<std::string> language_names;
    // The same languages named in BCP-47, which is what the public interface
    // speaks. Shorter than `language_names` whenever the package carries an
    // entry that is not a requestable language -- the dialect overrides.
    std::vector<std::string> language_tags;

    bool        frontend_present = false;
    std::string frontend_provider;

    // The Voice Profile capability snapshot (weights.h's
    // fill_voice_profile_capability, gated on voice_mode -- see that
    // function's own doc comment for why not has_speaker_encoder), all-zero
    // for a preset-catalog package. src/synthesize.cpp's shared_info copies
    // this straight into synth::ModelInfo::voice_profile.
    VoiceProfileInfo voice_profile;
};

// One synthesis. `codes` is kept because the Port Validation Contract compares
// and replays them: they are the seam between the autoregressive half, which
// draws from a random stream, and the codec, which is deterministic.
struct SynthesisOutput {
    uint64_t             frame_count = 0;
    std::vector<int32_t> codes;  // [frame_count, code_group_count], level-major
    std::vector<float>   audio;  // frame_count * hop_length samples, one channel

    // Captured only when the request asks for them. Each is frame-major.
    std::vector<float>              talker_logits;  // [frame_count, codec_vocab_size]
    std::vector<float>              talker_final;   // [frame_count, hidden_size]
    std::vector<std::vector<float>> talker_layers;  // one per requested probe layer

    // Where the wall clock went, so a backend decision is made against a
    // measurement rather than an assumption. See docs/backends.md, which
    // requires the cost of the discrete-output rule to be measured per family.
    double talker_seconds          = 0.0;
    double predictor_seconds       = 0.0;
    double codec_seconds           = 0.0;
    // The share of the predictor's time spent creating a scheduler and placing a
    // graph rather than computing one.
    double predictor_setup_seconds = 0.0;

    // Where each stage's nodes actually ran, summed over the whole synthesis.
    //
    // This is the evidence behind the Golden Manifest's `backend_placement`
    // check, which every case in every family has declared since the schema was
    // written and which nothing has ever read. Counting nodes is the only way to
    // tell a graph that ran on an accelerator from one that was placed there and
    // fell back: a device that reports as present proves nothing about a node.
    //
    // The rule this exists to enforce is docs/backends.md's: a sampled code is a
    // discrete output, so the talker and the code predictor must stay on the CPU
    // however the request asks for a backend, and only the codec may move.
    struct StagePlacement {
        uint64_t nodes             = 0;
        uint64_t accelerator_nodes = 0;
    };

    StagePlacement talker_placement;
    StagePlacement predictor_placement;
    StagePlacement codec_placement;
};

struct SynthesisRequest {
    std::vector<int32_t> token_ids;  // the tokenized assistant turn
    std::string          voice_id;
    std::string          language;   // "auto" selects the no-think prompt
    uint64_t             seed        = 0;
    bool                 sample      = true;
    // Zero means "whatever the package ships", which is the normal case. These
    // are not this port's numbers to choose: the checkpoint carries them, and
    // keeping a second copy here is how its repetition penalty went missing.
    float                temperature = 0.0f;
    uint32_t             top_k       = 0;
    float                top_p       = 0.0f;
    int                  threads     = 0;
    // The most frames this request may emit. The talker's cache is sized from
    // it, so it bounds memory as well as length; zero takes the family default.
    uint64_t             max_frames  = 0;

    // When set, each frame's codes are taken from here instead of sampled.
    //
    // This is the Port Validation Contract's replay seam. The oracle draws from
    // PyTorch's generator and this port from its own, so parity replays the
    // codes rather than reproducing them; everything downstream of the draw is
    // then compared on identical inputs. Laid out [frames, code_group_count].
    const std::vector<int32_t> * replay_codes  = nullptr;
    uint64_t                     replay_frames = 0;

    // Talker layers whose output is captured per frame, for comparison against
    // the oracle's probes. Empty captures nothing and leaves the graph alone.
    std::vector<uint32_t> probe_layers;

    // Non-null selects the x-vector path: the speaker slot's embedding comes
    // from a prepared Voice Profile instead of a preset Voice's codec token,
    // and `voice_id` is not consulted. Mutually exclusive with naming a
    // preset Voice at this rung -- a request carries one speaker source or
    // the other, never both -- which is why this is a separate field rather
    // than an alternate meaning for `voice_id`. Its length must equal
    // hparams.talker.hidden_size; enc_dim == hidden_size is enforced at load
    // (weights.cpp), so a Profile prepared against this same Model always
    // satisfies it.
    const std::vector<float> * x_vector = nullptr;
};

// The talker attends over the whole utterance, so its cache grows with it: at
// 28 layers, 8 key/value heads and a head width of 128, one frame costs 229 kB.
// The package declares a limit of fifteen million frames, which is a generic
// "no limit" rather than a length anyone synthesizes, and sizing the cache from
// it asks for three terabytes. This ceiling is what a request gets when it does
// not ask for less: 2048 frames is about 164 seconds of audio and 481 MB of
// cache. A request wanting more must say so.
constexpr uint64_t kDefaultMaxFrames = 2048;

class Model {
  public:
    static synth_status_t load_cpu(const std::string & path, std::unique_ptr<Model> & output);
    static synth_status_t load(const std::string &      path,
                               ggml_backend_device *    primary_device,
                               bool                     include_accelerators,
                               std::unique_ptr<Model> & output);

    ~Model();
    Model(const Model &)             = delete;
    Model & operator=(const Model &) = delete;

    synth_status_t                      get_info(ModelInfo & output) const;
    ggml_backend_device *               primary_device() const;
    std::shared_ptr<const TextFrontend> text_frontend() const;

    // What the core runtime needs by value: the frame geometry it reports, and
    // the vocabulary the *frontend's* ids index, which is the text tower's and
    // not the codec's -- the core range-checks the tokens it is handed.
    uint32_t samples_per_frame() const;
    uint32_t text_vocab_size() const;

    // Wraps text in the assistant turn the reference uses and tokenizes it. The
    // template is a fixed string rather than something the package carries, so it
    // lives with the family rather than with the caller.
    synth_status_t tokenize_request(const std::string & text, std::vector<int32_t> & token_ids) const;

    // Tokenizes a reference transcript for transcript-assisted cloning, which
    // is NOT tokenize_request over different text: the reference wraps a
    // reference transcript in a shorter turn (bpe.h's qwen_reference_turn) and
    // slices the tokenized result at 3 and -2 where a request is sliced at 3
    // and -5. The ids returned are what upstream passes as `ref_id`, so they
    // are the sliced ones -- the turn's own markers are not in them.
    //
    // Not a `std::vector<int32_t>` return: an empty or whitespace-only
    // transcript is SYNTH_ERR_INVALID_ARG per the design's §9 error table, and
    // the limit and package-defect paths in qwen_reference_transcript_ids have
    // their own statuses too.
    synth_status_t tokenize_reference_transcript(const std::string & text, std::vector<int32_t> & token_ids) const;

    // Resolves a preset Voice and the codec language token for a request. A
    // speaker carrying a dialect override wins over the requested language.
    synth_status_t resolve_voice(const std::string & voice_id,
                                 const std::string & language,
                                 uint32_t &          speaker_token,
                                 bool &              has_language,
                                 uint32_t &          language_token) const;

    synth_status_t run_synthesis(const SynthesisRequest & request, SynthesisOutput & output) const;

    // Turns a finished code stream into audio. Split out because the Port
    // Validation Contract replays codes: the autoregressive half draws from a
    // random stream and the codec does not, so this is the deterministic side of
    // the seam and is compared on its own.
    synth_status_t decode_codes(const std::vector<int32_t> & codes,
                                uint64_t                     frame_count,
                                int                          threads,
                                std::vector<float> &         audio) const;

    // Reference audio to a speaker embedding. Split out from Voice Profile
    // preparation the way decode_codes is split from run_synthesis: this half
    // is deterministic and is compared against the oracle on its own.
    synth_status_t prepare_x_vector(const std::vector<float> & pcm_24k,
                                    int                        threads,
                                    XVectorEncoding &          output,
                                    const char *&              out_diagnostic_code,
                                    const char *&              out_diagnostic_message) const;

    // Reference audio to the [16, T] reference code grid, for the ICL path.
    // Split out from Voice Profile preparation on exactly the reasoning
    // prepare_x_vector's own comment gives: this half is deterministic and is
    // compared against the oracle on its own. The two are siblings and not
    // alternatives -- an ICL Profile carries both an x-vector and a code grid.
    synth_status_t prepare_codec_reference(const std::vector<float> & pcm_24k,
                                           int                        threads,
                                           CodecEncoding &            output,
                                           const char *&              out_diagnostic_code,
                                           const char *&              out_diagnostic_message) const;

    // What arch/qwen3-tts/profile.h's create_x_vector_profile needs from a
    // live Model, exposed as two small accessors rather than that function
    // taking a `Model &` directly -- see its own header comment for why:
    // `synth::qwen3tts::Model` has a private constructor reachable only
    // through `load`/`load_cpu`, both of which need a real GGUF on disk, and
    // this family has no synthetic-package test harness yet, so a `Model &`
    // parameter there would force create_x_vector_profile's own unit tests to
    // depend on the ~2.5 GB real package. src/voice-profile.cpp's
    // create_qwen3_tts_profile_from_reference is the one caller that needs
    // these from a REAL Loaded Model rather than a synthetic HParams fixture.
    const HParams &               hparams() const;
    const SpeakerEncoderWeights & speaker_encoder_weights() const;
    // The speech tokenizer's encoder half, for the ICL path. Both empty on a
    // CustomVoice package, which carries neither -- see build_model_weights.
    // Exposed on the same reasoning as the accessor above: the graph it feeds
    // takes a weights struct rather than a Model, so a synthetic fixture can
    // drive it, and only a caller holding a REAL Loaded Model needs this.
    const CodecEncoderWeights &   codec_encoder_weights() const;

  private:
    // The language half of resolve_voice, for a request whose speaker is an
    // external Voice Profile rather than a preset Voice: there is no
    // PresetVoice to consult, so no dialect override can win over the
    // requested language. Factored out rather than duplicated so the two
    // paths share one implementation of that rule instead of a second copy
    // free to drift from it.
    synth_status_t resolve_language_only(const std::string & language,
                                         bool &              has_language,
                                         uint32_t &          language_token) const;

    struct Impl;
    explicit Model(std::unique_ptr<Impl> implementation);
    std::unique_ptr<Impl> implementation_;
};

}  // namespace synth::qwen3tts
