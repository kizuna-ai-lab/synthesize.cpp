#pragma once

#include "arch/omnivoice/generator-host.h"
#include "synthesize.h"
#include "text-frontend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_backend_device;

namespace synth::omnivoice {

struct ModelInfo {
    std::string family = "omnivoice";
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

    // Auto-voice is the unnamed package default; the Preset Voice Catalog is
    // deliberately empty. Voice identity arrives through profiles.
    bool                     has_package_default = true;
    std::vector<std::string> language_tags;  // already BCP-47; no name bridge

    bool        frontend_present = false;
    std::string frontend_provider;
};

// A family-internal synthesis request. Plan 2 reached this through the
// replay runner only; Plan 3's Task 5 adds the public path Model::synthesize
// builds one from, below. The prompt arrives as ids (the oracle's
// input/token_ids.i32 under replay, or assemble_prompt_ids's output on the
// public path) and the canvas length arrives fixed -- the duration
// estimator's output, or the oracle grid's frame count under replay.
struct SynthesisRequest {
    // Row 0's text region: style markers, language/instruct slots and the
    // wrapped text, already tokenized. Every codebook row repeats these ids.
    std::vector<int32_t>  prompt_text_ids;
    // Reference audio tokens, codebook-major [num_codebooks * frames]; empty
    // for auto-voice and voice-design requests. Plan 3's cloning encoder
    // produces these from audio; Plan 2's runner replays the oracle's.
    std::vector<int32_t>  reference_tokens;
    uint64_t              target_frames        = 0;
    uint32_t              num_step             = 0;  // 0 = the package's embedded default
    // The seed for this synthesis' one NormalRandomStream. Consumed only when
    // a resolved temperature below is positive; a fully greedy synthesis
    // (both temperatures resolve to 0) constructs no stream at all and never
    // reads this field -- the family's recorded zero-RNG property.
    uint64_t              seed                 = 0;
    // Temperature overrides for the mask-predict loop's two Gumbel draws
    // (generator-host.h's gumbel_perturb / choose_token_sampled). Negative
    // means "the package's own default governs" (HParams::generation);
    // 0.0f is a meaningful value -- greedy, not "unset". The replay runner
    // (tests/omnivoice_replay_real.cpp) pins both to 0.0f, which is what
    // keeps every Plan 2 golden grid exact; the public path Plan 3 adds
    // leaves both at -1.0f so the package's own sampling defaults govern.
    float                 position_temperature = -1.0f;
    float                 class_temperature    = -1.0f;
    // Stop after the step-0 conditional forward with the probe buffers filled;
    // the sampled golden cases compare only that forward in Plan 2.
    bool                  probe_only           = false;
    // Measure how narrowly the greedy loop's decisions were made and report the
    // narrowest one (see MarginReport). Off by default: nothing in a synthesis
    // needs it, and it is a screening instrument for golden-case selection.
    bool                  margin_report        = false;
    int                   threads              = 0;  // 0 = default_synthesis_threads()
    std::vector<uint32_t> probe_layers;              // layer indices probed at step 0
};

// MarginReport is declared in generator-host.h: filled when
// SynthesisRequest::margin_report is set, and its NaN-wins update rule
// (note_margin) lives there too, where it is unit-testable without a Model.

// The Reference Audio and Serialized Profile payload a cloning request
// carries. Defined in profile.h from Plan 3's Task 14 on; forward-declared
// here so this task's PublicSynthesisParams can carry the pointer this
// family's own request eventually threads through, without this file
// depending on a header that does not exist yet.
struct ClonePrompt;

// The public seam's own request shape: everything `src/synthesize.cpp`'s
// omnivoice branch collects from a `synth_request_t` before it ever touches
// a tensor. Distinct from SynthesisRequest -- that one already carries
// tokenized ids and a settled canvas length; this one carries the raw
// strings and rates Model::synthesize turns into those by calling
// assemble_prompt_ids and DurationEstimator itself.
//
// `clone` and `instruct` are pointers rather than values so "absent" and
// "empty string" stay distinguishable: an instruction of "" is a legal,
// meaningful request (the language-agnostic, instruction-free style), while
// a null pointer means the caller supplied neither. Both stay null through
// this task -- Task 14 threads Reference Audio through `clone`, Task 15
// threads the free-text instruction through `instruct` -- so the struct's
// shape does not change again when those land.
struct PublicSynthesisParams {
    std::string         text;                         // Linguistic Input, UTF-8
    std::string         language_tag;                 // resolved by core; may be empty
    const ClonePrompt * clone             = nullptr;  // Task 14 threads this; null now
    const std::string * instruct          = nullptr;  // Task 15 threads this; null now
    double              speaking_rate     = 1.0;
    uint64_t            seed              = 0;
    uint64_t            max_output_frames = 0;  // native frames; 0 = package cap
    int32_t             threads           = 0;
};

struct SynthesisOutput {
    // The canvas length, set from the request rather than from the loop, so the
    // probe-only path reports it too. Empty on neither path: a request that
    // reaches a forward has a canvas.
    uint64_t             frame_count = 0;
    // The committed grid, codebook-major [num_codebooks * frame_count] --
    // codebook c, frame t at c * frame_count + t, the oracle's codes/grid.i32
    // layout exactly.
    std::vector<int32_t> codes;
    // The decoded waveform with the no-reference volume branch applied
    // (peak-normalise-to-0.5), which is what the oracle returns to its caller
    // for auto-voice and voice-design requests. Plan 2 applies that branch
    // unconditionally, including to a request carrying reference tokens: the
    // other two arms of the oracle's chain key off a reference RMS that only
    // Plan 3's cloning path can supply. Empty on the probe-only path.
    std::vector<float>   audio;

    // Step-0 conditional probes, in ggml read-back order; the runner reorders
    // logits into the oracle's [C, S, V] layout on write.
    std::vector<float>              logits_step0;  // [positions][codebooks][vocab]
    std::vector<float>              final_hidden;  // [positions][hidden]
    std::vector<std::vector<float>> layer_hidden;  // one per requested layer

    // Summed over every graph a stage ran, not one graph's count: a greedy
    // decode makes two generator forwards per step (conditional and
    // unconditional), so `generator_placement.nodes` reads as ~64 x one graph's
    // node count on a 32-step run and only equals one graph's on the
    // probe-only path. `accelerator_nodes` is a total for the same reason --
    // the CPU-only rule is "this stays zero", which a sum states just as well.
    struct StagePlacement {
        uint64_t nodes             = 0;
        uint64_t accelerator_nodes = 0;
    };

    // Also totals across forwards, for the same reason.
    double         generator_seconds       = 0.0;
    double         generator_setup_seconds = 0.0;
    double         codec_seconds           = 0.0;
    StagePlacement generator_placement;
    StagePlacement codec_placement;

    // Left unmeasured unless the request asked for it, and on the probe-only
    // path there is no greedy loop to measure.
    MarginReport margin;
};

// The cloning path's full encode-chain result (Task 13): reference-encoder-
// host.h's hop-clip -> ref_rms -> quiet-boost -> resample -> semantic branch
// -> acoustic branch + fusion -> RVQ nearest-neighbour encode, run by
// Model::encode_reference below. A struct rather than a growing out-parameter
// list: the debugging taps alone (pcm_16k, semantic_mean, fused_latent,
// semantic_encoder_output) already match run_semantic_branch's/
// run_acoustic_and_fuse's own out_-parameter count, and the discrete result
// (tokens, frames, ref_rms, the margin pair) adds four more.
struct ReferenceEncoding {
    // The discrete decision: `frames` frames across every resolved codebook
    // level, codebook-major (level l, frame t at l * frames + t) -- the
    // oracle's own ref/tokens.i32 layout, and SynthesisRequest::reference_tokens'
    // own layout.
    std::vector<int32_t> tokens;
    uint64_t             frames = 0;

    // The reference's loudness, PRE-boost -- VoiceClonePrompt.ref_rms's own
    // contract (see reference-encoder-host.h's clip_and_boost_reference for
    // exactly which segment this is measured over). Task 14's volume arms
    // read this; encode_reference itself never rejects on it.
    float ref_rms = 0.0f;

    // RVQ margin instrumentation (diagnostic only -- the gate is exact
    // tokens): the narrowest best-vs-second-best dist gap over every
    // (level, frame) decision in this reference. `margin_measured` is false
    // only when this struct is left at its default (a non-OK
    // encode_reference call); a SYNTH_OK return always measured at least one
    // decision. `gaps` carries the FULL per-decision grid, `tokens`' own
    // codebook-major layout, so a caller diagnosing a token mismatch can
    // print the gap at that exact (level, frame) rather than only the
    // clip-wide minimum.
    float              narrowest_gap   = 0.0f;
    bool               margin_measured = false;
    std::vector<float> gaps;

    // Debugging/probe taps, mirroring run_semantic_branch's/
    // run_acoustic_and_fuse's own out_ parameters: pcm_16k/semantic_mean/
    // fused_latent are the oracle's own ref.pcm_16k/ref.semantic_mean/
    // ref.fused_latent probes; semantic_encoder_output carries no committed
    // oracle probe of its own (build_semantic_branch's PRIMARY return value,
    // the fusion's own semantic-side input) and is exposed for debugging the
    // same way run_semantic_branch's own out_semantic_encoder always has
    // been.
    std::vector<float> pcm_16k;
    std::vector<float> semantic_mean;
    std::vector<float> fused_latent;
    std::vector<float> semantic_encoder_output;
};

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
    // The device the Loaded Model actually holds, which the public interface
    // publishes. Present in Plan 1 even though every graph is Plan 2's: a model
    // that loads and cannot say where it lives fails a documented query.
    ggml_backend_device *               primary_device() const;
    std::shared_ptr<const TextFrontend> text_frontend() const;

    // The frame geometry the core reports, and the vocabulary the frontend's
    // ids index — the text tower's, which the core range-checks against.
    uint32_t samples_per_frame() const;  // the codec hop: 960
    uint32_t text_vocab_size() const;

    // The greedy mask-predict synthesis path. Deterministic: with the golden
    // parameters it makes no random draw at all, which is what the exact-token
    // gate stands on. The sampled path is Plan 3.
    synth_status_t run_synthesis(const SynthesisRequest & request, SynthesisOutput & output);

    // The public seam's family entry: raw text and rates in, delivered PCM
    // out. Assembles the row-0 prompt (assemble_prompt_ids), estimates the
    // canvas length (DurationEstimator, against the no-reference anchor pair
    // until Task 14 threads a real reference through `clone`), clamps it to
    // the effective frame limit -- an estimate past the limit is
    // SYNTH_ERR_OUTPUT_LIMIT, the limit being a cap on the canvas rather than
    // a target for it -- and then calls run_synthesis, which already
    // performs decode_codes and the no-reference volume branch. Auto-voice
    // and voice-design only this task: `params.clone` is unread until Task
    // 14, `params.instruct` until Task 15.
    synth_status_t synthesize(const PublicSynthesisParams & params, SynthesisOutput & output);

    // Decodes a committed grid (codebook-major [num_codebooks * frame_count],
    // values in [0, codebook_size)) to the RAW waveform -- no volume branch,
    // so the replay seam can apply the oracle's branch per case.
    //
    // `out_seconds`/`out_placement` report THIS call's own codec timing and
    // node placement when non-null. run_synthesis passes its own output
    // fields here directly; the replay seam calls this free-standing (there is
    // no SynthesisOutput in scope), and without these out-params its codec
    // pass left the caller's copy at its zero default -- the CPU-only
    // placement rule was vacuous for every case that never runs the greedy
    // loop.
    synth_status_t decode_codes(const std::vector<int32_t> &      codes,
                                uint64_t                          frame_count,
                                int                               threads,
                                std::vector<float> &              audio,
                                double *                          out_seconds   = nullptr,
                                SynthesisOutput::StagePlacement * out_placement = nullptr);

    // The cloning path's encode half, symmetric with decode_codes above: the
    // WHOLE chain from a raw 24 kHz mono reference to a committed token grid.
    // `pcm_24k` need not already be hop-aligned -- hop-clip (tail-clip to a
    // whole number of `hparams.codec.hop_length`-sample frames) is this
    // function's own first step (reference-encoder-host.h's
    // clip_and_boost_reference), followed by the quiet-reference boost, THEN
    // resample to 16 kHz (resample_24k_to_16k) and the HuBERT semantic branch
    // plus the codec's own SemanticEncoder (reference-encoder.h's
    // build_semantic_branch, orchestrated by run_semantic_branch), THEN the
    // DAC acoustic encoder over the clipped+boosted 24 kHz segment plus the
    // reference fusion Linear (build_acoustic_encoder/build_reference_fusion,
    // orchestrated by run_acoustic_and_fuse), THEN (Task 13) the RVQ
    // nearest-neighbour encode (reference-encoder-host.h's rvq_encode).
    //
    // `output` is cleared up front (`ReferenceEncoding{}`) and meaningful
    // ONLY when this returns SYNTH_OK; every field is documented at
    // ReferenceEncoding's own declaration.
    //
    // Returns SYNTH_ERR_INVALID_ARG for an empty `pcm_24k` or one shorter
    // than one hop (clip_and_boost_reference's own empty-result case),
    // whatever resample_24k_to_16k/run_semantic_branch/run_acoustic_and_fuse
    // themselves return on their own refusals, and SYNTH_ERR_INTERNAL when
    // rvq_encode refuses the package's own resolved quantizer shapes (a
    // wiring defect upstream of this call, not a runtime case this function
    // covers).
    synth_status_t encode_reference(const std::vector<float> & pcm_24k, int threads, ReferenceEncoding & output);

  private:
    struct Impl;
    explicit Model(std::unique_ptr<Impl> implementation);
    std::unique_ptr<Impl> implementation_;
};

}  // namespace synth::omnivoice
