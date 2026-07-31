// Drives the greedy mask-predict loop and the codec decode end to end on the
// synthetic package.
//
// What this is NOT: a parity test. The loop's numeric rules are pinned by
// tests/omnivoice_generator_host_test.cpp against hand-computable fixtures, the
// codec's by tests/omnivoice_codec_test.cpp against a PyTorch reference, and
// the pair as a whole is gated on exact equality with the oracle's 8 x T grids
// and on waveform parity over the 20 golden cases -- none of which a synthetic
// package can strengthen.
//
// What it IS: the only execution of the decode loop and of Model::decode_codes
// that the unit gate -- and therefore the SANITIZER gate -- ever sees. The
// real-package sweep needs a 3 GiB F32 GGUF that is deliberately not committed,
// so without this test every pointer computation the loop makes (the canvas
// refill into the target region of an 8-row prompt grid, the two logit offsets
// into a position-major read-back buffer, the unconditional canvas that is
// `frames` long where the conditional one is `total`) runs under ASan/UBSan
// exactly never, and so does the codec's committed input tensor. It also covers
// run_synthesis's request validation, its probe-only early return, and
// decode_codes' own refusals, none of which the parity gates reach.
//
// The fixture's weights are one repeated constant, so every logit at every
// position is bit-identical and the argmax is token 0 everywhere. That
// degeneracy is deliberate: it makes the whole committed grid exactly
// predictable, so a read that strays outside its buffer shows up as a non-zero
// token rather than needing a sanitizer to notice. It says nothing about the
// waveform's VALUES, which is why this file asserts the waveform's shape, its
// finiteness, the volume branch's fixed point and the wiring between the two
// stages -- and leaves every number to the codec test.

#include "arch/omnivoice/codec-host.h"
#include "arch/omnivoice/omnivoice.h"
#include "omnivoice_synthetic_package.h"
#include "test-assert.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

// The synthetic layout's, transcribed: two codebooks over a five-slot canvas
// whose last slot is the mask, four steps, guidance 2.0 (so the unconditional
// branch really runs), and sixteen frames of output headroom.
constexpr uint32_t kCodebooks    = 2;
constexpr int32_t  kMaskId       = 4;
constexpr uint64_t kMaxFrames    = 16;
constexpr uint32_t kPackageStep  = 4;
// The synthetic codec's hop, 2 * 3 = the product of its upsampling ratios, so
// one frame is six samples. Transcribed from the same small layout.
constexpr uint32_t kHop          = 6;
// The codec's codebook is four entries wide, one narrower than the canvas
// vocabulary -- so the mask id is also the first value decode_codes must
// refuse, which is the coincidence that makes this fixture worth having.
constexpr int32_t  kCodebookSize = 4;

// Text ids inside the small package's 40-token text vocabulary: the text_start
// and text_end markers around two ordinary tokens.
const std::vector<int32_t> kText{ 35, 1, 2, 36 };

synth::omnivoice::SynthesisRequest base_request(uint64_t frames) {
    synth::omnivoice::SynthesisRequest request;
    request.prompt_text_ids = kText;
    request.target_frames   = frames;
    return request;
}

// Every property the committed grid must have regardless of what the weights
// say: the right shape, the reported frame count, no surviving mask, and every
// token a legal code.
int check_grid(const synth::omnivoice::SynthesisOutput & output, uint64_t frames, const char * what) {
    if (output.frame_count != frames || output.codes.size() != size_t(kCodebooks) * frames) {
        std::fprintf(stderr, "%s: %llu frames and %zu codes, expected %llu and %zu\n", what,
                     (unsigned long long) output.frame_count, output.codes.size(), (unsigned long long) frames,
                     size_t(kCodebooks) * frames);
        return 1;
    }
    for (size_t index = 0; index < output.codes.size(); ++index) {
        const int32_t token = output.codes[index];
        if (token < 0 || token >= kMaskId) {
            std::fprintf(stderr, "%s: code %zu is %d, outside [0, %d)\n", what, index, token, kMaskId);
            return 1;
        }
        // Constant weights make every candidate's logits equal, so the ban on
        // the mask id leaves slot 0 as the argmax at every position. A stray
        // read would have to land on another all-0.03125 buffer to fake this.
        if (token != 0) {
            std::fprintf(stderr, "%s: code %zu is %d, not the constant-weight argmax 0\n", what, index, token);
            return 1;
        }
    }
    return 0;
}

// Every property the decoded waveform must have whatever the weights say: one
// hop of samples per frame and every sample finite. `normalised` additionally
// demands the no-reference volume branch's fixed point -- an exact peak of 0.5,
// which holds for any input the branch fires on, because the sample that
// attained the peak becomes peak / peak * 0.5 and no other can pass it.
int check_waveform(const std::vector<float> & audio, uint64_t frames, bool normalised, const char * what) {
    if (audio.size() != size_t(frames) * kHop) {
        std::fprintf(stderr, "%s: %zu samples for %llu frames, expected %zu\n", what, audio.size(),
                     (unsigned long long) frames, size_t(frames) * kHop);
        return 1;
    }
    float peak = 0.0f;
    for (size_t index = 0; index < audio.size(); ++index) {
        if (!std::isfinite(audio[index])) {
            std::fprintf(stderr, "%s: sample %zu is not finite\n", what, index);
            return 1;
        }
        peak = std::fmax(peak, std::fabs(audio[index]));
    }
    // A fixture quiet enough to skip the branch would make the assertion below
    // vacuous rather than false, so it is named separately.
    if (normalised && peak <= 1e-6f) {
        std::fprintf(stderr,
                     "%s: peak %g is under the volume branch's own guard, so this fixture no longer\n"
                     "  exercises the branch at all -- give the package louder weights\n",
                     what, double(peak));
        return 1;
    }
    if (normalised && peak != 0.5f) {
        std::fprintf(stderr, "%s: peak is %.9g, not the volume branch's exact 0.5\n", what, double(peak));
        return 1;
    }
    return 0;
}

int check_no_reference_run(synth::omnivoice::Model & model, uint64_t one_forward_nodes) {
    synth::omnivoice::SynthesisRequest request = base_request(5);
    // num_step 0 asks for the package's embedded default, which is 4 here.
    request.num_step                           = 0;
    synth::omnivoice::SynthesisOutput output;
    SYNTH_TEST_CHECK(model.run_synthesis(request, output) == SYNTH_OK);
    if (check_grid(output, 5, "no-reference run") != 0) {
        return 1;
    }
    // `generator_placement.nodes` accumulates across forwards, and the loop
    // makes two per step, so it must exceed the one-forward probe run's count
    // by a wide margin rather than equal it. (The two branches' graphs differ
    // in node count -- the unconditional one has no text embedding to merge --
    // so the total is not a clean multiple of anything.)
    SYNTH_TEST_CHECK(one_forward_nodes > 0);
    SYNTH_TEST_CHECK(output.generator_placement.nodes > one_forward_nodes * kPackageStep);
    SYNTH_TEST_CHECK(output.generator_placement.accelerator_nodes == 0);
    if (check_waveform(output.audio, 5, /*normalised=*/true, "no-reference run") != 0) {
        return 1;
    }
    // The codec really ran, on the CPU, and said how long it took. Zero placed
    // nodes would mean run_synthesis reported a stage it never reached.
    SYNTH_TEST_CHECK(output.codec_placement.nodes > 0);
    SYNTH_TEST_CHECK(output.codec_placement.accelerator_nodes == 0);
    SYNTH_TEST_CHECK(output.codec_seconds >= 0.0);

    // The wiring, stated as an identity rather than a hope: what run_synthesis
    // returned IS its own grid decoded and put through the shared volume
    // helper. An audio buffer left over from an earlier call, a volume branch
    // applied twice, or a decode of the wrong grid all break this and nothing
    // else in the unit gate would.
    std::vector<float> rebuilt;
    SYNTH_TEST_CHECK(model.decode_codes(output.codes, output.frame_count, 0, rebuilt) == SYNTH_OK);
    if (check_waveform(rebuilt, 5, /*normalised=*/false, "decode_codes replay") != 0) {
        return 1;
    }
    synth::omnivoice::apply_no_reference_volume(rebuilt);
    SYNTH_TEST_CHECK(rebuilt == output.audio);

    // Greedy decoding draws no random number, so a second run of the same
    // request is the same grid -- and therefore the same waveform. This is the
    // property the exact-token gate rests on, asserted where it can be asserted
    // cheaply.
    synth::omnivoice::SynthesisOutput again;
    SYNTH_TEST_CHECK(model.run_synthesis(request, again) == SYNTH_OK);
    SYNTH_TEST_CHECK(again.codes == output.codes);
    SYNTH_TEST_CHECK(again.audio == output.audio);
    return 0;
}

// decode_codes' own guards, which no parity gate reaches: every one of these
// grids would otherwise reach ggml_get_rows and read a row that exists but
// means nothing, or one that does not exist at all.
int check_decode_codes_refuses(synth::omnivoice::Model & model) {
    std::vector<float>         audio;
    // The legal shape first, so the refusals below differ from it in exactly
    // one way each: two codebook-major rows of four frames.
    const std::vector<int32_t> legal{ 0, 1, 2, 3, 3, 2, 1, 0 };
    SYNTH_TEST_CHECK(model.decode_codes(legal, 4, 0, audio) == SYNTH_OK);
    if (check_waveform(audio, 4, /*normalised=*/false, "decode_codes legal grid") != 0) {
        return 1;
    }
    // The same graph over the same input twice: the codec carries nothing
    // between calls.
    std::vector<float> repeated;
    SYNTH_TEST_CHECK(model.decode_codes(legal, 4, 0, repeated) == SYNTH_OK);
    SYNTH_TEST_CHECK(repeated == audio);

    // The mask id is not a code. It is also exactly codebook_size here, so this
    // one case covers both "a mask survived" and "past the table".
    std::vector<int32_t> masked = legal;
    masked[3]                   = kMaskId;
    SYNTH_TEST_CHECK(model.decode_codes(masked, 4, 0, audio) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(audio.empty());

    std::vector<int32_t> negative = legal;
    negative[7]                   = -1;
    SYNTH_TEST_CHECK(model.decode_codes(negative, 4, 0, audio) == SYNTH_ERR_INVALID_ARG);

    // Past the table by more than the mask, in case the guard was written as an
    // equality against the mask id rather than a range check.
    std::vector<int32_t> beyond = legal;
    beyond[0]                   = kCodebookSize + 9;
    SYNTH_TEST_CHECK(model.decode_codes(beyond, 4, 0, audio) == SYNTH_ERR_INVALID_ARG);

    // A grid that is not frames x codebooks, in both directions, plus the empty
    // canvas.
    SYNTH_TEST_CHECK(model.decode_codes(legal, 3, 0, audio) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(model.decode_codes(legal, 5, 0, audio) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(model.decode_codes(legal, 0, 0, audio) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(model.decode_codes({}, 4, 0, audio) == SYNTH_ERR_INVALID_ARG);
    return 0;
}

int check_reference_run(synth::omnivoice::Model & model) {
    // Three reference frames shift the target region three positions further
    // into each prompt row; the per-step refill has to land on the target and
    // nowhere near the reference tokens or the row end.
    synth::omnivoice::SynthesisRequest request = base_request(4);
    request.reference_tokens                   = { 0, 1, 2, 3, 2, 1 };  // codebook-major [2 x 3]
    synth::omnivoice::SynthesisOutput output;
    SYNTH_TEST_CHECK(model.run_synthesis(request, output) == SYNTH_OK);
    if (check_grid(output, 4, "reference run") != 0) {
        return 1;
    }
    // Reference tokens lengthen the prompt, not the canvas: the waveform is
    // still one hop per TARGET frame, and Plan 2 gives it the no-reference
    // branch regardless (the other two arms need a reference RMS that only
    // Plan 3's cloning path can supply).
    return check_waveform(output.audio, 4, /*normalised=*/true, "reference run");
}

int check_single_step_commits_everything(synth::omnivoice::Model & model) {
    // One step means the schedule's final-step rule -- commit the entire
    // remainder -- is the only rule that fires. If it did not, a mask would
    // survive and run_synthesis would refuse rather than return this grid.
    synth::omnivoice::SynthesisRequest request = base_request(6);
    request.num_step                           = 1;
    synth::omnivoice::SynthesisOutput output;
    SYNTH_TEST_CHECK(model.run_synthesis(request, output) == SYNTH_OK);
    if (check_grid(output, 6, "single-step run") != 0) {
        return 1;
    }
    return check_waveform(output.audio, 6, /*normalised=*/true, "single-step run");
}

// Also reports what one conditional forward places, which the loop run's
// accumulated total is measured against.
int check_probe_only_skips_the_loop(synth::omnivoice::Model & model, uint64_t & one_forward_nodes) {
    synth::omnivoice::SynthesisRequest request = base_request(5);
    request.probe_only                         = true;
    request.probe_layers                       = { 0, 1 };
    synth::omnivoice::SynthesisOutput output;
    SYNTH_TEST_CHECK(model.run_synthesis(request, output) == SYNTH_OK);
    // The probe path stops after ONE conditional forward: no grid, no
    // waveform, and the probes filled. The frame count is reported anyway --
    // the canvas length is settled by the request, not by the loop, and a probe
    // run that said zero frames while holding a canvas of probes was reading as
    // an empty synthesis.
    SYNTH_TEST_CHECK(output.codes.empty());
    SYNTH_TEST_CHECK(output.audio.empty());
    SYNTH_TEST_CHECK(output.frame_count == 5);
    SYNTH_TEST_CHECK(output.codec_placement.nodes == 0);
    SYNTH_TEST_CHECK(output.layer_hidden.size() == 2);
    SYNTH_TEST_CHECK(!output.logits_step0.empty());
    one_forward_nodes = output.generator_placement.nodes;
    return 0;
}

int check_requests_the_loop_refuses(synth::omnivoice::Model & model) {
    synth::omnivoice::SynthesisOutput output;

    synth::omnivoice::SynthesisRequest no_text = base_request(4);
    no_text.prompt_text_ids.clear();
    SYNTH_TEST_CHECK(model.run_synthesis(no_text, output) == SYNTH_ERR_INVALID_ARG);

    SYNTH_TEST_CHECK(model.run_synthesis(base_request(0), output) == SYNTH_ERR_INVALID_ARG);

    // One frame past the package's declared ceiling is a limit, not a bug.
    SYNTH_TEST_CHECK(model.run_synthesis(base_request(kMaxFrames + 1), output) == SYNTH_ERR_OUTPUT_LIMIT);

    // A reference stream that is not a whole number of codebook rows cannot be
    // laid out, and one carrying the mask id is a hole where a code belongs.
    synth::omnivoice::SynthesisRequest ragged = base_request(4);
    ragged.reference_tokens                   = { 0, 1, 2 };
    SYNTH_TEST_CHECK(model.run_synthesis(ragged, output) == SYNTH_ERR_INVALID_ARG);

    synth::omnivoice::SynthesisRequest masked_reference = base_request(4);
    masked_reference.reference_tokens                   = { 0, kMaskId };
    SYNTH_TEST_CHECK(model.run_synthesis(masked_reference, output) == SYNTH_ERR_INVALID_ARG);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <scratch-dir>\n", argv[0]);
        return 2;
    }
    const std::string path = std::string(argv[1]) + "/synthetic-decode-loop.gguf";
    SYNTH_TEST_CHECK(synth::omnivoice::testing::write_synthetic_package(path, {}));
    std::unique_ptr<synth::omnivoice::Model> model;
    SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(path, model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);

    int      failures          = 0;
    uint64_t one_forward_nodes = 0;
    failures += check_probe_only_skips_the_loop(*model, one_forward_nodes);
    failures += check_no_reference_run(*model, one_forward_nodes);
    failures += check_reference_run(*model);
    failures += check_single_step_commits_everything(*model);
    failures += check_requests_the_loop_refuses(*model);
    failures += check_decode_codes_refuses(*model);
    return failures == 0 ? 0 : 1;
}
