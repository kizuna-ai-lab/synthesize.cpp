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
#include "random-stream.h"
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
// The small package's text vocabulary width; the first id run_synthesis must
// refuse as out of range.
constexpr uint32_t kTextVocab    = 40;

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
// token a legal code. `expect_constant_argmax` additionally pins the
// constant-weight package's degenerate argmax-0 property; callers running
// against the varied-weights package (see SyntheticPackageOptions) pass
// false, since that fixture's whole point is that tokens are NOT
// predictable, and pinning a specific value there would be asserting on
// numbers this test has no business predicting.
int check_grid(const synth::omnivoice::SynthesisOutput & output,
               uint64_t                                  frames,
               const char *                              what,
               bool                                      expect_constant_argmax = true) {
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
        if (expect_constant_argmax && token != 0) {
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
    // so the total is not a clean multiple of anything.) Tightened into a
    // WINDOW rather than a bare floor: at most kPackageStep forwards of each
    // kind run, and the unconditional graph is strictly smaller than the
    // conditional one, so the true total sits below "every forward were
    // conditional-sized" -- the ceiling below -- as well as above the floor.
    SYNTH_TEST_CHECK(one_forward_nodes > 0);
    SYNTH_TEST_CHECK(output.generator_placement.nodes > one_forward_nodes * kPackageStep);
    SYNTH_TEST_CHECK(output.generator_placement.nodes < one_forward_nodes * kPackageStep * 2);
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

    // This request leaves the new temperature fields at their default -1.0f,
    // which resolves to the PACKAGE's own position_temperature (5.0, see the
    // metadata above) -- so the loop DOES draw random numbers here. But
    // `request.seed` also defaults (to 0), and a fixed seed draws the same
    // sequence every time, so two runs of the same request are still the same
    // grid -- and therefore the same waveform. This is the reproducibility
    // property the exact-token gate rests on (a real package is not
    // degenerate the way this fixture is, so its grid WOULD move if the seed
    // moved), asserted here where it can be asserted cheaply.
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

    // A text id outside the embedding table would reach ggml_get_rows as an
    // out-of-range row -- an abort, not a status -- so the loop refuses it at
    // the seam. One past the top and one below zero pin both edges.
    synth::omnivoice::SynthesisRequest overflowing_text = base_request(4);
    overflowing_text.prompt_text_ids.push_back(int32_t(kTextVocab));
    SYNTH_TEST_CHECK(model.run_synthesis(overflowing_text, output) == SYNTH_ERR_INVALID_ARG);

    synth::omnivoice::SynthesisRequest negative_text = base_request(4);
    negative_text.prompt_text_ids.push_back(-1);
    SYNTH_TEST_CHECK(model.run_synthesis(negative_text, output) == SYNTH_ERR_INVALID_ARG);

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

// Same seed, twice: two run_synthesis calls with the same request (seed 7,
// position_temperature 5.0f -- so the per-candidate position draw actually
// fires -- class_temperature 0.0f, so token CHOICE itself stays plain-greedy)
// commit the same grid. This is the reproducibility property the replay
// runner's pinned-zero contract and a real caller's expectations both rest
// on: fixing the seed reproduces the output, whatever the resolved
// temperatures are.
int check_sampled_same_seed_identical_grid(synth::omnivoice::Model & model) {
    synth::omnivoice::SynthesisRequest request = base_request(5);
    request.seed                               = 7;
    request.position_temperature               = 5.0f;
    request.class_temperature                  = 0.0f;

    synth::omnivoice::SynthesisOutput first;
    SYNTH_TEST_CHECK(model.run_synthesis(request, first) == SYNTH_OK);
    if (check_grid(first, 5, "sampled seed-7 run (first)") != 0) {
        return 1;
    }

    synth::omnivoice::SynthesisOutput second;
    SYNTH_TEST_CHECK(model.run_synthesis(request, second) == SYNTH_OK);
    SYNTH_TEST_CHECK(second.codes == first.codes);
    return 0;
}

// NOTE: this does NOT prove run_synthesis reads request.seed -- it predates
// (and is now redundant with) check_sampled_seed_actually_drives_the_grid
// below, which does, on the varied-weights package. Kept only because it
// documents WHY the constant-weight package's `output.codes` can never show
// a seed effect (see that comment), a fact the wiring test's own comment
// leans on. Seed 7 vs seed 8, otherwise identical requests: this fixture's
// `output.codes` CANNOT show the difference -- its whole point (see the file
// header) is constant weights, which make every candidate's guided log-prob
// tie across the entire vocabulary at every position, so choose_token's
// argmax always lands on token 0 regardless of which candidate a
// Gumbel-perturbed score commits first -- widening `frames` does not change
// this, because the degeneracy is in the WEIGHTS, not the canvas size. What
// follows only shows that two NormalRandomStreams seeded 7 and 8 draw
// different uniforms -- true regardless of whether run_synthesis ever reads
// its own seed argument, since these two streams are built by the test, not
// by the loop under test.
int check_sampled_different_seed_different_draws(synth::omnivoice::Model & model) {
    synth::omnivoice::SynthesisRequest seed7 = base_request(5);
    seed7.seed                               = 7;
    seed7.position_temperature               = 5.0f;
    seed7.class_temperature                  = 0.0f;
    synth::omnivoice::SynthesisRequest seed8 = seed7;
    seed8.seed                               = 8;

    synth::omnivoice::SynthesisOutput output7;
    synth::omnivoice::SynthesisOutput output8;
    SYNTH_TEST_CHECK(model.run_synthesis(seed7, output7) == SYNTH_OK);
    SYNTH_TEST_CHECK(model.run_synthesis(seed8, output8) == SYNTH_OK);
    // Confirmed identical for the reason in the comment above -- a property of
    // this fixture's degeneracy, not a regression.
    SYNTH_TEST_CHECK(output7.codes == output8.codes);

    synth::NormalRandomStream stream7(7);
    synth::NormalRandomStream stream8(8);
    SYNTH_TEST_CHECK(stream7.next_uniform() != stream8.next_uniform());
    return 0;
}

// The actual wiring proof the two checks above cannot give: on the
// constant-weight package, EVERY candidate's guided log-prob ties across the
// whole vocabulary at every position (build_guided's log-softmax makes any
// constant added to every entry cancel out), so choose_token's strict `>`
// tie-break always lands on token 0 no matter which order a Gumbel-perturbed
// score commits candidates in -- a hardcoded seed inside run_synthesis and a
// correctly-wired one would look IDENTICAL there. This uses the
// varied-weights package instead, where logits genuinely differ across
// positions and vocabulary entries, so the position draw's effect on commit
// ORDER feeds back through the canvas refill into later steps' forwards and
// produces a genuinely different token grid. A fresh SynthesisRequest is
// built per call (never reused, unlike the identical-grid check above): a
// hardcoded seed would still make every call from one shared, already-seeded
// request "look the same", which proves nothing about whether the field is
// read at all.
//
// Seeds 7 and 8 are verified (see task-4-report.md's fix-round-1 section) to
// actually diverge on this fixture as it stands; if a future change to the
// LCG walk or the small layout ever made them coincide, the fix is a
// different seed pair, not relaxing this assertion.
int check_sampled_seed_actually_drives_the_grid(synth::omnivoice::Model & varied_model) {
    auto seeded_request = [](uint64_t seed) {
        synth::omnivoice::SynthesisRequest request = base_request(5);
        request.seed                               = seed;
        request.position_temperature               = 5.0f;
        request.class_temperature                  = 0.0f;
        return request;
    };

    synth::omnivoice::SynthesisOutput seed7_first;
    SYNTH_TEST_CHECK(varied_model.run_synthesis(seeded_request(7), seed7_first) == SYNTH_OK);
    if (check_grid(seed7_first, 5, "varied-weights seed-7 run (first)", /*expect_constant_argmax=*/false) != 0) {
        return 1;
    }

    // Same seed, a SECOND fresh request object: still the same grid.
    synth::omnivoice::SynthesisOutput seed7_second;
    SYNTH_TEST_CHECK(varied_model.run_synthesis(seeded_request(7), seed7_second) == SYNTH_OK);
    SYNTH_TEST_CHECK(seed7_second.codes == seed7_first.codes);

    // A different seed, yet another fresh request object: a different grid.
    synth::omnivoice::SynthesisOutput seed8;
    SYNTH_TEST_CHECK(varied_model.run_synthesis(seeded_request(8), seed8) == SYNTH_OK);
    if (check_grid(seed8, 5, "varied-weights seed-8 run", /*expect_constant_argmax=*/false) != 0) {
        return 1;
    }
    SYNTH_TEST_CHECK(seed8.codes != seed7_first.codes);
    return 0;
}

// The request struct's new temperature fields DEFAULT to sampling
// (position_temperature resolves to this package's 5.0 unless overridden), so
// a caller wanting the old Plan-2 greedy behaviour back has to ask for it
// explicitly. This pins that asking explicitly reproduces exactly what an
// implicit (field-less) request produces on THIS fixture -- which is why
// tests/omnivoice_replay_real.cpp's pin to explicit 0.0f matters on a
// non-degenerate real package, even though the two are indistinguishable
// here.
int check_sampled_explicit_greedy_matches_defaulted_request(synth::omnivoice::Model & model) {
    synth::omnivoice::SynthesisRequest explicit_greedy = base_request(5);
    explicit_greedy.position_temperature               = 0.0f;
    explicit_greedy.class_temperature                  = 0.0f;

    synth::omnivoice::SynthesisOutput explicit_output;
    SYNTH_TEST_CHECK(model.run_synthesis(explicit_greedy, explicit_output) == SYNTH_OK);
    if (check_grid(explicit_output, 5, "explicit greedy run") != 0) {
        return 1;
    }

    // The plain field-less request: both new fields sit at -1.0f, so
    // position_temperature resolves to the package's 5.0 and the loop DOES
    // sample -- but this fixture's constant weights make every committed
    // token argmax-0 regardless (see the file header), so the grid is
    // unaffected. A real package is not degenerate this way, which is exactly
    // why the replay runner pins both fields rather than relying on this
    // equivalence.
    synth::omnivoice::SynthesisRequest defaulted = base_request(5);
    synth::omnivoice::SynthesisOutput  defaulted_output;
    SYNTH_TEST_CHECK(model.run_synthesis(defaulted, defaulted_output) == SYNTH_OK);
    SYNTH_TEST_CHECK(defaulted_output.codes == explicit_output.codes);
    return 0;
}

// margin_report demands a resolved temperature of exactly 0 in BOTH
// dimensions: a Gumbel-perturbed margin measures nothing meaningful, since
// "narrow" and "random" are not the same axis. The package's own default
// position_temperature (5.0) means even a field-less request already
// triggers the refusal.
int check_margin_report_refuses_positive_temperature(synth::omnivoice::Model & model) {
    synth::omnivoice::SynthesisOutput output;

    synth::omnivoice::SynthesisRequest defaulted = base_request(5);
    defaulted.margin_report                      = true;
    SYNTH_TEST_CHECK(model.run_synthesis(defaulted, output) == SYNTH_ERR_INVALID_ARG);

    // An explicit positive class_temperature refuses too, independent of
    // position_temperature.
    synth::omnivoice::SynthesisRequest class_positive = base_request(5);
    class_positive.margin_report                      = true;
    class_positive.position_temperature               = 0.0f;
    class_positive.class_temperature                  = 1.0f;
    SYNTH_TEST_CHECK(model.run_synthesis(class_positive, output) == SYNTH_ERR_INVALID_ARG);

    // Both temperatures explicitly zero: margin_report is legal again, and
    // actually measures something over this request's schedule.
    synth::omnivoice::SynthesisRequest explicit_greedy = base_request(5);
    explicit_greedy.margin_report                      = true;
    explicit_greedy.position_temperature               = 0.0f;
    explicit_greedy.class_temperature                  = 0.0f;
    SYNTH_TEST_CHECK(model.run_synthesis(explicit_greedy, output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.margin.measured);
    return 0;
}

// Carryover item 2: the guidance == 0 branch (the reference's own
// `guidance_scale != 0` split) skips the unconditional forward ENTIRELY --
// every step makes one forward instead of two. The default synthetic package
// pins guidance_scale to 2.0 (so guidance != 0 is what every other case in
// this file exercises), so this builds its own package with guidance_scale
// overridden to 0.0 via the harness's new option and compares its
// generator_placement.nodes against an identical-shape request on the
// ordinary (guidance != 0) package: fewer forwards is directly fewer placed
// nodes, the same counter the other checks in this file already read.
int check_guidance_zero_skips_uncond_forward(synth::omnivoice::Model & guided_model, const std::string & scratch_dir) {
    const std::string zero_guidance_path = scratch_dir + "/synthetic-decode-loop-zero-guidance.gguf";
    synth::omnivoice::testing::SyntheticPackageOptions options;
    options.guidance_scale = 0.0f;
    SYNTH_TEST_CHECK(synth::omnivoice::testing::write_synthetic_package(zero_guidance_path, options));
    std::unique_ptr<synth::omnivoice::Model> zero_guidance_model;
    SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(zero_guidance_path, zero_guidance_model) == SYNTH_OK);
    SYNTH_TEST_CHECK(zero_guidance_model != nullptr);

    synth::omnivoice::SynthesisOutput guided_output;
    SYNTH_TEST_CHECK(guided_model.run_synthesis(base_request(5), guided_output) == SYNTH_OK);

    synth::omnivoice::SynthesisOutput zero_guidance_output;
    SYNTH_TEST_CHECK(zero_guidance_model->run_synthesis(base_request(5), zero_guidance_output) == SYNTH_OK);
    if (check_grid(zero_guidance_output, 5, "guidance == 0 run") != 0) {
        return 1;
    }

    // Half the forwards (conditional only, no unconditional) means
    // meaningfully fewer placed nodes -- not just "not more", which a broken
    // zero-node run would also satisfy.
    SYNTH_TEST_CHECK(zero_guidance_output.generator_placement.nodes > 0);
    SYNTH_TEST_CHECK(zero_guidance_output.generator_placement.nodes < guided_output.generator_placement.nodes);
    return 0;
}

// Model::synthesize (Task 5's public seam entry): the only path in this file
// that goes from raw UTF-8 text to a delivered waveform through
// assemble_prompt_ids and DurationEstimator, rather than driving
// run_synthesis directly with pre-tokenized ids and a fixed frame count.
// Needs its own package: the default synthetic vocabulary ("tok0".."tok39")
// cannot tokenize real text at all (see SyntheticPackageOptions::
// ascii_text_vocab), which is also why none of this file's other checks
// exercise the frontend for real.
int check_public_synthesize(synth::omnivoice::Model & text_model) {
    // "a" alone is the frontend test's own boosted-floor example
    // (omnivoice_frontend_test.cpp's check_frame_truncation): weight 1.0
    // against the no-reference anchor's 14.1 lands the estimate at exactly
    // 16 frames -- this package's own ceiling (kMaxFrames) -- which doubles
    // as the boundary case: an estimate EQUAL to the limit is not a limit
    // violation.
    synth::omnivoice::PublicSynthesisParams request;
    request.text = "a";
    synth::omnivoice::SynthesisOutput output;
    SYNTH_TEST_CHECK(text_model.synthesize(request, output) == SYNTH_OK);
    SYNTH_TEST_CHECK(output.frame_count == kMaxFrames);
    if (check_waveform(output.audio, kMaxFrames, /*normalised=*/true, "public synthesize") != 0) {
        return 1;
    }

    // Same seed, same text: two independent Model::synthesize calls commit
    // the same grid. This is the public seam's own reproducibility contract
    // (a real package is not degenerate the way this fixture's constant
    // weights are, so its grid would move with the seed; here it is the
    // WIRING -- that the seed reaches run_synthesis at all -- Task 4 already
    // proved on the varied-weights package).
    synth::omnivoice::SynthesisOutput again;
    SYNTH_TEST_CHECK(text_model.synthesize(request, again) == SYNTH_OK);
    SYNTH_TEST_CHECK(again.codes == output.codes);
    SYNTH_TEST_CHECK(again.audio == output.audio);

    // "aa" doubles the weight and pushes the boosted estimate to 20 frames,
    // past this same package's 16-frame ceiling: the limit caps the
    // ESTIMATE before run_synthesis ever sees a request, rather than letting
    // the loop fall short of one.
    synth::omnivoice::PublicSynthesisParams too_long;
    too_long.text = "aa";
    synth::omnivoice::SynthesisOutput unused;
    SYNTH_TEST_CHECK(text_model.synthesize(too_long, unused) == SYNTH_ERR_OUTPUT_LIMIT);

    // Empty and whitespace-only text are refused before any estimate is
    // made: DurationEstimator's own floor (`max(1, int(...))`) would
    // otherwise hand back a one-frame canvas for a request that named no
    // Linguistic Input at all.
    synth::omnivoice::PublicSynthesisParams empty_text;
    SYNTH_TEST_CHECK(text_model.synthesize(empty_text, unused) == SYNTH_ERR_INVALID_ARG);
    synth::omnivoice::PublicSynthesisParams blank_text;
    blank_text.text = "   ";
    SYNTH_TEST_CHECK(text_model.synthesize(blank_text, unused) == SYNTH_ERR_INVALID_ARG);
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

    // A second package, weights varied via a fixed LCG walk rather than the
    // constant fill above: see check_sampled_seed_actually_drives_the_grid
    // for why the constant-weight package cannot prove run_synthesis reads
    // its own seed argument.
    const std::string varied_weights_path = std::string(argv[1]) + "/synthetic-decode-loop-varied-weights.gguf";
    synth::omnivoice::testing::SyntheticPackageOptions varied_options;
    varied_options.varied_weights = true;
    SYNTH_TEST_CHECK(synth::omnivoice::testing::write_synthetic_package(varied_weights_path, varied_options));
    std::unique_ptr<synth::omnivoice::Model> varied_model;
    SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(varied_weights_path, varied_model) == SYNTH_OK);
    SYNTH_TEST_CHECK(varied_model != nullptr);

    // A third package whose vocabulary can actually tokenize text: see
    // SyntheticPackageOptions::ascii_text_vocab. Only check_public_synthesize
    // needs it -- every other check in this file drives run_synthesis
    // directly with pre-tokenized ids.
    const std::string text_capable_path = std::string(argv[1]) + "/synthetic-decode-loop-text-capable.gguf";
    synth::omnivoice::testing::SyntheticPackageOptions text_options;
    text_options.ascii_text_vocab = true;
    SYNTH_TEST_CHECK(synth::omnivoice::testing::write_synthetic_package(text_capable_path, text_options));
    std::unique_ptr<synth::omnivoice::Model> text_model;
    SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(text_capable_path, text_model) == SYNTH_OK);
    SYNTH_TEST_CHECK(text_model != nullptr);

    int      failures          = 0;
    uint64_t one_forward_nodes = 0;
    failures += check_probe_only_skips_the_loop(*model, one_forward_nodes);
    failures += check_no_reference_run(*model, one_forward_nodes);
    failures += check_reference_run(*model);
    failures += check_single_step_commits_everything(*model);
    failures += check_requests_the_loop_refuses(*model);
    failures += check_decode_codes_refuses(*model);
    failures += check_sampled_same_seed_identical_grid(*model);
    failures += check_sampled_different_seed_different_draws(*model);
    failures += check_sampled_seed_actually_drives_the_grid(*varied_model);
    failures += check_sampled_explicit_greedy_matches_defaulted_request(*model);
    failures += check_margin_report_refuses_positive_temperature(*model);
    failures += check_guidance_zero_skips_uncond_forward(*model, std::string(argv[1]));
    failures += check_public_synthesize(*text_model);
    return failures == 0 ? 0 : 1;
}
