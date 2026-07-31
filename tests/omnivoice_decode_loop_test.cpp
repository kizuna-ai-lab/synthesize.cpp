// Drives the greedy mask-predict loop end to end on the synthetic package.
//
// What this is NOT: a parity test. The loop's numeric rules are pinned by
// tests/omnivoice_generator_host_test.cpp against hand-computable fixtures, and
// the loop as a whole is gated on exact equality with the oracle's 8 x T grids
// over the 17 greedy golden cases -- neither of which a synthetic package can
// strengthen.
//
// What it IS: the only execution of the decode loop that the unit gate -- and
// therefore the SANITIZER gate -- ever sees. The real-package sweep needs a
// 3 GiB F32 GGUF that is deliberately not committed, so without this test every
// pointer computation the loop makes (the canvas refill into the target region
// of an 8-row prompt grid, the two logit offsets into a position-major
// read-back buffer, the unconditional canvas that is `frames` long where the
// conditional one is `total`) runs under ASan/UBSan exactly never. It also
// covers run_synthesis's request validation and its probe-only early return,
// which the exact-token gate never reaches.
//
// The fixture's weights are one repeated constant, so every logit at every
// position is bit-identical and the argmax is token 0 everywhere. That
// degeneracy is deliberate: it makes the whole committed grid exactly
// predictable, so a read that strays outside its buffer shows up as a non-zero
// token rather than needing a sanitizer to notice.

#include "arch/omnivoice/omnivoice.h"
#include "omnivoice_synthetic_package.h"
#include "test-assert.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

// The synthetic layout's, transcribed: two codebooks over a five-slot canvas
// whose last slot is the mask, four steps, guidance 2.0 (so the unconditional
// branch really runs), and sixteen frames of output headroom.
constexpr uint32_t kCodebooks   = 2;
constexpr int32_t  kMaskId      = 4;
constexpr uint64_t kMaxFrames   = 16;
constexpr uint32_t kPackageStep = 4;

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
    // Task 12 owns the codec; the grid is the whole product today.
    SYNTH_TEST_CHECK(output.audio.empty());

    // Greedy decoding draws no random number, so a second run of the same
    // request is the same grid. This is the property the exact-token gate rests
    // on, asserted where it can be asserted cheaply.
    synth::omnivoice::SynthesisOutput again;
    SYNTH_TEST_CHECK(model.run_synthesis(request, again) == SYNTH_OK);
    SYNTH_TEST_CHECK(again.codes == output.codes);
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
    return check_grid(output, 4, "reference run");
}

int check_single_step_commits_everything(synth::omnivoice::Model & model) {
    // One step means the schedule's final-step rule -- commit the entire
    // remainder -- is the only rule that fires. If it did not, a mask would
    // survive and run_synthesis would refuse rather than return this grid.
    synth::omnivoice::SynthesisRequest request = base_request(6);
    request.num_step                           = 1;
    synth::omnivoice::SynthesisOutput output;
    SYNTH_TEST_CHECK(model.run_synthesis(request, output) == SYNTH_OK);
    return check_grid(output, 6, "single-step run");
}

// Also reports what one conditional forward places, which the loop run's
// accumulated total is measured against.
int check_probe_only_skips_the_loop(synth::omnivoice::Model & model, uint64_t & one_forward_nodes) {
    synth::omnivoice::SynthesisRequest request = base_request(5);
    request.probe_only                         = true;
    request.probe_layers                       = { 0, 1 };
    synth::omnivoice::SynthesisOutput output;
    SYNTH_TEST_CHECK(model.run_synthesis(request, output) == SYNTH_OK);
    // The probe path stops after ONE conditional forward: no grid, no frame
    // count, and the probes filled.
    SYNTH_TEST_CHECK(output.codes.empty());
    SYNTH_TEST_CHECK(output.frame_count == 0);
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
    return failures == 0 ? 0 : 1;
}
