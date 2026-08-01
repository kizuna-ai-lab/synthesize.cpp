// The family-internal 24 kHz -> 16 kHz resampler
// (src/arch/omnivoice/reference-encoder-host.{h,cpp}): a transcription of
// torchaudio.functional.resample(wav, 24000, 16000) at the pinned defaults.
// This is the first stage of a chain that ends in a DISCRETE decision (Task
// 13's RVQ nearest-neighbour, exact-token comparison against the oracle) --
// divergence introduced here compounds silently through HuBERT and the
// quantiser rather than failing loudly, so the fixtures below hold to a
// max_abs of 1e-6 against a +-1-scale signal, tightened to the value actually
// measured wherever that came out below the gate.
//
// Fixture provenance: every expected array below was produced by running the
// pinned torchaudio (2.11.0) in the locked venv:
//
//   uv run --project scripts/envs/omnivoice --locked python -c "
//   import torch, torchaudio, numpy as np
//   def resample(x):
//       return torchaudio.functional.resample(
//           torch.from_numpy(np.asarray(x, dtype=np.float32)), 24000, 16000).numpy()
//   # dc96_expected:      resample(np.ones(96, dtype=np.float32))
//   # impulse64_expected: resample(x) where x = np.zeros(64, dtype=np.float32); x[32] = 1.0
//   # mixed48_expected:   resample(mixed48_input), mixed48_input computed below
//   "
//
// (see the task's exploration transcript for the exact generating script;
// the expressions producing each input are documented next to that fixture).

#include "arch/omnivoice/reference-encoder-host.h"
#include "test-assert.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

// --------------------------------------------------------------------------
// Fixture (a): DC 1.0, 96 samples -> 64 samples. Exercises the kernel's
// steady-state gain (the pinned rolloff=0.99 anti-aliasing means the interior
// samples settle just under 1.0 rather than exactly at it) and the edge
// transient from zero-padding.
// --------------------------------------------------------------------------

int check_dc96_fixture() {
    const std::vector<float> input(96, 1.0f);
    // clang-format off
    const std::vector<float> expected = {
        8.302113414e-01f, 1.050367713e+00f, 9.769163728e-01f, 1.011090279e+00f, 9.962181449e-01f, 1.000874996e+00f,
        1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f,
        1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f,
        1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f,
        1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f,
        1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f,
        1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f,
        1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f,
        1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f,
        1.000422597e+00f, 1.000509024e+00f, 1.000422597e+00f, 1.000509024e+00f, 1.000424266e+00f, 1.000385880e+00f,
        9.979410768e-01f, 1.007709622e+00f, 9.819432497e-01f, 1.044140100e+00f,
    };
    // clang-format on
    SYNTH_TEST_CHECK(expected.size() == 64);

    std::vector<float> actual;
    SYNTH_TEST_CHECK(synth::omnivoice::resample_24k_to_16k(input, actual));
    SYNTH_TEST_CHECK(actual.size() == expected.size());
    const float measured = max_abs_diff(actual, expected);
    std::cout << "dc96 max_abs = " << measured << '\n';
    // Measured 0.0 after fix-round-1 (the scalar-promotion correction made
    // the kernel construction bit-exact against torch's own kernel tensor,
    // and this fixture's accumulation order happens to match too) -- was
    // 1.19209e-07 before the fix. Asserted exactly rather than with a
    // tolerance: a regression back to the wrong promotion model, or any
    // other change that perturbs this fixture, should fail loudly rather
    // than sneak under a nonzero bound.
    SYNTH_TEST_CHECK(measured == 0.0f);
    return 0;
}

// --------------------------------------------------------------------------
// Fixture (b): a single impulse at index 32 of 64 zero samples. Placed away
// from both edges so the output directly exposes the kernel's own taps
// (scaled and phase-interleaved) rather than the zero-padding transient --
// this is the fixture that would catch a wrong kernel table even if fixture
// (a)'s DC gain happened to still add up correctly.
// --------------------------------------------------------------------------

int check_impulse64_fixture() {
    std::vector<float> input(64, 0.0f);
    input[32] = 1.0f;
    // clang-format off
    const std::vector<float> expected = {
        0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,
        0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,
        0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,  3.191141549e-24f,
        3.191141549e-24f, -1.076447428e-03f,  7.250522263e-03f, -2.172334678e-02f,  5.090379342e-02f,
       -1.189598814e-01f,  5.438855290e-01f,  2.706917822e-01f, -9.356199950e-02f,  4.274800047e-02f,
       -1.795491949e-02f,  5.282599013e-03f, -3.660339571e-04f,  3.191141549e-24f,  3.191141549e-24f,
        0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,
        0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,
        0.000000000e+00f,  0.000000000e+00f,  0.000000000e+00f,
    };
    // clang-format on
    SYNTH_TEST_CHECK(expected.size() == 43);

    std::vector<float> actual;
    SYNTH_TEST_CHECK(synth::omnivoice::resample_24k_to_16k(input, actual));
    SYNTH_TEST_CHECK(actual.size() == expected.size());
    const float measured = max_abs_diff(actual, expected);
    std::cout << "impulse64 max_abs = " << measured << '\n';
    // Measured 0.0 after fix-round-1 (was 2.98023e-08 before the
    // scalar-promotion correction) -- see the dc96 fixture above for why
    // this is asserted exactly rather than with a tolerance.
    SYNTH_TEST_CHECK(measured == 0.0f);
    return 0;
}

// --------------------------------------------------------------------------
// Fixture (c): a short mixed signal, 48 samples -> 32 samples. Input is
// 0.5*sin(2*pi*1000*n/24000) + 0.3*sin(2*pi*7000*n/24000) - 0.05 for
// n = 0..47 (two tones plus a DC offset), exercising superposition across
// the kernel rather than a single frequency component.
// --------------------------------------------------------------------------

int check_mixed48_fixture() {
    // clang-format off
    const std::vector<float> input = {
        -5.000000075e-02f,  3.691872656e-01f,  5.000000075e-02f,  9.142135829e-02f,  6.428202987e-01f,
         5.106086135e-01f,  1.500000060e-01f,  5.106086135e-01f,  6.428202987e-01f,  9.142135829e-02f,
         5.000000075e-02f,  3.691872656e-01f, -5.000000075e-02f, -4.691872597e-01f, -1.500000060e-01f,
        -1.914213598e-01f, -7.428203225e-01f, -6.106086373e-01f, -2.500000000e-01f, -6.106086373e-01f,
        -7.428203225e-01f, -1.914213598e-01f, -1.500000060e-01f, -4.691872597e-01f, -5.000000075e-02f,
         3.691872656e-01f,  5.000000075e-02f,  9.142135829e-02f,  6.428202987e-01f,  5.106086135e-01f,
         1.500000060e-01f,  5.106086135e-01f,  6.428202987e-01f,  9.142135829e-02f,  5.000000075e-02f,
         3.691872656e-01f, -5.000000075e-02f, -4.691872597e-01f, -1.500000060e-01f, -1.914213598e-01f,
        -7.428203225e-01f, -6.106086373e-01f, -2.500000000e-01f, -6.106086373e-01f, -7.428203225e-01f,
        -1.914213598e-01f, -1.500000060e-01f, -4.691872597e-01f,
    };
    const std::vector<float> expected = {
         7.866234332e-02f,  1.901225895e-01f,  1.502883136e-01f,  6.299664974e-01f,  2.076667249e-01f,
         6.373097301e-01f,  1.305534989e-01f,  2.347492427e-01f, -5.002112687e-02f, -3.348001540e-01f,
        -2.305957973e-01f, -7.375323772e-01f, -3.053922653e-01f, -7.375324368e-01f, -2.305957526e-01f,
        -3.348000944e-01f, -5.002113432e-02f,  2.347492427e-01f,  1.305534989e-01f,  6.374814510e-01f,
         2.053499818e-01f,  6.374814510e-01f,  1.305534989e-01f,  2.347492427e-01f, -5.002112687e-02f,
        -3.348001540e-01f, -2.305958718e-01f, -7.373728156e-01f, -3.073747158e-01f, -7.309065461e-01f,
        -2.482313067e-01f, -2.948479652e-01f,
    };
    // clang-format on
    SYNTH_TEST_CHECK(input.size() == 48);
    SYNTH_TEST_CHECK(expected.size() == 32);

    std::vector<float> actual;
    SYNTH_TEST_CHECK(synth::omnivoice::resample_24k_to_16k(input, actual));
    SYNTH_TEST_CHECK(actual.size() == expected.size());
    const float measured = max_abs_diff(actual, expected);
    std::cout << "mixed48 max_abs = " << measured << '\n';
    // Measured 5.96046e-08 both before and after fix-round-1 (unchanged --
    // the kernel itself is now bit-exact against torch's; this residual is
    // the convolution's own accumulation order, an implementation-defined
    // non-goal). Tightened to <=1e-7f (~1.7x measured, comfortably within
    // the project's <=5x-measured discipline), down from the interim 1e-6.
    SYNTH_TEST_CHECK(measured <= 1e-7f);
    return 0;
}

// --------------------------------------------------------------------------
// Length arithmetic: target_length = ceil(new_freq * length / orig_freq) =
// ceil(2*length/3), measured against the pinned torchaudio for every case
// below (including the tiny ones, which torchaudio itself resamples without
// error -- there is nothing in the formula that requires a minimum length).
// --------------------------------------------------------------------------

int check_length_arithmetic() {
    struct Case {
        size_t input_len;
        size_t expected_output_len;
    };

    const Case cases[] = {
        { 960, 640 }, // exact: 960*2/3 = 640.0
        { 961, 641 }, // ceil(640.667) = 641
        { 1,   1   }, // ceil(0.667) = 1
        { 2,   2   }, // ceil(1.333) = 2
        { 3,   2   }, // exact: 3*2/3 = 2.0
    };
    for (const Case & c : cases) {
        const std::vector<float> input(c.input_len, 0.0f);
        std::vector<float>       output;
        SYNTH_TEST_CHECK(synth::omnivoice::resample_24k_to_16k(input, output));
        SYNTH_TEST_CHECK(output.size() == c.expected_output_len);
    }
    return 0;
}

int check_rejects_empty_input() {
    const std::vector<float> input;
    std::vector<float>       output = { 1.0f, 2.0f, 3.0f };
    SYNTH_TEST_CHECK(!synth::omnivoice::resample_24k_to_16k(input, output));
    SYNTH_TEST_CHECK(output.empty());
    return 0;
}

// --------------------------------------------------------------------------
// Real-signal gate: the oracle's own reference audio, resampled 24k -> 16k.
// Sentinel-guarded like every other real-payload check in this suite -- the
// golden directory is not committed, so this only runs where it has been
// materialized (build/goldens/omnivoice/omni-clone-en/{ref/pcm_24k.f32,
// ref/pcm_16k.f32}, from Task 9's clone-encode probes). Registered as its own
// `integration;omnivoice` ctest entry that passes the directory as argv[1];
// the plain `unit;omnivoice` entry runs this binary with no arguments and
// never reaches this function.
// --------------------------------------------------------------------------

bool read_f32_file(const std::string & path, std::vector<float> & values) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return false;
    }
    const std::streamoff bytes = input.tellg();
    if (bytes < 0 || bytes % std::streamoff(sizeof(float)) != 0) {
        return false;
    }
    values.resize(size_t(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    return input.good() || input.eof();
}

int check_real_signal_parity(const std::string & golden_dir) {
    std::vector<float> pcm_24k;
    std::vector<float> pcm_16k_expected;
    const std::string  pcm_24k_path = golden_dir + "/ref/pcm_24k.f32";
    const std::string  pcm_16k_path = golden_dir + "/ref/pcm_16k.f32";
    SYNTH_TEST_CHECK(read_f32_file(pcm_24k_path, pcm_24k));
    SYNTH_TEST_CHECK(read_f32_file(pcm_16k_path, pcm_16k_expected));
    SYNTH_TEST_CHECK(!pcm_24k.empty());
    SYNTH_TEST_CHECK(!pcm_16k_expected.empty());

    std::vector<float> actual;
    SYNTH_TEST_CHECK(synth::omnivoice::resample_24k_to_16k(pcm_24k, actual));
    SYNTH_TEST_CHECK(actual.size() == pcm_16k_expected.size());

    const float measured = max_abs_diff(actual, pcm_16k_expected);
    std::cout << "real-signal (omni-clone-en) pcm_24k(" << pcm_24k.size() << ") -> pcm_16k(" << pcm_16k_expected.size()
              << ") max_abs = " << measured << '\n';
    // Measured 1.19209e-07 after fix-round-1's scalar-promotion correction
    // (was 1.78814e-07 before it) -- this task's brief set an interim gate
    // of 1e-6 against a +-1-scale signal, but since the kernel is now
    // bit-exact against torch's own tensor, this file's OWN gate is
    // tightened to <=5e-7f (~4.2x measured, within the project's
    // <=5x-measured discipline) rather than left at the interim value. Task
    // 13 still owns committing the FINAL tolerance once the whole encode
    // chain exists to compare against -- this is this test's own gate, not
    // that one.
    SYNTH_TEST_CHECK(measured <= 5e-7f);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    SYNTH_TEST_CHECK(check_dc96_fixture() == 0);
    SYNTH_TEST_CHECK(check_impulse64_fixture() == 0);
    SYNTH_TEST_CHECK(check_mixed48_fixture() == 0);
    SYNTH_TEST_CHECK(check_length_arithmetic() == 0);
    SYNTH_TEST_CHECK(check_rejects_empty_input() == 0);
    if (argc >= 2) {
        SYNTH_TEST_CHECK(check_real_signal_parity(argv[1]) == 0);
    }
    return 0;
}
