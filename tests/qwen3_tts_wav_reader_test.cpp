// qwen3_tts_mel_driver.cpp's `read_wav`, exercised black-box through
// synthesize-qwen3-tts-mel-driver.
//
// SCOPE -- read this before trusting it further than it goes. Four files here
// parse WAV bytes by hand, and they are NOT four copies of one function:
//
//   - qwen3_tts_mel_driver.cpp `read_wav`     <- the subject of this file
//   - qwen3_tts_xvector_driver.cpp `read_wav`    byte-identical to it
//   - qwen3_tts_clone_real.cpp `read_wav_mono`   a DIFFERENT, smaller reader
//   - omnivoice_profile_test.cpp `read_wav_mono16`  another different one
//
// This test runs ONE binary. It pins the mel driver's copy and nothing else.
// The xvector copy is byte-identical today, but nothing here enforces that --
// editing its chunk bound out would leave this suite green. The other two are
// separate functions with their own names, signatures and accepted formats,
// and they do not all carry the same rule set: PR #11's review found an
// overflow in one of them that the other three never had. Each rule is
// hand-carried, not shared by construction. If you change a rule below, walk
// all four and re-diff the two `read_wav` bodies.
//
// The duplication is a recorded decision, stated at
// qwen3_tts_xvector_driver.cpp's own copy: every small driver that reads a WAV
// carries its own, and a WAV reader belongs in validation tooling rather than
// in src/ (docs/testing.md's port-validation split). So this file does not
// link a shared helper -- there isn't one, on purpose. It drives the driver as
// a subprocess instead, which is the only handle a `unit` test has on a reader
// that lives inside a translation unit with its own main(). Same shape as
// synthesize-shared-exports: a `unit` test whose subject is a built artifact
// rather than a linkable symbol.
//
// The mel driver is the one of the four this can reach at the `unit` tier at
// all: it takes a WAV path and an output path and nothing else. The other
// three need a real package (and, for two of them, an oracle dump). It is also
// the only one of the four either check target BUILDS: synthesize-check-unit
// walks an enumerated target list, and the mel driver is on it only because
// tests/CMakeLists.txt gives this test an explicit add_dependencies on it. The
// xvector driver is registered integration-only and is reached by neither
// check target; the remaining two have their add_executable guarded on local
// model artifacts existing at all.
//
// Every WAV below is generated here, byte by byte, rather than committed:
// tests/golden's rule is contracts, not payloads, and a deliberately
// malformed 60-byte binary in the tree would be a payload with no contract.

#include "test-assert.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef SYNTH_MEL_DRIVER
#    error "SYNTH_MEL_DRIVER must name the mel driver binary (tests/CMakeLists.txt)"
#endif

namespace {

// The rate the mel driver requires (its own pinned speaker-encoder params).
constexpr uint32_t kSampleRate = 24000;
// Comfortably past compute_log_mel's own (n_fft - hop_length) / 2 padding
// floor, so a valid file is rejected by nothing downstream of the reader.
constexpr uint32_t kFrameCount = 4096;

void put_bytes(std::vector<uint8_t> & out, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    out.insert(out.end(), bytes, bytes + size);
}

template <typename T> void put(std::vector<uint8_t> & out, T value) {
    put_bytes(out, &value, sizeof(value));
}

// A minimal, well-formed IEEE-float32 mono WAV. `declared_data_size` is the
// value written into the `data` chunk's own size field; it is the real
// payload length for a valid file and something larger for the
// over-declaration arm, which is the only way to build a file whose header
// lies about its own contents.
std::vector<uint8_t> wav_bytes(uint32_t frames, uint32_t declared_data_size, uint32_t declared_fmt_size) {
    std::vector<uint8_t> fmt;
    put<uint16_t>(fmt, 3);                // IEEE float
    put<uint16_t>(fmt, 1);                // mono
    put<uint32_t>(fmt, kSampleRate);
    put<uint32_t>(fmt, kSampleRate * 4);  // byte rate
    put<uint16_t>(fmt, 4);                // block align
    put<uint16_t>(fmt, 32);               // bits per sample

    std::vector<uint8_t> body;
    put_bytes(body, "fmt ", 4);
    put<uint32_t>(body, declared_fmt_size);
    put_bytes(body, fmt.data(), fmt.size());
    put_bytes(body, "data", 4);
    put<uint32_t>(body, declared_data_size);
    for (uint32_t index = 0; index < frames; ++index) {
        // A quiet ramp rather than silence: compute_log_mel has its own
        // digital-silence handling, and this file's subject is the parser.
        put<float>(body, 0.25f * float(index % 97) / 97.0f);
    }

    std::vector<uint8_t> bytes;
    put_bytes(bytes, "RIFF", 4);
    put<uint32_t>(bytes, uint32_t(4 + body.size()));
    put_bytes(bytes, "WAVE", 4);
    put_bytes(bytes, body.data(), body.size());
    return bytes;
}

// A well-formed file with one extra chunk appended AFTER `data`. `pad`
// controls whether the RIFF padding byte that must follow an odd-sized chunk
// is actually written -- the single byte that separates a legal odd chunk
// from a truncated one, and the reason the length bound has to count it.
//
// LAST, not first, and that placement is the whole isolation argument. An odd
// chunk placed anywhere but the end is followed by another chunk, and the
// reader's pad skip runs unconditionally on an odd size -- so with the bound
// left un-widened it skips one byte INTO the next chunk id, every later field
// is misread, and some downstream garbage size trips the un-widened bound and
// produces the identical "declares a chunk larger than the file holds"
// message. Measured, not assumed: the first draft of this helper spliced the
// chunk in ahead of `fmt ` and both arms below passed with the pad-widening
// deleted. With the odd chunk last there is nothing after it to misread, so
// the only thing that can refuse the unpadded file is the bound itself.
std::vector<uint8_t> wav_with_trailing_chunk(const char * chunk_id, uint32_t payload_size, bool pad) {
    std::vector<uint8_t> bytes = wav_bytes(kFrameCount, kFrameCount * 4, 16);
    put_bytes(bytes, chunk_id, 4);
    put<uint32_t>(bytes, payload_size);
    for (uint32_t index = 0; index < payload_size; ++index) {
        put<uint8_t>(bytes, uint8_t(index));
    }
    if (pad && (payload_size % 2 == 1)) {
        put<uint8_t>(bytes, 0);
    }
    const uint32_t riff_size = uint32_t(bytes.size() - 8);
    std::memcpy(bytes.data() + 4, &riff_size, sizeof(riff_size));
    return bytes;
}

// A PCM16 file whose `data` chunk declares an ODD byte count and really holds
// exactly that many bytes, followed by its pad byte -- so the file-length
// bound has nothing to say about it and the odd count is the only thing
// wrong. Payload content is arbitrary because no arm expects this file to be
// accepted.
std::vector<uint8_t> pcm16_wav_with_odd_data(uint32_t declared_data_size) {
    std::vector<uint8_t> fmt;
    put<uint16_t>(fmt, 1);                // PCM
    put<uint16_t>(fmt, 1);                // mono
    put<uint32_t>(fmt, kSampleRate);
    put<uint32_t>(fmt, kSampleRate * 2);  // byte rate
    put<uint16_t>(fmt, 2);                // block align
    put<uint16_t>(fmt, 16);               // bits per sample

    std::vector<uint8_t> body;
    put_bytes(body, "fmt ", 4);
    put<uint32_t>(body, uint32_t(16));
    put_bytes(body, fmt.data(), fmt.size());
    put_bytes(body, "data", 4);
    put<uint32_t>(body, declared_data_size);
    for (uint32_t index = 0; index < declared_data_size; ++index) {
        put<uint8_t>(body, uint8_t(index));
    }
    put<uint8_t>(body, 0);  // the pad byte an odd chunk requires

    std::vector<uint8_t> bytes;
    put_bytes(bytes, "RIFF", 4);
    put<uint32_t>(bytes, uint32_t(4 + body.size()));
    put_bytes(bytes, "WAVE", 4);
    put_bytes(bytes, body.data(), body.size());
    return bytes;
}

bool write_file(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
    }
    return bool(output);
}

std::string read_text(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return std::string();
    }
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

struct Run {
    bool        succeeded = false;  // the driver exited 0
    std::string diagnostics;        // whatever it wrote to stdout/stderr
};

// Runs the mel driver over `wav_path`. Both streams are captured so an arm
// can assert WHICH rule refused the file, not merely that something did --
// several of the reader's refusals share one exit code.
Run run_driver(const std::string & fixtures, const std::string & wav_path) {
    const std::string output_path = fixtures + "/mel.f32";
    const std::string log_path    = fixtures + "/driver.log";
    std::remove(output_path.c_str());
    std::remove(log_path.c_str());
    const std::string command =
        "\"" SYNTH_MEL_DRIVER "\" \"" + wav_path + "\" \"" + output_path + "\" > \"" + log_path + "\" 2>&1";
    Run run;
    run.succeeded   = std::system(command.c_str()) == 0;
    run.diagnostics = read_text(log_path);
    return run;
}

bool mentions(const Run & run, const char * needle) {
    return run.diagnostics.find(needle) != std::string::npos;
}

// --- A well-formed file is still accepted. The positive control for both
// rules below: a bound derived from the file's own length must not refuse a
// file whose chunks fit inside it, and the header check must not refuse a
// header that was fully read. Without this arm, "refuse everything" would
// pass the other two.
int test_a_well_formed_wav_is_accepted(const std::string & fixtures) {
    const std::string path = fixtures + "/valid.wav";
    SYNTH_TEST_CHECK(write_file(path, wav_bytes(kFrameCount, kFrameCount * 4, 16)));
    const Run run = run_driver(fixtures, path);
    if (!run.succeeded) {
        std::fprintf(stderr, "the driver refused a well-formed WAV: %s\n", run.diagnostics.c_str());
        return 1;
    }
    return 0;
}

// --- A `data` chunk that declares more bytes than the file holds is refused
// before anything is sized from it.
//
// This is the arm that has to fail if the bound is removed, and it does: with
// the resize sized from the claim alone, the read that follows falls short,
// std::vector leaves the untouched tail ZERO, `have_data` is set anyway, and
// the driver reports SUCCESS over a buffer whose second half is fabricated
// silence. So the assertion here is `!succeeded`, and it is the fabrication
// -- not the allocation -- that makes it load-bearing at a size small enough
// to run in milliseconds. (The allocation matters too: the same field can
// claim 4 GiB.)
//
// The declared size is deliberately only 16x the real payload rather than
// UINT32_MAX: a value near the type's ceiling would also be refused by the
// allocator itself under a sanitizer, which would let this arm pass with the
// bound deleted. A modest over-declaration can only be caught by the bound.
int test_an_over_declared_data_chunk_is_refused(const std::string & fixtures) {
    const std::string path = fixtures + "/over-declared-data.wav";
    SYNTH_TEST_CHECK(write_file(path, wav_bytes(kFrameCount, kFrameCount * 4 * 16, 16)));
    const Run run = run_driver(fixtures, path);
    if (run.succeeded) {
        std::fprintf(stderr, "the driver accepted a data chunk larger than the file: %s\n", run.diagnostics.c_str());
        return 1;
    }
    SYNTH_TEST_CHECK(mentions(run, "declares a chunk larger than the file holds"));
    return 0;
}

// --- The same rule for the `fmt` chunk, which allocates from the identical
// field one branch earlier. Separate from the arm above because the two
// allocations are separate statements: a fix applied to only one of them
// would leave this one passing.
int test_an_over_declared_fmt_chunk_is_refused(const std::string & fixtures) {
    const std::string path = fixtures + "/over-declared-fmt.wav";
    SYNTH_TEST_CHECK(write_file(path, wav_bytes(kFrameCount, kFrameCount * 4, 1u << 20)));
    const Run run = run_driver(fixtures, path);
    if (run.succeeded) {
        std::fprintf(stderr, "the driver accepted a fmt chunk larger than the file: %s\n", run.diagnostics.c_str());
        return 1;
    }
    SYNTH_TEST_CHECK(mentions(run, "declares a chunk larger than the file holds"));
    return 0;
}

// --- A file too short to hold the 12-byte RIFF/WAVE header is refused on the
// stream state, not on the contents of buffers that were never filled.
//
// The file starts with a real "RIFF" so the first 4-byte read SUCCEEDS and
// only the later ones fail -- that is what leaves `wave[4]` holding whatever
// was on the stack. Asserting the exit code alone would prove nothing here
// (indeterminate bytes are overwhelmingly unlikely to spell "WAVE", so the
// old code also refused this file, for the wrong reason and after reading
// them). Asserting WHICH message comes back is what makes the arm
// load-bearing: with the stream-state check removed, the driver reports "is
// not a RIFF/WAVE file" instead and this fails.
int test_a_short_header_is_refused_on_the_stream_state(const std::string & fixtures) {
    const std::string    path = fixtures + "/short-header.wav";
    std::vector<uint8_t> bytes;
    put_bytes(bytes, "RIFF", 4);
    put<uint16_t>(bytes, 0);  // two bytes of a four-byte size field, then EOF
    SYNTH_TEST_CHECK(write_file(path, bytes));
    const Run run = run_driver(fixtures, path);
    if (run.succeeded) {
        std::fprintf(stderr, "the driver accepted a 6-byte file: %s\n", run.diagnostics.c_str());
        return 1;
    }
    SYNTH_TEST_CHECK(mentions(run, "shorter than a 12-byte RIFF/WAVE header"));
    return 0;
}

// --- An empty file, the degenerate end of the same rule: not one of the
// three header reads succeeds, so all three buffers are indeterminate.
int test_an_empty_file_is_refused_on_the_stream_state(const std::string & fixtures) {
    const std::string path = fixtures + "/empty.wav";
    SYNTH_TEST_CHECK(write_file(path, std::vector<uint8_t>()));
    const Run run = run_driver(fixtures, path);
    if (run.succeeded) {
        std::fprintf(stderr, "the driver accepted an empty file: %s\n", run.diagnostics.c_str());
        return 1;
    }
    SYNTH_TEST_CHECK(mentions(run, "shorter than a 12-byte RIFF/WAVE header"));
    return 0;
}

// =============================================================================
// PR #11's review findings. Read the coverage note before trusting these four
// further than they go: the three defects the review named do not all live in
// the driver this file can reach.
//
//  - the odd-`data` HEAP OVERFLOW is omnivoice_profile_test.cpp's alone. That
//    reader sizes an int16_t vector as chunk_size/2 and then reads chunk_size
//    bytes into it; the mel driver sizes a char vector as chunk_size and reads
//    chunk_size, so it never had the bug. The arm below still pins that the
//    mel driver REFUSES such a file -- but it refuses it by a rule that was
//    already there, so deleting the new omnivoice rule cannot fail this arm.
//  - the short-`fmt ` over-read is omnivoice_profile_test.cpp's and
//    qwen3_tts_clone_real.cpp's. The mel driver's `chunk_size < 16` floor
//    already stood in front of its own fixed reads. Same caveat as above.
//  - the pad byte is the one finding that reaches this driver, and only in
//    its bound half: three of the four readers already skipped the pad byte,
//    including this one, and only omnivoice_profile_test.cpp did not. What is
//    NEW here is counting the pad in the length bound, and the last two arms
//    are load-bearing against exactly that.
//
// The first two arms are therefore regression coverage for a contract this
// driver already met, kept because they are the shapes the other three
// readers now have to meet too and this is the only place in the suite that
// can execute any of them at all.
// =============================================================================

// --- A `data` chunk with an odd byte count is refused.
int test_an_odd_data_chunk_is_refused(const std::string & fixtures) {
    const std::string path = fixtures + "/odd-data.wav";
    SYNTH_TEST_CHECK(write_file(path, pcm16_wav_with_odd_data(4097)));
    const Run run = run_driver(fixtures, path);
    if (run.succeeded) {
        std::fprintf(stderr, "the driver accepted an odd data chunk: %s\n", run.diagnostics.c_str());
        return 1;
    }
    SYNTH_TEST_CHECK(mentions(run, "not a whole number of interleaved frames"));
    return 0;
}

// --- A `fmt ` chunk declaring fewer than its 16 fixed bytes is refused.
// Declared 8 while 16 real bytes follow, so the file is long enough and only
// the declaration is wrong -- which is precisely the case that would leave a
// reader that trusts it standing 8 bytes into the next chunk.
int test_a_short_fmt_chunk_is_refused(const std::string & fixtures) {
    const std::string path = fixtures + "/short-fmt.wav";
    SYNTH_TEST_CHECK(write_file(path, wav_bytes(kFrameCount, kFrameCount * 4, 8)));
    const Run run = run_driver(fixtures, path);
    if (run.succeeded) {
        std::fprintf(stderr, "the driver accepted an 8-byte fmt chunk: %s\n", run.diagnostics.c_str());
        return 1;
    }
    SYNTH_TEST_CHECK(mentions(run, "fmt chunk too small"));
    return 0;
}

// --- An odd-sized chunk WITH its pad byte still parses. The positive half of
// the pad rule, and the half that catches an over-tightened bound: counting
// the pad byte must not make a legal file unreadable.
int test_an_odd_chunk_with_its_pad_byte_still_parses(const std::string & fixtures) {
    const std::string path = fixtures + "/junk-odd-padded.wav";
    SYNTH_TEST_CHECK(write_file(path, wav_with_trailing_chunk("JUNK", 5, /*pad=*/true)));
    const Run run = run_driver(fixtures, path);
    if (!run.succeeded) {
        std::fprintf(stderr, "the driver refused a legal odd chunk: %s\n", run.diagnostics.c_str());
        return 1;
    }
    return 0;
}

// --- The same chunk WITHOUT its pad byte is refused. Byte-for-byte identical
// to the arm above except for that one byte, so nothing else can be what
// changes the answer. Before the bound counted the pad, this file PARSED: the
// chunk's own five bytes are all present, so the un-widened bound accepted
// it, the reader skipped five and then one more past the end, and the loop
// simply ended with fmt and data already in hand.
int test_an_odd_chunk_missing_its_pad_byte_is_refused(const std::string & fixtures) {
    const std::string path = fixtures + "/junk-odd-unpadded.wav";
    SYNTH_TEST_CHECK(write_file(path, wav_with_trailing_chunk("JUNK", 5, /*pad=*/false)));
    const Run run = run_driver(fixtures, path);
    if (run.succeeded) {
        std::fprintf(stderr, "the driver accepted an odd chunk with no pad byte: %s\n", run.diagnostics.c_str());
        return 1;
    }
    SYNTH_TEST_CHECK(mentions(run, "declares a chunk larger than the file holds"));
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <fixtures-dir>\n", argv[0]);
        return 1;
    }
    const std::string fixtures = argv[1];

    SYNTH_TEST_CHECK(test_a_well_formed_wav_is_accepted(fixtures) == 0);
    SYNTH_TEST_CHECK(test_an_over_declared_data_chunk_is_refused(fixtures) == 0);
    SYNTH_TEST_CHECK(test_an_over_declared_fmt_chunk_is_refused(fixtures) == 0);
    SYNTH_TEST_CHECK(test_a_short_header_is_refused_on_the_stream_state(fixtures) == 0);
    SYNTH_TEST_CHECK(test_an_empty_file_is_refused_on_the_stream_state(fixtures) == 0);
    SYNTH_TEST_CHECK(test_an_odd_data_chunk_is_refused(fixtures) == 0);
    SYNTH_TEST_CHECK(test_a_short_fmt_chunk_is_refused(fixtures) == 0);
    SYNTH_TEST_CHECK(test_an_odd_chunk_with_its_pad_byte_still_parses(fixtures) == 0);
    SYNTH_TEST_CHECK(test_an_odd_chunk_missing_its_pad_byte_is_refused(fixtures) == 0);
    std::printf("qwen3-tts wav reader: ok\n");
    return 0;
}
