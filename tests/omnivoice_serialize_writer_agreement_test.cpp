// Plan 4's Task 5 -- carry-over item 5 from Plan 3's own ledger: pins the
// REAL serialize_clone_prompt/serialize_design_instruct writer output
// against the REAL kPrescanKnownKeys whitelist (profile.h) in a fast unit
// test. Until now, neither of this schema's two hand-transcriptions of "the
// keys this writer emits" was generated from, or mechanically checked
// against, the other:
//   * kPrescanKnownKeys (profile.h) is transcribed BY HAND from what
//     set_common_metadata/serialize_clone_prompt/serialize_design_instruct
//     (profile.cpp) actually write;
//   * tests/omnivoice_serialize_test.cpp's own common_kv_bytes helper is a
//     SECOND, independent hand-transcription of the same writer behavior,
//     used there to build synthetic malformed buffers for its own tamper
//     matrix -- not to check writer/whitelist agreement.
// The only place the REAL writer and the REAL whitelist actually meet is the
// model-guarded round-trip integration test (tests/omnivoice_profile_test.cpp),
// which needs a multi-GB real GGUF and is not in the fast unit loop.
//
// Two drift directions, and the second is the dangerous one:
//   * a key ADDED to the writer without a matching whitelist entry -> the
//     writer's own output gets rejected by its own loader. Loud, but only
//     ever exercised by the model-guarded test.
//   * a key REMOVED from the writer while left in the whitelist -> a
//     SILENTLY TOO PERMISSIVE whitelist (prescan_buffer would still accept a
//     buffer shaped like the writer USED to produce, not what it produces
//     now) -- a security-relevant loosening of the untrusted-bytes surface
//     this whitelist exists to defend, and no other fast test would notice.
//
// Method: build a ClonePrompt and a DesignInstruct payload entirely by hand
// -- neither serialize_clone_prompt nor serialize_design_instruct takes a
// Model argument at all, so no real (or synthetic) GGUF package is loaded
// here, the same "hand-built payload, no real encode chain" pattern
// tests/omnivoice_serialize_test.cpp's own make_clone_prompt helper and
// tests/omnivoice_profile_test.cpp's synthetic-package tests already use
// elsewhere in this family. Run them through the REAL writer functions,
// parse the REAL emitted bytes with ggml's own gguf_init_from_buffer -- a
// genuine GGUF reader, independent of profile.cpp's own hand-rolled
// prescan_buffer walk -- and compare the resulting key sets against
// profile.h's kPrescanKnownKeys.
//
// The assertion is three checks, not one equality, because the two kinds
// emit DISJOINT key sets by design: ClonePrompt's 3 own keys
// (transcript_text/ref_rms/language_tag) and DesignInstruct's 1 own key
// (instruct) never appear in the other kind's envelope, so neither kind's
// emitted set alone equals the whitelist.
//   1. clone-prompt's emitted keys ⊆ whitelist
//   2. design-instruct's emitted keys ⊆ whitelist
//   3. (clone-prompt's emitted keys ∪ design-instruct's emitted keys) == whitelist
// (1) and (2) catch a key ADDED to either writer: an emitted key with no
// whitelist entry breaks the subset relation. (3) is what actually catches
// the dangerous direction: a key REMOVED from either writer leaves the
// whitelist with an entry NEITHER kind emits any more, which only the
// union-equals-whitelist check (not (1) or (2), which only ever look at
// what IS emitted) can surface.

#include "arch/omnivoice/profile.h"
#include "gguf.h"
#include "synthesize.h"
#include "test-assert.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using synth::omnivoice::ClonePrompt;
using synth::omnivoice::DesignInstruct;
using synth::omnivoice::kPrescanKnownKeyCount;
using synth::omnivoice::kPrescanKnownKeys;

struct GgufContextDeleter {
    void operator()(gguf_context * context) const {
        if (context != nullptr) {
            gguf_free(context);
        }
    }
};

using OwnedGgufContext = std::unique_ptr<gguf_context, GgufContextDeleter>;

// Parses `bytes` with ggml's own gguf_init_from_buffer -- independent of
// profile.cpp's own prescan_buffer, which this test deliberately never calls
// -- and fills `out_keys` with the exact set of metadata key names found.
// `no_alloc = true` mirrors load_profile_from_memory's own call
// (profile.cpp): this test only ever reads metadata key NAMES, never a
// tensor payload byte. Returns false (leaving `out_keys` empty) only if
// `bytes` does not parse as a GGUF buffer at all -- which would itself mean
// serialize_clone_prompt/serialize_design_instruct produced something
// broken, not a whitelist question.
bool parse_key_set(const std::vector<uint8_t> & bytes, std::set<std::string> & out_keys) {
    out_keys.clear();
    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = nullptr;
    OwnedGgufContext ctx(gguf_init_from_buffer(bytes.data(), bytes.size(), init_params));
    if (ctx == nullptr) {
        return false;
    }
    const int64_t n_kv = gguf_get_n_kv(ctx.get());
    for (int64_t index = 0; index < n_kv; ++index) {
        out_keys.insert(gguf_get_key(ctx.get(), index));
    }
    return true;
}

void print_keys(const char * label, const std::set<std::string> & keys) {
    std::fprintf(stderr, "%s (%zu): {", label, keys.size());
    bool first = true;
    for (const std::string & key : keys) {
        std::fprintf(stderr, "%s%s", first ? "" : ", ", key.c_str());
        first = false;
    }
    std::fprintf(stderr, "}\n");
}

}  // namespace

int main() {
    // A hand-built ClonePrompt: 2 frames x 8 codebooks (serialize_clone_prompt's
    // own "non-empty multiple of 8" contract), otherwise arbitrary content --
    // this test cares about which METADATA KEYS the envelope carries, never
    // about any field's value.
    ClonePrompt clone_prompt;
    clone_prompt.reference_tokens.assign(2 * 8, 1);
    clone_prompt.transcript_text = "a";
    clone_prompt.ref_rms         = 0.5f;
    clone_prompt.language_tag    = "en";

    // A hand-built DesignInstruct: serialize_design_instruct performs no
    // vocabulary validation of its own (resolve_instruct is a SEPARATE,
    // creation-time step -- see profile.h's own header comment), so any
    // non-empty string exercises the writer identically.
    DesignInstruct design_instruct;
    design_instruct.instruct = "male";

    // Arbitrary but non-zero, so a reviewer scanning a hex dump of the
    // envelope bytes never mistakes this field for an all-zero placeholder
    // that was never set.
    uint8_t compatibility_id[32];
    for (size_t index = 0; index < sizeof(compatibility_id); ++index) {
        compatibility_id[index] = uint8_t(index);
    }

    std::vector<uint8_t> clone_bytes;
    SYNTH_TEST_CHECK(synth::omnivoice::serialize_clone_prompt(clone_prompt, compatibility_id, clone_bytes) == SYNTH_OK);
    SYNTH_TEST_CHECK(!clone_bytes.empty());

    std::vector<uint8_t> design_bytes;
    SYNTH_TEST_CHECK(synth::omnivoice::serialize_design_instruct(design_instruct, compatibility_id, design_bytes) ==
                     SYNTH_OK);
    SYNTH_TEST_CHECK(!design_bytes.empty());

    std::set<std::string> clone_keys;
    SYNTH_TEST_CHECK(parse_key_set(clone_bytes, clone_keys));
    std::set<std::string> design_keys;
    SYNTH_TEST_CHECK(parse_key_set(design_bytes, design_keys));

    std::set<std::string> whitelist_keys;
    for (size_t index = 0; index < kPrescanKnownKeyCount; ++index) {
        whitelist_keys.insert(kPrescanKnownKeys[index].key);
    }

    // Printed unconditionally (not gated on a failing check): this is the
    // "actual failure output" this task's own brief asks the mutation proof
    // to report, and the sets are cheap enough (a dozen short strings) that
    // printing them every run costs nothing.
    print_keys("clone-prompt emitted keys", clone_keys);
    print_keys("design-instruct emitted keys", design_keys);
    print_keys("kPrescanKnownKeys whitelist", whitelist_keys);

    std::set<std::string> emitted_union = clone_keys;
    emitted_union.insert(design_keys.begin(), design_keys.end());

    // (1) clone-prompt's emitted keys ⊆ whitelist -- a key ADDED to
    // serialize_clone_prompt (or set_common_metadata) without a matching
    // whitelist entry breaks this.
    SYNTH_TEST_CHECK(std::includes(whitelist_keys.begin(), whitelist_keys.end(), clone_keys.begin(), clone_keys.end()));

    // (2) design-instruct's emitted keys ⊆ whitelist -- the same check for
    // serialize_design_instruct (or set_common_metadata).
    SYNTH_TEST_CHECK(
        std::includes(whitelist_keys.begin(), whitelist_keys.end(), design_keys.begin(), design_keys.end()));

    // (3) the union of what both kinds actually emit equals the whitelist
    // exactly -- the check that catches the DANGEROUS direction: a key
    // REMOVED from either writer leaves a whitelist entry neither kind emits
    // any more, which (1) and (2) alone can never see (they only ever look
    // at what IS emitted, never at what the whitelist additionally allows).
    SYNTH_TEST_CHECK(emitted_union == whitelist_keys);

    return 0;
}
