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
// Method: build a ClonePrompt and a DesignInstruct payload entirely by hand
// -- neither serialize_clone_prompt nor serialize_design_instruct takes a
// Model argument at all, so no real (or synthetic) GGUF package is loaded
// here, the same "hand-built payload, no real encode chain" pattern
// tests/omnivoice_serialize_test.cpp's own make_clone_prompt helper and
// tests/omnivoice_profile_test.cpp's synthetic-package tests already use
// elsewhere in this family. Run them through the REAL writer functions,
// parse the REAL emitted bytes with ggml's own gguf_init_from_buffer -- a
// genuine GGUF reader, independent of profile.cpp's own hand-rolled
// prescan_buffer walk -- and check the resulting key sets/counts against
// profile.h's kPrescanKnownKeys, kPrescanKvCountClone, and
// kPrescanKvCountDesign.
//
// Fix-round-1 (reviewer finding): the original three checks below (the two
// subset checks and the union-equals-whitelist check) are blind to two
// further, realistic drifts, because none of the three cares WHICH kind
// emitted a given key, only whether it was emitted by "either":
//   * a key MOVED between the two writers (e.g. `language_tag` from clone to
//     design) leaves the union unchanged -- all three original checks stay
//     green -- while prescan_buffer (profile.cpp) would reject every real
//     envelope of BOTH kinds, since neither one's key set matches either of
//     prescan_buffer's own two accepted shapes any more.
//   * a key added correctly to a writer AND to kPrescanKnownKeys, but whose
//     matching kPrescanKvCountClone/kPrescanKvCountDesign is not bumped,
//     also leaves all three original checks green (the whitelist and the
//     writer agree on the key SET) while prescan_buffer's own separate `n_kv`
//     gate -- a plain scalar comparison, independent of which specific keys
//     are present -- rejects every real envelope of that kind.
// Two more checks close these, using profile.h's `PrescanKeyScope` (added in
// this same fix round) and the two count constants (moved to profile.h
// alongside kPrescanKnownKeys, exposed the same way and for the same reason):
//   4/5. per-kind EXACT set equality: clone's emitted keys must equal
//        exactly the kCommon+kCloneOnly-scoped entries of kPrescanKnownKeys;
//        design's must equal exactly the kCommon+kDesignOnly-scoped entries.
//        Built by filtering kPrescanKnownKeys on its own `scope` field --
//        not by re-typing "transcript_text"/"ref_rms"/"language_tag"/
//        "instruct" as fresh string literals here, which would make this
//        test a fourth hand-transcription of the same knowledge.
//   6/7. per-kind COUNT equality: clone_keys.size() == kPrescanKvCountClone,
//        design_keys.size() == kPrescanKvCountDesign -- the SAME two scalars
//        prescan_buffer's own `n_kv` gate compares against, so a count drift
//        independent of any actual key-set drift (the "added correctly
//        everywhere except the count constant" case above) is still caught.
//
// All seven booleans are computed and printed BEFORE any SYNTH_TEST_CHECK
// runs (see the "check:" lines below), so the failure evidence for a given
// mutation is never limited to whichever single check happens to run first
// and halt the process -- the stderr transcript always shows every check's
// own pass/fail verdict.

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
using synth::omnivoice::kPrescanKvCountClone;
using synth::omnivoice::kPrescanKvCountDesign;
using synth::omnivoice::PrescanKeyScope;

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

void print_check(const char * label, bool value) {
    std::fprintf(stderr, "check: %-55s -> %s\n", label, value ? "pass" : "FAIL");
}

// Filters kPrescanKnownKeys by `scope`, collecting every entry whose scope is
// `kCommon` or matches `only_scope` -- i.e. the exact expected key set for
// the kind `only_scope` names. Reused for both kinds below rather than
// writing the filter twice.
std::set<std::string> expected_keys_for(PrescanKeyScope only_scope) {
    std::set<std::string> keys;
    for (size_t index = 0; index < kPrescanKnownKeyCount; ++index) {
        const auto & spec = kPrescanKnownKeys[index];
        if (spec.scope == PrescanKeyScope::kCommon || spec.scope == only_scope) {
            keys.insert(spec.key);
        }
    }
    return keys;
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

    std::set<std::string> emitted_union = clone_keys;
    emitted_union.insert(design_keys.begin(), design_keys.end());

    // Expected per-kind sets, derived from kPrescanKnownKeys' own `scope`
    // field -- not from re-typed key-name literals (see this file's own
    // header comment on why that would be a fourth hand-transcription).
    const std::set<std::string> expected_clone_keys  = expected_keys_for(PrescanKeyScope::kCloneOnly);
    const std::set<std::string> expected_design_keys = expected_keys_for(PrescanKeyScope::kDesignOnly);

    // Every boolean this test checks, computed up front.
    const bool clone_is_subset =
        std::includes(whitelist_keys.begin(), whitelist_keys.end(), clone_keys.begin(), clone_keys.end());
    const bool design_is_subset =
        std::includes(whitelist_keys.begin(), whitelist_keys.end(), design_keys.begin(), design_keys.end());
    const bool union_matches_whitelist = (emitted_union == whitelist_keys);
    const bool clone_matches_expected  = (clone_keys == expected_clone_keys);
    const bool design_matches_expected = (design_keys == expected_design_keys);
    const bool clone_count_ok          = (int64_t(clone_keys.size()) == kPrescanKvCountClone);
    const bool design_count_ok         = (int64_t(design_keys.size()) == kPrescanKvCountDesign);

    // Printed unconditionally (not gated on a failing check): this is the
    // "actual failure output" a mutation proof needs to report, and the data
    // is cheap enough (a dozen short strings, seven booleans) that printing
    // it every run costs nothing. Because this all runs BEFORE any
    // SYNTH_TEST_CHECK below, every one of these lines is visible in the
    // transcript regardless of which check ends up halting the process.
    print_keys("clone-prompt emitted keys", clone_keys);
    print_keys("design-instruct emitted keys", design_keys);
    print_keys("kPrescanKnownKeys whitelist", whitelist_keys);
    print_keys("expected clone-prompt keys (scope filter)", expected_clone_keys);
    print_keys("expected design-instruct keys (scope filter)", expected_design_keys);
    print_check("(1) clone-prompt emitted keys subset of whitelist", clone_is_subset);
    print_check("(2) design-instruct emitted keys subset of whitelist", design_is_subset);
    print_check("(3) union(clone, design) == whitelist", union_matches_whitelist);
    print_check("(4) clone-prompt emitted keys == expected (scope) set", clone_matches_expected);
    print_check("(5) design-instruct emitted keys == expected (scope) set", design_matches_expected);
    print_check("(6) clone-prompt emitted key count == kPrescanKvCountClone", clone_count_ok);
    print_check("(7) design-instruct emitted key count == kPrescanKvCountDesign", design_count_ok);

    // (1)/(2): no key emitted that the whitelist lacks -- a key ADDED to
    // either writer without a matching whitelist entry breaks the subset
    // relation.
    SYNTH_TEST_CHECK(clone_is_subset);
    SYNTH_TEST_CHECK(design_is_subset);

    // (3): no whitelist entry neither kind emits -- the check that catches a
    // key REMOVED from a writer while left in the whitelist (a silently
    // too-permissive whitelist that (1)/(2) alone can never see, since they
    // only ever look at what IS emitted).
    SYNTH_TEST_CHECK(union_matches_whitelist);

    // (4)/(5): per-kind EXACT equality -- catches a key MOVED between the
    // two writers, which leaves (1), (2), AND (3) all green (the union of
    // what both kinds emit is unchanged by a move) but changes what EACH
    // individual kind emits.
    SYNTH_TEST_CHECK(clone_matches_expected);
    SYNTH_TEST_CHECK(design_matches_expected);

    // (6)/(7): per-kind COUNT equality against the SAME scalar constants
    // prescan_buffer's own `n_kv` gate uses -- catches a key added correctly
    // to a writer AND to kPrescanKnownKeys (with the correct `scope`, so (4)/
    // (5) above stay green) whose matching count constant was never bumped:
    // prescan_buffer would still reject every real envelope of that kind
    // over the stale `n_kv` comparison alone.
    SYNTH_TEST_CHECK(clone_count_ok);
    SYNTH_TEST_CHECK(design_count_ok);

    return 0;
}
