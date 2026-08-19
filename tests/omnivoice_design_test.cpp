// Description Text ("voice design") profiles (Plan 3's Task 15): the
// instruct vocabulary transcription and the create_from_description
// dispatch, against a synthetic package rather than the real multi-GB one --
// nothing here touches the frontend or a tensor, so a real model buys
// nothing a synthetic one does not already have. This is a genuine unit
// test (no real model required), unlike tests/omnivoice_profile_test.cpp
// (Task 14's Reference Audio equivalent), which is model-guarded because
// Model::encode_reference is a real forward pass; resolve_instruct is pure
// string processing and create_from_description never reaches the frontend
// either, so nothing here needed that heavier harness.
//
// Two halves:
//   1. synth::omnivoice::resolve_instruct() called directly -- the
//      vocabulary transcription itself, against
//      omnivoice/utils/voice_design.py:31-97 and
//      omnivoice/models/omnivoice.py's own _resolve_instruct (1492-1621),
//      both at the pinned revision (see profile.cpp's own citations).
//   2. synth_voice_profile_create_from_description() through the public C
//      seam, against a synthetic package -- the dispatcher's own checks
//      (description required, seed concreteness, description_language
//      validation) that resolve_instruct never sees.

#include "arch/omnivoice/omnivoice.h"
#include "arch/omnivoice/profile.h"
#include "omnivoice_synthetic_package.h"
#include "synthesize.h"
#include "test-assert.h"
#include "voice-profile-handle.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

namespace {

// ---------------------------------------------------------------------------
// Part 1: resolve_instruct() directly.
// ---------------------------------------------------------------------------

synth_status_t resolve(const std::string & description,
                       bool                use_zh,
                       std::string &       instruct,
                       std::string &       code,
                       std::string &       message) {
    std::shared_ptr<const synth::omnivoice::DesignInstruct> design;
    const char *                                            diagnostic_code = nullptr;
    std::string                                             diagnostic_message;
    const synth_status_t                                    status =
        synth::omnivoice::resolve_instruct(description, use_zh, design, diagnostic_code, diagnostic_message);
    instruct = status == SYNTH_OK ? design->instruct : std::string();
    code     = diagnostic_code != nullptr ? diagnostic_code : std::string();
    message  = diagnostic_message;
    return status;
}

int check_each_category_single_item() {
    std::string instruct, code, message;
    // gender
    SYNTH_TEST_CHECK(resolve("male", false, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "male");
    // age
    SYNTH_TEST_CHECK(resolve("teenager", false, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "teenager");
    // pitch
    SYNTH_TEST_CHECK(resolve("very high pitch", false, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "very high pitch");
    // style (whisper)
    SYNTH_TEST_CHECK(resolve("whisper", false, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "whisper");
    // accent: EN-only, and forces use_zh=false even when the baseline says
    // otherwise (omnivoice.py:1591-1592).
    SYNTH_TEST_CHECK(resolve("british accent", false, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "british accent");
    SYNTH_TEST_CHECK(resolve("british accent", true, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "british accent");
    // dialect: ZH-only, and forces use_zh=true even when the baseline says
    // otherwise (omnivoice.py:1589-1590) -- a single item has no separator
    // to prove that with, but the unify step still has to leave it
    // untranslated either way (no EN counterpart exists).
    SYNTH_TEST_CHECK(resolve("东北话", false, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "东北话");
    SYNTH_TEST_CHECK(resolve("东北话", true, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "东北话");
    return 0;
}

int check_category_conflict() {
    std::string instruct, code, message;
    SYNTH_TEST_CHECK(resolve("male, female", false, instruct, code, message) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(code == "voice_profile.instruct_category_conflict");
    SYNTH_TEST_CHECK(message.find("'male' vs 'female'") != std::string::npos);
    return 0;
}

int check_dialect_accent_mix() {
    std::string instruct, code, message;
    SYNTH_TEST_CHECK(resolve("河南话, british accent", false, instruct, code, message) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(code == "voice_profile.instruct_dialect_accent_mix");
    return 0;
}

int check_unknown_item() {
    std::string instruct, code, message;
    SYNTH_TEST_CHECK(resolve("sings well", false, instruct, code, message) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(code == "voice_profile.instruct_unknown_item");
    // Named verbatim, no difflib "did you mean" suggestion.
    SYNTH_TEST_CHECK(message.find("'sings well'") != std::string::npos);
    SYNTH_TEST_CHECK(message.find("did you mean") == std::string::npos);
    return 0;
}

int check_dialect_forces_zh_join() {
    std::string instruct, code, message;
    // Mixed-script input, individually valid items (validation is per-item
    // against the flat vocabulary regardless of which script each one was
    // written in): the EN gender term translates, the dialect item forces
    // the WHOLE result to Chinese and the full-width join separator, even
    // though the baseline requested here is English.
    SYNTH_TEST_CHECK(resolve("male, 河南话", false, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "男，河南话");
    return 0;
}

int check_golden_instructs_are_fixed_points() {
    std::string instruct, code, message;
    // tests/golden/omnivoice/omnivoice-0-6b.manifest.json's omni-design-en
    // case, language "en" -> use_zh=false.
    SYNTH_TEST_CHECK(resolve("female, young adult, high pitch", false, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "female, young adult, high pitch");
    // The same manifest's omni-design-zh case, language "zh" -> use_zh=true.
    SYNTH_TEST_CHECK(resolve("男，老年，低音调", true, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "男，老年，低音调");
    return 0;
}

int check_translation_direction() {
    std::string instruct, code, message;
    // The EN golden instruct, unified TO Chinese: proves EN_TO_ZH beyond the
    // identity case above.
    SYNTH_TEST_CHECK(resolve("female, young adult, high pitch", true, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "女，青年，高音调");
    // The ZH golden instruct, unified TO English: proves ZH_TO_EN.
    SYNTH_TEST_CHECK(resolve("男，老年，低音调", false, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "male, elderly, low pitch");
    return 0;
}

int check_width_comma_splitting() {
    std::string instruct, code, message;
    // Half-width comma, no surrounding spaces: still splits into two items
    // (a category conflict either way proves the split ran).
    SYNTH_TEST_CHECK(resolve("male,female", false, instruct, code, message) == SYNTH_ERR_INVALID_ARG);
    SYNTH_TEST_CHECK(code == "voice_profile.instruct_category_conflict");
    // Half-width comma, irregular ASCII whitespace around items and at the
    // string's own ends -- two DIFFERENT categories, so this one succeeds.
    SYNTH_TEST_CHECK(resolve("  male ,  young adult  ", false, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "male, young adult");
    // Full-width comma with extra spacing on one side, both items already
    // Chinese, order preserved (age before gender, as given).
    SYNTH_TEST_CHECK(resolve("青年，  男", true, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct == "青年，男");
    return 0;
}

int check_all_separators_resolves_empty() {
    // A description that is non-empty (voice-profile.cpp's own dispatcher
    // check runs on the RAW bytes, upstream of this function) but validates
    // to zero attribute items: upstream's own `_resolve_instruct` does not
    // raise here either -- the comma-split's post-filter drops every empty
    // piece, leaving nothing to unify or conflict-check, and the join over
    // zero items is "" (omnivoice.py:1544-1545, 1621). Distinct from
    // upstream's `None` return for a WHITESPACE-only instruct (an earlier,
    // separate early-return, omnivoice.py:1536-1541) only at the Python
    // type level -- both render as the same "None" style_text literal in
    // this family's grammar (frontend-host.cpp's style_text), so this port
    // represents both as an empty DesignInstruct::instruct rather than
    // carrying a second null state with no observable difference.
    std::string instruct, code, message;
    SYNTH_TEST_CHECK(resolve(",,,", false, instruct, code, message) == SYNTH_OK);
    SYNTH_TEST_CHECK(instruct.empty());
    return 0;
}

// ---------------------------------------------------------------------------
// Part 2: the public dispatch, against a synthetic package.
// ---------------------------------------------------------------------------

struct SeenDiagnostic {
    std::string code;
    std::string message;
    bool        seen = false;
};

void SYNTH_CALL record_diagnostic(void * user_data, const synth_diagnostic_t * diagnostic) {
    auto * seen = static_cast<SeenDiagnostic *>(user_data);
    seen->code.assign(diagnostic->code, static_cast<size_t>(diagnostic->code_size));
    seen->message.assign(diagnostic->message, static_cast<size_t>(diagnostic->message_size));
    seen->seen = true;
}

synth_diagnostic_sink_t make_sink(SeenDiagnostic & target) {
    synth_diagnostic_sink_t sink;
    synth_diagnostic_sink_init(&sink, sizeof(sink));
    sink.emit      = record_diagnostic;
    sink.user_data = &target;
    return sink;
}

synth_voice_description_params_t make_params(const char *                    description,
                                             const char *                    language_tag,
                                             uint64_t                        seed,
                                             const synth_diagnostic_sink_t * diagnostics) {
    synth_voice_description_params_t params;
    synth_voice_description_params_init(&params, sizeof(params));
    if (description != nullptr) {
        params.description      = description;
        params.description_size = std::strlen(description);
    }
    if (language_tag != nullptr) {
        params.language_tag      = language_tag;
        params.language_tag_size = std::strlen(language_tag);
    }
    params.seed        = seed;
    params.diagnostics = diagnostics;
    return params;
}

int check_dispatch(const char * scratch_dir) {
    const std::string package_path = std::string(scratch_dir) + "/synthetic-design.gguf";
    SYNTH_TEST_CHECK(synth::omnivoice::testing::write_synthetic_package(package_path, {}));

    synth_model_load_params_t load_params;
    synth_model_load_params_init(&load_params, sizeof(load_params));
    load_params.backend   = SYNTH_BACKEND_CPU;
    synth_model_t * model = nullptr;
    SYNTH_TEST_CHECK(synth_model_load(package_path.c_str(), &load_params, &model) == SYNTH_OK);

    // Capabilities: DESCRIPTION_TEXT is claimed unconditionally per family
    // (src/synthesize.cpp's shared_info), so a synthetic package sees the
    // same flags a real one does.
    synth_voice_profile_capabilities_t capabilities;
    synth_voice_profile_capabilities_init(&capabilities, sizeof(capabilities));
    SYNTH_TEST_CHECK(synth_model_get_voice_profile_capabilities(model, &capabilities) == SYNTH_OK);
    SYNTH_TEST_CHECK((capabilities.source_flags & SYNTH_PROFILE_SOURCE_DESCRIPTION_TEXT) != 0);
    SYNTH_TEST_CHECK(capabilities.description_language == SYNTH_REQUIREMENT_OPTIONAL);

    // --- Missing description -> INVALID_ARG, named diagnostic.
    {
        SeenDiagnostic                         diagnostic;
        const synth_diagnostic_sink_t          sink    = make_sink(diagnostic);
        const synth_voice_description_params_t params  = make_params(nullptr, "en", 0, &sink);
        synth_voice_profile_t *                profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(model, &params, &profile) ==
                         SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
        SYNTH_TEST_CHECK(diagnostic.seen);
        SYNTH_TEST_CHECK(diagnostic.code == "voice_profile.description_required");
    }

    // --- Empty (zero-length, non-null) description -> the same refusal.
    {
        const synth_voice_description_params_t params  = make_params("", "en", 0, nullptr);
        synth_voice_profile_t *                profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(model, &params, &profile) ==
                         SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
    }

    // --- A random (non-concrete) preparation seed -> INVALID_ARG, named
    // diagnostic (docs/voice-conditioning.md:57, and docs/c-interface.md's
    // v1 Description Text Profile Preparation section).
    {
        SeenDiagnostic                         diagnostic;
        const synth_diagnostic_sink_t          sink    = make_sink(diagnostic);
        const synth_voice_description_params_t params  = make_params("male", "en", SYNTH_SEED_RANDOM, &sink);
        synth_voice_profile_t *                profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(model, &params, &profile) ==
                         SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
        SYNTH_TEST_CHECK(diagnostic.seen);
        SYNTH_TEST_CHECK(diagnostic.code == "voice_profile.seed_must_be_concrete");
    }

    // --- A malformed language tag shape -> INVALID_ARG.
    {
        const synth_voice_description_params_t params  = make_params("male", "!!", 0, nullptr);
        synth_voice_profile_t *                profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(model, &params, &profile) ==
                         SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
    }

    // --- A well-formed but unsupported description language ("ja" is a
    // real BCP-47 tag but not one of this package's en/zh description
    // languages) -> SYNTH_ERR_UNSUPPORTED_INPUT, not UNSUPPORTED_LANGUAGE
    // (docs/c-interface.md's v1 Description Text Profile Preparation
    // section: validated against description-language
    // support, not the synthesis Language Capability Catalog).
    {
        SeenDiagnostic                         diagnostic;
        const synth_diagnostic_sink_t          sink    = make_sink(diagnostic);
        const synth_voice_description_params_t params  = make_params("male", "ja", 0, &sink);
        synth_voice_profile_t *                profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(model, &params, &profile) ==
                         SYNTH_ERR_UNSUPPORTED_INPUT);
        SYNTH_TEST_CHECK(profile == nullptr);
        SYNTH_TEST_CHECK(diagnostic.seen);
        SYNTH_TEST_CHECK(diagnostic.code == "voice_profile.description_language_unsupported");
    }

    // --- An unknown vocabulary item, through the FULL dispatcher rather
    // than resolve_instruct() directly: proves the dynamic diagnostic
    // message survives the C ABI's synth_diagnostic_t round trip.
    {
        SeenDiagnostic                         diagnostic;
        const synth_diagnostic_sink_t          sink    = make_sink(diagnostic);
        const synth_voice_description_params_t params  = make_params("sings well", "en", 0, &sink);
        synth_voice_profile_t *                profile = reinterpret_cast<synth_voice_profile_t *>(uintptr_t(1));
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(model, &params, &profile) ==
                         SYNTH_ERR_INVALID_ARG);
        SYNTH_TEST_CHECK(profile == nullptr);
        SYNTH_TEST_CHECK(diagnostic.seen);
        SYNTH_TEST_CHECK(diagnostic.code == "voice_profile.instruct_unknown_item");
        SYNTH_TEST_CHECK(diagnostic.message.find("'sings well'") != std::string::npos);
    }

    // --- The golden EN instruct, language omitted (null -> package default
    // "en"): succeeds, and the profile's own payload carries the fixed-point
    // canonical string.
    {
        const synth_voice_description_params_t params =
            make_params("female, young adult, high pitch", nullptr, 0, nullptr);
        synth_voice_profile_t * profile = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(model, &params, &profile) == SYNTH_OK);
        SYNTH_TEST_CHECK(profile != nullptr);
        SYNTH_TEST_CHECK(profile->model == model);
        SYNTH_TEST_CHECK(profile->family_tag == synth::ProfileFamilyTag::OmnivoiceDesign);
        const auto * design = static_cast<const synth::omnivoice::DesignInstruct *>(profile->payload.get());
        SYNTH_TEST_CHECK(design != nullptr);
        SYNTH_TEST_CHECK(design->instruct == "female, young adult, high pitch");
        synth_voice_profile_free(profile);
    }

    // --- The golden ZH instruct, language "zh" explicit: succeeds, fixed
    // point again.
    {
        const synth_voice_description_params_t params  = make_params("男，老年，低音调", "zh", 0, nullptr);
        synth_voice_profile_t *                profile = nullptr;
        SYNTH_TEST_CHECK(synth_voice_profile_create_from_description(model, &params, &profile) == SYNTH_OK);
        SYNTH_TEST_CHECK(profile != nullptr);
        const auto * design = static_cast<const synth::omnivoice::DesignInstruct *>(profile->payload.get());
        SYNTH_TEST_CHECK(design->instruct == "男，老年，低音调");
        synth_voice_profile_free(profile);
    }

    synth_model_free(model);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <scratch-dir>\n", argv[0]);
        return 2;
    }
    int failures = 0;
    failures += check_each_category_single_item();
    failures += check_category_conflict();
    failures += check_dialect_accent_mix();
    failures += check_unknown_item();
    failures += check_dialect_forces_zh_join();
    failures += check_golden_instructs_are_fixed_points();
    failures += check_translation_direction();
    failures += check_width_comma_splitting();
    failures += check_all_separators_resolves_empty();
    failures += check_dispatch(argv[1]);
    return failures == 0 ? 0 : 1;
}
