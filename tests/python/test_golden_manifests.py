"""Every committed Golden Manifest must satisfy the committed schema and the
loader rules stated in docs/port-validation.md.

These are family-independent contract checks: they run over whatever manifests
exist under tests/golden/ so a new Model Family inherits them automatically.
"""

import hashlib
import json
import pathlib
import unittest

import jsonschema

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "docs/schemas/synthesize-golden-manifest-v1.schema.json"
GOLDEN_ROOT = REPO_ROOT / "tests/golden"
UINT64_MAX = 2**64 - 1


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def manifest_paths():
    return sorted(GOLDEN_ROOT.glob("*/*.manifest.json"))


class GoldenManifestSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = load_json(SCHEMA_PATH)
        cls.manifests = [(p, load_json(p)) for p in manifest_paths()]

    def test_schema_is_itself_valid(self):
        jsonschema.Draft202012Validator.check_schema(self.schema)

    def test_at_least_one_manifest_is_committed(self):
        self.assertTrue(self.manifests, f"no manifests found under {GOLDEN_ROOT}")

    def test_manifests_conform_to_schema(self):
        validator = jsonschema.Draft202012Validator(self.schema)
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                errors = sorted(validator.iter_errors(manifest), key=lambda e: list(e.path))
                detail = "; ".join(
                    f"{list(e.path)}: {e.message}" for e in errors[:5]
                )
                self.assertEqual([], errors, f"{path.relative_to(REPO_ROOT)}: {detail}")

    def test_family_and_variant_match_path(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                self.assertEqual(path.parent.name, manifest["family"])
                self.assertEqual(path.name, f"{manifest['variant']}.manifest.json")

    def test_case_ids_are_unique(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                ids = [case["id"] for case in manifest["cases"]]
                self.assertEqual(sorted(set(ids)), sorted(ids), "duplicate case ids")

    def test_relations_reference_known_cases(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                known = {case["id"] for case in manifest["cases"]}
                for relation in manifest["relations"]:
                    unknown = set(relation["cases"]) - known
                    self.assertEqual(set(), unknown, f"unknown relation targets: {unknown}")

    def test_seeds_are_representable_u64(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                for case in manifest["cases"]:
                    seed = int(case["request"]["seed_u64"])
                    self.assertGreaterEqual(seed, 0)
                    self.assertLessEqual(seed, UINT64_MAX, f"{case['id']} seed exceeds uint64")

    def test_speaking_rate_range_is_ordered_and_contains_default(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                low, high = manifest["package_contract"]["speaking_rate_range"]
                self.assertLessEqual(low, high, "minimum speaking rate exceeds maximum")
                self.assertLessEqual(low, 1.0)
                self.assertLessEqual(1.0, high)

    def test_case_speaking_rates_are_inside_the_declared_range(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                low, high = manifest["package_contract"]["speaking_rate_range"]
                for case in manifest["cases"]:
                    rate = case["request"]["speaking_rate"]
                    self.assertTrue(
                        low <= rate <= high,
                        f"{case['id']} speaking_rate {rate} outside [{low}, {high}]",
                    )

    def test_single_valued_source_roles_are_not_duplicated(self):
        """A role may repeat, but never with the same locator.

        This asserted at most one `checkpoint` and one `config` until
        2026-07-27, when qwen3-tts arrived with two genuine checkpoints -- the
        talker and the speech tokenizer are separate files with separate
        digests, and pinning only one would leave the other unpinned in the
        contract that exists to pin them.

        The guard the old form actually provided was against copy-paste: the
        same artifact listed twice, or a second one whose locator was never
        updated. That is what is checked now, and it applies to every role
        rather than only to two, so a duplicated frontend resource is caught as
        well.
        """
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                locators = [artifact["locator"] for artifact in manifest["source"]["artifacts"]]
                duplicates = {loc for loc in locators if locators.count(loc) > 1}
                self.assertFalse(
                    duplicates, f"source artifacts repeat a locator: {sorted(duplicates)}"
                )

    def test_schema_refuses_a_reference_count_the_loader_would_reject(self):
        """The schema may not permit a manifest that converts into an unloadable package.

        Both loaders' read_profile_contract have required
        `max_reference_count == 1` since 2026-08-13, because every family that
        declares a `profile.reference` block reads `references[0]` and nothing
        else. The schema said `minimum: 1` until the same date, so a manifest
        declaring 2 was schema-valid, converted without complaint, and produced
        a GGUF that failed at load -- the documented contract and the enforced
        rule disagreeing, which this project treats as a defect in the
        contract.

        Asserted against a real committed manifest with only that one field
        mutated, so nothing else in the schema can be what rejects it: the
        unmutated copy is asserted valid first, in the same test. Relaxing the
        schema back to `minimum: 1` makes the second half of this fail.
        """
        validator = jsonschema.Draft202012Validator(self.schema)
        checked = 0
        for path, manifest in self.manifests:
            reference = manifest.get("package_contract", {}).get("profile", {}).get("reference")
            if reference is None:
                continue
            checked += 1
            with self.subTest(manifest=path.name):
                self.assertEqual(
                    [], sorted(validator.iter_errors(manifest), key=lambda e: list(e.path))
                )
                mutated = json.loads(json.dumps(manifest))
                mutated["package_contract"]["profile"]["reference"]["max_reference_count"] = 2
                self.assertTrue(
                    list(validator.iter_errors(mutated)),
                    f"{path.relative_to(REPO_ROOT)}: the schema accepts max_reference_count 2, "
                    "which read_profile_contract refuses at load",
                )
        self.assertTrue(checked, "no manifest declares a profile.reference block to check")

    def test_tolerance_file_is_committed(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                tolerance = REPO_ROOT / manifest["tolerance_file"]
                self.assertTrue(tolerance.is_file(), f"missing {manifest['tolerance_file']}")

    def test_tolerance_case_count_matches_manifest(self):
        """A tolerance file describing N cases must mean the manifest's N.

        The qwen3-tts file said 18 while its manifest had grown to 20 -- an
        honest historical number that read as a current claim. case_count is
        bookkeeping about the suite, so it tracks the suite.

        A file whose `tolerance_file` is shared by more than one Reference
        Model Variant (VITS, and qwen3-tts since 2026-08-11) cannot carry one
        flat top-level number that is right for both -- it records
        `variants.<name>.case_count` instead, and this checks that form too.
        Checking only the flat key left both VITS variants silently
        unchecked from the day that shape was introduced: neither vits.json
        variant carries a top-level `case_count` at all, so the old
        `"case_count" not in tolerance` guard skipped every VITS manifest
        every time, and nothing failed when this test's own module was
        exercised by making a `variants.*.case_count` wrong on purpose.

        A manifest whose variant is absent from a per-variant file, or present
        without a `case_count`, is a FAILURE rather than a skip. That was the
        same skip-shaped hole one level down: the branch that introduced this
        form fixed "the flat key is missing" and left "this variant is
        missing" silently passing, which is exactly how a third variant added
        to a shared file would arrive unchecked. All four committed
        per-variant entries carry the key today, so nothing legitimately
        needs the escape.
        """
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                if not manifest["cases"]:
                    # Same incrementally-built-suite exemption as
                    # test_upstream_examples_are_present: a manifest with no
                    # cases yet has no case_count for any tolerance file --
                    # shared or not -- to agree with. It does not need its own
                    # entry in a shared per-variant file before it has a case
                    # for that file to describe.
                    continue
                tolerance_path = REPO_ROOT / manifest["tolerance_file"]
                tolerance = json.loads(tolerance_path.read_text(encoding="utf-8"))
                if "variants" in tolerance:
                    variant = manifest["variant"]
                    entry = tolerance["variants"].get(variant)
                    self.assertIsNotNone(
                        entry,
                        f"{manifest['tolerance_file']}: no variants.{variant} entry for {path.name}; "
                        f"a per-variant tolerance file must describe every manifest that points at it")
                    self.assertIn(
                        "case_count", entry,
                        f"{manifest['tolerance_file']}: variants.{variant} carries no case_count "
                        f"for {path.name}")
                    self.assertEqual(
                        entry["case_count"], len(manifest["cases"]),
                        f"{manifest['tolerance_file']}: variants.{variant}.case_count "
                        f"disagrees with {path.name}")
                    continue
                # A flat file describes exactly one variant, so its
                # `case_count` is that manifest's. Kept optional only because
                # a flat file that never carried the key is a different,
                # older shape than a per-variant file missing an entry.
                if "case_count" not in tolerance:
                    continue
                self.assertEqual(
                    tolerance["case_count"], len(manifest["cases"]),
                    f"{manifest['tolerance_file']}: case_count disagrees with {path.name}")

    def test_environment_lock_is_committed(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                lock = REPO_ROOT / manifest["reference"]["environment_lock"]
                self.assertTrue(lock.is_file(), f"missing {manifest['reference']['environment_lock']}")

    def test_runner_is_committed(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                runner = REPO_ROOT / manifest["reference"]["runner"]
                self.assertTrue(runner.is_file(), f"missing {manifest['reference']['runner']}")

    def test_preset_voice_cases_name_catalog_entries(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                catalog = set(manifest["package_contract"]["voices"]["preset_ids"])
                for case in manifest["cases"]:
                    voice = case["voice"]
                    if voice["kind"] == "preset_voice":
                        self.assertIn(
                            voice["id"], catalog, f"{case['id']} selects an uncatalogued voice"
                        )

    def test_reference_audio_cases_name_a_pinned_source_artifact(self):
        """A clone case's reference must be one of the manifest's pinned artifacts.

        `input.reference.artifact` is a locator rather than a path, and a
        locator that matches nothing in `source.artifacts` is a reference whose
        bytes are not pinned by anything -- the case would silently compare
        against whatever the URL served that day. This is the same guard
        `test_preset_voice_cases_name_catalog_entries` provides for Voices: the
        case may only select from what the manifest declares.
        """
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                pinned = {
                    artifact["locator"]
                    for artifact in manifest["source"]["artifacts"]
                    if artifact["role"] == "reference-audio"
                }
                for case in manifest["cases"]:
                    reference = case["input"].get("reference")
                    if reference is None:
                        continue
                    self.assertIn(
                        reference["artifact"],
                        pinned,
                        f"{case['id']} references audio that no reference-audio "
                        f"source artifact pins",
                    )

    def test_case_language_tags_are_declared(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                declared = set(manifest["package_contract"]["language_tags"])
                for case in manifest["cases"]:
                    tag = case["input"]["language_tag"]
                    if tag is not None:
                        self.assertIn(tag, declared, f"{case['id']} uses undeclared {tag}")

    def test_case_input_kinds_are_declared(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                declared = set(manifest["package_contract"]["input_kinds"])
                for case in manifest["cases"]:
                    self.assertIn(case["input"]["kind"], declared, case["id"])

    def test_upstream_examples_are_present(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                if not manifest["cases"]:
                    # An incrementally-built suite -- see the schema's "cases"
                    # description -- has no first case yet to check. This is
                    # not a skip-shaped hole: a manifest with 1+ cases still
                    # runs the assertion below, so the moment a case is added
                    # without an upstream_example anywhere in the suite, this
                    # starts failing for it.
                    continue
                origins = {case["origin"]["kind"] for case in manifest["cases"]}
                self.assertIn(
                    "upstream_example",
                    origins,
                    "a suite must begin from at least one upstream-authored example",
                )

    def test_stochastic_packages_declare_replay_inputs(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                if not manifest["package_contract"]["stochastic"]:
                    continue
                for case in manifest["cases"]:
                    if "tensor_parity" not in case["checks"]:
                        continue
                    self.assertTrue(
                        case["oracle"]["stochastic_inputs"],
                        f"{case['id']} replays tensors but declares no stochastic inputs",
                    )

    def test_alternate_grids_are_committed_and_match_their_digests(self):
        """An admissible-grid witness is a contract only while its bytes are pinned.

        `oracle.alternate_grids` widens what a case will accept as exact, so the
        one thing that must not be possible is a witness whose content drifted
        from the digest the manifest records -- that would silently admit an
        output nobody enumerated. The file is committed (unlike the dumped
        oracle payloads, which are deliberately absent from git), so this check
        can run in the unit gate rather than only where models exist.
        """
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                for case in manifest["cases"]:
                    for entry in case["oracle"].get("alternate_grids", []):
                        witness = path.parent / entry["file"]
                        self.assertTrue(
                            witness.is_file(),
                            f"{case['id']}: alternate grid {entry['file']} is not committed",
                        )
                        digest = hashlib.sha256(witness.read_bytes()).hexdigest()
                        self.assertEqual(
                            entry["sha256"],
                            digest,
                            f"{case['id']}: {entry['file']} has sha256 {digest}",
                        )

    def test_omnivoice_primary_grids_are_digest_pinned(self):
        """Every omnivoice case pins its PRIMARY grid's sha256, not only fast-mode's alternate.

        The primary grid is never committed -- it lives in the oracle dump
        under `case_artifact_root`, which stays out of the tree -- but its
        digest is knowable ahead of time, and recording it turns "the oracle
        dump on this machine is the one the manifest describes" into
        something scripts/validate-omnivoice-replay.py checks (before
        comparing anything) rather than assumes. Scoped to the omnivoice
        manifest: `stochasticInput.sha256` is optional in the shared schema
        because vits/kokoro/qwen3-tts use the same type for their own
        stochastic inputs and do not (yet) pin one.
        """
        for path, manifest in self.manifests:
            if manifest["family"] != "omnivoice":
                continue
            with self.subTest(manifest=path.name):
                for case in manifest["cases"]:
                    grid_entries = [
                        entry for entry in case["oracle"]["stochastic_inputs"]
                        if entry["name"] == "codes.grid"
                    ]
                    self.assertEqual(
                        1, len(grid_entries),
                        f"{case['id']}: expected exactly one codes.grid stochastic input",
                    )
                    self.assertIn(
                        "sha256", grid_entries[0],
                        f"{case['id']}: codes.grid stochastic input has no pinned sha256",
                    )

    def test_generated_artifact_paths_stay_inside_the_case_root(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                for case in manifest["cases"]:
                    artifacts = list(case["expected"]["artifacts"])
                    artifacts += list(case["oracle"]["stochastic_inputs"])
                    if "artifact" in case["input"]:
                        artifacts.append(case["input"]["artifact"])
                    for artifact in artifacts:
                        relative = pathlib.PurePosixPath(artifact["path"])
                        self.assertFalse(relative.is_absolute(), artifact["path"])
                        self.assertNotIn("..", relative.parts, artifact["path"])


if __name__ == "__main__":
    unittest.main()
