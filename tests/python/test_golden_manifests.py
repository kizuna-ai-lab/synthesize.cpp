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

    def test_tolerance_file_is_committed(self):
        for path, manifest in self.manifests:
            with self.subTest(manifest=path.name):
                tolerance = REPO_ROOT / manifest["tolerance_file"]
                self.assertTrue(tolerance.is_file(), f"missing {manifest['tolerance_file']}")

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
