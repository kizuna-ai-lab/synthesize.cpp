from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_converter():
    path = PROJECT_ROOT / "scripts" / "convert-kokoro.py"
    spec = importlib.util.spec_from_file_location("synthesize_convert_kokoro", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load converter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


converter = load_converter()


def make_config() -> dict:
    return {
        "n_token": 178,
        "hidden_dim": 512,
        "style_dim": 128,
        "n_layer": 3,
        "n_mels": 80,
        "max_dur": 50,
        "dim_in": 64,
        "max_conv_dim": 512,
        "text_encoder_kernel_size": 5,
        "multispeaker": True,
        "dropout": 0.2,
        "plbert": {
            "hidden_size": 768,
            "num_attention_heads": 12,
            "intermediate_size": 2048,
            "max_position_embeddings": 512,
            "num_hidden_layers": 12,
            "dropout": 0.1,
        },
        "istftnet": {
            "upsample_rates": [10, 6],
            "upsample_kernel_sizes": [20, 12],
            "gen_istft_n_fft": 20,
            "gen_istft_hop_size": 5,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_initial_channel": 512,
        },
        "vocab": {"a": 43, "b": 44, "ˈ": 156},
    }


class ManifestContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (PROJECT_ROOT / "tests/golden/kokoro/kokoro-v1-0.manifest.json").read_text(
                encoding="utf-8"
            )
        )

    def test_accepts_the_pinned_variant(self) -> None:
        package = converter.validate_manifest(copy.deepcopy(self.manifest))
        self.assertEqual(package["language_tags"], ["en"])
        self.assertTrue(package["stochastic"])
        self.assertEqual(len(package["voices"]["preset_ids"]), converter.EXPECTED_VOICES)

    def test_rejects_identity_drift(self) -> None:
        drifts = {
            "schema": ("schema", "other-schema"),
            "family": ("family", "vits"),
            "variant": ("variant", "kokoro-v9-9"),
        }
        for label, (key, value) in drifts.items():
            with self.subTest(drift=label):
                manifest = copy.deepcopy(self.manifest)
                manifest[key] = value
                with self.assertRaises(converter.ConverterError):
                    converter.validate_manifest(manifest)

    def test_rejects_source_drift(self) -> None:
        for key, value in (
            ("repository", "https://github.com/someone/else"),
            ("revision", "0" * 40),
        ):
            with self.subTest(field=key):
                manifest = copy.deepcopy(self.manifest)
                manifest["source"][key] = value
                with self.assertRaises(converter.ConverterError):
                    converter.validate_manifest(manifest)

    def test_rejects_package_contract_drift(self) -> None:
        cases = {
            "input_kinds": lambda m: m["package_contract"].__setitem__(
                "input_kinds", ["text_utf8"]
            ),
            "frontend_mapping": lambda m: m["package_contract"]["frontend"].__setitem__(
                "phoneme_mapping", "delimited_symbol"
            ),
            "padding_rule": lambda m: m["package_contract"]["frontend"].__setitem__(
                "padding_rule", "interleaved_blank"
            ),
            "language": lambda m: m["package_contract"].__setitem__(
                "language_tags", ["zh-CN"]
            ),
            "not_stochastic": lambda m: m["package_contract"].__setitem__(
                "stochastic", False
            ),
            "sample_rate": lambda m: m["package_contract"]["native_audio"].__setitem__(
                "sample_rate_hz", 22050
            ),
            "stereo": lambda m: m["package_contract"]["native_audio"].__setitem__(
                "channels", 2
            ),
            "voice_mode": lambda m: m["package_contract"]["voices"].__setitem__(
                "mode", "fixed-default"
            ),
            "invented_default": lambda m: m["package_contract"]["voices"].__setitem__(
                "default_id", "af_heart"
            ),
            "voice_count": lambda m: m["package_contract"]["voices"].__setitem__(
                "preset_ids", ["af_heart"]
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(drift=label):
                manifest = copy.deepcopy(self.manifest)
                mutate(manifest)
                with self.assertRaises(converter.ConverterError):
                    converter.validate_manifest(manifest)

    def test_source_artifact_requires_exactly_one_role(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        self.assertEqual(
            converter.source_artifact(manifest, "checkpoint")["role"], "checkpoint"
        )
        manifest["source"]["artifacts"].append(
            copy.deepcopy(converter.source_artifact(manifest, "config"))
        )
        with self.assertRaises(converter.ConverterError):
            converter.source_artifact(manifest, "config")
        with self.assertRaises(converter.ConverterError):
            converter.source_artifact(manifest, "absent-role")


class ConfigContractTest(unittest.TestCase):
    def test_accepts_the_pinned_config(self) -> None:
        config = converter.validate_config(make_config())
        self.assertEqual(config["n_token"], 178)

    def test_rejects_dimension_drift(self) -> None:
        cases = {
            "n_token": lambda c: c.__setitem__("n_token", 179),
            "hidden_dim": lambda c: c.__setitem__("hidden_dim", 256),
            "style_dim": lambda c: c.__setitem__("style_dim", 64),
            "max_dur": lambda c: c.__setitem__("max_dur", 40),
            "single_speaker": lambda c: c.__setitem__("multispeaker", False),
            "plbert_heads": lambda c: c["plbert"].__setitem__("num_attention_heads", 8),
            "plbert_layers": lambda c: c["plbert"].__setitem__("num_hidden_layers", 6),
            "upsample_rates": lambda c: c["istftnet"].__setitem__(
                "upsample_rates", [8, 8]
            ),
            "n_fft": lambda c: c["istftnet"].__setitem__("gen_istft_n_fft", 16),
            "hop": lambda c: c["istftnet"].__setitem__("gen_istft_hop_size", 4),
        }
        for label, mutate in cases.items():
            with self.subTest(drift=label):
                config = make_config()
                mutate(config)
                with self.assertRaises(converter.ConverterError):
                    converter.validate_config(config)

    def test_samples_per_frame_is_derived_not_assumed(self) -> None:
        config = make_config()
        config["istftnet"]["upsample_rates"] = [10, 6]
        config["istftnet"]["gen_istft_hop_size"] = 5
        converter.validate_config(config)
        self.assertEqual(converter.SAMPLES_PER_FRAME, 2 * 10 * 6 * 5)

    def test_missing_sections_are_rejected(self) -> None:
        for key in ("plbert", "istftnet"):
            with self.subTest(section=key):
                config = make_config()
                config[key] = None
                with self.assertRaises(converter.ConverterError):
                    converter.validate_config(config)


class SymbolTableTest(unittest.TestCase):
    def test_densifies_a_sparse_vocabulary(self) -> None:
        config = make_config()
        symbols = converter.build_symbol_table(config)
        self.assertEqual(len(symbols), 178)
        self.assertEqual(symbols[43], "a")
        self.assertEqual(symbols[44], "b")
        self.assertEqual(symbols[156], "ˈ")
        self.assertEqual(symbols[0], "")
        self.assertEqual(sum(1 for value in symbols if value), 3)

    def test_matches_the_real_vocabulary_size(self) -> None:
        config = json.loads(
            (
                PROJECT_ROOT / "reports/porting/kokoro/kokoro-v1-0/intake.json"
            ).read_text(encoding="utf-8")
        )
        entries = config["architecture"]["dims"]["vocab_entries"]
        self.assertEqual(entries, 114)

    def test_rejects_malformed_vocabularies(self) -> None:
        cases = {
            "empty": {},
            "pad_remap": {"a": 0},
            "out_of_range": {"a": 999},
            "multi_scalar": {"ab": 12},
            "duplicate_id": {"a": 43, "b": 43},
        }
        for label, vocab in cases.items():
            with self.subTest(vocab=label):
                config = make_config()
                config["vocab"] = vocab
                with self.assertRaises(converter.ConverterError):
                    converter.build_symbol_table(config)


class WeightNormTest(unittest.TestCase):
    def _padded_state(self, extra: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pad to the expected pair count so shape rules can be tested alone."""
        state = dict(extra)
        pairs = sum(1 for name in extra if name.endswith("weight_g"))
        for index in range(converter.EXPECTED_WEIGHT_NORM_PAIRS - pairs):
            state[f"pad.{index}.weight_g"] = torch.ones(2, 1, 1)
            state[f"pad.{index}.weight_v"] = torch.ones(2, 3, 1)
        return state

    def test_fuses_dimension_zero_and_accounts_sources(self) -> None:
        torch.manual_seed(0)
        v = torch.randn(4, 3, 5)
        g = torch.randn(4, 1, 1).abs() + 0.5
        state = self._padded_state(
            {"block.weight_g": g, "block.weight_v": v, "block.bias": torch.randn(4)}
        )
        logical = converter.normalize_weight_norm(state)
        by_name = {item.name: item for item in logical}
        fused = by_name["block.weight"]
        self.assertEqual(fused.transform, "weight_norm_fusion")
        self.assertEqual(set(fused.source_names), {"block.weight_g", "block.weight_v"})
        torch.testing.assert_close(fused.tensor, torch._weight_norm(v, g, 0))
        self.assertEqual(by_name["block.bias"].transform, "identity")
        accounted = sum(len(item.source_names) for item in logical)
        self.assertEqual(accounted, len(state))

    def test_rejects_unpaired_pairs(self) -> None:
        with self.assertRaises(converter.ConverterError):
            converter.normalize_weight_norm({"block.weight_g": torch.ones(2, 1, 1)})

    def test_rejects_a_wrong_pair_count(self) -> None:
        state = {"block.weight_g": torch.ones(2, 1, 1), "block.weight_v": torch.ones(2, 3, 1)}
        with self.assertRaises(converter.ConverterError):
            converter.normalize_weight_norm(state)

    def test_rejects_fused_and_unfused_collision(self) -> None:
        state = self._padded_state(
            {
                "block.weight_g": torch.ones(2, 1, 1),
                "block.weight_v": torch.ones(2, 3, 1),
                "block.weight": torch.ones(2, 3, 1),
            }
        )
        with self.assertRaises(converter.ConverterError):
            converter.normalize_weight_norm(state)

    def test_rejects_non_dimension_zero_norm(self) -> None:
        state = self._padded_state(
            {"block.weight_g": torch.ones(2, 3, 1), "block.weight_v": torch.ones(2, 3, 1)}
        )
        with self.assertRaises(converter.ConverterError):
            converter.normalize_weight_norm(state)


class SkipAndCountTest(unittest.TestCase):
    def test_only_the_unused_pooler_is_skipped(self) -> None:
        self.assertIsNotNone(converter.skip_reason("bert.pooler.weight"))
        self.assertIsNotNone(converter.skip_reason("bert.pooler.bias"))
        for keep in (
            "bert.embeddings.word_embeddings.weight",
            "bert.encoder.embedding_hidden_mapping_in.weight",
            "predictor.lstm.weight_ih_l0",
            "decoder.generator.conv_post.weight",
            "text_encoder.embedding.weight",
        ):
            self.assertIsNone(converter.skip_reason(keep), keep)

    def test_expected_counts_are_internally_consistent(self) -> None:
        # 548 entries = 89 fused pairs (178 entries) + 370 plain, of which 2 are
        # skipped, so 368 + 89 = 457 emitted model tensors.
        plain = converter.EXPECTED_CHECKPOINT_ENTRIES - 2 * converter.EXPECTED_WEIGHT_NORM_PAIRS
        emitted = plain - converter.EXPECTED_SKIPPED + converter.EXPECTED_WEIGHT_NORM_PAIRS
        self.assertEqual(emitted, converter.EXPECTED_MODEL_TENSORS)

    def test_output_preparation_reports_counts_and_skips(self) -> None:
        logical = [
            converter.LogicalTensor("bert.pooler.weight", ("bert.pooler.weight",), torch.ones(2), "identity"),
            converter.LogicalTensor("bert.pooler.bias", ("bert.pooler.bias",), torch.ones(2), "identity"),
        ]
        logical += [
            converter.LogicalTensor(f"t.{i}", (f"t.{i}",), torch.ones(2), "identity")
            for i in range(converter.EXPECTED_MODEL_TENSORS)
        ]
        outputs, skipped = converter.prepare_output_tensors(logical)
        self.assertEqual(len(outputs), converter.EXPECTED_MODEL_TENSORS)
        self.assertEqual(len(skipped), converter.EXPECTED_SKIPPED)
        self.assertTrue(all(item["reason"] for item in skipped))

    def test_output_preparation_rejects_a_wrong_emitted_count(self) -> None:
        logical = [
            converter.LogicalTensor(f"t.{i}", (f"t.{i}",), torch.ones(2), "identity")
            for i in range(3)
        ]
        with self.assertRaises(converter.ConverterError):
            converter.prepare_output_tensors(logical)

    def test_size_label(self) -> None:
        self.assertEqual(converter.compute_size_label(81_763_410), "82M")
        self.assertEqual(converter.compute_size_label(1_500_000_000), "1.5B")
        self.assertEqual(converter.compute_size_label(5_000), "5K")


class TensorNameTest(unittest.TestCase):
    def test_albert_path_is_shortened(self) -> None:
        source = (
            "bert.encoder.albert_layer_groups.0.albert_layers.0."
            "full_layer_layer_norm.weight"
        )
        self.assertEqual(converter.canonical_name(source), "bert.layer.full_layer_layer_norm.weight")

    def test_other_names_are_untouched(self) -> None:
        for name in (
            "bert.embeddings.word_embeddings.weight",
            "predictor.shared.weight_ih_l0_reverse",
            "decoder.generator.resblocks.5.convs2.2.weight",
            "voice.af_heart",
        ):
            self.assertEqual(converter.canonical_name(name), name)

    def test_rename_is_injective_over_the_albert_group(self) -> None:
        sources = [
            converter.ALBERT_SOURCE_PREFIX + suffix
            for suffix in (
                "attention.query.weight",
                "attention.key.weight",
                "attention.LayerNorm.bias",
                "ffn_output.weight",
                "full_layer_layer_norm.bias",
            )
        ]
        mapped = [converter.canonical_name(name) for name in sources]
        self.assertEqual(len(set(mapped)), len(sources))
        self.assertTrue(all(name.startswith(converter.ALBERT_CANONICAL_PREFIX) for name in mapped))

    def test_over_long_names_are_rejected(self) -> None:
        # GGML truncates at a fixed field width, so an over-long name produces a
        # GGUF that the runtime cannot open. The converter must refuse to write.
        import numpy as np

        long_name = "a" * converter.GGML_MAX_NAME
        outputs = [
            converter.OutputTensor(
                long_name, (long_name,), long_name, np.zeros(1, dtype="float32"), "identity"
            )
        ]
        with self.assertRaises(converter.ConverterError):
            converter.check_name_lengths(outputs)

    def test_names_at_the_limit_are_accepted(self) -> None:
        import numpy as np

        name = "b" * (converter.GGML_MAX_NAME - 1)
        outputs = [converter.OutputTensor(name, (name,), name, np.zeros(1, dtype="float32"), "identity")]
        converter.check_name_lengths(outputs)

    def test_every_upstream_name_fits_after_renaming(self) -> None:
        # The full checkpoint path set, reconstructed from the committed report
        # when it exists, must survive the mapping within the field width.
        report = PROJECT_ROOT / "reports/convert/kokoro/kokoro-v1-0-F32.json"
        if not report.is_file():
            self.skipTest("converter report is a generated artifact")
        payload = json.loads(report.read_text(encoding="utf-8"))
        for tensor in payload["tensors"]:
            self.assertLess(len(tensor["name"]), converter.GGML_MAX_NAME, tensor["name"])


class VoicepackTest(unittest.TestCase):
    def _manifest_with(self, tmp: Path, ids: list[str], digests: dict[str, str]) -> dict:
        return {
            "source": {
                "artifacts": [
                    {
                        "role": "frontend-resource",
                        "locator": f"https://example.invalid/voices/{voice}.pt",
                        "sha256": digests[voice],
                    }
                    for voice in ids
                ]
            }
        }

    def _write_pack(self, path: Path, shape=(510, 1, 256)) -> str:
        torch.save(torch.zeros(*shape), path)
        return converter.sha256_file(path)

    def test_loads_and_squeezes_every_pinned_voicepack(self) -> None:
        ids = [f"v{index:02d}" for index in range(converter.EXPECTED_VOICES)]
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            digests = {voice: self._write_pack(tmp / f"{voice}.pt") for voice in ids}
            manifest = self._manifest_with(tmp, ids, digests)
            outputs, records = converter.load_voicepacks(manifest, tmp, ids)
        self.assertEqual(len(outputs), converter.EXPECTED_VOICES)
        self.assertEqual(len(records), converter.EXPECTED_VOICES)
        self.assertEqual(outputs[0].name, "voice.v00")
        self.assertEqual(list(outputs[0].array.shape), [510, 256])
        self.assertEqual(outputs[0].transform, "squeeze_singleton_batch")

    def test_rejects_a_digest_mismatch(self) -> None:
        ids = [f"v{index:02d}" for index in range(converter.EXPECTED_VOICES)]
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            digests = {voice: self._write_pack(tmp / f"{voice}.pt") for voice in ids}
            digests[ids[0]] = "0" * 64
            manifest = self._manifest_with(tmp, ids, digests)
            with self.assertRaises(converter.ConverterError):
                converter.load_voicepacks(manifest, tmp, ids)

    def test_rejects_a_wrong_pack_shape(self) -> None:
        ids = [f"v{index:02d}" for index in range(converter.EXPECTED_VOICES)]
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            digests = {}
            for index, voice in enumerate(ids):
                shape = (510, 1, 128) if index == 0 else (510, 1, 256)
                digests[voice] = self._write_pack(tmp / f"{voice}.pt", shape)
            manifest = self._manifest_with(tmp, ids, digests)
            with self.assertRaises(converter.ConverterError):
                converter.load_voicepacks(manifest, tmp, ids)

    def test_rejects_a_missing_file(self) -> None:
        ids = [f"v{index:02d}" for index in range(converter.EXPECTED_VOICES)]
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            digests = {voice: self._write_pack(tmp / f"{voice}.pt") for voice in ids}
            manifest = self._manifest_with(tmp, ids, digests)
            (tmp / f"{ids[-1]}.pt").unlink()
            with self.assertRaises(converter.ConverterError):
                converter.load_voicepacks(manifest, tmp, ids)

    def test_rejects_an_incomplete_manifest_pin(self) -> None:
        ids = [f"v{index:02d}" for index in range(converter.EXPECTED_VOICES)]
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            digests = {voice: self._write_pack(tmp / f"{voice}.pt") for voice in ids}
            manifest = self._manifest_with(tmp, ids[:-1], digests)
            with self.assertRaises(converter.ConverterError):
                converter.load_voicepacks(manifest, tmp, ids)


class MetadataContractTest(unittest.TestCase):
    def test_manifest_declares_every_voice_the_converter_will_emit(self) -> None:
        manifest = json.loads(
            (PROJECT_ROOT / "tests/golden/kokoro/kokoro-v1-0.manifest.json").read_text(
                encoding="utf-8"
            )
        )
        preset_ids = manifest["package_contract"]["voices"]["preset_ids"]
        pinned = {
            artifact["locator"].rsplit("/", 1)[-1][: -len(".pt")]
            for artifact in manifest["source"]["artifacts"]
            if artifact["role"] == "frontend-resource"
        }
        self.assertEqual(set(preset_ids), pinned)
        self.assertEqual(len(preset_ids), converter.EXPECTED_VOICES)


if __name__ == "__main__":
    unittest.main()
