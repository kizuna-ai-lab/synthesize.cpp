from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_converter():
    path = PROJECT_ROOT / "scripts" / "convert-vits.py"
    spec = importlib.util.spec_from_file_location("synthesize_convert_vits", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load converter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


converter = load_converter()


def make_vctk_manifest(ljs_manifest: dict) -> dict:
    manifest = copy.deepcopy(ljs_manifest)
    manifest["variant"] = "vits-vctk"
    checkpoint = next(
        item for item in manifest["source"]["artifacts"] if item["role"] == "checkpoint"
    )
    checkpoint["locator"] = (
        "https://drive.google.com/uc?id=11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru"
    )
    checkpoint["sha256"] = (
        "ab981615c443d935fc3a89b08137df544a1175bad99bcbbc9f59e7c3d4930043"
    )
    config = next(
        item for item in manifest["source"]["artifacts"] if item["role"] == "config"
    )
    config["repository_path"] = "configs/vctk_base.json"
    config["sha256"] = (
        "85d00f84166ee9d078419e3c3a061341b172a27e3c1f57669d2290fac77dd253"
    )
    voices = manifest["package_contract"]["voices"]
    voices["mode"] = "preset-catalog"
    voices["default_id"] = None
    voices["preset_ids"] = [f"speaker-{index:03d}" for index in range(109)]
    return manifest


class ConverterContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (PROJECT_ROOT / "tests/golden/vits/vits-ljspeech.manifest.json").read_text(
                encoding="utf-8"
            )
        )

    def test_manifest_contract_accepts_pinned_variant(self) -> None:
        package = converter.validate_manifest(copy.deepcopy(self.manifest))
        self.assertEqual(package["input_kinds"], ["phonemes_utf8", "token_ids"])
        self.assertEqual(
            package["frontend"],
            {
                "provider": "synthesize.symbol_map",
                "contract_version": 1,
                "phoneme_mapping": "unicode_scalar",
            },
        )
        self.assertEqual(package["native_audio"]["sample_rate_hz"], 22050)

    def test_manifest_contract_accepts_vctk_preset_catalog(self) -> None:
        package = converter.validate_manifest(make_vctk_manifest(self.manifest))
        self.assertEqual(package["voices"]["mode"], "preset-catalog")
        self.assertEqual(package["voices"]["preset_ids"][0], "speaker-000")
        self.assertEqual(package["voices"]["preset_ids"][-1], "speaker-108")

    def test_vctk_metadata_emits_complete_preset_catalog(self) -> None:
        class RecordingWriter:
            def __init__(self) -> None:
                self.values = {}

            def __getattr__(self, name):
                if not name.startswith("add_"):
                    raise AttributeError(name)

                def record(key, *values):
                    if isinstance(key, str) and key.startswith("synthesize.") and values:
                        self.values[key] = values[0]

                return record

        manifest = make_vctk_manifest(self.manifest)
        package = converter.validate_manifest(manifest)
        writer = RecordingWriter()
        expected = converter.add_package_metadata(
            writer,
            manifest,
            package,
            "checkpoint-sha256",
            "config-sha256",
            {"source_parameter_count": 39_700_208},
        )
        self.assertEqual(expected["general.name"], "VITS VCTK")
        self.assertEqual(
            expected["synthesize.capabilities.input_flags"],
            converter.INPUT_PHONEMES_UTF8 | converter.INPUT_TOKEN_IDS,
        )
        self.assertTrue(expected["synthesize.frontend.present"])
        self.assertEqual(
            writer.values["synthesize.frontend.provider"],
            "synthesize.symbol_map",
        )
        self.assertFalse(expected["synthesize.voice.has_package_default"])
        self.assertEqual(expected["synthesize.voice.preset_count"], 109)
        self.assertEqual(writer.values["synthesize.voice.0.id"], "speaker-000")
        self.assertEqual(writer.values["synthesize.voice.108.id"], "speaker-108")
        self.assertEqual(writer.values["synthesize.voice.108.flags"], 0)

    def test_manifest_contract_rejects_identity_and_type_drift(self) -> None:
        mutations = (
            lambda value: value.__setitem__("schema", "future-schema"),
            lambda value: value.__setitem__("family", "other"),
            lambda value: value.__setitem__("variant", "vits-unknown"),
            lambda value: value["source"].__setitem__("revision", "mutable-main"),
            lambda value: value["reference"].__setitem__("dtype", "float16"),
            lambda value: value["package_contract"].__setitem__(
                "max_input_tokens", True
            ),
            lambda value: value["package_contract"].__setitem__(
                "speaking_rate_range", [0.5, 2.0]
            ),
            lambda value: value["package_contract"]["frontend"].__setitem__(
                "provider", "unknown"
            ),
        )
        for mutate in mutations:
            manifest = copy.deepcopy(self.manifest)
            mutate(manifest)
            with (
                self.subTest(mutate=mutate),
                self.assertRaises(converter.ConverterError),
            ):
                converter.validate_manifest(manifest)

    def test_source_artifact_requires_exactly_one_role(self) -> None:
        checkpoint = converter.source_artifact(self.manifest, "checkpoint")
        self.assertEqual(checkpoint["role"], "checkpoint")
        with self.assertRaises(converter.ConverterError):
            converter.source_artifact(self.manifest, "missing")
        duplicated = copy.deepcopy(self.manifest)
        duplicated["source"]["artifacts"].append(copy.deepcopy(checkpoint))
        with self.assertRaises(converter.ConverterError):
            converter.source_artifact(duplicated, "checkpoint")

    def test_config_validation_derives_checked_hparams(self) -> None:
        package = converter.validate_manifest(copy.deepcopy(self.manifest))
        config = {
            "data": {
                "n_speakers": 0,
                "sampling_rate": 22050,
                "hop_length": 4,
                "add_blank": True,
            },
            "model": {
                "gin_channels": 0,
                "use_sdp": True,
                "resblock": "1",
                "inter_channels": 2,
                "hidden_channels": 4,
                "filter_channels": 8,
                "n_heads": 2,
                "n_layers": 2,
                "kernel_size": 3,
                "upsample_initial_channel": 8,
                "resblock_kernel_sizes": [3],
                "resblock_dilation_sizes": [[1, 3, 5]],
                "upsample_rates": [2, 2],
                "upsample_kernel_sizes": [4, 4],
            },
            "train": {"segment_size": 8},
        }
        hparams = converter.validate_config(config, ["a", "b", "c"], package)
        self.assertEqual(hparams["vocab_size"], 3)
        self.assertEqual(hparams["hop_length"], 4)
        self.assertEqual(hparams["text_attention_window"], 4)

        invalid = copy.deepcopy(config)
        invalid["model"]["n_heads"] = True
        with self.assertRaises(converter.ConverterError):
            converter.validate_config(invalid, ["a"], package)

    def test_config_validation_accepts_vctk_global_conditioning(self) -> None:
        package = converter.validate_manifest(make_vctk_manifest(self.manifest))
        config = json.loads(
            (PROJECT_ROOT / "models/upstream/vits-source/configs/vctk_base.json").read_text(
                encoding="utf-8"
            )
        )
        hparams = converter.validate_config(config, ["a", "b", "c"], package)
        self.assertEqual(hparams["speaker_count"], 109)
        self.assertEqual(hparams["gin_channels"], 256)
        invalid = copy.deepcopy(config)
        invalid["model"]["upsample_rates"] = [2, 3]
        with self.assertRaises(converter.ConverterError):
            converter.validate_config(invalid, ["a"], package)

    def test_weight_norm_fuses_dimension_zero_and_accounts_sources(self) -> None:
        g = torch.tensor([[[2.0]], [[3.0]]])
        v = torch.arange(1, 13, dtype=torch.float32).reshape(2, 2, 3)
        state = {
            "layer.bias": torch.tensor([1.0, 2.0]),
            "layer.weight_g": g,
            "layer.weight_v": v,
        }
        logical = converter.normalize_weight_norm(state)
        self.assertEqual(
            [item.name for item in logical], ["layer.bias", "layer.weight"]
        )
        fused = logical[1]
        self.assertEqual(fused.source_names, ("layer.weight_g", "layer.weight_v"))
        self.assertEqual(fused.transform, "weight_norm_fusion")
        torch.testing.assert_close(fused.tensor, torch._weight_norm(v, g, 0))

    def test_weight_norm_rejects_unpaired_colliding_and_wrong_shapes(self) -> None:
        with self.assertRaises(converter.ConverterError):
            converter.normalize_weight_norm({"x.weight_g": torch.ones(1, 1)})
        with self.assertRaises(converter.ConverterError):
            converter.normalize_weight_norm(
                {
                    "x.weight": torch.ones(1, 1),
                    "x.weight_g": torch.ones(1, 1),
                    "x.weight_v": torch.ones(1, 1),
                }
            )
        with self.assertRaises(converter.ConverterError):
            converter.normalize_weight_norm(
                {
                    "x.weight_g": torch.ones(2, 2),
                    "x.weight_v": torch.ones(2, 2),
                }
            )

    def test_tensor_mapping_covers_each_runtime_namespace(self) -> None:
        cases = {
            "enc_p.emb.weight": "text_encoder.token_embedding.weight",
            "enc_p.encoder.attn_layers.2.conv_q.bias": "text_encoder.blocks.2.attention.query.bias",
            "enc_p.encoder.attn_layers.1.emb_rel_v": "text_encoder.blocks.1.attention.relative_value.weight",
            "enc_p.encoder.norm_layers_2.3.gamma": "text_encoder.blocks.3.ffn_norm.weight",
            "enc_p.encoder.ffn_layers.4.conv_1.weight": "text_encoder.blocks.4.ffn.input.weight",
            "dp.pre.weight": "duration_predictor.pre.weight",
            "dp.convs.norms_2.1.beta": "duration_predictor.dds.blocks.1.pointwise_norm.bias",
            "dp.flows.5.pre.bias": "duration_predictor.flows.1.pre.bias",
            "flow.flows.6.post.weight": "flow.blocks.3.projection.weight",
            "flow.flows.2.enc.in_layers.1.bias": "flow.blocks.1.wn.layers.1.input.bias",
            "dec.conv_pre.weight": "decoder.pre.weight",
            "dec.ups.2.bias": "decoder.upsample.2.transpose_conv.bias",
            "dec.resblocks.5.convs2.1.weight": "decoder.upsample.1.resblocks.2.conv2.1.weight",
        }
        hparams = {"resblock_kernel_sizes": [3, 7, 11]}
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(
                    converter.map_runtime_tensor(source, hparams), expected
                )
        with self.assertRaises(converter.ConverterError):
            converter.map_runtime_tensor("unknown.weight", hparams)

    def test_tensor_mapping_covers_vctk_conditioning(self) -> None:
        cases = {
            "emb_g.weight": "voice.embedding.weight",
            "dp.cond.weight": "duration_predictor.conditioning.weight",
            "dp.cond.bias": "duration_predictor.conditioning.bias",
            "flow.flows.4.enc.cond_layer.weight": "flow.blocks.2.conditioning.weight",
            "flow.flows.4.enc.cond_layer.bias": "flow.blocks.2.conditioning.bias",
            "dec.cond.weight": "decoder.conditioning.weight",
            "dec.cond.bias": "decoder.conditioning.bias",
        }
        hparams = {"resblock_kernel_sizes": [3, 7, 11]}
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(
                    converter.map_runtime_tensor(source, hparams), expected
                )

    def test_skip_rules_and_size_labels(self) -> None:
        self.assertEqual(
            converter.skip_reason("enc_q.pre.weight"), "training-only posterior encoder"
        )
        self.assertEqual(
            converter.skip_reason("dp.post_proj.weight"),
            "training-only stochastic-duration posterior",
        )
        self.assertEqual(
            converter.skip_reason("dp.flows.1.pre.weight"),
            "unused first ConvFlow removed by upstream reverse inference",
        )
        self.assertIsNone(converter.skip_reason("enc_p.emb.weight"))
        self.assertEqual(converter.compute_size_label(999_999), "1000K")
        self.assertEqual(converter.compute_size_label(36_321_072), "36M")
        self.assertEqual(converter.compute_size_label(1_250_000_000), "1.2B")

    def test_output_preparation_converts_and_reports_skips(self) -> None:
        logical = [
            converter.LogicalTensor(
                "enc_p.emb.weight",
                ("enc_p.emb.weight",),
                torch.tensor([[1.0, 2.0]]),
                "identity",
            ),
            converter.LogicalTensor(
                "enc_q.pre.bias",
                ("enc_q.pre.bias",),
                torch.tensor([0.0]),
                "identity",
            ),
        ]
        with mock.patch.object(converter, "EXPECTED_EMITTED_TENSORS", 1):
            outputs, skipped = converter.prepare_output_tensors(
                logical, {"resblock_kernel_sizes": [3]}
            )
        self.assertEqual(
            [item.name for item in outputs], ["text_encoder.token_embedding.weight"]
        )
        self.assertEqual(outputs[0].array.dtype, np.float32)
        self.assertEqual(skipped[0]["logical_source_name"], "enc_q.pre.bias")


if __name__ == "__main__":
    unittest.main()
