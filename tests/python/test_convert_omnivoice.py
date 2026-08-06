"""Focused tests for the OmniVoice converter's silent-failure rules.

Every rule tested here fails *quietly* if it is wrong: the converter still
writes a file, the file still loads, and the audio is wrong, absent, or
subtly detuned. They are the reason this test file exists rather than a smoke
run.

The checkpoint's own numbers are read from the committed intake inventory
(`reports/porting/omnivoice/omnivoice-0-6b/tensor-inventory.json`) rather than
restated here, so a checkpoint whose layout moves fails these tests instead of
passing them against a stale transcription.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
INVENTORY_PATH = (
    REPO_ROOT / "reports" / "porting" / "omnivoice" / "omnivoice-0-6b" / "tensor-inventory.json"
)
sys.path.insert(0, str(SCRIPTS))

import dump_reference_omnivoice_pytorch as dump_reference  # noqa: E402
import omnivoice_pinned_inputs  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "convert_omnivoice", SCRIPTS / "convert-omnivoice.py"
)
convert = importlib.util.module_from_spec(_spec)
# Registered before execution because @dataclass resolves its own module through
# sys.modules; without this the decorator raises while the file is still loading.
sys.modules["convert_omnivoice"] = convert
_spec.loader.exec_module(convert)


def inventory() -> dict[str, dict[str, dict]]:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def codebook(index: int, entries: int = 4, dim: int = 3) -> dict[str, torch.Tensor]:
    """One RVQ quantizer as the checkpoint stores it: a live table plus training state."""
    base = f"quantizer.quantizers.{index}.codebook."
    return {
        base + "embed": torch.arange(entries * dim, dtype=torch.float32).reshape(entries, dim),
        base + "embed_avg": torch.ones(entries, dim, dtype=torch.float32),
        base + "cluster_size": torch.full((entries,), 7.0),
        base + "inited": torch.ones(1, dtype=torch.float32),
    }


def weight_norm_pair(
    base: str = "semantic_model.encoder.pos_conv_embed.conv",
    channels: int = 8,
    groups: int = 2,
    kernel: int = 4,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """A real torch weight-norm parametrization and the weight it stands for."""
    torch.manual_seed(20260730)
    module = torch.nn.utils.parametrizations.weight_norm(
        torch.nn.Conv1d(channels, channels, kernel, groups=groups), name="weight", dim=2
    )
    tensors = {
        f"{base}.parametrizations.weight.original0":
            module.parametrizations.weight.original0.detach().clone(),
        f"{base}.parametrizations.weight.original1":
            module.parametrizations.weight.original1.detach().clone(),
        f"{base}.bias": module.bias.detach().clone(),
    }
    return tensors, module.weight.detach().clone()


class SkipRuleTests(unittest.TestCase):
    """What the package must NOT carry, and why each omission is not silent."""

    def test_drops_fc1_and_decoder_semantic(self) -> None:
        """Both feed the training-only semantic reconstruction loss.

        `HiggsAudioV2TokenizerModel.encode` and `.decode` reach neither, so
        carrying them would add weight to the package that no graph can read.
        """
        tensors = {
            "fc1.weight": torch.ones(2, 2),
            "fc1.bias": torch.ones(2),
            "decoder_semantic.conv1.weight": torch.ones(2, 2, 3),
            "fc2.weight": torch.ones(2, 2),
            "encoder_semantic.conv.weight": torch.ones(2, 2, 3),
        }
        conversion = convert.Conversion()
        kept = convert.drop_by_prefix(tensors, convert.DROP_PREFIXES, conversion, "codec.")

        self.assertEqual(sorted(kept), ["encoder_semantic.conv.weight", "fc2.weight"])
        self.assertEqual(len(conversion.skipped), len(convert.DROP_PREFIXES))
        for entry in conversion.skipped:
            self.assertTrue(entry["reason"], "every skip must carry a reason")
            self.assertTrue(entry["logical_source_name"].startswith("codec."))

    def test_missing_drop_targets_is_an_error(self) -> None:
        """A layout move must stop the conversion, not quietly widen the package."""
        tensors = {"fc2.weight": torch.ones(2, 2)}
        with self.assertRaises(convert.ConverterError) as caught:
            convert.drop_by_prefix(tensors, ("fc1.",), convert.Conversion(), "codec.")
        self.assertIn("fc1.", str(caught.exception))

    def test_codebook_layer_offsets_buffer_is_skipped_not_emitted(self) -> None:
        """The generator's only non-F32 tensor is a derivable index buffer."""
        offsets = torch.arange(convert.NUM_CODEBOOKS, dtype=torch.int64) * convert.AUDIO_VOCAB
        tensors = {
            "codebook_layer_offsets": offsets,
            "audio_heads.weight": torch.ones(2, 2),
        }
        conversion = convert.Conversion()
        kept = convert.drop_derivable_offsets(tensors, conversion)

        self.assertEqual(sorted(kept), ["audio_heads.weight"])
        self.assertEqual(len(conversion.skipped), 1)
        self.assertIn("derivable", conversion.skipped[0]["reason"])
        self.assertIn("I64", conversion.skipped[0]["reason"])

    def test_offsets_that_are_not_derivable_stop_the_conversion(self) -> None:
        """Dropping a buffer whose values moved would delete real information."""
        tensors = {
            "codebook_layer_offsets": torch.zeros(convert.NUM_CODEBOOKS, dtype=torch.int64),
        }
        with self.assertRaises(convert.ConverterError) as caught:
            convert.drop_derivable_offsets(tensors, convert.Conversion())
        self.assertIn("codebook_layer_offsets", str(caught.exception))

    def test_ema_buffers_are_skipped_when_embed_exists(self) -> None:
        """Unlike qwen3-tts, the RVQ table is live; the accumulators are dead weight.

        `HiggsAudioV2TokenizerEuclideanCodebook.decode` is
        `F.embedding(embed_ind, self.embed)` -- the table is read directly. The
        three sibling buffers are k-means training state; carrying `embed_avg`
        as if it were a weight would double the quantizer for nothing.
        """
        tensors = {**codebook(0), **codebook(1), "fc.weight": torch.ones(2, 2)}
        conversion = convert.Conversion()
        kept = convert.skip_codebook_training_buffers(tensors, conversion, "codec.")

        self.assertEqual(
            sorted(kept),
            [
                "fc.weight",
                "quantizer.quantizers.0.codebook.embed",
                "quantizer.quantizers.1.codebook.embed",
            ],
        )
        self.assertTrue(torch.equal(
            kept["quantizer.quantizers.0.codebook.embed"],
            codebook(0)["quantizer.quantizers.0.codebook.embed"],
        ))
        # Three buffers per quantizer, each with its own recorded reason.
        self.assertEqual(len(conversion.skipped), 6)
        for entry in conversion.skipped:
            self.assertTrue(entry["reason"])

    def test_a_codebook_without_a_live_table_is_an_error(self) -> None:
        """Accumulators with no table means this checkpoint needs reconstruction.

        Emitting nothing for that quantizer loses it silently; the graph would
        then read whatever tensor the catalog resolved to next.
        """
        tensors = dict(codebook(0))
        del tensors["quantizer.quantizers.0.codebook.embed"]
        with self.assertRaises(convert.ConverterError) as caught:
            convert.skip_codebook_training_buffers(tensors, convert.Conversion(), "codec.")
        self.assertIn("embed", str(caught.exception))

    def test_a_checkpoint_with_no_codebook_at_all_is_an_error(self) -> None:
        with self.assertRaises(convert.ConverterError) as caught:
            convert.skip_codebook_training_buffers(
                {"fc.weight": torch.ones(2, 2)}, convert.Conversion(), "codec."
            )
        self.assertIn("no RVQ codebook", str(caught.exception))


class WeightNormFoldTests(unittest.TestCase):
    """HuBERT's positional convolution is stored as a weight-norm parametrization.

    Emitting `original0`/`original1` verbatim gives the graph a magnitude and a
    direction where it expects a kernel. Nothing downstream can tell.
    """

    def test_folds_g_v_into_a_plain_weight(self) -> None:
        tensors, expected = weight_norm_pair()
        conversion = convert.Conversion()
        folded = convert.fold_weight_norm(tensors, conversion, "codec.")

        target = "semantic_model.encoder.pos_conv_embed.conv.weight"
        self.assertEqual(sorted(folded), [
            "semantic_model.encoder.pos_conv_embed.conv.bias", target,
        ])
        self.assertEqual(folded[target].shape, expected.shape)
        self.assertEqual(len(conversion.transformed), 1)
        self.assertEqual(conversion.transformed[0]["output"], target)
        self.assertIn("weight norm", conversion.transformed[0]["reason"])

    def test_fold_matches_torch_weight_norm(self) -> None:
        """Bit-identical, not merely close.

        The oracle recomputes this weight on every forward through the same
        operator. A fold that only agreed to a few ulps would move the argmax
        cascade this family commits at every step.
        """
        for channels, groups, kernel in ((8, 2, 4), (16, 4, 3), (6, 1, 5)):
            with self.subTest(channels=channels, groups=groups, kernel=kernel):
                tensors, expected = weight_norm_pair(
                    channels=channels, groups=groups, kernel=kernel
                )
                folded = convert.fold_weight_norm(tensors, convert.Conversion(), "codec.")
                actual = folded["semantic_model.encoder.pos_conv_embed.conv.weight"]
                self.assertTrue(torch.equal(actual, expected))

    def test_orphan_g_without_v_is_an_error(self) -> None:
        tensors, _ = weight_norm_pair()
        del tensors["semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"]
        with self.assertRaises(convert.ConverterError) as caught:
            convert.fold_weight_norm(tensors, convert.Conversion(), "codec.")
        self.assertIn("original1", str(caught.exception))

    def test_orphan_v_without_g_is_an_error(self) -> None:
        tensors, _ = weight_norm_pair()
        del tensors["semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0"]
        with self.assertRaises(convert.ConverterError) as caught:
            convert.fold_weight_norm(tensors, convert.Conversion(), "codec.")
        self.assertIn("original0", str(caught.exception))

    def test_an_ambiguous_magnitude_shape_is_an_error(self) -> None:
        """The norm dimension is recovered from g's one non-singleton axis."""
        tensors, _ = weight_norm_pair()
        base = "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight."
        tensors[base + "original0"] = torch.ones(8, 1, 4)
        with self.assertRaises(convert.ConverterError) as caught:
            convert.fold_weight_norm(tensors, convert.Conversion(), "codec.")
        self.assertIn("dimension", str(caught.exception))

    def test_a_checkpoint_with_no_parametrization_is_an_error(self) -> None:
        """This family's codec always carries one; none means the layout moved."""
        with self.assertRaises(convert.ConverterError) as caught:
            convert.fold_weight_norm(
                {"semantic_model.encoder.pos_conv_embed.conv.bias": torch.ones(4)},
                convert.Conversion(),
                "codec.",
            )
        self.assertIn("weight norm", str(caught.exception))

    def test_a_fold_that_would_overwrite_a_plain_weight_is_an_error(self) -> None:
        tensors, _ = weight_norm_pair()
        tensors["semantic_model.encoder.pos_conv_embed.conv.weight"] = torch.ones(4)
        with self.assertRaises(convert.ConverterError) as caught:
            convert.fold_weight_norm(tensors, convert.Conversion(), "codec.")
        self.assertIn("overwrite", str(caught.exception))


class NameLengthTests(unittest.TestCase):
    """GGML stores a tensor name in a fixed 64-byte field and truncates past it.

    A truncated name is not findable by the name the catalog asks for, and two
    names that differ only past the cut become one tensor. 195 of this codec's
    527 names sit within six characters of the limit and the longest overruns
    it by eighteen.
    """

    def emitted_names(self) -> list[str]:
        """Every name the converter will hand to `shorten_name`."""
        data = inventory()
        names = list(data["generator"])
        names.remove("codebook_layer_offsets")
        for name in data["codec"]:
            if any(name.startswith(prefix) for prefix in convert.DROP_PREFIXES):
                continue
            if name.endswith((".embed_avg", ".cluster_size", ".inited")):
                continue  # RVQ training buffers the converter skips
            if convert.PARAMETRIZATION_INFIX in name:
                if name.endswith(".original1"):
                    continue
                name = convert.folded_weight_name(name)
            names.append("codec." + name)
        return names

    def test_emitted_names_match_the_report_count(self) -> None:
        # 798 is output.emitted_tensor_count (312 generator + 486 codec) in
        # the conversion report reports/convert/omnivoice/omnivoice-0-6b-F32
        # .json -- a local artifact, not committed -- from the conversion
        # that produced the pinned package (GGUF sha256 3ecaa5e2..., recorded
        # in full in the porting log).
        self.assertEqual(len(self.emitted_names()), 798,
                         "the helper no longer models what the converter emits")

    def test_every_inventory_name_fits_after_prefixing(self) -> None:
        for name in self.emitted_names():
            shortened = convert.shorten_name(name, convert.Conversion())
            self.assertLess(len(shortened), convert.GGML_MAX_NAME, name)

    def test_shortened_names_stay_distinct(self) -> None:
        shortened = [convert.shorten_name(n, convert.Conversion()) for n in self.emitted_names()]
        self.assertEqual(len(set(shortened)), len(shortened), "a shortening merged two tensors")

    def test_every_shortening_rule_fires_on_a_real_name(self) -> None:
        names = self.emitted_names()
        for long_form, _ in convert.NAME_SHORTENINGS:
            with self.subTest(rule=long_form):
                self.assertTrue(
                    [n for n in names if long_form in n],
                    f"{long_form!r} matches nothing in this checkpoint; a rename nobody needs",
                )

    def test_every_shortening_rule_is_load_bearing(self) -> None:
        """A rule that fixes no overflow is a gratuitous rename of the catalog."""
        names = self.emitted_names()
        for index, (long_form, _) in enumerate(convert.NAME_SHORTENINGS):
            with self.subTest(rule=long_form):
                without = [r for i, r in enumerate(convert.NAME_SHORTENINGS) if i != index]
                overflowing = []
                for name in names:
                    reduced = name
                    for old, new in without:
                        reduced = reduced.replace(old, new)
                    if len(reduced) >= convert.GGML_MAX_NAME:
                        overflowing.append(reduced)
                self.assertTrue(
                    overflowing,
                    f"{long_form!r} fixes no overflow; drop it rather than rename the catalog",
                )

    def test_overflowing_name_without_a_rule_stops_conversion(self) -> None:
        conversion = convert.Conversion()
        with self.assertRaises(convert.ConverterError) as caught:
            convert.shorten_name("codec." + "x" * convert.GGML_MAX_NAME, conversion)
        self.assertIn("GGML truncates", str(caught.exception))

    def test_a_name_already_short_enough_is_untouched(self) -> None:
        conversion = convert.Conversion()
        self.assertEqual(convert.shorten_name("llm.norm.weight", conversion), "llm.norm.weight")
        self.assertEqual(conversion.renamed, [], "an unchanged name is not a rename")

    def test_a_rename_is_recorded(self) -> None:
        conversion = convert.Conversion()
        name = "codec.semantic_model.encoder.layers.11.feed_forward.intermediate_dense.weight"
        shortened = convert.shorten_name(name, conversion)
        self.assertEqual(conversion.renamed, [{"output": shortened, "from": name}])


class DtypeTests(unittest.TestCase):
    """This checkpoint is F32 throughout; anything else is a layout change."""

    def test_f32_stays_f32(self) -> None:
        array, dtype = convert.numpy_of(torch.ones(4, dtype=torch.float32))
        self.assertEqual(dtype.name, "F32")
        self.assertEqual(array.dtype.itemsize, 4)

    def test_i64_and_other_dtypes_are_rejected(self) -> None:
        for dtype in (torch.int64, torch.bfloat16, torch.float16, torch.int32):
            with self.subTest(dtype=dtype):
                with self.assertRaises(convert.ConverterError):
                    convert.numpy_of(torch.ones(4, dtype=dtype))

    def test_the_inventory_carries_exactly_one_non_f32_tensor(self) -> None:
        """The single I64 buffer is the one the converter derives instead."""
        data = inventory()
        non_f32 = {
            name: entry["dtype"]
            for name, entry in {**data["generator"], **data["codec"]}.items()
            if entry["dtype"] != "F32"
        }
        self.assertEqual(non_f32, {"codebook_layer_offsets": "I64"})

    def test_non_finite_values_are_refused(self) -> None:
        with self.assertRaises(convert.ConverterError) as caught:
            convert.convert_file_from_tensors(  # use the module's existing test seam;
                {"x": torch.tensor([float("nan")])}, "", convert.Conversion())
        self.assertIn("non-finite", str(caught.exception))


class ConfigGuardTests(unittest.TestCase):
    """A config whose layout moved must fail with a diagnosis, not a bare KeyError."""

    def test_a_present_key_is_returned(self) -> None:
        self.assertEqual(convert.require_config({"llm_config": 1}, "llm_config", "config.json"), 1)

    def test_a_missing_key_names_the_key_and_where(self) -> None:
        with self.assertRaises(convert.ConverterError) as caught:
            convert.require_config({}, "llm_config", "config.json")
        message = str(caught.exception)
        self.assertIn("llm_config", message)
        self.assertIn("config.json", message)


class CodecGeometryTests(unittest.TestCase):
    """`audio_tokenizer/config.json` disagrees with its own weights three ways.

    The tensors win. A converter that sized the codebooks from the config would
    write n_codebooks 9 and codebook_dim 8 into a package holding eight
    [1024, 64] tables, and the loader would build a graph around the metadata.
    """

    def test_geometry_is_measured_from_the_tables(self) -> None:
        tensors = {}
        for index in range(3):
            tensors[f"quantizer.quantizers.{index}.codebook.embed"] = torch.zeros(1024, 64)
        geometry = convert.measure_codec_geometry(tensors)
        self.assertEqual(geometry.quantizer_count, 3)
        self.assertEqual(geometry.codebook_size, 1024)
        self.assertEqual(geometry.codebook_dim, 64)

    def test_the_real_inventory_measures_eight_by_1024_by_64(self) -> None:
        shapes = {
            name: entry["shape"]
            for name, entry in inventory()["codec"].items()
            if name.endswith(".codebook.embed")
        }
        self.assertEqual(len(shapes), 8)
        self.assertEqual(sorted({tuple(s) for s in shapes.values()}), [(1024, 64)])

    def test_tables_of_different_shapes_are_refused(self) -> None:
        tensors = {
            "quantizer.quantizers.0.codebook.embed": torch.zeros(1024, 64),
            "quantizer.quantizers.1.codebook.embed": torch.zeros(512, 64),
        }
        with self.assertRaises(convert.ConverterError) as caught:
            convert.measure_codec_geometry(tensors)
        self.assertIn("shape", str(caught.exception))

    def test_a_gap_in_the_quantizer_indices_is_refused(self) -> None:
        tensors = {
            "quantizer.quantizers.0.codebook.embed": torch.zeros(1024, 64),
            "quantizer.quantizers.2.codebook.embed": torch.zeros(1024, 64),
        }
        with self.assertRaises(convert.ConverterError) as caught:
            convert.measure_codec_geometry(tensors)
        self.assertIn("contiguous", str(caught.exception))

    def test_config_disagreements_are_recorded_rather_than_trusted(self) -> None:
        codec_config = {
            "codebook_dim": 64,
            "codebook_size": 1024,
            "sample_rate": 24000,
            "acoustic_model_config": {
                "n_codebooks": 9, "codebook_dim": 8, "sampling_rate": 16000,
            },
        }
        geometry = convert.CodecGeometry(quantizer_count=8, codebook_size=1024, codebook_dim=64)
        found = convert.config_disagreements(codec_config, geometry)

        fields = sorted(entry["field"] for entry in found)
        self.assertEqual(fields, [
            "acoustic_model_config.codebook_dim",
            "acoustic_model_config.n_codebooks",
            "acoustic_model_config.sampling_rate",
        ])
        for entry in found:
            self.assertIn("resolution", entry)

    def test_a_config_that_agrees_reports_nothing(self) -> None:
        codec_config = {
            "codebook_dim": 64,
            "codebook_size": 1024,
            "sample_rate": 24000,
            "acoustic_model_config": {
                "n_codebooks": 8, "codebook_dim": 64, "sampling_rate": 24000,
            },
        }
        geometry = convert.CodecGeometry(quantizer_count=8, codebook_size=1024, codebook_dim=64)
        self.assertEqual(convert.config_disagreements(codec_config, geometry), [])


class GenerationDefaultsTests(unittest.TestCase):
    """The decoding defaults come from the pinned package, never a second copy.

    `num_step`, `guidance_scale` and `t_shift` decide what the model says, not
    merely how it sounds: they are the mask-predict schedule. A restated table
    drifts from the package the oracle ran.
    """

    @unittest.skipUnless(
        importlib.util.find_spec("omnivoice"),
        "the omnivoice package is only present in scripts/envs/omnivoice",
    )
    def test_defaults_are_read_from_the_upstream_dataclass(self) -> None:
        defaults = convert.read_generation_defaults()
        for key in ("num_step", "guidance_scale", "t_shift", "layer_penalty_factor",
                    "position_temperature", "class_temperature"):
            self.assertIn(key, defaults)

        from omnivoice import OmniVoiceGenerationConfig  # noqa: PLC0415
        upstream = OmniVoiceGenerationConfig()
        for key, value in defaults.items():
            self.assertEqual(value, getattr(upstream, key))

    def test_an_absent_package_is_an_error_not_a_fallback_table(self) -> None:
        with self.assertRaises(convert.ConverterError) as caught:
            convert.read_generation_defaults(module_name="omnivoice_not_installed_anywhere")
        self.assertIn("omnivoice_not_installed_anywhere", str(caught.exception))

    def test_zero_num_step_is_refused(self) -> None:
        defaults = {
            "num_step": 0, "guidance_scale": 2.0, "t_shift": 0.1,
            "layer_penalty_factor": 5.0, "position_temperature": 5.0,
            "class_temperature": 0.0,
        }
        with self.assertRaises(convert.ConverterError) as caught:
            convert.validate_generation_defaults(defaults)
        self.assertIn("num_step", str(caught.exception))

    def test_a_missing_default_is_refused(self) -> None:
        with self.assertRaises(convert.ConverterError) as caught:
            convert.validate_generation_defaults({"num_step": 32})
        self.assertIn("guidance_scale", str(caught.exception))

    def test_a_non_finite_schedule_value_is_refused(self) -> None:
        defaults = {
            "num_step": 32, "guidance_scale": float("nan"), "t_shift": 0.1,
            "layer_penalty_factor": 5.0, "position_temperature": 5.0,
            "class_temperature": 0.0,
        }
        with self.assertRaises(convert.ConverterError) as caught:
            convert.validate_generation_defaults(defaults)
        self.assertIn("guidance_scale", str(caught.exception))


class OutputFrameCeilingTests(unittest.TestCase):
    """package_contract.max_output_frames must be PCM frames, not codec frames.

    PR #6's review found the committed manifest itself had this exact class of
    error: 750, this port's codec-frame ceiling (30 s at 25 Hz), written where
    720000 native PCM frames belonged. These tests pin the guard that now
    catches it at conversion time rather than at a synthesis call three layers
    away.
    """

    def test_the_committed_manifest_declares_a_plausible_pcm_ceiling(self) -> None:
        manifest = json.loads(
            (REPO_ROOT / "tests" / "golden" / "omnivoice" / "omnivoice-0-6b.manifest.json")
            .read_text(encoding="utf-8")
        )
        convert.validate_output_frame_ceiling(manifest["package_contract"])

    def test_a_codec_frame_count_is_refused(self) -> None:
        """The exact defect: 750 codec frames mistaken for PCM frames."""
        with self.assertRaises(convert.ConverterError) as caught:
            convert.validate_output_frame_ceiling({"max_output_frames": 750})
        message = str(caught.exception)
        self.assertIn("750", message)
        self.assertIn("PCM", message)

    def test_a_value_under_one_second_of_native_pcm_is_refused(self) -> None:
        with self.assertRaises(convert.ConverterError) as caught:
            convert.validate_output_frame_ceiling({"max_output_frames": convert.SAMPLE_RATE - 1})
        self.assertIn("PCM", str(caught.exception))

    def test_exactly_one_second_is_accepted(self) -> None:
        # convert.SAMPLE_RATE (24000) is a whole multiple of HOP_LENGTH (960).
        convert.validate_output_frame_ceiling({"max_output_frames": convert.SAMPLE_RATE})

    def test_a_value_not_landing_on_a_codec_frame_boundary_is_refused(self) -> None:
        with self.assertRaises(convert.ConverterError) as caught:
            convert.validate_output_frame_ceiling({"max_output_frames": convert.SAMPLE_RATE + 1})
        message = str(caught.exception)
        self.assertIn("hop_length", message)
        self.assertIn(str(convert.HOP_LENGTH), message)


class PinnedInputTests(unittest.TestCase):
    """Every local file the conversion reads or republishes is pinned.

    The codec's LICENSE is the one that matters most here: it is republished
    beside the artifact and its digest is recorded in the report as
    authoritative, so a locally edited grant would be carried into the package
    and vouched for. Hashing the copy against its own source proves nothing.
    """

    def manifest_with(self, artifacts: list[dict[str, str]]) -> dict[str, object]:
        return {"source": {"artifacts": artifacts}}

    def license_pin(self) -> omnivoice_pinned_inputs.PinnedInput:
        return omnivoice_pinned_inputs.PinnedInput(
            "license", "audio_tokenizer/LICENSE", "codec_license"
        )

    def license_path(self, weights_dir: Path) -> Path:
        return omnivoice_pinned_inputs.resolve_local(weights_dir, self.license_pin())

    def test_the_codec_license_is_one_of_the_pinned_inputs(self) -> None:
        entries = {entry.sha256_key: entry for entry in omnivoice_pinned_inputs.PINNED_INPUTS}
        self.assertIn("codec_license", entries)
        entry = entries["codec_license"]
        self.assertEqual(entry.relative_path, "audio_tokenizer/LICENSE")
        self.assertEqual(entry.role, "license")

    def test_the_committed_manifest_pins_the_audited_license_digest(self) -> None:
        """The pin the intake audited, resolved the way the converter resolves it.

        The manifest carries two `license` artifacts; the "/resolve/" marker
        must select the codec's rather than the source repository's Apache
        grant, whose raw.githubusercontent.com locator carries no such marker.
        """
        manifest = json.loads(
            (REPO_ROOT / "tests" / "golden" / "omnivoice" / "omnivoice-0-6b.manifest.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(
            convert.pinned_digest(manifest, self.license_pin()),
            "ac933dc084d119bd20401956b90d11ae87c248b2da62622cd580d82cdf2fa049",
        )

    def test_a_license_that_does_not_match_its_pin_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory)
            path = self.license_path(weights)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"a locally edited grant\n")
            manifest = self.manifest_with([{
                "role": "license",
                "locator": "https://example.invalid/resolve/rev/audio_tokenizer/LICENSE",
                "sha256": "ac933dc084d119bd20401956b90d11ae87c248b2da62622cd580d82cdf2fa049",
            }])
            with self.assertRaises(convert.ConverterError) as caught:
                convert.verify_pinned_inputs(manifest, weights, [self.license_pin()])
            message = str(caught.exception)
            self.assertIn(str(path), message, "the error must name the offending file")
            self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(), message)

    def test_a_manifest_with_no_license_artifact_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory)
            path = self.license_path(weights)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"some grant\n")
            manifest = self.manifest_with([{
                "role": "checkpoint",
                "locator": "https://example.invalid/resolve/rev/model.safetensors",
                "sha256": "0" * 64,
            }])
            with self.assertRaises(convert.ConverterError) as caught:
                convert.verify_pinned_inputs(manifest, weights, [self.license_pin()])
            self.assertIn("license", str(caught.exception))

    def test_two_matching_license_artifacts_are_refused(self) -> None:
        """An ambiguous pin must not be resolved by picking the first one."""
        duplicate = {
            "role": "license",
            "locator": "https://example.invalid/resolve/rev/audio_tokenizer/LICENSE",
            "sha256": "0" * 64,
        }
        with self.assertRaises(convert.ConverterError) as caught:
            convert.pinned_digest(self.manifest_with([duplicate, dict(duplicate)]),
                                  self.license_pin())
        self.assertIn("exactly one", str(caught.exception))

    def test_a_missing_pinned_input_is_refused_before_anything_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory)
            path = self.license_path(weights)
            manifest = self.manifest_with([{
                "role": "license",
                "locator": "https://example.invalid/resolve/rev/audio_tokenizer/LICENSE",
                "sha256": "0" * 64,
            }])
            with self.assertRaises(convert.ConverterError) as caught:
                convert.verify_pinned_inputs(manifest, weights, [self.license_pin()])
            self.assertIn(str(path), str(caught.exception))

    def test_a_matching_input_returns_its_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory)
            path = self.license_path(weights)
            path.parent.mkdir(parents=True)
            payload = b"the audited grant\n"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            manifest = self.manifest_with([{
                "role": "license",
                "locator": "https://example.invalid/resolve/rev/audio_tokenizer/LICENSE",
                "sha256": digest,
            }])
            self.assertEqual(
                convert.verify_pinned_inputs(manifest, weights, [self.license_pin()]),
                {"codec_license": digest},
            )


class SharedPinnedInputTableTests(unittest.TestCase):
    """Plan 3 carryover item 4: one pinned-input table, two consumers.

    `convert-omnivoice.py` used to carry a private six-entry list and
    `dump_reference_omnivoice_pytorch.py` inferred its own idea of the same
    set from "/resolve/" markers in manifest locator URLs, with nothing
    checking the two agreed. Both now import `omnivoice_pinned_inputs` and
    walk its `PINNED_INPUTS`; this is what keeps that true instead of a
    private second copy creeping back into either script.
    """

    def build_weights_and_manifest(self, directory: Path) -> dict[str, object]:
        """A weights directory and a matching manifest covering every pin.

        Built by walking `PINNED_INPUTS` itself rather than a restated
        six-item list, so a table edited to add or drop an entry keeps this
        test honest about what it is actually exercising.
        """
        artifacts = []
        for pin in omnivoice_pinned_inputs.PINNED_INPUTS:
            local = omnivoice_pinned_inputs.resolve_local(directory, pin)
            local.parent.mkdir(parents=True, exist_ok=True)
            payload = f"payload for {pin.sha256_key}\n".encode()
            local.write_bytes(payload)
            artifacts.append({
                "role": pin.role,
                "locator": f"https://example.invalid/resolve/rev/{pin.relative_path}",
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        return {"source": {"artifacts": artifacts}}

    def test_both_consumers_import_the_identical_table_object(self) -> None:
        self.assertIs(convert.omnivoice_pinned_inputs, omnivoice_pinned_inputs)
        self.assertIs(dump_reference.omnivoice_pinned_inputs, omnivoice_pinned_inputs)
        self.assertIs(
            convert.omnivoice_pinned_inputs.PINNED_INPUTS,
            omnivoice_pinned_inputs.PINNED_INPUTS,
        )
        self.assertIs(
            dump_reference.omnivoice_pinned_inputs.PINNED_INPUTS,
            omnivoice_pinned_inputs.PINNED_INPUTS,
        )

    def test_the_converter_verifies_exactly_the_shared_table_by_default(self) -> None:
        """`verify_pinned_inputs` with no explicit list walks the shared table.

        This is the converter's startup-verification list: `main()` calls
        `verify_pinned_inputs(manifest, weights_dir)` with no third argument,
        so whatever this default resolves to is what a real conversion checks.
        """
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory)
            manifest = self.build_weights_and_manifest(weights)
            digests = convert.verify_pinned_inputs(manifest, weights)
        self.assertEqual(
            set(digests), {pin.sha256_key for pin in omnivoice_pinned_inputs.PINNED_INPUTS}
        )

    def test_the_dumper_digest_checks_exactly_the_shared_table(self) -> None:
        """`dump_reference_omnivoice_pytorch.verify_pinned_inputs` walks the same table.

        It used to infer a private set of paths from "/resolve/" markers with
        no reference to what the converter pins; this drives it the same way
        the test above drives the converter and checks the same six relative
        paths come back verified.
        """
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory)
            manifest = self.build_weights_and_manifest(weights)
            verified = dump_reference.verify_pinned_inputs(manifest, weights)
        self.assertEqual(
            sorted(verified),
            sorted(pin.relative_path for pin in omnivoice_pinned_inputs.PINNED_INPUTS),
        )


class LicenseCarriageTests(unittest.TestCase):
    """The codec's grant must never separate from the converted artifact.

    `audio_tokenizer/LICENSE` is the sole grant for the Higgs Audio 2 codec
    weights -- its own model card says "[More Information Needed]" -- and the
    agreement requires redistribution to carry its text.
    """

    def test_copies_the_codec_license_byte_identically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights" / "audio_tokenizer"
            weights.mkdir(parents=True)
            payload = b"BOSON HIGGS AUDIO 2 COMMUNITY LICENSE AGREEMENT\n\xc2\xa0text\n"
            (weights / "LICENSE").write_bytes(payload)
            output = root / "out" / "model.gguf"
            output.parent.mkdir(parents=True)
            digest = hashlib.sha256(payload).hexdigest()

            records = convert.carry_licenses(root / "weights", output, Path.cwd(), digest)

            copied = output.parent / convert.CODEC_LICENSE_NAME
            self.assertEqual(copied.read_bytes(), payload)
            self.assertIn(digest, [r.get("sha256") for r in records])
            statements = " ".join(r.get("statement", "") for r in records)
            self.assertIn("CC-BY-NC", statements)

    def test_the_copy_is_checked_against_the_pin_not_against_itself(self) -> None:
        """Re-hashing the source after the copy compares a file with itself."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights" / "audio_tokenizer"
            weights.mkdir(parents=True)
            (weights / "LICENSE").write_bytes(b"a locally edited grant\n")
            output = root / "out" / "model.gguf"
            output.parent.mkdir(parents=True)
            with self.assertRaises(convert.ConverterError) as caught:
                convert.carry_licenses(root / "weights", output, Path.cwd(), "0" * 64)
            self.assertIn(convert.CODEC_LICENSE_NAME, str(caught.exception))

    def test_a_missing_codec_license_stops_the_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights" / "audio_tokenizer").mkdir(parents=True)
            output = root / "out" / "model.gguf"
            output.parent.mkdir(parents=True)
            with self.assertRaises(convert.ConverterError) as caught:
                convert.carry_licenses(root / "weights", output, Path.cwd(), "0" * 64)
            self.assertIn("LICENSE", str(caught.exception))


class VerifyGgufShapeTests(unittest.TestCase):
    """Carry-over item 7 (Plan 1's Task 3): `verify_gguf` must catch a shape
    change, not merely an element-count change.

    A tensor written transposed keeps the same element count -- a 2x3 array
    and its 3x2 transpose both hold six F32 values -- so a check built on
    `np.prod(shape) == size` cannot tell them apart. `verify_gguf` compares
    the full GGML shape tuple instead. These tests write a real tensor
    through `GGUFWriter`, the same object the converter itself uses, then
    feed `verify_gguf` an `OutputTensor` that names the same tensor but
    disagrees about its shape while agreeing about its element count.
    """

    def write_minimal_gguf(self, path: Path, name: str, array: np.ndarray) -> None:
        writer = convert.GGUFWriter(str(path), convert.ARCH_KEY)
        writer.add_tensor(name, array, raw_dtype=convert.GGMLQuantizationType.F32)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_a_matching_shape_and_dtype_is_accepted(self) -> None:
        """Positive control: proves the harness itself is sound."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            array = np.arange(6, dtype=np.float32).reshape(2, 3)
            self.write_minimal_gguf(path, "x", array)
            outputs = [convert.OutputTensor("x", array, convert.GGMLQuantizationType.F32, "test")]
            convert.verify_gguf(path, outputs)  # must not raise

    def test_a_transposed_same_element_count_tensor_is_caught(self) -> None:
        """The exact defect: 2x3 and 3x2 both hold six F32 values."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            written = np.arange(6, dtype=np.float32).reshape(2, 3)
            self.write_minimal_gguf(path, "x", written)

            transposed = np.ascontiguousarray(written.T)  # (3, 2): same 6 elements, wrong shape
            self.assertEqual(transposed.size, written.size, "the harness must keep counts equal")
            outputs = [
                convert.OutputTensor("x", transposed, convert.GGMLQuantizationType.F32, "test")
            ]

            with self.assertRaises(convert.ConverterError) as caught:
                convert.verify_gguf(path, outputs)
            self.assertIn("shape", str(caught.exception))

    def test_an_element_count_only_check_would_have_missed_it(self) -> None:
        """Documents why the fix matters: the pre-fix rule this replaced was
        `int(np.prod(tensor.shape)) != output.array.size` (see
        scripts/convert-qwen3-tts.py's `verify_gguf`, which still runs it).
        That rule cannot distinguish the transposed tensor from the one that
        was actually written.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            written = np.arange(6, dtype=np.float32).reshape(2, 3)
            self.write_minimal_gguf(path, "x", written)

            reader = convert.GGUFReader(str(path))
            tensor = {t.name: t for t in reader.tensors}["x"]
            transposed = np.ascontiguousarray(written.T)

            # The old, insufficient rule sees no problem:
            self.assertEqual(int(np.prod(tensor.shape)), transposed.size)
            # ...yet the shapes genuinely disagree, which is what the current
            # (fixed) check in `verify_gguf` catches instead.
            self.assertNotEqual(
                tuple(int(d) for d in tensor.shape), tuple(reversed(transposed.shape))
            )


class ConverterEntryPointOrderingTests(unittest.TestCase):
    """Carry-over item 7 (Plan 1's Task 3), step 2: a missing license must
    cost nothing, not a multi-gigabyte GGUF write.

    `main()`'s `verify_pinned_inputs` call already covers all six pinned
    inputs -- including the codec's LICENSE -- before a single tensor is
    read, and `carry_licenses` (the license-copy step) now runs ahead of the
    `atomic_output_path` block too. This drives `main()` itself through its
    real `sys.argv` seam, so it is the wiring that gets tested rather than
    any one guard function in isolation: a future edit that reorders
    `main()` again, or drops the license from the pinned set, would be
    caught here without needing a full conversion to notice.

    The five non-license pinned inputs are ordinary bytes, not real weights;
    nothing downstream of `verify_pinned_inputs` is ever reached because the
    missing license is the first fatal problem -- which is exactly the
    property under test.
    """

    def build_weights_missing_license(self, root: Path) -> tuple[Path, dict[str, object]]:
        """Every pinned input present and digest-matching except the license."""
        weights = root / "weights"
        artifacts = []
        for pin in omnivoice_pinned_inputs.PINNED_INPUTS:
            if pin.sha256_key == "codec_license":
                continue  # deliberately absent
            local = omnivoice_pinned_inputs.resolve_local(weights, pin)
            local.parent.mkdir(parents=True, exist_ok=True)
            payload = f"payload for {pin.sha256_key}\n".encode()
            local.write_bytes(payload)
            artifacts.append({
                "role": pin.role,
                "locator": f"https://example.invalid/resolve/rev/{pin.relative_path}",
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        manifest = {"family": convert.ARCH_KEY, "source": {"artifacts": artifacts}}
        return weights, manifest

    def test_a_missing_license_fails_before_any_output_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, manifest = self.build_weights_missing_license(root)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_path = root / "out" / "model.gguf"

            argv = [
                "convert-omnivoice.py",
                "--manifest", str(manifest_path),
                "--weights-dir", str(weights),
                "--output", str(output_path),
            ]
            saved_argv = sys.argv
            sys.argv = argv
            try:
                with self.assertRaises(convert.ConverterError) as caught:
                    convert.main()
            finally:
                sys.argv = saved_argv

            self.assertIn("LICENSE", str(caught.exception))
            self.assertFalse(output_path.exists(), "the GGUF must not exist after this failure")
            self.assertFalse(
                output_path.parent.exists(),
                "the output directory is only created inside atomic_output_path, which this "
                "failure must never reach",
            )


class ReportShapeTests(unittest.TestCase):
    def test_tensor_report_states_ggml_shape_not_source_shape(self) -> None:
        array, dtype = convert.numpy_of(torch.zeros(2, 3, dtype=torch.float32))
        entry = convert.OutputTensor("x", array, dtype, "src").report()
        self.assertEqual(entry["ggml_shape"], [3, 2])
        self.assertEqual(entry["elements"], 6)
        self.assertIn("sha256", entry)


if __name__ == "__main__":
    unittest.main()
