"""Focused tests for the qwen3-tts converter's silent-failure rules.

Every rule tested here fails *quietly* if it is wrong: the converter still
writes a file, the file still loads, and the audio is wrong or absent. They are
the reason this test file exists rather than a smoke run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

_spec = importlib.util.spec_from_file_location(
    "convert_qwen3_tts", SCRIPTS / "convert-qwen3-tts.py"
)
convert = importlib.util.module_from_spec(_spec)
# Registered before execution because @dataclass resolves its own module through
# sys.modules; without this the decorator raises while the file is still loading.
sys.modules["convert_qwen3_tts"] = convert
_spec.loader.exec_module(convert)


def decoder_pair(layer: int, entries: int = 4, dim: int = 3) -> dict[str, torch.Tensor]:
    base = f"decoder.quantizer.rvq_first.vq.layers.{layer}._codebook."
    return {
        base + "embedding_sum": torch.arange(entries * dim, dtype=torch.float32).reshape(entries, dim),
        base + "cluster_usage": torch.arange(1, entries + 1, dtype=torch.float32),
    }


def encoder_pair(layer: int, entries: int = 4, dim: int = 3) -> dict[str, torch.Tensor]:
    base = f"encoder.quantizer.acoustic_residual_vector_quantizer.layers.{layer}.codebook."
    return {
        base + "embed_sum": torch.ones(entries, dim, dtype=torch.float32),
        base + "cluster_usage": torch.full((entries,), 2.0),
    }


class CodebookReconstructionTests(unittest.TestCase):
    """The checkpoint stores EMA accumulators, not codebooks."""

    def test_reconstructs_the_table_from_the_accumulator_pair(self) -> None:
        tensors = {**decoder_pair(0), **encoder_pair(0)}
        conversion = convert.Conversion()
        out = convert.reconstruct_codebooks(tensors, conversion)

        key = "decoder.quantizer.rvq_first.vq.layers.0._codebook.codebook"
        self.assertIn(key, out)
        expected = tensors[
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"
        ] / tensors[
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"
        ].clamp(min=convert.RVQ_EPS).unsqueeze(1)
        self.assertTrue(torch.equal(out[key], expected))

    def test_consumes_the_accumulators_so_they_cannot_reach_the_package(self) -> None:
        tensors = {**decoder_pair(0), **encoder_pair(0)}
        out = convert.reconstruct_codebooks(tensors, convert.Conversion())
        leaked = [n for n in out if n.endswith(("embedding_sum", "embed_sum", "cluster_usage"))]
        self.assertEqual(leaked, [], "raw accumulators must not survive conversion")

    def test_handles_both_naming_conventions(self) -> None:
        """The decoder says embedding_sum, the encoder says embed_sum.

        The shipped package carries only the decoder half, but the rule matches
        either spelling so that a package which carries both -- or only the
        other -- is reconstructed rather than half-emitted.
        """
        tensors = {**decoder_pair(0), **encoder_pair(0)}
        out = convert.reconstruct_codebooks(tensors, convert.Conversion())
        self.assertIn("decoder.quantizer.rvq_first.vq.layers.0._codebook.codebook", out)
        self.assertIn(
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.codebook",
            out,
        )

    def test_one_convention_alone_is_fine(self) -> None:
        """The shipped package carries only the decoder half.

        Requiring both spellings to appear would reject it. What must hold is
        that nothing was left behind, which the next two tests cover.
        """
        out = convert.reconstruct_codebooks(dict(decoder_pair(0)), convert.Conversion())
        self.assertIn("decoder.quantizer.rvq_first.vq.layers.0._codebook.codebook", out)

    def test_rejects_an_accumulator_the_rule_did_not_match(self) -> None:
        """A renamed accumulator must stop the conversion, not pass through.

        Emitting one as a weight would hand the graph EMA sums where it expects
        a codebook, and nothing downstream can tell the difference.
        """
        tensors = {**decoder_pair(0)}
        base = "decoder.quantizer.rvq_rest.vq.layers.0._codebook."
        tensors[base + "embedding_total"] = torch.ones(4, 3)
        tensors[base + "cluster_usage"] = torch.ones(4)
        with self.assertRaises(convert.ConverterError) as caught:
            convert.reconstruct_codebooks(tensors, convert.Conversion())
        self.assertIn("not reconstructed", str(caught.exception))

    def test_rejects_a_checkpoint_with_no_codebooks_at_all(self) -> None:
        with self.assertRaises(convert.ConverterError) as caught:
            convert.reconstruct_codebooks({"decoder.pre_conv.conv.weight": torch.ones(2)},
                                          convert.Conversion())
        self.assertIn("no RVQ codebook", str(caught.exception))

    def test_rejects_an_accumulator_with_no_usage_counts(self) -> None:
        tensors = {**decoder_pair(0), **encoder_pair(0)}
        del tensors["decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"]
        with self.assertRaises(convert.ConverterError) as caught:
            convert.reconstruct_codebooks(tensors, convert.Conversion())
        self.assertIn("cluster_usage", str(caught.exception))

    def test_rejects_mismatched_entry_counts(self) -> None:
        tensors = {**decoder_pair(0), **encoder_pair(0)}
        base = "decoder.quantizer.rvq_first.vq.layers.0._codebook."
        tensors[base + "cluster_usage"] = torch.ones(3)
        with self.assertRaises(convert.ConverterError) as caught:
            convert.reconstruct_codebooks(tensors, convert.Conversion())
        self.assertIn("usage counts", str(caught.exception))

    def test_zero_usage_does_not_divide_by_zero(self) -> None:
        """An unused codebook entry has zero usage; RVQ_EPS is what saves it."""
        tensors = {**decoder_pair(0), **encoder_pair(0)}
        base = "decoder.quantizer.rvq_first.vq.layers.0._codebook."
        tensors[base + "cluster_usage"] = torch.zeros(4)
        out = convert.reconstruct_codebooks(tensors, convert.Conversion())
        table = out[base + "codebook"]
        self.assertTrue(torch.isfinite(table).all(), "zero usage must not produce inf or nan")

    def test_records_every_reconstruction_in_the_report(self) -> None:
        conversion = convert.Conversion()
        convert.reconstruct_codebooks({**decoder_pair(0), **encoder_pair(0)}, conversion)
        self.assertEqual(len(conversion.transformed), 2)
        for entry in conversion.transformed:
            self.assertIn("EMA accumulators", entry["reason"])


class TensorNameLengthTests(unittest.TestCase):
    """GGML stores a tensor name in a fixed 64-byte field and truncates past it.

    A truncated name is not findable by the name the catalog asks for, and the
    package still loads, so nothing reports the problem. Five paths overflow in
    CustomVoice's talker-plus-codec-decoder conversion; carrying the codec's
    encoder half for Base introduces its own, additional overflowing paths
    (`EncoderTensorNameShorteningTests` below covers those).
    """

    def test_shortens_the_paths_that_overflow(self) -> None:
        conversion = convert.Conversion()
        name = "codec.decoder.pre_transformer.layers.7.post_attention_layernorm.weight"
        self.assertGreaterEqual(len(name), convert.GGML_MAX_NAME)
        shortened = convert.shorten_name(name, conversion)
        self.assertEqual(shortened, "codec.decoder.pre_transformer.layers.7.post_attn_norm.weight")
        self.assertLess(len(shortened), convert.GGML_MAX_NAME)
        self.assertEqual(conversion.renamed, [{"output": shortened, "from": name}])

    def test_shortens_every_overflowing_pattern_in_this_checkpoint(self) -> None:
        for name in (
            "codec.decoder.pre_transformer.layers.7.post_attention_layernorm.weight",
            "codec.decoder.pre_transformer.layers.7.self_attn_layer_scale.scale",
            "codec.decoder.quantizer.rvq_first.vq.layers.0._codebook.codebook",
            "talker.code_predictor.model.layers.4.post_attention_layernorm.weight",
            "codec.encoder.encoder_transformer.layers.0.input_layernorm.weight",
            "codec.encoder.quantizer.acoustic_residual_vector_quantizer.output_proj.weight",
            "codec.encoder.quantizer.semantic_residual_vector_quantizer.output_proj.weight",
            "codec.encoder.quantizer.acoustic_residual_vector_quantizer.layers.30.codebook.codebook",
        ):
            shortened = convert.shorten_name(name, convert.Conversion())
            self.assertLess(len(shortened), convert.GGML_MAX_NAME, name)

    def test_a_name_left_over_the_limit_stops_the_conversion(self) -> None:
        conversion = convert.Conversion()
        with self.assertRaises(convert.ConverterError) as caught:
            convert.shorten_name("codec." + "x" * convert.GGML_MAX_NAME, conversion)
        self.assertIn("GGML truncates", str(caught.exception))

    def test_a_name_already_short_enough_is_untouched(self) -> None:
        conversion = convert.Conversion()
        self.assertEqual(convert.shorten_name("talker.model.norm.weight", conversion),
                         "talker.model.norm.weight")
        self.assertEqual(conversion.renamed, [], "an unchanged name is not a rename")


class SourceDtypeTests(unittest.TestCase):
    """The talker is BF16 and must stay BF16; upcasting it is a silent change."""

    def test_bf16_is_carried_through_as_bf16(self) -> None:
        tensor = torch.tensor([1.5, -2.25, 0.0], dtype=torch.bfloat16)
        array, dtype = convert.numpy_of(tensor)
        self.assertEqual(dtype.name, "BF16")
        self.assertEqual(array.dtype.itemsize, 2)

    def test_bf16_conversion_preserves_the_exact_bit_pattern(self) -> None:
        tensor = torch.randn(64, dtype=torch.float32).to(torch.bfloat16)
        array, _ = convert.numpy_of(tensor)
        expected = tensor.contiguous().view(torch.uint16).numpy()
        self.assertTrue((array == expected).all())

    def test_f32_stays_f32(self) -> None:
        array, dtype = convert.numpy_of(torch.ones(4, dtype=torch.float32))
        self.assertEqual(dtype.name, "F32")

    def test_unhandled_dtypes_are_rejected_rather_than_coerced(self) -> None:
        with self.assertRaises(convert.ConverterError):
            convert.numpy_of(torch.ones(4, dtype=torch.int64))


class ReportShapeTests(unittest.TestCase):
    def test_tensor_report_states_ggml_shape_not_source_shape(self) -> None:
        array, dtype = convert.numpy_of(torch.zeros(2, 3, dtype=torch.float32))
        entry = convert.OutputTensor("x", array, dtype, "src").report()
        self.assertEqual(entry["ggml_shape"], [3, 2])
        self.assertEqual(entry["elements"], 6)
        self.assertIn("sha256", entry)


class VariantDiscriminationTests(unittest.TestCase):
    """The two variants differ by a whole subsystem, not by a label."""

    def test_base_config_declares_the_speaker_encoder(self) -> None:
        profile = convert.variant_profile({
            "tts_model_type": "base",
            "tts_model_size": "0b6",
            "speaker_encoder_config": {"enc_dim": 1024, "sample_rate": 24000},
        })
        self.assertTrue(profile.carries_speaker_encoder)
        self.assertTrue(profile.carries_codec_encoder)

    def test_custom_voice_config_carries_neither(self) -> None:
        profile = convert.variant_profile({"tts_model_type": "custom_voice", "tts_model_size": "0b6"})
        self.assertFalse(profile.carries_speaker_encoder)
        self.assertFalse(profile.carries_codec_encoder)

    def test_base_config_without_a_speaker_encoder_is_refused(self) -> None:
        with self.assertRaises(convert.ConverterError):
            convert.variant_profile({"tts_model_type": "base", "tts_model_size": "0b6"})

    def test_an_unknown_model_type_is_refused(self) -> None:
        with self.assertRaises(convert.ConverterError):
            convert.variant_profile({"tts_model_type": "voice_design", "tts_model_size": "1b7"})


class EncoderCodebookMeasurementTests(unittest.TestCase):
    """Stage 1 measured 16 encoder codebooks as bit-identical to the decoder's.

    Both halves are carried in the package in full regardless of what this
    measures: an alias with no in-package record of where it points is a name
    a consumer cannot resolve, so nothing here is ever removed. The
    measurement is kept as a reported fact -- "we looked and both are
    carried" is a different statement from "we never looked".
    """

    def test_identical_tables_are_recorded_but_both_are_kept(self) -> None:
        shared = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        tensors = {
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.codebook": shared,
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.codebook": shared.clone(),
        }
        conversion = convert.Conversion()
        convert.measure_shared_codebooks(tensors, conversion)

        self.assertEqual(len(tensors), 2, "both tensors must still be present; nothing is removed")
        self.assertEqual(len(conversion.measured_shared_codebooks), 1)
        self.assertEqual(
            conversion.measured_shared_codebooks[0]["encoder_tensor"],
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.codebook",
        )
        self.assertEqual(
            conversion.measured_shared_codebooks[0]["decoder_tensor"],
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.codebook",
        )

    def test_tables_with_no_decoder_match_are_not_recorded(self) -> None:
        """Not every encoder codebook has a same-index decoder counterpart at all.

        This is the case the measurement must not misreport: an encoder table
        with no matching decoder table is neither an error nor a duplicate --
        it is simply carried, unremarked.
        """
        tensors = {
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.codebook":
                torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.codebook":
                torch.ones(4, 3, dtype=torch.float32),
        }
        conversion = convert.Conversion()
        convert.measure_shared_codebooks(tensors, conversion)

        self.assertEqual(len(tensors), 2, "both tensors must still be present; nothing is removed")
        self.assertEqual(conversion.measured_shared_codebooks, [])


class EncoderTensorNameShorteningTests(unittest.TestCase):
    """Carrying the codec's encoder half for Base introduced its own
    overflowing tensor-name patterns, on top of the five CustomVoice already
    had. The four new `NAME_SHORTENINGS` entries this needed only ever fired
    during a real conversion before this class existed -- CI cannot run one,
    so each entry gets a focused test of its own.
    """

    def test_encoder_transformer_is_shortened(self) -> None:
        name = "codec.encoder.encoder_transformer.layers.0.input_layernorm.weight"
        self.assertGreaterEqual(len(name), convert.GGML_MAX_NAME)
        shortened = convert.shorten_name(name, convert.Conversion())
        self.assertEqual(shortened, "codec.encoder.enc_transformer.layers.0.input_layernorm.weight")
        self.assertNotIn("encoder_transformer", shortened)
        self.assertLess(len(shortened), convert.GGML_MAX_NAME)

    def test_acoustic_quantizer_is_shortened_to_rvq(self) -> None:
        name = "codec.encoder.quantizer.acoustic_residual_vector_quantizer.output_proj.weight"
        self.assertGreaterEqual(len(name), convert.GGML_MAX_NAME)
        shortened = convert.shorten_name(name, convert.Conversion())
        self.assertEqual(shortened, "codec.encoder.quantizer.acoustic_rvq.output_proj.weight")
        self.assertNotIn("acoustic_residual_vector_quantizer", shortened)
        self.assertLess(len(shortened), convert.GGML_MAX_NAME)

    def test_semantic_quantizer_is_shortened_to_rvq(self) -> None:
        name = "codec.encoder.quantizer.semantic_residual_vector_quantizer.output_proj.weight"
        self.assertGreaterEqual(len(name), convert.GGML_MAX_NAME)
        shortened = convert.shorten_name(name, convert.Conversion())
        self.assertEqual(shortened, "codec.encoder.quantizer.semantic_rvq.output_proj.weight")
        self.assertNotIn("semantic_residual_vector_quantizer", shortened)
        self.assertLess(len(shortened), convert.GGML_MAX_NAME)

    def test_reconstructed_encoder_codebook_name_is_shortened(self) -> None:
        name = "codec.encoder.quantizer.acoustic_residual_vector_quantizer.layers.30.codebook.codebook"
        self.assertGreaterEqual(len(name), convert.GGML_MAX_NAME)
        shortened = convert.shorten_name(name, convert.Conversion())
        self.assertEqual(shortened, "codec.encoder.quantizer.acoustic_rvq.layers.30.codebook")
        self.assertLess(len(shortened), convert.GGML_MAX_NAME)

    def test_the_new_codebook_rule_does_not_collide_with_the_decoders_underscored_one(self) -> None:
        """`.codebook.codebook` and `._codebook.codebook` must stay two distinct
        rules.

        The decoder's post-reconstruction name keeps the underscore
        (`._codebook.codebook`, from its `_codebook` module); the encoder's has
        none (`.codebook.codebook`, from its `codebook` module). A rule
        careless about that -- for instance one that also matched
        `._codebook.codebook` -- could make two different source tensors
        collapse onto the same output name, which is exactly the truncation
        failure this whole mechanism exists to prevent.
        """
        decoder_name = "codec.decoder.quantizer.rvq_first.vq.layers.0._codebook.codebook"
        encoder_name = "codec.encoder.quantizer.acoustic_residual_vector_quantizer.layers.30.codebook.codebook"
        decoder_shortened = convert.shorten_name(decoder_name, convert.Conversion())
        encoder_shortened = convert.shorten_name(encoder_name, convert.Conversion())

        self.assertNotEqual(decoder_shortened, encoder_shortened)
        self.assertEqual(decoder_shortened, "codec.decoder.quantizer.rvq_first.vq.layers.0.codebook")
        self.assertTrue(encoder_shortened.endswith(".codebook"))
        self.assertLess(len(decoder_shortened), convert.GGML_MAX_NAME)
        self.assertLess(len(encoder_shortened), convert.GGML_MAX_NAME)


if __name__ == "__main__":
    unittest.main()
