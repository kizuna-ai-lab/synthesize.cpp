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
    package still loads, so nothing reports the problem. Five of this
    checkpoint's paths overflowed.
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


if __name__ == "__main__":
    unittest.main()
