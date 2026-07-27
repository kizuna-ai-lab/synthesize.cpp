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

        A rule keyed on one name silently skips the other side's codebooks and
        the converter reports nothing, which is the defect this guards.
        """
        tensors = {**decoder_pair(0), **encoder_pair(0)}
        out = convert.reconstruct_codebooks(tensors, convert.Conversion())
        self.assertIn("decoder.quantizer.rvq_first.vq.layers.0._codebook.codebook", out)
        self.assertIn(
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.codebook",
            out,
        )

    def test_rejects_a_checkpoint_carrying_only_one_convention(self) -> None:
        with self.assertRaises(convert.ConverterError) as caught:
            convert.reconstruct_codebooks(dict(decoder_pair(0)), convert.Conversion())
        self.assertIn("both RVQ naming conventions", str(caught.exception))

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
