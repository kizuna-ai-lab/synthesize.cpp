"""Focused tests for the qwen3-tts converter's silent-failure rules.

Every rule tested here fails *quietly* if it is wrong: the converter still
writes a file, the file still loads, and the audio is wrong or absent. They are
the reason this test file exists rather than a smoke run.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
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
    """The three variants differ by a whole subsystem, not by a label."""

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
        # `voice_design` was this test's example until Stage 3 Plan 1 made it a
        # supported variant. The guard is what matters, not the example, so it
        # moved to a type upstream does not ship.
        with self.assertRaises(convert.ConverterError):
            convert.variant_profile({"tts_model_type": "dialogue", "tts_model_size": "1b7"})

    def test_voice_design_config_carries_neither_encoder(self) -> None:
        profile = convert.variant_profile({"tts_model_type": "voice_design", "tts_model_size": "1b7"})
        self.assertEqual(profile.model_type, "voice_design")
        self.assertFalse(profile.carries_speaker_encoder)
        self.assertFalse(profile.carries_codec_encoder)
        self.assertEqual(profile.size_label, "1.7B")

    def test_voice_design_config_with_a_speaker_encoder_is_refused(self) -> None:
        # The check runs in the direction CustomVoice's does: a voice_design
        # checkpoint that carried an encoder would be a different model than
        # the one this arm was written against.
        with self.assertRaises(convert.ConverterError) as caught:
            convert.variant_profile({
                "tts_model_type": "voice_design",
                "tts_model_size": "1b7",
                "speaker_encoder_config": {"enc_dim": 1024, "sample_rate": 24000},
            })
        self.assertIn("speaker_encoder_config", str(caught.exception))


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

    def test_the_two_layer_scale_rules_are_a_matched_pair(self) -> None:
        """`.self_attn_layer_scale.` and `.mlp_layer_scale.` shorten together.

        Only the first is a length fix: at 66 characters it overflows
        `GGML_MAX_NAME`, while its sibling fits at 60 and would be emitted
        unchanged if its rule did not exist. The `.mlp_layer_scale.` entry is
        there for symmetry -- one long name and one short one beside it reads
        as a mistake -- and `src/arch/qwen3-tts/catalog.cpp` resolves both
        short forms (`self_attn_scale.scale`, `mlp_scale.scale`).

        Asserted because a rule that fires on a name which fits anyway is
        exactly the rule someone later deletes as redundant, and deleting it
        renames a tensor the catalog then cannot find -- with no error from
        either side, since neither length limit is involved.
        """
        attn = "codec.decoder.pre_transformer.layers.0.self_attn_layer_scale.scale"
        mlp = "codec.decoder.pre_transformer.layers.0.mlp_layer_scale.scale"
        self.assertGreaterEqual(len(attn), convert.GGML_MAX_NAME)
        self.assertLess(len(mlp), convert.GGML_MAX_NAME)

        self.assertEqual(
            convert.shorten_name(attn, convert.Conversion()),
            "codec.decoder.pre_transformer.layers.0.self_attn_scale.scale")
        self.assertEqual(
            convert.shorten_name(mlp, convert.Conversion()),
            "codec.decoder.pre_transformer.layers.0.mlp_scale.scale")

    def test_a_name_that_cannot_be_shortened_enough_is_refused(self) -> None:
        """The rule set's floor: emitting a name GGML would truncate is refused.

        A truncated name is not a load error -- the catalog simply never finds
        the tensor -- so this raises rather than warning.
        """
        name = "codec.encoder." + "x" * convert.GGML_MAX_NAME + ".weight"
        with self.assertRaises(convert.ConverterError):
            convert.shorten_name(name, convert.Conversion())


class BaseCatalogTests(unittest.TestCase):
    """A Base package advertises no Voice it can select and says so positively."""

    def test_speaker_metadata_is_refused_when_the_checkpoint_has_no_speakers(self) -> None:
        talker = {"spk_id": {}, "spk_is_dialect": {}}
        with self.assertRaises(convert.ConverterError):
            convert.speaker_catalog(talker, preset_ids=["aiden"])

    def test_an_empty_checkpoint_and_an_empty_manifest_agree(self) -> None:
        names, token_ids, dialects = convert.speaker_catalog(
            {"spk_id": {}, "spk_is_dialect": {}}, preset_ids=[])
        self.assertEqual((names, token_ids, dialects), ([], [], []))

    def test_the_mel_front_end_parameters_are_the_ones_the_reference_uses(self) -> None:
        params = convert.speaker_encoder_metadata({"enc_dim": 1024, "sample_rate": 24000})
        self.assertEqual(params["mel_bins"], 128)
        self.assertEqual(params["n_fft"], 1024)
        self.assertEqual(params["hop_length"], 256)
        self.assertEqual(params["win_length"], 1024)
        self.assertEqual(params["fmin"], 0.0)
        self.assertEqual(params["fmax"], 12000.0)


def golden_manifest(variant: str) -> dict:
    path = REPO_ROOT / "tests" / "golden" / "qwen3-tts" / f"qwen3-tts-12hz-0-6b-{variant}.manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


class LicenseLinkTests(unittest.TestCase):
    """Each variant's package must link its own Hugging Face model page.

    CustomVoice's manifest carries a GitHub URL as `source.repository` (where
    the code lives); the license link must come from the checkpoint's own
    pinned locator instead, or every package would credit whichever variant's
    repository field happened to be a Hugging Face URL.
    """

    def test_base_links_its_own_repository(self) -> None:
        link = convert.license_link_for(golden_manifest("base"))
        self.assertEqual(link, "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base")

    def test_customvoice_links_its_own_repository_not_the_github_source(self) -> None:
        link = convert.license_link_for(golden_manifest("customvoice"))
        self.assertEqual(link, "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

    def test_the_talker_checkpoint_is_matched_unambiguously_not_by_list_order(self) -> None:
        """Both manifests carry a second "checkpoint"-role artifact --
        `speech_tokenizer/model.safetensors` -- whose locator also contains
        the substring "model.safetensors". Putting it first must not change
        the answer: `talker_checkpoint_locator` matches on the talker's own
        path shape (no `speech_tokenizer/` segment), not on being listed first.
        """
        manifest = golden_manifest("base")
        artifacts = manifest["source"]["artifacts"]
        checkpoint_artifacts = [a for a in artifacts if a["role"] == "checkpoint"]
        self.assertEqual(len(checkpoint_artifacts), 2, "fixture assumption: two checkpoint artifacts")
        reordered = dict(manifest)
        reordered["source"] = dict(manifest["source"])
        reordered["source"]["artifacts"] = list(reversed(artifacts))
        self.assertEqual(
            convert.talker_checkpoint_locator(reordered),
            convert.talker_checkpoint_locator(manifest),
        )

    def test_an_ambiguous_manifest_is_refused(self) -> None:
        manifest = {"source": {"artifacts": [
            {"role": "checkpoint", "locator": "https://huggingface.co/x/y/resolve/rev/model.safetensors"},
            {"role": "checkpoint", "locator": "https://huggingface.co/x/z/resolve/rev/model.safetensors"},
        ]}}
        with self.assertRaises(convert.ConverterError):
            convert.talker_checkpoint_locator(manifest)

    def test_a_manifest_with_no_talker_checkpoint_is_refused(self) -> None:
        manifest = {"source": {"artifacts": [
            {"role": "checkpoint", "locator": "https://huggingface.co/x/y/resolve/rev/speech_tokenizer/model.safetensors"},
        ]}}
        with self.assertRaises(convert.ConverterError):
            convert.talker_checkpoint_locator(manifest)


class CompatibilityIdTests(unittest.TestCase):
    """Known-value vector from the real Base package's own digests.

    Pins the exact formula and input order -- schema/version as one hashed
    piece, then talker, codec, config digests in that order -- against the
    id the shipped Base package actually carries today, so a change to any
    of those (reordering, rejoining, hashing a different representation)
    fails this test loudly instead of silently changing what compatibility
    means for an already-shipped package.
    """

    def test_matches_the_id_the_base_package_ships_today(self) -> None:
        result = convert.compatibility_id(
            "qwen3-tts-voice-clone", 1,
            (
                "180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6",  # talker
                "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",  # codec
                "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011",  # config
            ),
        )
        self.assertEqual(result, "34d4de22a329b6bc8347cb952b6fa16513320012628598ab59743679cc16806e")

    def test_digest_order_matters(self) -> None:
        """Swapping two digests must not land on the same id by coincidence."""
        forward = convert.compatibility_id("s", 1, ("a" * 64, "b" * 64, "c" * 64))
        swapped = convert.compatibility_id("s", 1, ("b" * 64, "a" * 64, "c" * 64))
        self.assertNotEqual(forward, swapped)

    def test_schema_and_version_are_both_part_of_the_id(self) -> None:
        base = convert.compatibility_id("s", 1, ("a" * 64,))
        self.assertNotEqual(base, convert.compatibility_id("t", 1, ("a" * 64,)))
        self.assertNotEqual(base, convert.compatibility_id("s", 2, ("a" * 64,)))


class ProfileMetadataEmissionTests(unittest.TestCase):
    """The Voice Profile contract is emitted for a variant with a speaker
    encoder and withheld from one without -- checked against a real written
    and re-read GGUF, not against `add_metadata`'s source code, since a KV
    that never makes it to the file is exactly the kind of failure that is
    silent everywhere except here.
    """

    @staticmethod
    def _minimal_add_metadata_args(carries_speaker_encoder: bool, model_type: str = "test") -> tuple:
        """The smallest fixture `add_metadata` accepts without raising.

        Field values are arbitrary except where a converter rule constrains
        them (e.g. the codec hop/frame-rate/quantizer-count cross-checks).
        `model_type` defaults to a value no real variant uses, matching the
        pre-existing behaviour of every caller that does not pass it; tests
        that need `profile_source_names`/`profile_schema_name` to resolve as
        a real variant would (`base`, `voice_design`, `custom_voice`) pass it
        explicitly, alongside the matching `carries_speaker_encoder`.
        """
        manifest = {
            "variant": "qwen3-tts-test-variant",
            "source": {
                "repository": "https://huggingface.co/Test/Repo",
                "revision": "deadbeef",
                "artifacts": [
                    {"role": "checkpoint",
                     "locator": "https://huggingface.co/Test/Repo/resolve/deadbeef/model.safetensors"},
                ],
            },
            "package_contract": {
                "language_tags": ["auto", "english"],
                "voices": {
                    "mode": "profile-sources" if carries_speaker_encoder else "preset-catalog",
                    "default_id": None,
                    "preset_ids": [] if carries_speaker_encoder else ["voice1"],
                },
                "native_audio": {"sample_rate_hz": 24000, "channels": 1, "sample_format": "f32le"},
                "speaking_rate_range": [1.0, 1.0],
                "max_input_tokens": 1024,
                "max_output_frames": 100,
            },
        }
        if carries_speaker_encoder:
            manifest["package_contract"]["profile"] = {
                "reference": {
                    "target_sample_rate": 24000,
                    "target_channels": 1,
                    "min_frames_per_clip": 24000,
                    "max_frames_per_clip": 720000,
                    "max_total_frames": 720000,
                    "max_reference_count": 1,
                }
            }

        talker_config = {
            "num_hidden_layers": 1, "hidden_size": 4, "num_attention_heads": 1,
            "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 4,
            "vocab_size": 10, "text_vocab_size": 10, "text_hidden_size": 4,
            "num_code_groups": 1, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "code_predictor_config": {
                "num_hidden_layers": 1, "hidden_size": 4, "num_attention_heads": 1,
                "num_key_value_heads": 1, "head_dim": 4, "vocab_size": 10, "num_code_groups": 1,
                # Deliberately DIFFERENT from talker_config["intermediate_size"]
                # (4) above: this is the value the Task 6 architecture-gap fix
                # is about (docs/porting/families/qwen3-tts.md, "A genuine
                # architecture gap, found and closed, not converted around"),
                # and equal fixture values would let a regression that quietly
                # re-inherited the talker's own number pass unnoticed.
                "intermediate_size": 8,
            },
            "spk_id": {} if carries_speaker_encoder else {"voice1": 0},
            "spk_is_dialect": {} if carries_speaker_encoder else {"voice1": ""},
            "codec_language_id": {"english": 0},
            "codec_bos_id": 7, "codec_eos_token_id": 8, "codec_pad_id": 9,
            "codec_think_id": 10, "codec_nothink_id": 11,
            "codec_think_bos_id": 12, "codec_think_eos_id": 13,
        }
        config = {
            "talker_config": talker_config,
            "tts_bos_token_id": 1, "tts_eos_token_id": 2, "tts_pad_token_id": 3,
            "im_start_token_id": 4, "im_end_token_id": 5, "assistant_token_id": 6,
        }
        if carries_speaker_encoder:
            config["speaker_encoder_config"] = {"enc_dim": 1024, "sample_rate": 24000}

        codec_config = {
            "decode_upsample_rate": convert.SAMPLES_PER_FRAME,
            "input_sample_rate": int(convert.SAMPLES_PER_FRAME * convert.FRAME_RATE_HZ),
            "decoder_config": {
                "upsample_rates": [convert.SAMPLES_PER_FRAME], "upsampling_ratios": [1],
                "latent_dim": 4, "decoder_dim": 4, "codebook_dim": 2, "codebook_size": 16,
                "num_quantizers": 1, "num_semantic_quantizers": 0,
                "hidden_size": 4, "intermediate_size": 4, "num_hidden_layers": 1,
                "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 4,
                "sliding_window": 4, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            },
        }

        generation_config = {
            "do_sample": True, "temperature": 1.0, "top_k": 50, "top_p": 0.9,
            "repetition_penalty": 1.1, "subtalker_temperature": 1.0,
            "subtalker_top_k": 50, "subtalker_top_p": 0.9,
        }

        digests = {"talker": "1" * 64, "codec": "2" * 64, "config": "3" * 64, "generation_config": "4" * 64}
        profile = convert.VariantProfile(
            model_type, carries_speaker_encoder, carries_speaker_encoder, "Test Display Name", "0.0B")
        return (manifest, config, {}, codec_config, generation_config, {"a": 0, "b": 1}, [], digests, profile)

    def _written_metadata(self, carries_speaker_encoder: bool, model_type: str = "test") -> dict:
        """Every KV of a real written-and-re-read GGUF, as plain Python values.

        Values, not just key names: an emission that writes the right keys
        with the wrong contents (a catalog ordered by token id against names
        ordered alphabetically, say) is silent everywhere else.
        """
        args = self._minimal_add_metadata_args(carries_speaker_encoder, model_type)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.gguf"
            writer = convert.GGUFWriter(str(path), convert.ARCH_KEY)
            convert.add_metadata(writer, *args)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            reader = convert.GGUFReader(str(path))
            # Materialized inside the block: the reader memory-maps the file,
            # which the TemporaryDirectory removes on exit.
            return {name: field.contents() for name, field in reader.fields.items()}

    def _written_keys(self, carries_speaker_encoder: bool, model_type: str = "test") -> set[str]:
        return set(self._written_metadata(carries_speaker_encoder, model_type))

    def test_a_variant_with_a_speaker_encoder_carries_the_profile_block(self) -> None:
        keys = self._written_keys(carries_speaker_encoder=True)
        for key in (
            "synthesize.profile.schema",
            "synthesize.profile.schema_version",
            "synthesize.profile.compatibility_id",
            "synthesize.reference.target_sample_rate",
            "synthesize.reference.target_channels",
            "synthesize.reference.min_frames_per_clip",
            "synthesize.reference.max_frames_per_clip",
            "synthesize.reference.max_total_frames",
            "synthesize.reference.max_reference_count",
            "synthesize.qwen3-tts.speaker_encoder.enc_dim",
            "synthesize.qwen3-tts.speaker_encoder.sample_rate",
            "synthesize.qwen3-tts.speaker_encoder.mel_bins",
            "synthesize.qwen3-tts.speaker_encoder.n_fft",
            "synthesize.qwen3-tts.speaker_encoder.hop_length",
            "synthesize.qwen3-tts.speaker_encoder.win_length",
            "synthesize.qwen3-tts.speaker_encoder.fmin",
            "synthesize.qwen3-tts.speaker_encoder.fmax",
        ):
            self.assertIn(key, keys, key)

    # The check above pins only that the three `synthesize.profile.*` keys
    # REACH the file. Their values went unasserted until 2026-08-19: a
    # converter that wrote the wrong schema name, a schema_version of 0, or an
    # id hashed over the generation_config digest instead of the config one
    # passed every test in this file. The id is the load-bearing one --
    # src/arch/qwen3-tts/profile.cpp:1361 refuses a serialized Profile whose
    # stored id differs from the model's with SYNTH_ERR_UNSUPPORTED_VOICE, so
    # a wrong id here is not a cosmetic mismatch: it is a Profile that stops
    # loading against the very package it was produced for.
    def test_the_profile_block_carries_the_variants_own_schema_and_id(self) -> None:
        metadata = self._written_metadata(carries_speaker_encoder=True, model_type="base")
        self.assertEqual(metadata["synthesize.profile.schema"], "qwen3-tts-voice-clone")
        self.assertEqual(metadata["synthesize.profile.schema_version"], 1)
        # A known value over the fixture's own digests ("1"*64, "2"*64, "3"*64),
        # deliberately NOT a re-call of `compatibility_id`: recomputing with the
        # same helper the emission uses would still pass if `add_metadata` fed
        # it the wrong pieces, which is the regression this test exists for.
        # `CompatibilityIdTests` pins the formula itself, against the id the
        # shipped Base package really carries.
        self.assertEqual(
            metadata["synthesize.profile.compatibility_id"],
            "bbeb77755504bcba298f656b3b8a221957e5f21e03a72d171689e7711a531741",
        )

    def test_the_schema_name_is_hashed_into_the_compatibility_id(self) -> None:
        """Two variants sharing every digest must not share an id.

        The schema name is one of the hashed pieces, so VoiceDesign and the
        clone schema diverge here even though this fixture hands both the
        identical talker/codec/config digests. Were the schema dropped from
        the formula, a Description Text Profile would satisfy
        profile.cpp:1361 against a clone package built from the same weights.
        """
        design = self._written_metadata(carries_speaker_encoder=False, model_type="voice_design")
        self.assertEqual(design["synthesize.profile.schema"], "qwen3-tts-voice-design")
        self.assertEqual(design["synthesize.profile.schema_version"], 1)
        self.assertEqual(
            design["synthesize.profile.compatibility_id"],
            "49c4cf86b4794be7400ada07c5a3af876563319f6dbe597f15ae086b3c20fd38",
        )
        clone = self._written_metadata(carries_speaker_encoder=True, model_type="base")
        self.assertNotEqual(
            design["synthesize.profile.compatibility_id"],
            clone["synthesize.profile.compatibility_id"],
        )

    def test_a_variant_without_a_speaker_encoder_carries_none_of_it(self) -> None:
        keys = self._written_keys(carries_speaker_encoder=False)
        leaked = [
            key for key in keys
            if key.startswith("synthesize.profile.")
            or key.startswith("synthesize.reference.")
            or key.startswith("synthesize.qwen3-tts.speaker_encoder.")
        ]
        self.assertEqual(leaked, [])

    # The Preset Voice Catalog is the other half of the same conditional, and
    # the half nothing checked: `add_metadata` emits the three
    # `synthesize.qwen3-tts.speakers.*` arrays under `if speaker_names:`,
    # where the published CustomVoice path used to be straight-line code.
    # Replacing that condition with `if False:` -- which drops the whole
    # catalog from a CustomVoice package -- left the entire Python suite
    # green, because every case here only asserted what a Base package does
    # NOT carry. The published variant's own emission is what these two
    # assert, from opposite directions.
    def test_a_preset_catalog_variant_carries_its_speaker_catalog(self) -> None:
        metadata = self._written_metadata(carries_speaker_encoder=False)
        self.assertEqual(metadata["synthesize.voice.mode"], "preset-catalog")
        self.assertEqual(metadata["synthesize.voice.preset_count"], 1)
        # One speaker, and the three arrays are one catalog read index by
        # index: same length, and each entry's own value.
        self.assertEqual(metadata["synthesize.qwen3-tts.speakers.names"], ["voice1"])
        self.assertEqual(metadata["synthesize.qwen3-tts.speakers.token_ids"], [0])
        self.assertEqual(metadata["synthesize.qwen3-tts.speakers.dialect_override"], [""])

    def test_a_profile_sources_variant_carries_no_speaker_catalog(self) -> None:
        metadata = self._written_metadata(carries_speaker_encoder=True)
        self.assertEqual(metadata["synthesize.voice.mode"], "profile-sources")
        self.assertEqual(metadata["synthesize.voice.preset_count"], 0)
        # Absent, not present-and-empty: an empty
        # `synthesize.qwen3-tts.speakers.*` array would be a Catalog that
        # exists and selects nothing, which is not what this variant declares
        # (src/arch/qwen3-tts/weights.cpp's read_voices reads the mode, not
        # the arrays' length).
        for key in (
            "synthesize.qwen3-tts.speakers.names",
            "synthesize.qwen3-tts.speakers.token_ids",
            "synthesize.qwen3-tts.speakers.dialect_override",
        ):
            self.assertNotIn(key, metadata, key)

    # `ProfileSourceDeclarationTests` below checks `profile_source_names` and
    # `profile_schema_name` in isolation, against bare `VariantProfile`
    # objects; it never calls `add_metadata`. Neither does anything else in
    # this class check the `synthesize.voice.profile_sources` key specifically
    # -- the two tests above assert only the `synthesize.profile.*`,
    # `synthesize.reference.*` and `synthesize.qwen3-tts.speaker_encoder.*`
    # prefixes, which is a different KV. So a regression that dropped the
    # `writer.add_array("synthesize.voice.profile_sources", ...)` call, or
    # misspelled the key, would pass every other test in this file. These
    # three close that gap, one per real variant, checking the written VALUE
    # and not just presence -- a key present with the wrong contents is the
    # failure mode a presence-only check cannot see.
    def test_base_writes_profile_sources_as_reference_audio(self) -> None:
        metadata = self._written_metadata(carries_speaker_encoder=True, model_type="base")
        self.assertEqual(metadata["synthesize.voice.profile_sources"], ["reference-audio"])

    def test_voice_design_writes_profile_sources_as_description_text_and_no_reference_contract(self) -> None:
        metadata = self._written_metadata(carries_speaker_encoder=False, model_type="voice_design")
        self.assertEqual(metadata["synthesize.voice.profile_sources"], ["description-text"])
        # VoiceDesign can serialize a Profile (the schema/compatibility_id
        # keys are present) but takes no reference audio and carries no
        # speaker encoder, so neither block belongs in its package.
        leaked = [
            key for key in metadata
            if key.startswith("synthesize.reference.")
            or key.startswith("synthesize.qwen3-tts.speaker_encoder.")
        ]
        self.assertEqual(leaked, [])

    def test_custom_voice_writes_no_profile_sources_and_no_schema(self) -> None:
        metadata = self._written_metadata(carries_speaker_encoder=False, model_type="custom_voice")
        # A preset-catalog package prepares nothing, so it carries no
        # contract at all -- not an empty one, which is a different claim.
        self.assertNotIn("synthesize.voice.profile_sources", metadata)
        self.assertNotIn("synthesize.profile.schema", metadata)

    # The Task 6 architecture-gap fix (docs/porting/families/qwen3-tts.md, "A
    # genuine architecture gap, found and closed, not converted around") rests
    # on `synthesize.qwen3-tts.code_predictor.intermediate_size` reaching the
    # package -- without it, weights.cpp falls back to the talker's own
    # intermediate_size, which is silently wrong at 1.7B (2048-wide talker,
    # 1024-wide code predictor). Nothing before this test asserted the KV
    # reaches a real written-and-re-read GGUF at all: deleting the emitting
    # tuple entry in `add_metadata` (scripts/convert-qwen3-tts.py) left all 55
    # converter tests and all 342 Python tests green.
    def test_code_predictor_intermediate_size_is_emitted_from_its_own_config(self) -> None:
        metadata = self._written_metadata(carries_speaker_encoder=True)
        self.assertIn("synthesize.qwen3-tts.code_predictor.intermediate_size", metadata)
        # 8, not the talker's 4 (see the fixture comment above): proves the
        # value is read from code_predictor_config, not inherited from
        # talker_config the way the pre-Task-6 catalog did.
        self.assertEqual(metadata["synthesize.qwen3-tts.code_predictor.intermediate_size"], 8)


class ProfileSourceDeclarationTests(unittest.TestCase):
    """A package declares which Profile sources it implements, positively.

    Before Stage 3 the runtime inferred this from the Voice Mode, which worked
    only while `profile-sources` meant exactly one variant. It now means two,
    whose sources differ, so the package has to say.
    """

    def test_base_declares_reference_audio(self) -> None:
        profile = convert.variant_profile({
            "tts_model_type": "base",
            "tts_model_size": "0b6",
            "speaker_encoder_config": {"enc_dim": 1024, "sample_rate": 24000},
        })
        self.assertEqual(convert.profile_source_names(profile), ["reference-audio"])
        self.assertEqual(convert.profile_schema_name(profile), "qwen3-tts-voice-clone")

    def test_voice_design_declares_description_text(self) -> None:
        profile = convert.variant_profile({"tts_model_type": "voice_design", "tts_model_size": "1b7"})
        self.assertEqual(convert.profile_source_names(profile), ["description-text"])
        self.assertEqual(convert.profile_schema_name(profile), "qwen3-tts-voice-design")

    def test_custom_voice_declares_none(self) -> None:
        # A preset-catalog package prepares nothing, so it carries no contract
        # at all -- not an empty one, which is a different claim.
        profile = convert.variant_profile({"tts_model_type": "custom_voice", "tts_model_size": "0b6"})
        self.assertEqual(convert.profile_source_names(profile), [])
        self.assertIsNone(convert.profile_schema_name(profile))


class VariantKindAgreementTests(unittest.TestCase):
    """The variant string and the checkpoint's `tts_model_type` have to agree.

    They are two independent statements about one package, and they reach the
    writer from two different origins: `synthesize.model_variant` is copied from
    the intake manifest, `synthesize.voice.profile_sources` is derived from
    `tts_model_type`. Nothing made them agree, so a manifest naming the wrong
    variant produced a package that called itself one thing and declared the
    Voice Profile sources of another. What that is is a MISLABELLED package, not
    an unimplemented one -- neither statement is the implementation, and
    Description Text has no tensor footprint that could corroborate either.
    """

    # `custom_voice` and `voice_design` both carry neither encoder; only `base`
    # carries a speaker encoder. Matching the real `variant_profile` output
    # matters because the profile is what the check reads `model_type` off.
    @staticmethod
    def _profile(model_type: str) -> "convert.VariantProfile":
        carries = model_type == "base"
        return convert.VariantProfile(model_type, carries, carries, "Display Name", "0.0B")

    COMMITTED = (
        ("qwen3-tts-12hz-0-6b-base", "base"),
        ("qwen3-tts-12hz-0-6b-customvoice", "custom_voice"),
        ("qwen3-tts-12hz-1-7b-voicedesign", "voice_design"),
    )

    def test_each_committed_variant_agrees_with_its_own_model_type(self) -> None:
        for variant, model_type in self.COMMITTED:
            with self.subTest(variant=variant):
                convert.check_variant_kind(variant, self._profile(model_type))

    def test_a_variant_naming_another_kind_is_refused(self) -> None:
        """Every cross pair among the three real kinds, both directions."""
        for variant, own_type in self.COMMITTED:
            for _, other_type in self.COMMITTED:
                if other_type == own_type:
                    continue
                with self.subTest(variant=variant, model_type=other_type):
                    with self.assertRaises(convert.ConverterError):
                        convert.check_variant_kind(variant, self._profile(other_type))

    def test_a_future_size_under_a_known_kind_is_accepted(self) -> None:
        """Reading the kind SEGMENT, not the whole string, is what buys this.

        A table of whole variant strings would refuse all three of these for no
        reason other than not having been updated to name them.
        """
        for variant, model_type in (
            ("qwen3-tts-24hz-3b-base", "base"),
            ("qwen3-tts-24hz-3b-customvoice", "custom_voice"),
            ("qwen3-tts-24hz-3b-voicedesign", "voice_design"),
        ):
            with self.subTest(variant=variant):
                convert.check_variant_kind(variant, self._profile(model_type))

    def test_an_unrecognized_kind_is_governed_by_nothing(self) -> None:
        """Known-value, not an enumeration lock: a kind this build has never
        heard of passes whatever `model_type` it arrives with. A new kind is a
        Voice Mode plus a source, so it has to be added here deliberately."""
        for variant in ("qwen3-tts-24hz-3b-dialogue", "qwen3-tts-test-variant", "synthetic"):
            for model_type in ("base", "custom_voice", "voice_design"):
                with self.subTest(variant=variant, model_type=model_type):
                    convert.check_variant_kind(variant, self._profile(model_type))

    # The two below drive `add_metadata` against a real GGUFWriter rather than
    # calling the check directly: the point of putting the guard at the top of
    # `add_metadata` is that the one path to a written package cannot route
    # around it, and only exercising that path proves it.
    @staticmethod
    def _args_with_variant(variant: str, model_type: str) -> tuple:
        args = ProfileMetadataEmissionTests._minimal_add_metadata_args(
            carries_speaker_encoder=(model_type == "base"), model_type=model_type)
        args[0]["variant"] = variant
        return args

    def test_the_write_path_refuses_a_mismatched_variant(self) -> None:
        args = self._args_with_variant("qwen3-tts-12hz-1-7b-voicedesign", "custom_voice")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.gguf"
            writer = convert.GGUFWriter(str(path), convert.ARCH_KEY)
            with self.assertRaises(convert.ConverterError):
                convert.add_metadata(writer, *args)
            # Refused before anything reached the file, not after.
            self.assertFalse(path.exists())

    def test_the_write_path_carries_an_agreeing_variant_through(self) -> None:
        args = self._args_with_variant("qwen3-tts-12hz-0-6b-customvoice", "custom_voice")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.gguf"
            writer = convert.GGUFWriter(str(path), convert.ARCH_KEY)
            convert.add_metadata(writer, *args)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            reader = convert.GGUFReader(str(path))
            # Materialized inside the block: the reader memory-maps the file,
            # which the TemporaryDirectory removes on exit.
            metadata = {name: field.contents() for name, field in reader.fields.items()}
        self.assertEqual(metadata["synthesize.model_variant"], "qwen3-tts-12hz-0-6b-customvoice")
        self.assertNotIn("synthesize.voice.profile_sources", metadata)


if __name__ == "__main__":
    unittest.main()
