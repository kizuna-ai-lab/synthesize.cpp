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
    def _minimal_add_metadata_args(carries_speaker_encoder: bool) -> tuple:
        """The smallest fixture `add_metadata` accepts without raising.

        Field values are arbitrary except where a converter rule constrains
        them (e.g. the codec hop/frame-rate/quantizer-count cross-checks).
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
            "test", carries_speaker_encoder, carries_speaker_encoder, "Test Display Name", "0.0B")
        return (manifest, config, {}, codec_config, generation_config, {"a": 0, "b": 1}, [], digests, profile)

    def _written_metadata(self, carries_speaker_encoder: bool) -> dict:
        """Every KV of a real written-and-re-read GGUF, as plain Python values.

        Values, not just key names: an emission that writes the right keys
        with the wrong contents (a catalog ordered by token id against names
        ordered alphabetically, say) is silent everywhere else.
        """
        args = self._minimal_add_metadata_args(carries_speaker_encoder)
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

    def _written_keys(self, carries_speaker_encoder: bool) -> set[str]:
        return set(self._written_metadata(carries_speaker_encoder))

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


if __name__ == "__main__":
    unittest.main()
