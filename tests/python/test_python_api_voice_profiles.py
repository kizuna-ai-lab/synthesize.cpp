import array
import importlib
import sys
import types
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = REPO / "bindings/python/src/synthesize_cpp"


def unsupported_profile_capabilities():
    return {
        "source_flags": 0,
        "reference_transcript": 0,
        "reference_language": 0,
        "description_language": 0,
        "max_reference_count": 0,
        "profile_schema": None,
        "profile_schema_version": 0,
        "profile_compatibility_id": bytes(32),
        "reference_target_sample_rate": 0,
        "reference_target_channel_count": 0,
        "min_reference_frames_per_clip": 0,
        "max_reference_frames_per_clip": 0,
        "max_reference_total_frames": 0,
    }


class FakeNative(types.ModuleType):
    def __init__(self):
        super().__init__("synthesize_cpp._native")
        self.calls = []
        self.next_handle = 0

    def model_load(self, path, backend, device_index, diagnostics):
        self.next_handle += 1
        return ("model", self.next_handle)

    def model_metadata(self, handle):
        return {
            "device": {
                "index": None, "name": "CPU", "description": "CPU",
                "kind": "cpu", "device_id": None, "memory_total": 1,
                "memory_free": 1, "device_type": 0, "flags": 1,
            },
            "capabilities": {
                "input_flags": 4, "capability_flags": 0,
                "output_sample_rate": 22050, "output_channel_count": 1,
                "min_speaking_rate": 1.0, "max_speaking_rate": 1.0,
                "max_input_tokens": 512, "max_output_frames": 1000,
            },
            "preset_voices": (),
            "languages": (),
            "voice_profile_capabilities": unsupported_profile_capabilities(),
        }

    def model_close(self, handle):
        self.calls.append(("model_close", handle))

    def context_create(self, model):
        self.next_handle += 1
        return ("context", self.next_handle)

    def context_close(self, context):
        self.calls.append(("context_close", context))

    def synthesize(self, *arguments):
        self.calls.append(("synthesize",) + arguments)
        return {
            "audio_owner": array.array("f", [0.0]),
            "frame_count": 1,
            "sample_rate": 22050,
            "channel_count": 1,
            "frames_emitted": 1,
            "actual_seed": 0,
            "result_flags": 0,
            "resolved_language_tag": None,
            "resolved_voice_id": None,
        }

    def voice_profile_create_from_reference(self, model, references, diagnostics):
        self.calls.append(("reference", model, references, diagnostics))
        self.next_handle += 1
        return ("profile", self.next_handle)

    def voice_profile_create_from_description(
        self, model, description, language, seed, diagnostics
    ):
        self.calls.append(
            ("description", model, description, language, seed, diagnostics)
        )
        self.next_handle += 1
        return ("profile", self.next_handle)

    def voice_profile_create_random(self, model, seed, diagnostics):
        self.calls.append(("random", model, seed, diagnostics))
        self.next_handle += 1
        return ("profile", self.next_handle)

    def voice_profile_load(self, model, data, diagnostics):
        self.calls.append(("load", model, data, diagnostics))
        self.next_handle += 1
        return ("profile", self.next_handle)

    def voice_profile_serialize(self, profile, diagnostics):
        self.calls.append(("serialize", profile, diagnostics))
        return bytes((1, 2, 3, 4))

    def voice_profile_close(self, profile):
        self.calls.append(("profile_close", profile))


class PythonApiVoiceProfileTests(unittest.TestCase):
    def tearDown(self):
        for name in tuple(sys.modules):
            if name == "synthesize_cpp" or name.startswith("synthesize_cpp."):
                sys.modules.pop(name, None)

    def load_models_module(self):
        package = types.ModuleType("synthesize_cpp")
        package.__path__ = [str(PACKAGE_SOURCE)]
        package.__package__ = "synthesize_cpp"
        sys.modules["synthesize_cpp"] = package
        native = FakeNative()
        sys.modules[native.__name__] = native
        return importlib.import_module("synthesize_cpp.models"), native

    def test_reference_pcm_is_borrowed_without_adapter_copy(self):
        module, native = self.load_models_module()
        voices = importlib.import_module("synthesize_cpp.voice_profiles")
        pcm = array.array("f", [0.0, 0.25, -0.25, 0.5])
        reference = voices.VoiceReference(
            pcm, sample_rate=16000, channel_count=1,
            transcript="hello", language="en",
        )
        model = module.Model("voice.gguf")
        profile = model.create_voice_profile_from_reference([reference])

        call = native.calls[0]
        self.assertEqual(call[0], "reference")
        self.assertIs(call[2][0][0], pcm)
        self.assertEqual(call[2][0][1:], (16000, 1, "hello", "en"))
        self.assertIs(profile.model, model)
        profile.close()
        profile.close()
        model.close()
        self.assertEqual(
            [item[0] for item in native.calls],
            ["reference", "profile_close", "model_close"],
        )

    def test_all_sources_and_zero_copy_serialized_buffer(self):
        module, native = self.load_models_module()
        model = module.Model("voice.gguf")
        description = model.create_voice_profile_from_description(
            "warm narrator", language="en", seed=7
        )
        random = model.create_random_voice_profile(seed=8)
        serialized = bytearray((9, 8, 7))
        loaded = model.load_voice_profile(serialized)

        view = description.serialize()
        self.assertTrue(view.readonly)
        self.assertEqual(view.tolist(), [1, 2, 3, 4])
        self.assertEqual(native.calls[0][0:5], (
            "description", ("model", 1), "warm narrator", "en", 7
        ))
        self.assertEqual(native.calls[1][0:3], ("random", ("model", 1), 8))
        self.assertIs(native.calls[2][2], serialized)

        model.close()
        self.assertIs(description.model, model)
        description.close()
        random.close()
        loaded.close()

    def test_source_validation(self):
        module, _ = self.load_models_module()
        voices = importlib.import_module("synthesize_cpp.voice_profiles")
        model = module.Model("voice.gguf")
        with self.assertRaisesRegex(ValueError, "at least one"):
            model.create_voice_profile_from_reference([])
        with self.assertRaisesRegex(TypeError, "VoiceReference"):
            model.create_voice_profile_from_reference([object()])
        with self.assertRaisesRegex(TypeError, "description must be str"):
            model.create_voice_profile_from_description(b"warm narrator")
        with self.assertRaisesRegex(ValueError, "concrete"):
            model.create_random_voice_profile(seed=2**64 - 1)
        with self.assertRaisesRegex(ValueError, "channel_count"):
            voices.VoiceReference(
                array.array("f", [0.0]), sample_rate=16000, channel_count=3
            )
        model.close()

    # Until 2026-08-19 test_source_validation above asserted a ValueError on an
    # empty description, pinning a hardcoded check in models.py. That check was
    # wrong: whether an empty description is legal is a per-Model-Variant
    # semantic rule the Adapter cannot know (docs/c-interface.md's v1
    # Description Text section -- non-empty is the DEFAULT, and Qwen3-TTS
    # VoiceDesign accepts an empty one as its unconditioned path), so the
    # Adapter made the C layer's own answer unreachable in BOTH directions:
    # a variant that accepts empty could not be reached, and a variant that
    # refuses it (OmniVoice) could not report its own refusal.
    #
    # THE ASSERTION IS RE-SITED, NOT DELETED, and it had to be: this file
    # drives a FakeNative stub, so nothing here reaches the C layer and no
    # test in this file can assert what a real variant does with "". What is
    # testable here is the Adapter's two halves of that contract, and both are
    # pinned below -- the empty string reaches the native entry point verbatim
    # (so the C layer gets to decide at all), and a native refusal is surfaced
    # rather than swallowed. The C layer's own decision is pinned where it can
    # be: tests/qwen3_tts_design_profile_test.cpp's
    # test_create_from_description_accepts_both_empty_description_spellings
    # for the accepting variant, and the OmniVoice arm's own refusal tests for
    # the refusing one.
    def test_empty_description_reaches_native_and_a_native_refusal_is_reported(self):
        module, native = self.load_models_module()
        model = module.Model("voice.gguf")

        # Half one: pass-through. The Adapter does not gate on emptiness, so
        # the C layer receives "" verbatim and decides for its own variant.
        profile = model.create_voice_profile_from_description("")
        self.assertEqual(native.calls[0][0:5], ("description", ("model", 1), "", None, 0))
        profile.close()

        # Half two: error mapping. A variant whose C arm refuses an empty
        # description returns SYNTH_ERR_INVALID_ARG, which
        # bindings/python/src/native_loader.c's synth_set_status_error turns
        # into a RuntimeError -- the exception type asserted here. The Adapter
        # must re-raise it as SynthesizeError with its own context, not let the
        # bare RuntimeError escape and not convert it to ValueError.
        def refuse(*arguments):
            raise RuntimeError(
                "synth_voice_profile_create_from_description failed: invalid argument (-2)"
            )

        native.voice_profile_create_from_description = refuse
        with self.assertRaises(module.SynthesizeError) as raised:
            model.create_voice_profile_from_description("")
        self.assertIn("could not prepare Voice Profile from Description Text", str(raised.exception))
        self.assertIn("invalid argument", str(raised.exception))
        model.close()

    def test_synthesis_selects_one_model_bound_profile(self):
        module, native = self.load_models_module()
        first_model = module.Model("first.gguf")
        second_model = module.Model("second.gguf")
        profile = first_model.create_random_voice_profile(seed=3)
        other_profile = second_model.create_random_voice_profile(seed=4)
        context = first_model.create_context()

        context.synthesize_tokens([1], voice_profile=profile)
        synthesis_call = next(
            call for call in native.calls if call[0] == "synthesize"
        )
        self.assertEqual(synthesis_call[-1], ("profile", 3))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            context.synthesize_tokens(
                [1], voice="alice", voice_profile=profile
            )
        with self.assertRaisesRegex(ValueError, "different Loaded Model"):
            context.synthesize_tokens([1], voice_profile=other_profile)
        profile.close()
        with self.assertRaisesRegex(module.SynthesizeError, "closed"):
            context.synthesize_tokens([1], voice_profile=profile)

        context.close()
        other_profile.close()
        first_model.close()
        second_model.close()

    def test_profile_remains_usable_until_it_closes_even_after_model_close(self):
        module, native = self.load_models_module()
        model = module.Model("voice.gguf")
        context = model.create_context()
        profile = model.create_random_voice_profile(seed=9)

        model.close()
        self.assertTrue(model.closed)
        self.assertFalse(profile.closed)

        serialization = profile.serialize()
        self.assertEqual(serialization.tolist(), [1, 2, 3, 4])

        result = context.synthesize_tokens([1], voice_profile=profile)
        self.assertEqual(result.frames_emitted, 1)

        profile.close()
        context.close()
        self.assertTrue(profile.closed)
        self.assertEqual(
            [item[0] for item in native.calls],
            [
                "random",
                "model_close",
                "serialize",
                "synthesize",
                "profile_close",
                "context_close",
            ],
        )


if __name__ == "__main__":
    unittest.main()
