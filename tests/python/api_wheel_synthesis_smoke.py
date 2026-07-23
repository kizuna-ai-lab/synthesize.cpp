import argparse
import array
import math
from pathlib import Path

import synthesize_cpp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()

    tokens = array.array("i")
    with arguments.tokens.open("rb") as source:
        tokens.fromfile(source, arguments.tokens.stat().st_size // tokens.itemsize)

    backend = (
        synthesize_cpp.Backend.CUDA
        if arguments.backend == "cuda"
        else synthesize_cpp.Backend.CPU
    )
    with synthesize_cpp.Model(arguments.model, backend=backend) as model:
        assert model.device.kind == arguments.backend
        with model.create_context() as context:
            result = context.synthesize_tokens(tokens, seed=42)
            cancellation_polls = 0

            def cancel_now():
                nonlocal cancellation_polls
                cancellation_polls += 1
                return True

            try:
                context.synthesize_tokens(tokens, should_cancel=cancel_now)
            except synthesize_cpp.SynthesizeError as error:
                assert "cancelled" in str(error), error
            else:
                raise AssertionError("cancelled synthesis unexpectedly succeeded")
            assert cancellation_polls == 1

            class CallbackError(Exception):
                pass

            def fail_cancellation():
                raise CallbackError("cancellation callback failed")

            try:
                context.synthesize_tokens(tokens, should_cancel=fail_cancellation)
            except CallbackError as error:
                assert str(error) == "cancellation callback failed"
            else:
                raise AssertionError("callback exception was not re-raised")

            try:
                context.synthesize_text("Hello")
            except synthesize_cpp.SynthesizeError as error:
                assert "unsupported input" in str(error), error
            else:
                raise AssertionError("VITS accepted unavailable raw-text input")

            phoneme_result = context.synthesize_phonemes("ˈeɪ.", seed=42)
            assert phoneme_result.audio.frame_count > 0
            assert phoneme_result.actual_seed == 42

            profile_sources = (
                lambda: model.create_voice_profile_from_reference([
                    synthesize_cpp.VoiceReference(
                        array.array("f", [0.0] * 8000),
                        sample_rate=8000,
                        channel_count=1,
                    )
                ]),
                lambda: model.create_voice_profile_from_description(
                    "warm narrator", language="en", seed=7
                ),
                lambda: model.create_random_voice_profile(seed=8),
                lambda: model.load_voice_profile(b"GGUF"),
            )
            for prepare_profile in profile_sources:
                try:
                    prepare_profile()
                except synthesize_cpp.SynthesizeError as error:
                    assert "unsupported voice" in str(error), error
                else:
                    raise AssertionError("VITS created an unsupported Voice Profile")

    assert result.audio.samples.readonly
    assert result.audio.samples.format == "f"
    assert len(result.audio.samples) == result.audio.frame_count * result.audio.channel_count
    assert result.audio.frame_count > 0
    assert result.audio.sample_rate == 22050
    assert result.audio.channel_count == 1
    assert result.frames_emitted == result.audio.frame_count
    assert result.actual_seed == 42
    assert result.flags == synthesize_cpp.SynthesisResultFlags.SEED_USED
    assert result.resolved_language_tag == "en"
    assert result.resolved_voice_id is None
    assert all(math.isfinite(value) for value in result.audio.samples[:64])

    diagnostics = []
    try:
        synthesize_cpp.Model(
            arguments.model,
            backend=synthesize_cpp.Backend.VULKAN,
            diagnostics=diagnostics.append,
        )
    except synthesize_cpp.SynthesizeError as error:
        assert "backend error" in str(error), error
    else:
        raise AssertionError("CPU Provider accepted Vulkan model loading")
    assert len(diagnostics) == 1
    assert diagnostics[0].level is synthesize_cpp.DiagnosticLevel.ERROR
    assert diagnostics[0].status is synthesize_cpp.Status.BACKEND
    assert diagnostics[0].code == "backend.unavailable"

    class DiagnosticError(Exception):
        pass

    def fail_diagnostic(diagnostic):
        raise DiagnosticError(diagnostic.code)

    try:
        synthesize_cpp.Model(
            arguments.model,
            backend=synthesize_cpp.Backend.VULKAN,
            diagnostics=fail_diagnostic,
        )
    except DiagnosticError as error:
        assert str(error) == "backend.unavailable"
    else:
        raise AssertionError("diagnostic callback exception was not re-raised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
