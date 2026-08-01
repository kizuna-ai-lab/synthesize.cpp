"""Synthesize one Model Family's package through the installed API wheel.

`api_wheel_synthesis_smoke.py` is a VITS contract test: it asserts that
family's sample rate, its fixed-default Voice, and the Voice Profile paths it
refuses. This one asserts only what holds for every family, so a new family is
registered here rather than given a script of its own.

What it covers is the part a family can reach the C interface without: the
binding's own capability reporting, Voice resolution, and seed contract.
"""

import argparse
import math
from pathlib import Path

import synthesize_cpp


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", default=None, help="comma-separated token IDs")
    parser.add_argument(
        "--text",
        default=None,
        help=(
            "UTF-8 text input, for a package whose only supported input kind "
            "is text_utf8 (mutually exclusive with --tokens)"
        ),
    )
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--samples-per-frame", type=int, required=True)
    parser.add_argument(
        "--voice",
        default=None,
        help="preset Voice to request; omit for a package with a default",
    )
    parser.add_argument(
        "--seed-changes-length",
        action="store_true",
        help=(
            "the family's randomness reaches its duration prediction, so a "
            "different seed may change the output length"
        ),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if (arguments.tokens is None) == (arguments.text is None):
        raise SystemExit("exactly one of --tokens or --text is required")
    tokens = None
    if arguments.tokens is not None:
        tokens = [int(value) for value in arguments.tokens.split(",") if value.strip()]
        assert tokens, "no token IDs were supplied"

    with synthesize_cpp.Model(arguments.model) as model:
        capabilities = model.capabilities
        assert capabilities.output_sample_rate == arguments.sample_rate, (
            f"model reports {capabilities.output_sample_rate} Hz, "
            f"expected {arguments.sample_rate}"
        )
        assert capabilities.output_channel_count == 1

        voice_ids = {voice.id for voice in model.preset_voices}
        if arguments.voice is not None:
            assert arguments.voice in voice_ids, (
                f"{arguments.voice} is not in the preset catalog"
            )

        with model.create_context() as context:

            def synthesize(*, voice, seed):
                if tokens is not None:
                    return context.synthesize_tokens(tokens, voice=voice, seed=seed)
                return context.synthesize_text(arguments.text, voice=voice, seed=seed)

            result = synthesize(voice=arguments.voice, seed=7)
            audio = result.audio
            assert audio.sample_rate == arguments.sample_rate
            assert audio.channel_count == 1
            assert audio.frame_count > 0
            assert audio.frame_count % arguments.samples_per_frame == 0, (
                f"{audio.frame_count} samples is not a whole number of "
                f"{arguments.samples_per_frame}-sample frames"
            )
            assert all(math.isfinite(value) for value in audio.samples)
            assert result.actual_seed == 7
            if arguments.voice is not None:
                assert result.resolved_voice_id == arguments.voice

            # The seed contract has to survive the binding, not only the C
            # interface: the same seed repeats and a different one does not.
            baseline = bytes(memoryview(audio.samples))
            repeat = synthesize(voice=arguments.voice, seed=7)
            assert bytes(memoryview(repeat.audio.samples)) == baseline
            other = synthesize(voice=arguments.voice, seed=8)
            assert bytes(memoryview(other.audio.samples)) != baseline
            # Whether the length moves with the seed is a family property, not
            # a universal one. VITS seeds its duration predictor, so its output
            # length legitimately varies; Kokoro draws only after the durations
            # are resolved, and OmniVoice's length comes from a deterministic
            # text-length estimate rather than the draw at all, so neither's
            # length must move.
            if arguments.seed_changes_length:
                assert other.audio.frame_count % arguments.samples_per_frame == 0
            else:
                assert other.audio.frame_count == audio.frame_count

            # A catalog with no default must refuse a request that names no
            # Voice; one with a default must accept it. Only a package that
            # was given a Voice to request here (Kokoro) exercises the
            # refusal side: a package with a package default and nothing to
            # name (VITS, OmniVoice) never reaches this block, because every
            # call above already ran with voice=None and had to succeed for
            # the script to get this far -- that success IS the
            # unnamed-default-accepted proof for those families.
            if arguments.voice is not None:
                try:
                    context.synthesize_tokens(tokens, seed=7)
                except synthesize_cpp.SynthesizeError as error:
                    assert "unsupported voice" in str(error), error
                else:
                    raise AssertionError(
                        "a catalog without a default accepted a request naming no Voice"
                    )

    print(
        f"family smoke ok: {audio.frame_count} samples at {audio.sample_rate} Hz"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
