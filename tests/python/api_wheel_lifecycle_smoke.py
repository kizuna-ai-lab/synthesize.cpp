import argparse
from pathlib import Path

import synthesize_cpp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    arguments = parser.parse_args()

    cpu_index = next(
        device.index
        for device in synthesize_cpp.backend_devices()
        if device.device_type is synthesize_cpp.DeviceType.CPU
    )
    model = synthesize_cpp.Model(
        arguments.model,
        backend=synthesize_cpp.Backend.CPU,
        device_index=cpu_index,
    )
    first = model.create_context()
    assert model.device.index is None
    assert model.device.kind == "cpu"
    assert model.capabilities.input_flags == (
        synthesize_cpp.InputFlags.PHONEMES_UTF8 | synthesize_cpp.InputFlags.TOKEN_IDS
    )
    assert model.capabilities.capability_flags == (
        synthesize_cpp.ModelCapabilityFlags.SPEAKING_RATE
        | synthesize_cpp.ModelCapabilityFlags.STOCHASTIC
    )
    assert model.capabilities.output_sample_rate == 22050
    assert model.capabilities.output_channel_count == 1
    assert model.preset_voices == ()
    assert len(model.languages) == 1
    assert model.languages[0].tag == "en"
    assert model.languages[0].flags == (
        synthesize_cpp.LanguageFlags.DEFAULT
        | synthesize_cpp.LanguageFlags.REGIONAL_FALLBACK
    )
    assert model.voice_profile_capabilities.source_flags == 0
    assert model.voice_profile_capabilities.reference_transcript is (
        synthesize_cpp.Requirement.UNSUPPORTED
    )
    assert model.voice_profile_capabilities.reference_language is (
        synthesize_cpp.Requirement.UNSUPPORTED
    )
    assert model.voice_profile_capabilities.description_language is (
        synthesize_cpp.Requirement.UNSUPPORTED
    )
    assert model.voice_profile_capabilities.profile_schema is None
    assert model.voice_profile_capabilities.profile_compatibility_id == bytes(32)
    with model.create_context() as second:
        assert second.model is model
        assert not second.closed
    assert second.closed

    model.close()
    model.close()
    assert model.closed
    assert model.languages[0].tag == "en"
    try:
        model.create_context()
    except synthesize_cpp.SynthesizeError as error:
        assert "Model is closed" in str(error), error
    else:
        raise AssertionError("closed Model created another Context")

    first.close()
    first.close()
    assert first.closed

    try:
        synthesize_cpp.Model(arguments.model.parent / "missing.gguf")
    except synthesize_cpp.SynthesizeError as error:
        assert "file not found" in str(error), error
    else:
        raise AssertionError("missing model unexpectedly loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
