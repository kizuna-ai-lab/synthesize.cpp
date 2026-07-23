import argparse
import importlib.metadata
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("default", "cu13"), default="default")
    arguments = parser.parse_args()

    environment_before = dict(os.environ)
    import synthesize_cpp
    from synthesize_cpp import _native, _runtime

    assert synthesize_cpp.__version__ == "0.1.0"
    assert synthesize_cpp.__all__ == [
        "Audio",
        "Backend",
        "BackendDevice",
        "Context",
        "DeviceFlags",
        "DeviceType",
        "Diagnostic",
        "DiagnosticCallback",
        "DiagnosticLevel",
        "InputFlags",
        "LanguageCapability",
        "LanguageFlags",
        "Model",
        "ModelCapabilities",
        "ModelCapabilityFlags",
        "PresetVoice",
        "PresetVoiceFlags",
        "ProviderError",
        "Requirement",
        "SynthesisResult",
        "SynthesisResultFlags",
        "SynthesizeError",
        "Status",
        "VoiceProfile",
        "VoiceProfileCapabilities",
        "VoiceProfileSourceFlags",
        "VoiceReference",
        "__version__",
        "backend_available",
        "backend_devices",
    ]
    expected_distribution = {
        "default": "synthesize-cpp-native",
        "cu13": "synthesize-cpp-native-cu13",
    }[arguments.provider]
    assert _runtime._provider.provider_id == arguments.provider
    assert _runtime._provider.distribution == expected_distribution
    assert _runtime._native_info == {
        "path": str(_runtime._provider.library_path),
        "abi_version": 1,
        "version": (0, 1, 0),
    }
    assert _native.backend_available(1) is True
    assert _native.backend_available(3) is (arguments.provider == "cu13")
    public_devices = synthesize_cpp.backend_devices()
    assert isinstance(public_devices, tuple) and public_devices
    assert any(device.kind == "cpu" for device in public_devices)
    assert synthesize_cpp.backend_available(synthesize_cpp.Backend.CPU) is True
    assert importlib.metadata.version("synthesize-cpp") == "0.1.0"
    assert importlib.metadata.version("synthesize-cpp-native") == "0.1.0"
    assert importlib.metadata.version(expected_distribution) == "0.1.0"
    assert dict(os.environ) == environment_before
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
