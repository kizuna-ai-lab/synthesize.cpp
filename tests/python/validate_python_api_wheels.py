import argparse
import email.parser
import zipfile
from pathlib import Path


def wheel_metadata(archive, suffix):
    names = archive.namelist()
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise AssertionError(f"expected one {suffix}, found {matches}")
    return archive.read(matches[0]).decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-wheel", type=Path, required=True)
    parser.add_argument("--provider-wheel", type=Path, required=True)
    arguments = parser.parse_args()

    with zipfile.ZipFile(arguments.api_wheel) as archive:
        names = set(archive.namelist())
        for required in (
            "synthesize_cpp/__init__.py",
            "synthesize_cpp/audio.py",
            "synthesize_cpp/_build_contract.py",
            "synthesize_cpp/_providers.py",
            "synthesize_cpp/_runtime.py",
            "synthesize_cpp/devices.py",
            "synthesize_cpp/diagnostics.py",
            "synthesize_cpp/errors.py",
            "synthesize_cpp/models.py",
            "synthesize_cpp/metadata.py",
            "synthesize_cpp/voice_profiles.py",
        ):
            assert required in names, required
        extensions = [
            name
            for name in names
            if name.startswith("synthesize_cpp/_native.")
            and (name.endswith(".so") or name.endswith(".pyd"))
        ]
        assert len(extensions) == 1, extensions
        assert ".abi3." in extensions[0], extensions[0]
        assert not any(name.endswith(".in") for name in names), names
        assert not any(
            "libsynthesize" in name or "libggml" in name for name in names
        ), "API wheel must not contain a Native Provider runtime"

        wheel = wheel_metadata(archive, ".dist-info/WHEEL")
        assert "Root-Is-Purelib: false" in wheel, wheel
        assert "Tag: cp311-abi3-" in wheel, wheel
        metadata_text = wheel_metadata(archive, ".dist-info/METADATA")
        metadata = email.parser.Parser().parsestr(metadata_text)
        assert metadata["Name"] == "synthesize-cpp", metadata["Name"]
        assert metadata["Version"] == "0.1.0", metadata["Version"]
        assert metadata["Requires-Python"] == ">=3.11"
        requirements = metadata.get_all("Requires-Dist") or []
        assert "synthesize-cpp-native==0.1.0.*" in requirements, requirements

    with zipfile.ZipFile(arguments.provider_wheel) as archive:
        names = set(archive.namelist())
        assert "synthesize_cpp_native/_native/libsynthesize.so" in names
        assert "synthesize_cpp_native/_contract.py" in names
        entry_points = wheel_metadata(archive, ".dist-info/entry_points.txt")
        assert "[synthesize_cpp.native]" in entry_points, entry_points
        assert "default = synthesize_cpp_native:descriptor" in entry_points

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
