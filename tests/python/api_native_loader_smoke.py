import argparse
import importlib.util
import sys
from pathlib import Path


def load_extension(extension_dir):
    candidates = sorted(extension_dir.glob("_native*"))
    if len(candidates) != 1:
        raise AssertionError(f"expected one _native extension, found {candidates}")
    spec = importlib.util.spec_from_file_location(
        "synthesize_cpp._native", candidates[0]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--good-library", type=Path, required=True)
    parser.add_argument("--missing-symbol-library", type=Path, required=True)
    arguments = parser.parse_args()

    native = load_extension(arguments.extension_dir)
    assert native.limited_api_hex() == 0x030B0000

    for path, message in (
        ("relative/libsynthesize.so", "absolute"),
        (str(arguments.good_library.parent / "missing.so"), "could not load"),
        (str(arguments.missing_symbol_library), "synth_status_string"),
    ):
        try:
            native.load(path)
        except RuntimeError as error:
            assert message in str(error), error
        else:
            raise AssertionError(f"loader unexpectedly accepted {path}")

    info = native.load(str(arguments.good_library))
    assert info == {
        "path": str(arguments.good_library),
        "abi_version": 1,
        "version": (0, 1, 0),
    }
    assert native.backend_available(1) is True
    assert native.backend_available(3) is False
    devices = native.backend_devices()
    assert isinstance(devices, tuple) and devices
    for index, device in enumerate(devices):
        assert device["index"] == index
        assert isinstance(device["name"], str) and device["name"]
        assert isinstance(device["description"], str)
        assert isinstance(device["kind"], str) and device["kind"]
        assert device["device_id"] is None or isinstance(device["device_id"], str)
        assert isinstance(device["memory_total"], int)
        assert isinstance(device["memory_free"], int)
        assert isinstance(device["device_type"], int)
        assert isinstance(device["flags"], int)
    assert any(device["kind"] == "cpu" for device in devices)
    assert native.load(str(arguments.good_library)) == info

    second_path = arguments.good_library.parent / "libsynthesize-copy.so"
    try:
        native.load(str(second_path))
    except RuntimeError as error:
        assert "already loaded" in str(error), error
    else:
        raise AssertionError("loader accepted a second process-global library")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
