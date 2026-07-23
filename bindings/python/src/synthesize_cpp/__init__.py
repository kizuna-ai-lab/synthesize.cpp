"""Python Adapter for the stable synthesize.cpp C Interface."""

from . import _build_contract
from .errors import ProviderError
from .errors import SynthesizeError
from . import _runtime as _runtime
from .audio import Audio
from .audio import SynthesisResult
from .audio import SynthesisResultFlags
from .devices import Backend
from .devices import BackendDevice
from .devices import DeviceFlags
from .devices import DeviceType
from .devices import backend_available
from .devices import backend_devices
from .diagnostics import Diagnostic
from .diagnostics import DiagnosticCallback
from .diagnostics import DiagnosticLevel
from .diagnostics import Status
from .models import Context
from .models import Model
from .metadata import InputFlags
from .metadata import LanguageCapability
from .metadata import LanguageFlags
from .metadata import ModelCapabilities
from .metadata import ModelCapabilityFlags
from .metadata import PresetVoice
from .metadata import PresetVoiceFlags
from .voice_profiles import Requirement
from .voice_profiles import VoiceProfile
from .voice_profiles import VoiceProfileCapabilities
from .voice_profiles import VoiceProfileSourceFlags
from .voice_profiles import VoiceReference


__version__ = _build_contract.VERSION

__all__ = [
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
