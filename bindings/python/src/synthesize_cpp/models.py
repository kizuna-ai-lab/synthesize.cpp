"""Owned Loaded Model and Synthesis Context wrappers."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable
from typing import Self

from . import _native
from .devices import Backend
from .devices import BackendDevice
from .devices import _device_from_native
from .errors import SynthesizeError
from .diagnostics import DiagnosticCallback
from .diagnostics import _diagnostic_adapter
from .metadata import LanguageCapability
from .metadata import ModelCapabilities
from .metadata import PresetVoice
from .metadata import _capabilities_from_native
from .metadata import _languages_from_native
from .metadata import _preset_voices_from_native
from .voice_profiles import VoiceProfile
from .voice_profiles import VoiceProfileCapabilities
from .voice_profiles import VoiceReference
from .voice_profiles import _concrete_profile_seed
from .voice_profiles import _references_for_native
from .voice_profiles import _voice_profile_capabilities_from_native
from .audio import SynthesisResult
from .audio import _synthesize_tokens
from .audio import _synthesize_phonemes
from .audio import _synthesize_text


class Model:
    """A loaded native Model with explicit and context-manager cleanup."""

    __slots__ = (
        "_capabilities",
        "_device",
        "_handle",
        "_languages",
        "_lock",
        "_preset_voices",
        "_voice_profile_capabilities",
        "backend",
        "device_index",
        "path",
    )

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        backend: Backend = Backend.AUTO,
        device_index: int = -1,
        diagnostics: DiagnosticCallback | None = None,
    ) -> None:
        self._handle = None
        self._lock = threading.RLock()
        if not isinstance(backend, Backend):
            raise TypeError("backend must be a synthesize_cpp.Backend value")
        if isinstance(device_index, bool) or not isinstance(device_index, int):
            raise TypeError("device_index must be an integer")
        if device_index < -1:
            raise ValueError("device_index must be -1 or a non-negative index")
        if device_index > 2**31 - 1:
            raise OverflowError("device_index does not fit int32_t")
        path_value = os.fspath(path)
        if not isinstance(path_value, str):
            raise TypeError("model path must resolve to str, not bytes")

        self.path = Path(path_value)
        self.backend = backend
        self.device_index = device_index
        native_diagnostics = _diagnostic_adapter(diagnostics)
        try:
            self._handle = _native.model_load(
                path_value, int(backend), device_index, native_diagnostics
            )
        except RuntimeError as error:
            raise SynthesizeError(
                f"could not load Model {path_value!r}: {error}"
            ) from error
        try:
            metadata = _native.model_metadata(self._handle)
            self._device = _device_from_native(metadata["device"])
            self._capabilities = _capabilities_from_native(
                metadata["capabilities"]
            )
            self._preset_voices = _preset_voices_from_native(
                metadata["preset_voices"]
            )
            self._languages = _languages_from_native(metadata["languages"])
            self._voice_profile_capabilities = (
                _voice_profile_capabilities_from_native(
                    metadata["voice_profile_capabilities"]
                )
            )
        except Exception as error:
            handle = self._handle
            self._handle = None
            _native.model_close(handle)
            raise SynthesizeError(
                f"could not read Loaded Model metadata: {error}"
            ) from error

    @property
    def closed(self) -> bool:
        return self._handle is None

    @property
    def device(self) -> BackendDevice:
        return self._device

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def preset_voices(self) -> tuple[PresetVoice, ...]:
        return self._preset_voices

    @property
    def languages(self) -> tuple[LanguageCapability, ...]:
        return self._languages

    @property
    def voice_profile_capabilities(self) -> VoiceProfileCapabilities:
        return self._voice_profile_capabilities

    def create_voice_profile_from_reference(
        self,
        references: object,
        *,
        diagnostics: DiagnosticCallback | None = None,
    ) -> VoiceProfile:
        native_references = _references_for_native(references)  # type: ignore[arg-type]
        native_diagnostics = _diagnostic_adapter(diagnostics)
        with self._lock:
            if self._handle is None:
                raise SynthesizeError("Model is closed")
            try:
                handle = _native.voice_profile_create_from_reference(
                    self._handle, native_references, native_diagnostics
                )
            except RuntimeError as error:
                raise SynthesizeError(
                    f"could not prepare Voice Profile from Reference Audio: {error}"
                ) from error
            return VoiceProfile._from_native(self, handle)

    def create_voice_profile_from_description(
        self,
        description: str,
        *,
        language: str | None = None,
        seed: int = 0,
        diagnostics: DiagnosticCallback | None = None,
    ) -> VoiceProfile:
        if not isinstance(description, str):
            raise TypeError("description must be str")
        if not description:
            raise ValueError("description must not be empty")
        if language is not None and not isinstance(language, str):
            raise TypeError("language must be str or None")
        native_seed = _concrete_profile_seed(seed)
        native_diagnostics = _diagnostic_adapter(diagnostics)
        with self._lock:
            if self._handle is None:
                raise SynthesizeError("Model is closed")
            try:
                handle = _native.voice_profile_create_from_description(
                    self._handle,
                    description,
                    language,
                    native_seed,
                    native_diagnostics,
                )
            except RuntimeError as error:
                raise SynthesizeError(
                    f"could not prepare Voice Profile from Description Text: {error}"
                ) from error
            return VoiceProfile._from_native(self, handle)

    def create_random_voice_profile(
        self,
        *,
        seed: int = 0,
        diagnostics: DiagnosticCallback | None = None,
    ) -> VoiceProfile:
        native_seed = _concrete_profile_seed(seed)
        native_diagnostics = _diagnostic_adapter(diagnostics)
        with self._lock:
            if self._handle is None:
                raise SynthesizeError("Model is closed")
            try:
                handle = _native.voice_profile_create_random(
                    self._handle, native_seed, native_diagnostics
                )
            except RuntimeError as error:
                raise SynthesizeError(
                    f"could not prepare random Voice Profile: {error}"
                ) from error
            return VoiceProfile._from_native(self, handle)

    def load_voice_profile(
        self,
        data: object,
        *,
        diagnostics: DiagnosticCallback | None = None,
    ) -> VoiceProfile:
        try:
            view = memoryview(data)
        except TypeError as error:
            raise TypeError("serialized Voice Profile must support the buffer protocol") from error
        if view.nbytes == 0:
            raise ValueError("serialized Voice Profile must not be empty")
        view.release()
        native_diagnostics = _diagnostic_adapter(diagnostics)
        with self._lock:
            if self._handle is None:
                raise SynthesizeError("Model is closed")
            try:
                handle = _native.voice_profile_load(
                    self._handle, data, native_diagnostics
                )
            except RuntimeError as error:
                raise SynthesizeError(
                    f"could not load serialized Voice Profile: {error}"
                ) from error
            return VoiceProfile._from_native(self, handle)

    def create_context(self) -> Context:
        """Create an independent mutable Synthesis Context for this Model."""
        with self._lock:
            if self._handle is None:
                raise SynthesizeError("Model is closed")
            try:
                handle = _native.context_create(self._handle)
            except RuntimeError as error:
                raise SynthesizeError(
                    f"could not create Synthesis Context: {error}"
                ) from error
            return Context._from_native(self, handle)

    def close(self) -> None:
        """Close this Python Model; native release may await retained Contexts."""
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native.model_close(handle)

    def __enter__(self) -> Self:
        if self.closed:
            raise SynthesizeError("Model is closed")
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Context:
    """Mutable synthesis state created by :meth:`Model.create_context`."""

    __slots__ = ("_handle", "_lock", "_model")

    def __init__(self) -> None:
        raise TypeError("Context objects are created by Model.create_context()")

    @classmethod
    def _from_native(cls, model: Model, handle: object) -> Context:
        instance = cls.__new__(cls)
        instance._handle = handle
        instance._lock = threading.RLock()
        instance._model = model
        return instance

    @property
    def model(self) -> Model:
        return self._model

    @property
    def closed(self) -> bool:
        return self._handle is None

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native.context_close(handle)

    def _voice_profile_handle(
        self, voice: str | None, voice_profile: VoiceProfile | None
    ) -> object | None:
        if voice_profile is None:
            return None
        if not isinstance(voice_profile, VoiceProfile):
            raise TypeError("voice_profile must be a VoiceProfile or None")
        if voice is not None:
            raise ValueError("voice and voice_profile are mutually exclusive")
        if voice_profile.model is not self._model:
            raise ValueError("Voice Profile belongs to a different Loaded Model")
        return voice_profile._handle_for_call()

    def synthesize_tokens(
        self,
        tokens: object,
        *,
        language: str | None = None,
        voice: str | None = None,
        seed: int | None = 0,
        speaking_rate: float = 1.0,
        max_output_frames: int | None = None,
        diagnostics: DiagnosticCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
        voice_profile: VoiceProfile | None = None,
    ) -> SynthesisResult:
        """Synchronously synthesize token IDs into native-owned F32 PCM."""
        with self._lock:
            if self._handle is None:
                raise SynthesizeError("Synthesis Context is closed")
            profile_handle = self._voice_profile_handle(voice, voice_profile)
            return _synthesize_tokens(
                self._handle,
                tokens,
                language=language,
                voice=voice,
                seed=seed,
                speaking_rate=speaking_rate,
                max_output_frames=max_output_frames,
                diagnostics=diagnostics,
                should_cancel=should_cancel,
                voice_profile=profile_handle,
            )

    def synthesize_text(
        self,
        text: str,
        *,
        language: str | None = None,
        voice: str | None = None,
        seed: int | None = 0,
        speaking_rate: float = 1.0,
        max_output_frames: int | None = None,
        diagnostics: DiagnosticCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
        voice_profile: VoiceProfile | None = None,
    ) -> SynthesisResult:
        """Synchronously synthesize UTF-8 text through the Model frontend."""
        with self._lock:
            if self._handle is None:
                raise SynthesizeError("Synthesis Context is closed")
            profile_handle = self._voice_profile_handle(voice, voice_profile)
            return _synthesize_text(
                self._handle,
                text,
                language=language,
                voice=voice,
                seed=seed,
                speaking_rate=speaking_rate,
                max_output_frames=max_output_frames,
                diagnostics=diagnostics,
                should_cancel=should_cancel,
                voice_profile=profile_handle,
            )

    def synthesize_phonemes(
        self,
        phonemes: str,
        *,
        language: str | None = None,
        voice: str | None = None,
        seed: int | None = 0,
        speaking_rate: float = 1.0,
        max_output_frames: int | None = None,
        diagnostics: DiagnosticCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
        voice_profile: VoiceProfile | None = None,
    ) -> SynthesisResult:
        """Synchronously synthesize the Model frontend's phoneme notation."""
        with self._lock:
            if self._handle is None:
                raise SynthesizeError("Synthesis Context is closed")
            profile_handle = self._voice_profile_handle(voice, voice_profile)
            return _synthesize_phonemes(
                self._handle,
                phonemes,
                language=language,
                voice=voice,
                seed=seed,
                speaking_rate=speaking_rate,
                max_output_frames=max_output_frames,
                diagnostics=diagnostics,
                should_cancel=should_cancel,
                voice_profile=profile_handle,
            )

    def __enter__(self) -> Self:
        if self.closed:
            raise SynthesizeError("Synthesis Context is closed")
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
