# synthesize-cpp

The single public Python Adapter for synthesize.cpp. It discovers one compatible
Native Provider, validates its release and public-header contract before loading
native code, and exposes the stable synthesis API without accelerator-specific
imports.

The default dependency supplies the CPU-capable Provider. Accelerator extras,
such as `synthesize-cpp[cu13]`, add complete alternative Providers; selection
remains process-global and deterministic.

```python
from array import array
import synthesize_cpp

with synthesize_cpp.Model(
    "model.gguf", backend=synthesize_cpp.Backend.CPU
) as model:
    print(model.device, model.capabilities, model.languages)
    with model.create_context() as context:
        result = context.synthesize_tokens(array("i", [1, 2, 3]), seed=42)

# The read-only F32 view retains the native Audio Buffer after Context/Model close.
pcm = result.audio.samples
```

The same Context also exposes `synthesize_text()` and `synthesize_phonemes()`;
availability is reported by `model.capabilities.input_flags`. Synthesis accepts
cooperative cancellation and structured diagnostic callbacks.

`model.voice_profile_capabilities` reports the four language-neutral preparation
sources. Separate Model methods prepare from one or more borrowed F32
`VoiceReference` buffers, Description Text, a concrete Random Seed, or opaque
serialized bytes. Every successful method returns the same immutable,
model-bound `VoiceProfile`, and `VoiceProfile.serialize()` exposes native-owned
GGUF bytes through a read-only `memoryview`. A Context selects either a Preset
Voice ID or a Voice Profile. Model-family support remains authoritative: the
current LJSpeech VITS package reports no runtime Voice Profile sources and fails
these preparation methods explicitly.

Native streaming Audio Sink callbacks remain a later Adapter slice; the C
Interface continues to be the sole native seam.
