---
status: accepted
---

# Begin with the VITS model family

synthesize.cpp will implement VITS as its first supported model family. VITS was selected over first targeting a spectrogram-only model, a diffusion or flow-matching model, or a codec language model because it provides a representative end-to-end TTS graph, has inspectable reference implementations, remains practical for native CPU inference, and creates a path to multiple independently validated descendants such as Piper, MMS-TTS, and MeloTTS. This decision selects an inference architecture, not a language set or blanket compatibility with every VITS-derived checkpoint.
