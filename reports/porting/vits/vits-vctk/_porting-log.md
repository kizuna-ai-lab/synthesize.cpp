# vits-vctk Porting Log

## 2026-07-23 — F32 multi-speaker variant completed

- Pinned the official VCTK configuration and checkpoint from the existing VITS
  upstream release. The checkpoint has 858 state entries, 39,700,208 parameters,
  109 speakers, and 256 conditioning channels.
- Extended the converter by architecture rather than language. The final F32 GGUF
  contains 473 tensors: 111 text encoder, 116 stochastic duration, 88 acoustic
  flow, 157 decoder, and one 109-by-256 speaker embedding table. Two conversions
  from the formal manifest are byte-identical at 120,407,360 bytes with SHA-256
  `cc947a9a4f0907b5015e264998ebd58277f60d08eb1f965edb92e6f6169203bc`.
- Added the private `voice-conditioning` module. It selects a speaker row with a
  graph I32 input and supplies conditioning to the duration predictor after its
  pre-convolution, to every acoustic-flow WN block, and to the waveform decoder
  after its pre-convolution. The public C Interface remains model-family neutral.
- Exposed stable Preset Voice IDs `speaker-000` through `speaker-108`. VCTK has no
  invented default: a missing or unknown Voice returns `SYNTH_ERR_UNSUPPORTED_VOICE`.
  Successful results return the resolved Voice ID. Voice Profile creation remains
  unsupported for this package because an upstream speaker index is a preset,
  not a portable cloned-voice profile.
- Materialized 12 upstream PyTorch cases with 216 files. The suite covers first,
  middle, last, and upstream-example speakers, three seeds, both speaking-rate
  boundaries, minimal, punctuation, normalization, and long inputs.
- CPU source-F32 validation passes all seven stages. The duration path is exact in
  12/12 cases; final PCM worst max-absolute drift is `1.7652847e-4`.
- CUDA 13.3 strict-FP32 validation passes all seven stages on both physical Linux
  hosts. Duration structure is exact in 12/12 cases. Final PCM worst max-absolute
  drift is `4.3350458e-4` on DGX Spark GB10 (`sm_121a`) and `4.7563761e-4` on RTX
  4070 SUPER (`sm_89`).
- The public multi-speaker integration test verifies the 109-entry catalog,
  missing and invalid Voice errors, resolved Voice metadata, same-seed
  repeatability, and different-speaker output. The registered VCTK integration
  suite passes 8/8. Release CPU, ASan/UBSan, DGX CUDA, and RTX CUDA unit gates pass
  40/40, 40/40, 37/37, and 37/37 respectively; locked Python tests pass 86/86.
- Public synthesis uses the project random stream, while oracle parity runners
  replay PyTorch's recorded duration and latent noise. Therefore an equal numeric
  seed is repeatable within each runtime but does not promise PyTorch-identical
  PCM. This does not affect the stage parity result.
- Per ADR 0017, perceptual quality evaluation remains deferred. The upstream code
  is MIT, but the separately downloaded checkpoint has no explicit redistribution
  terms. Local conversion and validation are complete; upload to
  `handy-computer/vits-vctk-gguf` remains blocked until checkpoint redistribution
  permission is established.

## 2026-07-23 — F16 and Q8_MIXED functional profiles completed

- Added a deterministic C++ quantizer with strict, versioned VITS profile
  policies. The loader rejects mismatched profile metadata, file type, tensor
  storage type, or layout before model-buffer allocation.
- F16 keeps 350 text, duration, voice, normalization, bias, and scalar tensors in
  F32 and stores 123 flow/decoder weights in F16. Its 74,208,032-byte package has
  SHA-256
  `88506dc57c82fd580a1c8fc57573cdf4ad97a6752da8f511ebce3df1a91e6d05`.
- Q8_MIXED keeps those 350 tensors in F32, stores four native-layout decoder
  transpose-convolution weights in F16, and stores 119 ordinary flow/decoder
  matrices in Q8_0. Its 55,047,200-byte package has SHA-256
  `6f45d73da1d98384365988088d93f8d7341607217114764de06e3bddadc96af1`.
  Q8 convolution weights use packed `[kernel * input_channels, output_channels]`
  matrices and execute directly through GGML matmul without a persistent F32
  weight copy.
- Both packages are byte-identical across two complete quantizer runs. All seven
  12-case stages pass on one-thread CPU, DGX Spark CUDA 13.3, and RTX 4070 SUPER
  CUDA 13.3. Duration structure remains exact in every case, and both CUDA
  placement records contain zero executable CPU fallback nodes.
- Final PCM drift is recorded as diagnostic evidence only. No numerical tolerance,
  perceptual quality, naturalness, or speaker-similarity acceptance claim is made.
- The final release CPU and ASan/UBSan gates pass 43/43; DGX CUDA passes 39/39,
  while the RTX configuration passes 40/40 because it additionally enables the
  host quantizer CLI contract test. The locked Python suite passes 88/88. Real
  F16 and Q8_MIXED public synthesis also passes under ASan/UBSan.
- At this checkpoint, model publication remained blocked by the earlier
  checkpoint-term policy; local implementation and functional validation were
  complete.

## 2026-07-23 — Hugging Face payload prepared; organization authorization blocked

- The maintainer changed project policy: an official checkpoint with no separately
  stated weight terms is now treated as having no additional redistribution
  restriction. The missing statement and this assumption remain explicitly
  disclosed in the generated model card.
- Added the transcribe.cpp-style YAML-to-README card workflow under
  `scripts/hf_cards/`. Three new tests validate artifact size and SHA-256,
  required validation/quality metadata, and the actual VCTK publication payload.
  The locked Python suite now passes 93/93.
- Generated `models/vits-vctk/README.md` beside the three flat GGUF files. A
  second generator run in check mode reproduced it exactly.
- Authenticated Hugging Face user `jiangzhuo9357` attempted to create the public
  repository `handy-computer/vits-vctk-gguf`. Hugging Face returned HTTP 403:
  the account cannot create a model in the `handy-computer` namespace. No remote
  repository or partial upload was created.
- Publication is ready and blocked only on granting this account organization
  create/write access (or activating an already-authorized credential). The
  target is intentionally not changed to a personal namespace.

## 2026-07-23 — Published under the maintainer's personal namespace

- The maintainer clarified that synthesize.cpp models publish under the personal
  Hugging Face account `jiangzhuo9357`, not the unrelated `handy-computer`
  organization. The source-controlled target and all generated download links
  were changed to `jiangzhuo9357/vits-vctk-gguf`.
- Added a Hugging Face `license_name` slug contract after the Hub correctly
  rejected the initial prose value during metadata preflight. The new test fails
  invalid mixed-case or spaced names before an upload attempt.
- Created the public model repository and uploaded the flat README, F32, F16,
  and Q8_MIXED payload in commit
  `64e5bd152da7a132441ac39868bdcaf008daad38`.
- Queried that exact revision through the Hub API. The remote repository is
  public, the file list is complete, the README exactly matches the generated
  local card, and all three remote GGUF sizes and SHA-256 values match the
  source artifacts.

## 2026-07-23 — Built-in phoneme frontend and package refresh

- Reused the architecture-level private `TextFrontend` seam; no language branch
  or VCTK-specific public API was added. The package now advertises
  `PHONEMES_UTF8 | TOKEN_IDS` and maps Unicode phoneme scalars with the embedded
  178-entry symbol table before the existing speaker-conditioned graph.
- Regenerated F32, F16, and Q8_MIXED twice each. Every pair is byte-identical.
  The refreshed packages are 120,407,552, 74,208,224, and 55,047,392 bytes with
  SHA-256 values
  `b9e69b257cc600679a45e4197614f36ec678156180e2d366f7e1dcb6668b2be0`,
  `ff11efb1106834efb3609647e68642b48a58dbbdbabbc776d3afd83cf46085af`,
  and
  `149438d3a6c817ca6e4ab803a67207a41f62097cc8f586520355610801fb0542`.
- Compared all 473 tensor names, types, shapes, and payload hashes with the
  previous published F32 package. There are zero tensor-payload differences;
  only frontend metadata changed.
- With `speaker-004`, public phoneme input `ˈeɪ.` produces byte-identical PCM to
  its resolved token sequence for all three profiles. All 42 one-thread CPU and
  42 DGX CUDA stage/profile checks pass with exact duration structure and zero
  executable CUDA fallback nodes.
- The common Release, Python, sanitizer, and DGX CUDA gates pass. The RTX 4070
  SUPER host has now been revalidated, and the refreshed package was
  republished to `jiangzhuo9357/vits-vctk-gguf` at revision
  `eb27a41f7a01dbe3ec88dabcd834844e9383a5e7`.

## 2026-07-23 — VoiceProfileAPI and integration gate refresh

- Added boundary tests for the public VoiceProfile C API so undersized
  size-tagged structures are rejected before the implementation is entered.
- Added a Python lifecycle regression that keeps a voice profile usable after
  its parent model closes until the profile itself is explicitly closed.
- Rebuilt the DGX Spark unit gate. A missing executable registration for
  `synthesize-abi-initializer-bounds-test` was corrected by building the target
  directly; the full unit gate then passed 43/43.
- Configured a separate integration build with the local LJSpeech and VCTK
  GGUFs plus the materialized golden payloads. The integration gate passed
  18/18, covering the LJSpeech public lifecycle, public synthesis, CLI smoke,
  all seven LJSpeech golden stage checks, the optional VCTK public path, and
  all seven VCTK golden checks.

## 2026-07-23 — Unit gate dependency fix

- Added `synthesize-abi-initializer-bounds-test` and `synthesize-cli` to the
  `synthesize-check-unit` dependency set so the unit gate no longer relies on
  prebuilt artifacts.
- Re-ran `cmake --build build/dev-dgx-spark --target synthesize-check-unit`;
  the clean rebuild now compiles the missing targets itself and passes 43/43.
