# Qwen3-TTS Stage 2, Plan 1: the Base Package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `Qwen/Qwen3-TTS-12Hz-0.6B-Base` into a loadable synthesize.cpp Model Package that carries the ECAPA-TDNN speaker encoder and both halves of the speech tokenizer, and make the runtime accept a package with an empty Preset Voice Catalog.

**Architecture:** The Base checkpoint is the CustomVoice checkpoint plus a 76-tensor speaker encoder, minus every preset speaker. Conversion therefore becomes variant-aware rather than forked: one converter, discriminated on `config.json`'s `tts_model_type`. On the C++ side no new graph is built in this plan — the tensors are catalogued and shape-checked at load, and the Voice Profile contract is read and validated, so the package is whole from its first cut and Plans 2 and 3 add graphs against metadata that already exists.

**Tech Stack:** Python 3.12 + `gguf` + `safetensors` + `torch` in the locked `scripts/envs/qwen3-tts` environment; C++17 with GGML/GGUF; CTest under the `unit` label; `uv` for every Python entry point.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-11-qwen3-tts-stage-2-design.md`. Family record: `docs/porting/families/qwen3-tts.md`.
- Reference Model Variant: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` at revision `5d83992436eae1d760afd27aff78a71d676296fc`, `license: apache-2.0`.
- Variant slug, used verbatim in every path and metadata value: `qwen3-tts-12hz-0-6b-base`.
- Tensor counts that must hold: talker 402, speaker encoder 76 (478 in `model.safetensors`), speech tokenizer 496 of which encoder-side 225.
- Frame geometry: `encode_downsample_rate` = `decode_upsample_rate` = 1920 samples at 24 kHz = 12.5 Hz.
- `enc_dim` = 1024 = talker `hidden_size`. `encoder_valid_num_quantizers` = 16 = `num_code_groups`.
- Reference Audio limits are expressed in **samples at the target sample rate**, matching `scripts/convert-omnivoice.py` (24000 = 1 s). Do not express them in codec frames. `docs/model-packages.md` calls the unit a Reference Frame Equivalent; the established value is one sample at 24 kHz.
- Values for this package: `min_frames_per_clip` 24000 (1 s), `max_frames_per_clip` 720000 (30 s), `max_total_frames` 720000, `max_reference_count` 1. The two duration bounds are provisional pending Task 3's measurement; `max_reference_count` is an upstream fact and is not provisional.
- Profile Schema identity: `qwen3-tts-voice-clone`, version 1.
- Every tensor name must stay under `GGML_MAX_NAME` (64); the converter already refuses to emit one at or over it.
- No quantization policy in the converter: it emits the source-dtype artifact only.
- Run the unit gate after every task: `cmake --build build --target synthesize-check-unit`.
- Format before every commit: `scripts/ci/clang-format.sh --fix`. Never format `ggml/`.

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-base/intake.json` | create | Pinned revision, per-file digests, tensor census, measured limits |
| `scripts/dump_reference_qwen3_tts_base.py` | create | Base oracle: x-vector, reference codes, clone prompt, end-to-end, both modes |
| `tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json` | create | Golden Manifest: provenance, package contract, cases, relations |
| `tests/tolerances/qwen3-tts.json` | modify | Add the base variant's stage keys |
| `scripts/convert-qwen3-tts.py` | modify | Variant-aware conversion; speaker encoder; both codec halves; profile contract |
| `tests/python/test_convert_qwen3_tts.py` | modify | Rules that fail silently: variant discrimination, codebook dedup, catalog emptiness |
| `src/arch/qwen3-tts/weights.h` | modify | `SpeakerEncoderParams`, `CodecEncoderParams`, `ProfileContract`, voice mode |
| `src/arch/qwen3-tts/weights.cpp` | modify | Read and validate the above; accept an empty Catalog |
| `src/arch/qwen3-tts/catalog.cpp` | modify | Cover the two new tensor namespaces with shape checks |
| `src/arch/qwen3-tts/model.cpp` | modify | Honest capability snapshot for a Catalog-less package |
| `tests/qwen3_tts_metadata_test.cpp` | modify | Base-variant metadata contract |
| `tests/qwen3_tts_catalog_test.cpp` | modify | New namespaces resolve and are shape-checked |
| `tests/qwen3_tts_voice_required_test.cpp` | create | Synthesis without a Voice Profile fails explicitly |
| `tests/CMakeLists.txt` | modify | Register the new test |

---

### Task 1: Pin the Base checkpoint and record its census

**Files:**
- Create: `reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-base/intake.json`

**Interfaces:**
- Produces: the digests Task 4 pins in the manifest and Task 7 hashes into the Compatibility ID. Keys: `weights.files[].{path,sha256,bytes}`, `census.{talker,speaker_encoder,codec_encoder,codec_decoder}`.

- [ ] **Step 1: Download the checkpoint**

```bash
uv run --project scripts/envs/qwen3-tts --locked python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base',
                  revision='5d83992436eae1d760afd27aff78a71d676296fc',
                  local_dir='models/qwen3-tts-12hz-0-6b-base')
"
```

- [ ] **Step 2: Write the census script and run it**

```bash
uv run --project scripts/envs/qwen3-tts --locked python - <<'PY' > reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-base/intake.json
import hashlib, json, collections
from pathlib import Path
from safetensors import safe_open

root = Path("models/qwen3-tts-12hz-0-6b-base")
files = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files.append({"path": str(path.relative_to(root)), "sha256": digest, "bytes": path.stat().st_size})

def census(path):
    with safe_open(str(path), framework="pt") as h:
        return collections.Counter(k.split(".")[0] for k in h.keys())

talker = census(root / "model.safetensors")
codec = census(root / "speech_tokenizer" / "model.safetensors")
print(json.dumps({
    "schema": "synthesize-intake-v1",
    "family": "qwen3-tts",
    "variant": "qwen3-tts-12hz-0-6b-base",
    "upstream": {
        "repository": "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "revision": "5d83992436eae1d760afd27aff78a71d676296fc",
        "license": "apache-2.0",
    },
    "weights": {"files": files},
    "census": {"model.safetensors": dict(talker), "speech_tokenizer": dict(codec)},
}, indent=2))
PY
```

- [ ] **Step 3: Verify the census against the constants this plan asserts**

Expected: `model.safetensors` shows `talker` 402 and `speaker_encoder` 76; `speech_tokenizer` shows `encoder` and `decoder` prefixes summing to 496. If any number differs, stop — the checkpoint moved and every downstream task's assertion is wrong.

- [ ] **Step 4: Commit**

```bash
git add reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-base/intake.json
git commit -m "qwen3-tts: pin the Base checkpoint and record its tensor census"
```

---

### Task 2: Base oracle runner

**Files:**
- Create: `scripts/dump_reference_qwen3_tts_base.py`

**Interfaces:**
- Consumes: the local checkpoint from Task 1.
- Produces: per-case directories under `build/goldens/qwen3-tts/<case-id>/` containing `speaker/x_vector.f32`, `codes/reference.i32`, `prompt/icl_embed.f32`, `codes/semantic.i32`, `codes/acoustic.i32`, `audio.pcm`. Task 4's manifest names exactly these paths.

- [ ] **Step 1: Write the runner**

Model it on `scripts/dump_reference_qwen3_tts_pytorch.py` — same argument shape, same atomic writes, same seeding. The Base-specific body:

```python
model = Qwen3TTSModel.from_pretrained(
    args.weights_dir, device_map=args.device, dtype=torch.bfloat16,
    attn_implementation="eager",
)

# Both switches, or the trajectory is silently non-deterministic. Setting only
# one of them is the Stage 1 trap.
prompt_items = model.create_voice_clone_prompt(
    ref_audio=(ref_wav, ref_sr),
    ref_text=None if case["x_vector_only"] else case["ref_text"],
    x_vector_only_mode=case["x_vector_only"],
)
wavs, sr = model.generate_voice_clone(
    text=case["text"], language=case["language"],
    voice_clone_prompt=prompt_items,
    do_sample=False, subtalker_dosample=False,
)
```

Dump, in this order, before synthesis so a failure downstream still leaves the deterministic stages on disk:

```python
write_f32(out / "speaker" / "x_vector.f32", prompt_items[0].ref_spk_embedding)
if not case["x_vector_only"]:
    write_i32(out / "codes" / "reference.i32", prompt_items[0].ref_code)
```

- [ ] **Step 2: Run it on one x-vector-only case**

Run: `uv run --project scripts/envs/qwen3-tts --locked python scripts/dump_reference_qwen3_tts_base.py --weights-dir models/qwen3-tts-12hz-0-6b-base --case base-xvector-en`
Expected: `build/goldens/qwen3-tts/base-xvector-en/speaker/x_vector.f32` is exactly 4096 bytes (1024 F32 values), and no `codes/reference.i32` exists.

- [ ] **Step 3: Run it on one ICL case**

Run: the same command with `--case base-icl-en`
Expected: `codes/reference.i32` exists, its length is a multiple of 16 int32 values, and `element_count / 16` equals `ceil(reference_samples / 1920)`.

- [ ] **Step 4: Record the measured duration bounds**

Run the ICL case at 0.5 s, 1 s, 3 s, 10 s and 30 s of reference audio. Record in `reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-base/intake.json` under a new `reference_bounds` key: for each duration, the reference-code frame count and whether the output is intelligible. This is what makes the spec's provisional 1 s / 30 s bounds measured rather than assumed. If 1 s produces unusable audio, raise `min_frames_per_clip` here and in Task 6 rather than shipping a bound the model does not honor.

- [ ] **Step 5: Commit**

```bash
git add scripts/dump_reference_qwen3_tts_base.py reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-base/intake.json
git commit -m "qwen3-tts: add the Base oracle runner and measure the reference duration bounds"
```

---

### Task 3: Golden Manifest for the Base variant

**Files:**
- Create: `tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json`
- Modify: `tests/tolerances/qwen3-tts.json`

**Interfaces:**
- Consumes: Task 1's digests, Task 2's artifact paths and measured bounds.
- Produces: `manifest["variant"]`, `manifest["source"]["artifacts"]`, `manifest["package_contract"]` — the three things the converter reads.

- [ ] **Step 1: Author the manifest**

Copy `qwen3-tts-12hz-0-6b-customvoice.manifest.json` as the structural model. It must validate against `docs/schemas/synthesize-golden-manifest-v1.schema.json`, which requires at least 12 cases plus `relations`. The `reference_audio` voice kind already exists in the schema — no schema change is needed.

`package_contract` differs from Stage 1's in exactly these fields:

```json
"voices": { "mode": "profile-only", "default_id": null, "preset_ids": [] },
"profile": {
  "schema": "qwen3-tts-voice-clone",
  "schema_version": 1,
  "sources": ["reference_audio", "serialized_profile"],
  "reference": {
    "target_sample_rate": 24000,
    "target_channels": 1,
    "min_frames_per_clip": 24000,
    "max_frames_per_clip": 720000,
    "max_total_frames": 720000,
    "max_reference_count": 1
  }
}
```

The case set, all of which carry `"voice": {"kind": "reference_audio"}`:

| id | mode | coverage |
| --- | --- | --- |
| `base-upstream-clone-en` | ICL | upstream-example, baseline — the README's `clone.wav` and its transcript |
| `base-xvector-en` | x-vector | x-vector-only path, no transcript |
| `base-icl-en` | ICL | transcript-assisted path |
| `base-icl-zh` | ICL | second language |
| `base-icl-ja` | ICL | third language |
| `base-xvector-zh` | x-vector | mode against language |
| `base-ref-min` | ICL | reference at `min_frames_per_clip` exactly |
| `base-ref-max` | ICL | reference at `max_frames_per_clip` exactly |
| `base-text-short` | ICL | one-word target text |
| `base-text-long` | ICL | target text near `max_input_tokens` |
| `base-seed-one` | ICL | seed 1 |
| `base-seed-forty-two` | ICL | seed 42 |

`relations` gets one entry: `artifact_differs` over `["base-icl-en", "base-seed-one", "base-seed-forty-two"]` on `audio.pcm` for phase `public_request`, which is what proves the seed reaches the sampler.

- [ ] **Step 2: Validate the manifest against the schema**

```bash
uv run --project scripts/envs/qwen3-tts --locked python -c "
import json, jsonschema
schema = json.load(open('docs/schemas/synthesize-golden-manifest-v1.schema.json'))
manifest = json.load(open('tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json'))
jsonschema.validate(manifest, schema)
print('manifest valid,', len(manifest['cases']), 'cases')
"
```
Expected: `manifest valid, 12 cases`

- [ ] **Step 3: Add the base variant's tolerance keys**

In `tests/tolerances/qwen3-tts.json`, add stage entries for `speaker_encoder` and `codec_encoder` under the existing profile-by-backend-by-stage keying. Leave the numbers at the file's existing default for a new stage; Plans 2 and 3 tighten them against measurement.

- [ ] **Step 4: Commit**

```bash
git add tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json tests/tolerances/qwen3-tts.json
git commit -m "qwen3-tts: commit the Base Golden Manifest and its tolerance keys"
```

---

### Task 4: Converter — discriminate the variant

**Files:**
- Modify: `scripts/convert-qwen3-tts.py`
- Test: `tests/python/test_convert_qwen3_tts.py`

**Interfaces:**
- Produces: `convert.variant_profile(config) -> VariantProfile` with fields `model_type: str`, `carries_speaker_encoder: bool`, `carries_codec_encoder: bool`, `display_name: str`, `size_label: str`. Tasks 5 and 6 branch on it.

- [ ] **Step 1: Write the failing test**

```python
class VariantDiscriminationTests(unittest.TestCase):
    """The two variants differ by a whole subsystem, not by a label."""

    def test_base_config_declares_the_speaker_encoder(self) -> None:
        profile = convert.variant_profile({
            "tts_model_type": "base",
            "tts_model_size": "0b6",
            "speaker_encoder_config": {"enc_dim": 1024, "sample_rate": 24000},
        })
        self.assertTrue(profile.carries_speaker_encoder)
        self.assertTrue(profile.carries_codec_encoder)

    def test_custom_voice_config_carries_neither(self) -> None:
        profile = convert.variant_profile({"tts_model_type": "custom_voice", "tts_model_size": "0b6"})
        self.assertFalse(profile.carries_speaker_encoder)
        self.assertFalse(profile.carries_codec_encoder)

    def test_base_config_without_a_speaker_encoder_is_refused(self) -> None:
        with self.assertRaises(convert.ConverterError):
            convert.variant_profile({"tts_model_type": "base", "tts_model_size": "0b6"})

    def test_an_unknown_model_type_is_refused(self) -> None:
        with self.assertRaises(convert.ConverterError):
            convert.variant_profile({"tts_model_type": "voice_design", "tts_model_size": "1b7"})
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run --project scripts/envs/qwen3-tts --locked python -m unittest tests.python.test_convert_qwen3_tts.VariantDiscriminationTests -v`
Expected: FAIL, `module 'convert_qwen3_tts' has no attribute 'variant_profile'`

- [ ] **Step 3: Implement it**

```python
@dataclass(frozen=True)
class VariantProfile:
    model_type: str
    carries_speaker_encoder: bool
    carries_codec_encoder: bool
    display_name: str
    size_label: str


def variant_profile(config: dict[str, Any]) -> VariantProfile:
    """Decide what this checkpoint carries from what it declares.

    The two supported variants differ by a whole subsystem: Base ships a
    76-tensor ECAPA-TDNN speaker encoder and needs the tokenizer's encoder half
    to turn reference audio into codes; CustomVoice ships neither and resolves
    speakers as codec-vocabulary token ids. Keying that on the declared type and
    then checking the declaration against the config is what keeps a future
    variant from silently converting as whichever branch it fell into.
    """
    model_type = str(config.get("tts_model_type", ""))
    has_encoder_config = "speaker_encoder_config" in config
    if model_type == "base":
        if not has_encoder_config:
            raise ConverterError(
                "config declares tts_model_type=base but carries no speaker_encoder_config; "
                "a Base package without a speaker encoder cannot prepare a Voice Profile"
            )
        return VariantProfile("base", True, True, "Qwen3-TTS 12Hz 0.6B Base", "0.6B")
    if model_type == "custom_voice":
        if has_encoder_config:
            raise ConverterError(
                "config declares tts_model_type=custom_voice but carries a speaker_encoder_config"
            )
        return VariantProfile("custom_voice", False, False, "Qwen3-TTS 12Hz 0.6B CustomVoice", "0.6B")
    raise ConverterError(f"unsupported tts_model_type {model_type!r}")
```

- [ ] **Step 4: Run the tests**

Run: `uv run --project scripts/envs/qwen3-tts --locked python -m unittest tests.python.test_convert_qwen3_tts -v`
Expected: PASS, and every pre-existing test in the file still passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/convert-qwen3-tts.py tests/python/test_convert_qwen3_tts.py
git commit -m "qwen3-tts: discriminate Base from CustomVoice on what the config declares"
```

---

### Task 5: Converter — carry the speaker encoder and both codec halves

**Files:**
- Modify: `scripts/convert-qwen3-tts.py` (`convert_file`, `main`)
- Test: `tests/python/test_convert_qwen3_tts.py`

**Interfaces:**
- Consumes: `variant_profile` from Task 4.
- Produces: tensors named `speaker_encoder.*` and `codec.encoder.*` in the GGUF; `Conversion.deduplicated: list[dict[str, str]]`.

- [ ] **Step 1: Write the failing test for the dedup decision**

```python
class EncoderCodebookDuplicationTests(unittest.TestCase):
    """Stage 1 measured 16 encoder codebooks as bit-identical to the decoder's.

    Carrying both halves makes that measurable again rather than theoretical. The
    outcome ruled out is duplicated bytes nobody decided on.
    """

    def test_identical_tables_are_stored_once_and_recorded(self) -> None:
        shared = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        tensors = {
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.codebook": shared,
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.codebook": shared.clone(),
        }
        conversion = convert.Conversion()
        out = convert.deduplicate_codebooks(tensors, conversion)

        self.assertEqual(len(out), 1)
        self.assertEqual(len(conversion.deduplicated), 1)
        self.assertEqual(
            conversion.deduplicated[0]["alias"],
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.codebook",
        )

    def test_diverged_tables_are_both_carried(self) -> None:
        tensors = {
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.codebook":
                torch.arange(12, dtype=torch.float32).reshape(4, 3),
            "encoder.quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.codebook":
                torch.ones(4, 3, dtype=torch.float32),
        }
        conversion = convert.Conversion()
        out = convert.deduplicate_codebooks(tensors, conversion)

        self.assertEqual(len(out), 2)
        self.assertEqual(conversion.deduplicated, [])
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run --project scripts/envs/qwen3-tts --locked python -m unittest tests.python.test_convert_qwen3_tts.EncoderCodebookDuplicationTests -v`
Expected: FAIL, `module 'convert_qwen3_tts' has no attribute 'deduplicate_codebooks'`

- [ ] **Step 3: Implement it**

Add `deduplicated: list[dict[str, str]] = field(default_factory=list)` to `Conversion`, then:

```python
def deduplicate_codebooks(tensors: dict[str, torch.Tensor],
                          conversion: Conversion) -> dict[str, torch.Tensor]:
    """Store one copy of any encoder codebook that still equals a decoder one.

    Measured, not assumed: equality is tested at this revision rather than
    carried over from Stage 1's measurement. A diverged pair is carried twice
    and says so, so the package never contains bytes nobody decided on.
    """
    decoder_tables = {
        name: tensor for name, tensor in tensors.items()
        if name.startswith("decoder.") and name.endswith(".codebook")
    }
    kept: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if name.startswith("encoder.") and name.endswith(".codebook"):
            twin = next(
                (d for d, table in decoder_tables.items()
                 if table.shape == tensor.shape and torch.equal(table, tensor)),
                None,
            )
            if twin is not None:
                conversion.deduplicated.append({
                    "alias": name,
                    "stored_as": twin,
                    "reason": "bit-identical to the decoder's table at this revision",
                })
                continue
        kept[name] = tensor
    return kept
```

Call it from `convert_file` for the codec file, after `reconstruct_codebooks`, only when the variant carries the encoder half. In `main`, replace the unconditional drop with the variant decision:

```python
    profile = variant_profile(config)
    convert_file(talker_path, "", conversion, reconstruct=False)
    convert_file(codec_path, "codec.", conversion, reconstruct=True,
                 drop_prefix=None if profile.carries_codec_encoder else "encoder.",
                 deduplicate=profile.carries_codec_encoder)
```

- [ ] **Step 4: Run the tests**

Run: `uv run --project scripts/envs/qwen3-tts --locked python -m unittest tests.python.test_convert_qwen3_tts -v`
Expected: PASS

- [ ] **Step 5: Convert the Base checkpoint end to end**

```bash
uv run --project scripts/envs/qwen3-tts --locked python scripts/convert-qwen3-tts.py \
  --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-base.manifest.json \
  --weights-dir models/qwen3-tts-12hz-0-6b-base \
  --output models/qwen3-tts-12hz-0-6b-base/qwen3-tts-12hz-0-6b-base-BF16.gguf
```
Expected: it writes the file and prints a tensor count. Read the printed count against the census: talker 402 + speaker encoder 76 + codec decoder + codec encoder, minus `initialized` flags, minus deduplicated aliases. Record the arithmetic in the commit message; a count you cannot account for means a rule silently dropped a region.

- [ ] **Step 6: Re-convert CustomVoice and prove it is byte-identical**

```bash
uv run --project scripts/envs/qwen3-tts --locked python scripts/convert-qwen3-tts.py \
  --manifest tests/golden/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice.manifest.json \
  --weights-dir models/qwen3-tts-12hz-0-6b-customvoice \
  --output /tmp/customvoice-recheck.gguf
sha256sum /tmp/customvoice-recheck.gguf \
  models/qwen3-tts-12hz-0-6b-customvoice/qwen3-tts-12hz-0-6b-customvoice-BF16.gguf
```
Expected: the two digests match. They must — the published Stage 1 package was built by this script, and a converter change that alters its output silently invalidates a live artifact.

- [ ] **Step 7: Commit**

```bash
git add scripts/convert-qwen3-tts.py tests/python/test_convert_qwen3_tts.py
git commit -m "qwen3-tts: carry the speaker encoder and both codec halves for Base"
```

---

### Task 6: Converter — empty Catalog and the Voice Profile contract

**Files:**
- Modify: `scripts/convert-qwen3-tts.py` (`add_metadata`)
- Test: `tests/python/test_convert_qwen3_tts.py`

**Interfaces:**
- Produces these metadata keys, which Tasks 7 and 8 read: `synthesize.voice.mode` = `profile-only`, `synthesize.voice.preset_count` = 0, `synthesize.profile.schema`, `synthesize.profile.schema_version`, `synthesize.profile.compatibility_id`, `synthesize.reference.target_sample_rate`, `synthesize.reference.target_channels`, `synthesize.reference.min_frames_per_clip`, `synthesize.reference.max_frames_per_clip`, `synthesize.reference.max_total_frames`, `synthesize.reference.max_reference_count`, `synthesize.qwen3-tts.speaker_encoder.{enc_dim,sample_rate,mel_bins,n_fft,hop_length,win_length,fmin,fmax}`.

- [ ] **Step 1: Write the failing test**

```python
class BaseCatalogTests(unittest.TestCase):
    """A Base package advertises no Voice it can select and says so positively."""

    def test_speaker_metadata_is_refused_when_the_checkpoint_has_no_speakers(self) -> None:
        talker = {"spk_id": {}, "spk_is_dialect": {}}
        with self.assertRaises(convert.ConverterError):
            convert.speaker_catalog(talker, preset_ids=["aiden"])

    def test_an_empty_checkpoint_and_an_empty_manifest_agree(self) -> None:
        names, token_ids, dialects = convert.speaker_catalog(
            {"spk_id": {}, "spk_is_dialect": {}}, preset_ids=[])
        self.assertEqual((names, token_ids, dialects), ([], [], []))

    def test_the_mel_front_end_parameters_are_the_ones_the_reference_uses(self) -> None:
        params = convert.speaker_encoder_metadata({"enc_dim": 1024, "sample_rate": 24000})
        self.assertEqual(params["mel_bins"], 128)
        self.assertEqual(params["n_fft"], 1024)
        self.assertEqual(params["hop_length"], 256)
        self.assertEqual(params["win_length"], 1024)
        self.assertEqual(params["fmin"], 0.0)
        self.assertEqual(params["fmax"], 12000.0)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run --project scripts/envs/qwen3-tts --locked python -m unittest tests.python.test_convert_qwen3_tts.BaseCatalogTests -v`
Expected: FAIL, `no attribute 'speaker_catalog'`

- [ ] **Step 3: Implement it**

```python
# modeling_qwen3_tts.py:1941 -- the speaker encoder consumes a 128-bin mel, not
# a waveform. These five numbers are the front end's whole contract and they are
# carried in the package rather than written into the C++ port.
SPEAKER_MEL = {"mel_bins": 128, "n_fft": 1024, "hop_length": 256,
               "win_length": 1024, "fmin": 0.0, "fmax": 12000.0}


def speaker_encoder_metadata(config: dict[str, Any]) -> dict[str, Any]:
    enc_dim = int(config["enc_dim"])
    sample_rate = int(config["sample_rate"])
    if sample_rate != 24000:
        raise ConverterError(f"speaker encoder declares {sample_rate} Hz; the mel front end is pinned to 24000")
    return {"enc_dim": enc_dim, "sample_rate": sample_rate, **SPEAKER_MEL}


def speaker_catalog(talker: dict[str, Any], preset_ids: list[str]):
    """Cross-check the manifest's Catalog against the checkpoint's speaker table.

    Stage 1's hazard was ordering; this variant's is emptiness. Both directions
    are checked, so a Base package cannot inherit a CustomVoice catalog and a
    CustomVoice package cannot lose one.
    """
    if set(preset_ids) != set(talker["spk_id"]):
        raise ConverterError(
            f"the manifest lists {sorted(preset_ids)} but the checkpoint carries "
            f"{sorted(talker['spk_id'])}"
        )
    names = list(preset_ids)
    return (names,
            [int(talker["spk_id"][n]) for n in names],
            [str(talker["spk_is_dialect"][n]) if talker["spk_is_dialect"][n] else "" for n in names])
```

In `add_metadata`, write `voices["mode"]` from the manifest rather than assuming `preset-catalog`, emit the speaker arrays only when non-empty, and add the profile block for a variant that carries a speaker encoder:

```python
    if profile.carries_speaker_encoder:
        writer.add_string("synthesize.profile.schema", "qwen3-tts-voice-clone")
        writer.add_uint32("synthesize.profile.schema_version", 1)
        reference = package["profile"]["reference"]
        for key in ("target_sample_rate", "target_channels"):
            writer.add_uint32(f"synthesize.reference.{key}", int(reference[key]))
        for key in ("min_frames_per_clip", "max_frames_per_clip",
                    "max_total_frames", "max_reference_count"):
            writer.add_uint64(f"synthesize.reference.{key}", int(reference[key]))
        for key, value in speaker_encoder_metadata(config["speaker_encoder_config"]).items():
            if isinstance(value, float):
                writer.add_float32(f"synthesize.qwen3-tts.speaker_encoder.{key}", value)
            else:
                writer.add_uint32(f"synthesize.qwen3-tts.speaker_encoder.{key}", int(value))
        # Profile Compatibility ID: sha256 over the family compatibility
        # manifest -- schema identity plus the fingerprints of every weight and
        # config that changes what prepared conditioning means.
        compat = hashlib.sha256()
        for piece in ("qwen3-tts-voice-clone/1", digests["talker"], digests["codec"], digests["config"]):
            compat.update(piece.encode())
        writer.add_string("synthesize.profile.compatibility_id", compat.hexdigest())
```

- [ ] **Step 4: Run the tests and re-convert both variants**

Run: `uv run --project scripts/envs/qwen3-tts --locked python -m unittest tests.python.test_convert_qwen3_tts -v`
Expected: PASS. Then re-run both conversions from Task 5 steps 5 and 6; the CustomVoice digest must still match the published artifact.

- [ ] **Step 5: Commit**

```bash
git add scripts/convert-qwen3-tts.py tests/python/test_convert_qwen3_tts.py
git commit -m "qwen3-tts: declare the empty Catalog and the Voice Profile contract"
```

---

### Task 7: Load and validate the new metadata

**Files:**
- Modify: `src/arch/qwen3-tts/weights.h`, `src/arch/qwen3-tts/weights.cpp`
- Test: `tests/qwen3_tts_metadata_test.cpp`

**Interfaces:**
- Produces: `HParams::speaker_encoder` (`SpeakerEncoderParams`), `HParams::profile` (`ProfileContract`), `HParams::voice_mode` (`VoiceMode` enum). Task 8 reads `voice_mode`; Plans 2 and 3 read the other two.

- [ ] **Step 1: Write the failing test**

Add to `tests/qwen3_tts_metadata_test.cpp`, alongside the existing cases:

```cpp
// A Base package: no speakers at all, and a Voice Profile contract instead.
GgufContext base_metadata() {
    GgufContext c = valid_metadata();
    gguf_context * g = c.get();
    gguf_set_val_str(g, "synthesize.model_variant", "qwen3-tts-12hz-0-6b-base");
    gguf_set_val_str(g, "synthesize.voice.mode", "profile-only");
    gguf_set_val_u32(g, "synthesize.voice.preset_count", 0);
    set_string_array(g, "synthesize.qwen3-tts.speakers.names", {});
    gguf_set_val_str(g, "synthesize.profile.schema", "qwen3-tts-voice-clone");
    gguf_set_val_u32(g, "synthesize.profile.schema_version", 1);
    gguf_set_val_str(g, "synthesize.profile.compatibility_id", std::string(64, 'a').c_str());
    gguf_set_val_u32(g, "synthesize.reference.target_sample_rate", 24000);
    gguf_set_val_u32(g, "synthesize.reference.target_channels", 1);
    gguf_set_val_u64(g, "synthesize.reference.min_frames_per_clip", 24000);
    gguf_set_val_u64(g, "synthesize.reference.max_frames_per_clip", 720000);
    gguf_set_val_u64(g, "synthesize.reference.max_total_frames", 720000);
    gguf_set_val_u64(g, "synthesize.reference.max_reference_count", 1);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.enc_dim", 1024);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.sample_rate", 24000);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.mel_bins", 128);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.n_fft", 1024);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.hop_length", 256);
    gguf_set_val_u32(g, "synthesize.qwen3-tts.speaker_encoder.win_length", 1024);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.speaker_encoder.fmin", 0.0f);
    gguf_set_val_f32(g, "synthesize.qwen3-tts.speaker_encoder.fmax", 12000.0f);
    return c;
}

void test_base_package_loads_without_any_preset_voice() {
    GgufContext c = base_metadata();
    synth::qwen3tts::HParams hparams;
    ASSERT_TRUE(synth::qwen3tts::read_hparams(c.get(), hparams) == SYNTH_OK);
    ASSERT_TRUE(hparams.preset_voices.empty());
    ASSERT_TRUE(hparams.voice_mode == synth::qwen3tts::VoiceMode::ProfileOnly);
    ASSERT_TRUE(hparams.profile.max_reference_count == 1);
}

// enc_dim feeds the talker's prompt slot directly. A package whose speaker
// encoder is a different width would build a prompt of the wrong shape.
void test_speaker_embedding_width_must_equal_the_talker_hidden_size() {
    GgufContext c = base_metadata();
    gguf_set_val_u32(c.get(), "synthesize.qwen3-tts.speaker_encoder.enc_dim", 512);
    synth::qwen3tts::HParams hparams;
    ASSERT_TRUE(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
}

void test_profile_only_mode_with_presets_is_refused() {
    GgufContext c = base_metadata();
    gguf_set_val_u32(c.get(), "synthesize.voice.preset_count", 9);
    synth::qwen3tts::HParams hparams;
    ASSERT_TRUE(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
}

void test_preset_catalog_mode_with_no_presets_is_still_refused() {
    GgufContext c = valid_metadata();
    gguf_set_val_u32(c.get(), "synthesize.voice.preset_count", 0);
    synth::qwen3tts::HParams hparams;
    ASSERT_TRUE(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
}

// A reference bound of zero would let a Profile be prepared from no audio.
void test_zero_reference_bounds_are_refused() {
    GgufContext c = base_metadata();
    gguf_set_val_u64(c.get(), "synthesize.reference.min_frames_per_clip", 0);
    synth::qwen3tts::HParams hparams;
    ASSERT_TRUE(synth::qwen3tts::read_hparams(c.get(), hparams) != SYNTH_OK);
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cmake --build build --target synthesize-qwen3-tts-metadata-test && ctest --test-dir build -R '^synthesize-qwen3-tts-metadata-test$' --output-on-failure`
Expected: compile error — `VoiceMode` and `HParams::profile` do not exist.

- [ ] **Step 3: Implement it**

In `weights.h`:

```cpp
enum class VoiceMode : uint32_t {
    PresetCatalog,  // speakers are codec-vocabulary token ids
    ProfileOnly,    // no selectable Voice; every request carries a Voice Profile
};

// The ECAPA-TDNN speaker encoder Base variants carry. Its output width equals
// the talker's hidden size because the x-vector substitutes directly for the
// prompt's speaker embedding -- there is no projection between them.
struct SpeakerEncoderParams {
    uint32_t enc_dim     = 0;
    uint32_t sample_rate = 0;
    uint32_t mel_bins    = 0;
    uint32_t n_fft       = 0;
    uint32_t hop_length  = 0;
    uint32_t win_length  = 0;
    float    fmin        = 0.0f;
    float    fmax        = 0.0f;
};

// Read and validated here so the package is whole from its first cut; the Voice
// Profile module enforces these from Plan 2 on.
struct ProfileContract {
    std::string schema;
    uint32_t    schema_version = 0;
    std::string compatibility_id_hex;  // 64 hex chars = 32 bytes
    uint32_t    reference_sample_rate = 0;
    uint32_t    reference_channels    = 0;
    uint64_t    min_frames_per_clip   = 0;
    uint64_t    max_frames_per_clip   = 0;
    uint64_t    max_total_frames      = 0;
    uint64_t    max_reference_count   = 0;
};
```

Add to `HParams`: `VoiceMode voice_mode = VoiceMode::PresetCatalog;`, `SpeakerEncoderParams speaker_encoder;`, `ProfileContract profile;`, `bool has_speaker_encoder = false;`.

In `weights.cpp`, replace the `read_voices` guard at line 377:

```cpp
    if (mode == "preset-catalog") {
        hparams.voice_mode = VoiceMode::PresetCatalog;
        if (preset_count == 0) {
            std::fprintf(stderr, "qwen3-tts: preset-catalog mode with no presets\n");
            return false;
        }
    } else if (mode == "profile-only") {
        // Base variants carry no selectable Voice at all: upstream ships an
        // empty spk_id table. The package says so positively rather than
        // arriving as a catalog that happens to be empty, so a truncated
        // catalog cannot be mistaken for this.
        hparams.voice_mode = VoiceMode::ProfileOnly;
        if (preset_count != 0) {
            std::fprintf(stderr, "qwen3-tts: profile-only mode with %u presets\n", preset_count);
            return false;
        }
        hparams.preset_voices.clear();
        return !hparams.has_package_default;
    } else {
        std::fprintf(stderr, "qwen3-tts: unsupported voice mode %s\n", mode.c_str());
        return false;
    }
```

Add `read_profile_contract` and `read_speaker_encoder`, called from `read_hparams` only when `synthesize.profile.schema` is present. Validate: schema equals `qwen3-tts-voice-clone`; `compatibility_id_hex.size() == 64` and every character is a lowercase hex digit; every reference bound is non-zero; `min_frames_per_clip <= max_frames_per_clip <= max_total_frames`; `reference_channels == 1`; `enc_dim == hparams.talker.hidden_size`; `sample_rate == hparams.codec.sample_rate`; `n_fft` and `win_length` non-zero with `hop_length < win_length`; `fmax > fmin` and `fmax <= sample_rate / 2`.

- [ ] **Step 4: Run the test**

Run: `ctest --test-dir build -R '^synthesize-qwen3-tts-metadata-test$' --output-on-failure`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/qwen3-tts/weights.h src/arch/qwen3-tts/weights.cpp tests/qwen3_tts_metadata_test.cpp
git commit -m "qwen3-tts: load the empty Catalog, the speaker encoder, and the profile contract"
```

---

### Task 8: Catalogue the new tensors

**Files:**
- Modify: `src/arch/qwen3-tts/catalog.cpp`, `src/arch/qwen3-tts/catalog.h`
- Test: `tests/qwen3_tts_catalog_test.cpp`

**Interfaces:**
- Consumes: `HParams::speaker_encoder`, `HParams::has_speaker_encoder`.
- Produces: catalog entries for `speaker_encoder.*` and `codec.encoder.*` with expected shapes, so no region of the package is present-but-unread.

- [ ] **Step 1: Write the failing test**

The existing file builds a synthetic package from a hand-written `Entry` list and
resolves it with `build_model_weights`. Follow that shape exactly — the hand-written
list is deliberately a second, independent statement of the contract.

```cpp
// Stage 1 left the catalog covering exactly what a graph could reach. Base adds
// two regions, and the catalog's own sweep makes cataloguing them mandatory
// rather than optional: "a tensor the catalog never asked for is an error".
void test_base_package_resolves_with_the_speaker_encoder() {
    synth::qwen3tts::HParams hparams = base_hparams();
    Context context = make_context();
    populate(context.get(), base_entries(hparams));   // includes speaker_encoder.* and codec.encoder.*

    synth::qwen3tts::ModelWeights weights;
    ASSERT_TRUE(synth::qwen3tts::build_model_weights(context.get(), nullptr, hparams, weights) == SYNTH_OK);
}

// fc maps the pooled statistics onto the embedding the prompt slot takes, so its
// output extent is enc_dim. A package that disagrees builds a prompt of the
// wrong width, which is wrong audio rather than an error.
void test_speaker_encoder_fc_shape_is_checked_against_enc_dim() {
    synth::qwen3tts::HParams hparams = base_hparams();
    Context context = make_context();
    std::vector<Entry> entries = base_entries(hparams);
    entry(entries, "speaker_encoder.fc.weight").ne[0] = 512;   // declared enc_dim is 1024
    populate(context.get(), entries);

    synth::qwen3tts::ModelWeights weights;
    ASSERT_TRUE(synth::qwen3tts::build_model_weights(context.get(), nullptr, hparams, weights) != SYNTH_OK);
}

// The sweep is the reason this matters: a CustomVoice package that somehow
// carried speaker-encoder tensors would be refused rather than silently
// carrying a region nobody resolves.
void test_uncatalogued_speaker_encoder_tensors_are_refused() {
    synth::qwen3tts::HParams hparams = custom_voice_hparams();   // has_speaker_encoder == false
    Context context = make_context();
    std::vector<Entry> entries = custom_voice_entries(hparams);
    entries.push_back(Entry{ "speaker_encoder.fc.weight", { 1024, 3072, 1 } });
    populate(context.get(), entries);

    synth::qwen3tts::ModelWeights weights;
    ASSERT_TRUE(synth::qwen3tts::build_model_weights(context.get(), nullptr, hparams, weights) != SYNTH_OK);
}

void test_expected_tensor_count_grows_by_both_new_regions() {
    const uint64_t custom = synth::qwen3tts::expected_tensor_count(custom_voice_hparams());
    const uint64_t base   = synth::qwen3tts::expected_tensor_count(base_hparams());
    // 76 speaker-encoder tensors plus the tokenizer's encoder half, less any
    // codebook the converter stored once. Task 1's census fixes the number;
    // assert it here so a silent change to either region fails a test.
    ASSERT_TRUE(base > custom + 76);
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cmake --build build --target synthesize-qwen3-tts-catalog-test && ctest --test-dir build -R '^synthesize-qwen3-tts-catalog-test$' --output-on-failure`
Expected: FAIL — the names are absent from the catalog.

- [ ] **Step 3: Implement it**

Extend the catalog builder with the two namespaces, gated on `hparams.has_speaker_encoder`. Derive every shape from metadata rather than hard-coding: `fc.weight` is `[enc_dim, 3 * mfa_channels, 1]`, `asp.conv.weight` is `[2 * mfa_channels, attention_channels, 1]`, and the res2net blocks from the block widths already in the file. Take the exact per-tensor shapes from Task 1's census — read them out of the checkpoint header rather than from this plan, and fail the load on any mismatch.

- [ ] **Step 4: Run the test**

Run: `ctest --test-dir build -R '^synthesize-qwen3-tts-catalog-test$' --output-on-failure`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/qwen3-tts/catalog.h src/arch/qwen3-tts/catalog.cpp tests/qwen3_tts_catalog_test.cpp
git commit -m "qwen3-tts: catalogue the speaker encoder and the codec encoder"
```

---

### Task 9: An honest capability snapshot, and synthesis that refuses to guess

**Files:**
- Modify: `src/arch/qwen3-tts/model.cpp`
- Create: `tests/qwen3_tts_voice_required_test.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `HParams::voice_mode`, `HParams::profile`.
- Produces: a `synth::ModelInfo` whose `preset_voice_ids` is empty, whose `voice_profile.source_flags` carries `SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO | SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE`, and a synthesis path that fails rather than selecting a Voice.

- [ ] **Step 1: Write the failing test**

```cpp
// The Base package is the project's first that cannot synthesize without
// prepared conditioning. The failure has to be explicit: silently selecting a
// Voice would mean shipping audio in a voice nobody asked for.
void test_request_without_a_voice_profile_is_refused() {
    synth_context_t * context = context_for(base_model());
    synth_synthesis_request_t request = text_request("hello");
    // no voice_id, no voice profile
    ASSERT_TRUE(synth_synthesize(context, &request, nullptr) == SYNTH_ERR_UNSUPPORTED_VOICE);
}

void test_request_naming_a_voice_id_is_refused_because_none_exist() {
    synth_context_t * context = context_for(base_model());
    synth_synthesis_request_t request = text_request("hello");
    request.voice_id = "vivian";
    ASSERT_TRUE(synth_synthesize(context, &request, nullptr) == SYNTH_ERR_UNSUPPORTED_VOICE);
}

void test_capability_snapshot_reports_no_preset_voices_and_reference_audio() {
    const synth_model_info_t * info = model_info(base_model());
    ASSERT_TRUE(info->preset_voice_count == 0);
    ASSERT_TRUE((info->voice_profile.source_flags & SYNTH_PROFILE_SOURCE_REFERENCE_AUDIO) != 0);
}
```

`SYNTH_ERR_UNSUPPORTED_VOICE` (status 10) is the **only** voice error the ABI
defines — there is no `SYNTH_ERR_VOICE_NOT_FOUND`, despite the comment above
`find_preset_voice` in `src/arch/qwen3-tts/weights.h` naming one. Do not add a
status code for this plan: both refusals are the same public status, and what
distinguishes them belongs in the diagnostic sink's message. Correct that stale
comment while you are in the file.

- [ ] **Step 2: Run it and watch it fail**

Run: `cmake --build build --target synthesize-qwen3-tts-voice-required-test && ctest --test-dir build -R '^synthesize-qwen3-tts-voice-required-test$' --output-on-failure`
Expected: FAIL

- [ ] **Step 3: Implement it**

In `model.cpp`'s shared-info builder, emit an empty `preset_voice_ids` and set `voice_profile.source_flags` from `hparams.has_speaker_encoder`, together with `reference_transcript = SYNTH_REQUIREMENT_OPTIONAL`, `reference_language = SYNTH_REQUIREMENT_OPTIONAL`, and the reference target and limit values from `hparams.profile`. In `resolve_voice`, return the Catalog-less error before the preset lookup when `voice_mode == VoiceMode::ProfileOnly`.

Note for the implementer: `SYNTH_PROFILE_SOURCE_SERIALIZED_PROFILE` is claimed alongside Reference Audio because every successfully prepared v1 Profile can be serialized — the same rule `src/synthesize.cpp` documents for OmniVoice. Preparation itself still returns `SYNTH_ERR_UNSUPPORTED_VOICE` until Plan 2 lands the graph; that is a dispatch gap, not a capability lie, and Plan 2 closes it.

- [ ] **Step 4: Register the test**

In `tests/CMakeLists.txt`, beside line 164:

```cmake
synth_add_unit_test(synthesize-qwen3-tts-voice-required-test qwen3_tts_voice_required_test.cpp)
```

- [ ] **Step 5: Run the whole unit gate**

Run: `cmake --build build --target synthesize-check-unit`
Expected: PASS, including every Stage 1 test.

- [ ] **Step 6: Run the sanitizer gate**

```bash
cmake -S . -B build-sanitize -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON \
  -DSYNTH_BUILD_INTEGRATION_TESTS=OFF -DSYNTH_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-sanitize --target synthesize-check-unit
```
Expected: PASS with no sanitizer diagnostics.

- [ ] **Step 7: Commit**

```bash
scripts/ci/clang-format.sh --fix
git add src/arch/qwen3-tts/model.cpp tests/qwen3_tts_voice_required_test.cpp tests/CMakeLists.txt
git commit -m "qwen3-tts: report a Catalog-less package honestly and refuse to guess a Voice"
```

---

### Task 10: Load the real package end to end

**Files:**
- Modify: `docs/porting/families/qwen3-tts.md`

- [ ] **Step 1: Load the converted Base package through the public interface**

```bash
cmake -S . -B build-integration -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_INTEGRATION_TESTS=ON \
  -DSYNTH_QWEN3_TTS_BASE_TEST_MODEL=models/qwen3-tts-12hz-0-6b-base/qwen3-tts-12hz-0-6b-base-BF16.gguf
cmake --build build-integration --target synthesize-qwen3-tts-public-real
```
Expected: the model loads, reports zero Preset Voices, and reports Reference Audio among its Profile sources. Synthesis is expected to fail with the Catalog-less error — Plan 2 is what makes it succeed.

- [ ] **Step 2: Record the outcome in the family document**

Extend `docs/porting/families/qwen3-tts.md` with a Stage 2 section: the pinned Base revision and digests, the tensor census, the dedup measurement's result, the measured reference-duration bounds from Task 2 step 4, and the emitted tensor count with the arithmetic that accounts for it. Update the document's `Status:` line to say Stage 2 Plan 1 is done and what remains.

- [ ] **Step 3: Commit**

```bash
git add docs/porting/families/qwen3-tts.md
git commit -m "qwen3-tts: record Stage 2 Plan 1 -- the Base package loads"
```

---

## Notes for Plan 2

Plan 2 adds the mel front end, the ECAPA graph, Voice Profile preparation and serialization, and the prompt-slot wiring. Two things this plan deliberately leaves for it: `synth_voice_profile_create_from_reference` still returns `SYNTH_ERR_UNSUPPORTED_VOICE` for this family, and the tolerance entries added in Task 3 still carry the default numbers rather than measured ones.
