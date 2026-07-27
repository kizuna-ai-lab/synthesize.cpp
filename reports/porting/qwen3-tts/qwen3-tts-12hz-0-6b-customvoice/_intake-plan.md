# Qwen3-TTS Stage 1 Intake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete stage `1-intake` for the Qwen3-TTS family's first Reference
Model Variant, producing a committed intake record, a locked CPU oracle
environment, and answers to the six open questions in the family plan.

**Architecture:** Intake pins upstream source and weight revisions, measures
artifact digests, locks a project-owned `uv` environment that can run the
upstream reference on CPU, extracts capability and architecture facts from the
loaded model rather than from documentation, and records everything as
`intake.json` plus a porting log. It writes no runtime code and adds no CTest
target; its gate is that every recorded number is reproducible from the
committed lockfile.

**Tech Stack:** Python 3.12, `uv` 0.11.28, PyTorch CPU, transformers 4.57.3,
upstream `qwen-tts` package pinned by git revision.

## Global Constraints

- Family stable key: `qwen3-tts`. Variant publication slug:
  `qwen3-tts-12hz-0-6b-customvoice`.
- Upstream source pin: `https://github.com/QwenLM/Qwen3-TTS` at
  `022e286b98fbec7e1e916cb940cdf532cd9f488e` (committed 2026-03-17).
- Weights pin: `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` at
  revision `85e237c12c027371202489a0ec509ded67b5e4b5` (2026-01-29),
  `license: apache-2.0`.
- The oracle runs on **CPU** with `attn_implementation="eager"` and
  `dtype=torch.float32`. Never FlashAttention, never bfloat16, never CUDA —
  `docs/port-validation.md` Phase 1 requires CPU reference capture.
- **This family's discrete-output cost has to be measured during intake, not at
  stage 4.** `docs/backends.md` requires every stage producing a discrete value,
  and everything feeding it, to run on CPU on every Execution Backend, because
  CUDA F32 matmuls compute at TF32 and that is enough to move a value across a
  rounding boundary. For Kokoro that was two stages of seven and cost 95 percent
  of synthesis time; for VITS one graph and 29 percent. This family samples a
  token every frame, and each sampled token conditions the next, so the held
  portion is most of the Talker rather than a prefix of it. Task 4 therefore
  measures the CPU path deliberately: if greedy CPU decoding of the smoke case is
  far off real time, that is a finding about whether this family can satisfy the
  rule at all, and it belongs in the intake record rather than being discovered
  after the graph is written.
- **Read "Findings From Reading qwentts.cpp" in `docs/porting/families/qwen3-tts.md`
  before Task 3.** A source read of the reference port recorded three conversion
  rules that produce no error and wrong output when missed -- the RVQ codebooks
  are EMA accumulators and must be reconstructed, convolution kernels are forced
  to F16 at load because ARM's im2col is strict about kernel dtype, and
  SnakeBeta's alpha and beta pass through `exp()` every forward. It also records
  that the 15 acoustic codebooks each carry a private embedding table and a
  private linear head, which the configuration alone does not reveal, and that
  deep talker layers need a cosine tolerance rather than max-abs. Task 3 Step 5
  reads the upstream source; these are the specific things to confirm there.
- Commit contracts, not payloads. Checkpoints go to the git-ignored
  `models/qwen3-tts-12hz-0-6b-customvoice/` cache. No training corpus is
  downloaded, and no checkpoint bytes are committed.
- `intake.json` uses `"schema": "synthesize-port-intake-v1"`, matching
  `reports/porting/kokoro/kokoro-v1-0/intake.json`.
- Host for the record: `aarch64` Linux, NVIDIA GB10. Record the host but do not
  use the GPU.
- This stage adds no C or C++ code, therefore no CTest registration. Do not
  invent a test target to satisfy `testing.md`; that gate begins at stage
  `3-convert`.

## File Structure

| Path | Responsibility |
| --- | --- |
| `scripts/envs/qwen3-tts/pyproject.toml` | Declares the pinned CPU oracle dependency set for this family. |
| `scripts/envs/qwen3-tts/uv.lock` | Locks the exact resolution so every recorded number is reproducible. |
| `reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/intake.json` | The machine-readable intake record. |
| `reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/_porting-log.md` | The human-readable narrative of what intake found. |
| `docs/porting/families/qwen3-tts.md` | Updated Status line and a new Reference Contract section. |
| `models/qwen3-tts-12hz-0-6b-customvoice/` | Git-ignored local checkpoint cache. |

---

### Task 1: Lock the CPU oracle environment

**Files:**
- Create: `scripts/envs/qwen3-tts/pyproject.toml`
- Create: `scripts/envs/qwen3-tts/uv.lock`

**Interfaces:**
- Consumes: nothing.
- Produces: a `uv` environment at `scripts/envs/qwen3-tts/` in which
  `from qwen_tts import Qwen3TTSModel` succeeds on CPU. Every later task runs
  its Python through `uv run --project scripts/envs/qwen3-tts`.

- [x] **Step 1: Write the environment declaration**

Create `scripts/envs/qwen3-tts/pyproject.toml`. The `qwen-tts` package is not on
PyPI, so it is pinned by git revision. `gradio` is excluded because it is only
used by the upstream demo CLI, and `sox` is excluded because it wraps a system
binary this path does not call.

```toml
[project]
name = "synthesize-qwen3-tts-reference"
version = "0.0.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "accelerate==1.12.0",
    "einops==0.8.1",
    "gguf==0.19.0",
    "huggingface-hub==0.30.2",
    "librosa==0.11.0",
    "numpy==1.26.4",
    "qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS@022e286b98fbec7e1e916cb940cdf532cd9f488e",
    "scipy==1.13.1",
    "soundfile==0.13.1",
    "torch>=2.7,<3",
    "torchaudio",
    "transformers==4.57.3",
]

[tool.uv]
package = false
```

- [x] **Step 2: Resolve and lock**

```bash
uv lock --project scripts/envs/qwen3-tts
```

Expected: `scripts/envs/qwen3-tts/uv.lock` is created. If resolution fails on
`onnxruntime` or another transitive dependency for aarch64, add the failing
package with an explicit version to `dependencies` and re-run. Record any such
addition and its reason in the porting log in Task 5.

- [x] **Step 3: Verify the import path works on CPU**

```bash
uv run --project scripts/envs/qwen3-tts python -c "
import torch, transformers, qwen_tts
from qwen_tts import Qwen3TTSModel
print('torch       ', torch.__version__)
print('transformers', transformers.__version__)
print('cuda_visible', torch.cuda.is_available())
print('import ok   ', Qwen3TTSModel.__name__)
"
```

Expected: prints versions and `import ok Qwen3TTSModel` with no traceback.
`cuda_visible` may print either value; the oracle will select CPU explicitly and
does not depend on this.

If `import qwen_tts` fails on a missing module, add that module to
`dependencies`, re-run Step 2, and repeat this step.

- [x] **Step 4: Record the resolved versions**

```bash
uv run --project scripts/envs/qwen3-tts python -c "
import importlib.metadata as m
for p in ['torch','transformers','accelerate','numpy','librosa','soundfile','einops']:
    print(f'{p}=={m.version(p)}')
"
```

Keep this output. Task 5 copies it verbatim into `intake.json` under
`reference_environment`.

- [x] **Step 5: Commit**

```bash
git add scripts/envs/qwen3-tts/pyproject.toml scripts/envs/qwen3-tts/uv.lock
git commit -m "Lock the qwen3-tts CPU reference environment"
```

---

### Task 2: Pin artifacts, measure digests, and audit the license

**Files:**
- Create: `models/qwen3-tts-12hz-0-6b-customvoice/` (git-ignored download cache)

**Interfaces:**
- Consumes: the environment from Task 1.
- Produces: a local checkpoint directory, a per-file SHA-256 table, and a
  license finding. Task 5 records all three.

- [x] **Step 1: Confirm the cache path is git-ignored**

```bash
git check-ignore -v models/qwen3-tts-12hz-0-6b-customvoice || echo "NOT IGNORED"
```

Expected: a line naming the `.gitignore` rule that covers it. If it prints
`NOT IGNORED`, stop and add the rule before downloading anything — committing
checkpoint bytes violates the Global Constraints.

- [x] **Step 2: Download the pinned revision**

The variant repository bundles the talker weights, the `speech_tokenizer/`
codec, and the byte-level BPE vocabulary, so this is the only download needed.

```bash
uv run --project scripts/envs/qwen3-tts python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
    revision='85e237c12c027371202489a0ec509ded67b5e4b5',
    local_dir='models/qwen3-tts-12hz-0-6b-customvoice',
)
print(p)
"
```

Expected: the path is printed and the directory contains `config.json`,
`generation_config.json`, `model.safetensors`, `merges.txt`, `vocab.json`,
`tokenizer_config.json`, `preprocessor_config.json`, and a `speech_tokenizer/`
subdirectory.

- [x] **Step 3: Measure every downloaded file**

```bash
uv run --project scripts/envs/qwen3-tts python -c "
import hashlib, json, pathlib
root = pathlib.Path('models/qwen3-tts-12hz-0-6b-customvoice')
out = []
for f in sorted(root.rglob('*')):
    if f.is_file() and '.cache' not in f.parts:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        out.append({'path': str(f.relative_to(root)), 'bytes': f.stat().st_size, 'sha256': h})
print(json.dumps(out, indent=2))
"
```

Keep this JSON array. Task 5 embeds it under `weights.files`.

- [x] **Step 4: Audit the license**

```bash
uv run --project scripts/envs/qwen3-tts python - <<'PY'
import urllib.request
for repo in ['Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', 'Qwen/Qwen3-TTS-Tokenizer-12Hz']:
    url = f'https://huggingface.co/{repo}/raw/main/README.md'
    card = urllib.request.urlopen(url).read().decode('utf-8')
    head = card.split('---')[1] if card.startswith('---') else ''
    lic = [l for l in head.splitlines() if l.startswith('license')]
    hits = [l for l in card.splitlines()
            if any(k in l.lower() for k in ('licen', 'non-commercial', 'cc-by', 'restrict'))]
    print(repo)
    print('  frontmatter:', lic or 'NONE')
    print('  prose hits :', hits[:5] or 'none')
PY
```

Expected: `license: apache-2.0` in frontmatter for both, and no prose line
asserting a non-commercial or training-data-derived restriction.

**This step is a gate, and it was run before Step 2 rather than after.** The
ordering in this plan would have discovered a blocking restriction only after
pulling 2.5 GB; running it first cannot change the answer. It was also re-run
against the downloaded card, because Step 4 as written fetches `main` while the
plan pins revision `85e237c1` -- those are not necessarily the same bytes.
If a restriction appears, stop the whole plan and
report it — this is the exact failure that removed OmniVoice from consideration,
and `docs/porting/families/qwen3-tts.md` records that a downstream port's
README is not a license source.

- [x] **Step 5: Verify the upstream source license**

```bash
uv run --project scripts/envs/qwen3-tts python -c "
import urllib.request
u='https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/022e286b98fbec7e1e916cb940cdf532cd9f488e/LICENSE'
print(urllib.request.urlopen(u).read().decode('utf-8')[:200])
"
```

Expected: the Apache License 2.0 header.

Nothing is committed by this task; its outputs are inputs to Task 5.

---

### Task 3: Extract capabilities and architecture from the loaded model

**Files:**
- None created. This task produces measurements.

**Interfaces:**
- Consumes: the environment from Task 1 and the checkpoint from Task 2.
- Produces: the Preset Voice Catalog contents, the advertised language list, and
  the config dimensions. Resolves open questions 1, 2, and 4 in
  `docs/porting/families/qwen3-tts.md`.

- [x] **Step 1: Enumerate speakers and languages**

Upstream exposes `get_supported_speakers()` and `get_supported_languages()` on
the model, which is authoritative where documentation is not.

```bash
uv run --project scripts/envs/qwen3-tts python - <<'PY'
import json, torch
from qwen_tts import Qwen3TTSModel
tts = Qwen3TTSModel.from_pretrained(
    'models/qwen3-tts-12hz-0-6b-customvoice',
    device_map='cpu', dtype=torch.float32, attn_implementation='eager',
)
spk = tts.model.get_supported_speakers()
lang = tts.model.get_supported_languages()
print(json.dumps({'speakers': sorted(spk), 'n_speakers': len(spk),
                  'languages': sorted(lang), 'n_languages': len(lang)}, indent=2,
                 ensure_ascii=False))
PY
```

Keep this output. It becomes the Preset Voice Catalog basis and answers open
question 4.

If `from_pretrained` raises because it requires `trust_remote_code`, re-run with
`trust_remote_code=True` added to the call and note that the checkpoint carries
executable modelling code — that fact must be recorded, because
`docs/scope.md` forbids loading executable code from a Model Package and the
converter must therefore reimplement rather than import it.

- [x] **Step 2: Dump the configuration**

```bash
uv run --project scripts/envs/qwen3-tts python - <<'PY'
import json
cfg = json.load(open('models/qwen3-tts-12hz-0-6b-customvoice/config.json'))
gen = json.load(open('models/qwen3-tts-12hz-0-6b-customvoice/generation_config.json'))
tok = json.load(open('models/qwen3-tts-12hz-0-6b-customvoice/speech_tokenizer/config.json'))
print('=== talker/config.json ==='); print(json.dumps(cfg, indent=2)[:4000])
print('=== generation_config.json ==='); print(json.dumps(gen, indent=2))
print('=== speech_tokenizer/config.json ==='); print(json.dumps(tok, indent=2)[:4000])
PY
```

Keep this output. Read from it and record: talker layer count, hidden width,
head counts, RoPE base, any `mrope_section` value, the codec codebook count and
size, the frame rate, and the hop length. These are the facts open questions 1
and 2 ask to confirm against upstream rather than against a third-party port.

- [x] **Step 3: Inventory the tensors**

```bash
uv run --project scripts/envs/qwen3-tts python - <<'PY'
from safetensors import safe_open
import collections
for name in ['model.safetensors', 'speech_tokenizer/model.safetensors']:
    p = f'models/qwen3-tts-12hz-0-6b-customvoice/{name}'
    with safe_open(p, framework='pt') as f:
        keys = list(f.keys())
        total = 0
        prefixes = collections.Counter()
        for k in keys:
            sl = f.get_slice(k)
            n = 1
            for d in sl.get_shape():
                n *= d
            total += n
            prefixes[k.split('.')[0]] += 1
        print(f'{name}: {len(keys)} tensors, {total:,} parameters')
        for pre, c in sorted(prefixes.items()):
            print(f'    {pre}: {c}')
PY
```

Keep this output. It is the basis for the converter's tensor catalog at stage
`3-convert` and confirms the real parameter count for the record.

- [x] **Step 4: Record whether the talker and codec are separable**

From the Step 3 prefixes, note whether the speaker embedding rows for
CustomVoice live in the talker file and whether any ECAPA-TDNN speaker-encoder
tensors are present. The family plan states CustomVoice carries no speaker
encoder; confirm or correct that here.

- [x] **Step 5: Read the upstream codec and attention source**

Open questions 1 and 2 exist specifically because the family plan's codec
description came from a third-party port and is marked *(second-hand)*. A
config dump does not answer them; the upstream implementation does.

```bash
cd /tmp && rm -rf qwen3tts-src && \
git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS qwen3tts-src && \
cd qwen3tts-src && \
git fetch --depth 1 origin 022e286b98fbec7e1e916cb940cdf532cd9f488e && \
git checkout 022e286b98fbec7e1e916cb940cdf532cd9f488e && \
wc -l qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py \
      qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py \
      qwen_tts/core/models/modeling_qwen3_tts.py
```

Read those three files and record, for the intake:

1. **Codec topology (open question 1).** The encoder downsampling stage ratios,
   the decoder upsampling ratios, the ConvNeXt upsample factor, the DAC decoder
   strides and channel widths, the residual-unit dilations, and whether the
   Snake or SnakeBeta activation applies `exp()` to its alpha and beta at
   forward time. The last one decides whether those factors can be folded once
   at load, which the converter depends on.
2. **RoPE application (open question 2).** Locate where rotary embeddings are
   applied in the talker, record whether a multi-section or multimodal rope is
   used and with what section widths, and record whether all sections receive
   the same position values for a text-plus-codec synthesis timeline. If they
   do, a 1-D collapse is exactly equivalent and that is the finding; if they do
   not, say so — the family plan's `(second-hand)` claim would then be wrong and
   stage `4-cpp` must implement the sectioned form.

```bash
cd /tmp/qwen3tts-src && \
grep -nE "rope|rotary|mrope|position_ids" qwen_tts/core/models/modeling_qwen3_tts.py | head -40
```

- [x] **Step 6: Record speaker dialect overrides**

Open question 4 asks for the speakers *and* their dialect overrides.

```bash
uv run --project scripts/envs/qwen3-tts python - <<'PY'
import torch
from qwen_tts import Qwen3TTSModel
tts = Qwen3TTSModel.from_pretrained(
    'models/qwen3-tts-12hz-0-6b-customvoice',
    device_map='cpu', dtype=torch.float32, attn_implementation='eager',
)
m = tts.model
for attr in dir(m):
    if 'speaker' in attr.lower() or 'dialect' in attr.lower():
        print(attr)
PY
```

Inspect whatever attribute holds the speaker table and record, per speaker, any
dialect or accent tag attached to it. Per `docs/languages.md`, an undocumented
accent stays unknown and is never inferred, so record only tags the model or its
card actually carries.

Nothing is committed by this task; its outputs are inputs to Task 5.

---

### Task 4: Deterministic CPU oracle smoke

**Files:**
- None created under version control. Audio goes to the ignored cache.

**Interfaces:**
- Consumes: the environment from Task 1 and the checkpoint from Task 2.
- Produces: proof that the upstream reference runs on CPU, the native sample
  rate and frame count, and a determinism finding. Resolves open questions 3 and
  5.

This is the riskiest task. Upstream's own example targets CUDA with bfloat16 and
FlashAttention 2; none of those are available or permitted here.

It is also where this family's viability under the discrete-output rule gets
decided. Step 1's real-time factor is not just a CPU practicality datum: because
every sampled token conditions the next, the rule in `docs/backends.md` would hold
most of the Talker on CPU, so the CPU number *is* approximately the pinned CUDA
number. Record it as both. If it lands far from real time, say so in the intake
record as a finding about the family rather than as a note about the oracle —
Kokoro cost 95 percent and VITS 29 percent for holding two stages and one graph
respectively, and this family holds far more.

- [x] **Step 1: Run one greedy synthesis on CPU**

`do_sample=False` was assumed here to select greedy decoding on both the talker
and the sub-talker. **That is wrong** -- it governs the Talker only, and the
sub-talker's `subtalker_dosample` defaults to `True`. Both must be set to
`False`, or the run is silently non-deterministic. Corrected 2026-07-27;
which is what makes an autoregressive oracle reproducible. Upstream defaults are
`do_sample=True, top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05`.

The speaker and language are selected deterministically from the model's own
catalog rather than hardcoded, so this script runs unmodified and the same
choice reproduces on any machine. English is preferred when advertised because
the smoke text is English; the first sorted entry is the fallback.

```bash
uv run --project scripts/envs/qwen3-tts python - <<'PY'
import time, torch, numpy as np, soundfile as sf
from qwen_tts import Qwen3TTSModel
tts = Qwen3TTSModel.from_pretrained(
    'models/qwen3-tts-12hz-0-6b-customvoice',
    device_map='cpu', dtype=torch.float32, attn_implementation='eager',
)
langs = sorted(tts.model.get_supported_languages())
lang = next((l for l in langs if l.lower().startswith('english')), langs[0])
speaker = sorted(tts.model.get_supported_speakers())[0]
print('chosen_language', lang)
print('chosen_speaker ', speaker)
t0 = time.time()
wavs, sr = tts.generate_custom_voice(
    text='Qwen3-TTS is awesome!',
    language=lang,
    speaker=speaker,
    do_sample=False,
    max_new_tokens=2048,
)
dt = time.time() - t0
w = np.asarray(wavs[0])
print('sample_rate  ', sr)
print('frames       ', w.shape)
print('dtype        ', w.dtype)
print('duration_s   ', w.shape[0] / sr)
print('wall_s       ', round(dt, 3))
print('rtf          ', round(dt / (w.shape[0] / sr), 3))
print('finite       ', bool(np.isfinite(w).all()))
print('peak         ', float(np.abs(w).max()))
sf.write('models/qwen3-tts-12hz-0-6b-customvoice/_smoke_a.wav', w, sr)
PY
```

Record `chosen_language` and `chosen_speaker`; Task 5 puts both in
`intake.json` under `oracle_smoke`.

Expected: a finite waveform, `sample_rate 24000`, and a printed real-time
factor. Record `rtf` — it is the CPU practicality evidence open question 3 asks
for, and the family plan's accepted risks depend on it.

If this fails because CPU lacks a kernel used by the model, record the exact
error and stop; a CPU oracle is mandatory and its absence is a finding that
changes the family plan, not something to work around by moving to CUDA.

- [x] **Step 2: Repeat the identical greedy call**

Re-run the exact command from Step 1, writing to `_smoke_b.wav` instead of
`_smoke_a.wav`.

- [x] **Step 3: Compare the two runs**

```bash
uv run --project scripts/envs/qwen3-tts python - <<'PY'
import numpy as np, soundfile as sf
a, sra = sf.read('models/qwen3-tts-12hz-0-6b-customvoice/_smoke_a.wav')
b, srb = sf.read('models/qwen3-tts-12hz-0-6b-customvoice/_smoke_b.wav')
print('same_rate  ', sra == srb)
print('same_length', a.shape == b.shape)
if a.shape == b.shape:
    print('max_abs_diff', float(np.abs(a - b).max()))
    print('identical   ', bool(np.array_equal(a, b)))
PY
```

Expected under greedy decoding: identical output. Record the result either way.

A greedy oracle that is *not* reproducible against itself would mean the
stochastic-replay approach in `docs/porting/families/qwen3-tts.md` needs more
than a captured code sequence, which is a material finding for stage
`5-port-validate`.

- [x] **Step 4: Measure the sampled path for the stochastic capability**

```bash
uv run --project scripts/envs/qwen3-tts python - <<'PY'
import torch, numpy as np
from qwen_tts import Qwen3TTSModel
tts = Qwen3TTSModel.from_pretrained(
    'models/qwen3-tts-12hz-0-6b-customvoice',
    device_map='cpu', dtype=torch.float32, attn_implementation='eager',
)
langs = sorted(tts.model.get_supported_languages())
lang = next((l for l in langs if l.lower().startswith('english')), langs[0])
speaker = sorted(tts.model.get_supported_speakers())[0]
outs = []
for _ in range(2):
    wavs, sr = tts.generate_custom_voice(
        text='Qwen3-TTS is awesome!', language=lang, speaker=speaker,
        do_sample=True, max_new_tokens=2048,
    )
    outs.append(np.asarray(wavs[0]))
print('lengths', [o.shape[0] for o in outs])
n = min(o.shape[0] for o in outs)
print('max_abs_diff_over_common_prefix', float(np.abs(outs[0][:n] - outs[1][:n]).max()))
PY
```

Expected: two unseeded sampled runs differ, and may differ in length. This
establishes that the package sets the stochastic capability, matching how
Kokoro's intake recorded its own stochastic behaviour.

- [x] **Step 5: Decide the audio delivery claim**

Using the Step 1 and Step 2 evidence plus the `CONTEXT.md` definitions, decide
whether Stage 1 claims **Chunked Audio Delivery** only, or whether the causal
codec decoder justifies a **Native Streaming Synthesis** claim. The upstream
call used here is non-streaming, so the default answer is Chunked Audio
Delivery, and a Native Streaming Synthesis claim would need its own validated
evidence at a later stage. Write the decision down for Task 5; this answers open
question 5.

---

### Task 5: Write the intake record and update the family plan

**Files:**
- Create: `reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/intake.json`
- Create: `reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/_porting-log.md`
- Modify: `docs/porting/families/qwen3-tts.md`

**Interfaces:**
- Consumes: the recorded outputs of Tasks 1 through 4.
- Produces: the committed intake contract that stage `2-oracle` builds on.

- [x] **Step 1: Write `intake.json`**

Mirror the top-level shape of
`reports/porting/kokoro/kokoro-v1-0/intake.json`. Read that file first, then
write this one with the same key names so both are queryable by the same tools.

Required top-level keys and what fills them:

```text
schema               "synthesize-port-intake-v1"
recorded_on          the date intake ran, ISO yyyy-mm-dd
family               "qwen3-tts"
variant              "qwen3-tts-12hz-0-6b-customvoice"
status               a short state string, e.g. "intake_complete_oracle_smoke_passed"
publication_status   "not_started"
source               repository, revision 022e286b..., license finding from Task 2 Step 5
weights              repository, revision 85e237c1..., files array from Task 2 Step 3
architecture         dims, module inventory, audio facts from Task 3 Steps 2 and 3
stochastic_behavior  is_stochastic, sources, measured values from Task 4 Steps 3 and 4
license_audit        the Task 2 Step 4 result for both repositories, plus a
                     training_data_note recording that Alibaba does not disclose
                     its corpora and the Apache-2.0 grant is the basis relied on
frontend_requirements  byte-level BPE from the variant's own vocab.json and
                     merges.txt; runtime_text_input claim for Stage 1
reference_environment  host, uv version, and the versions printed in Task 1 Step 4
oracle_smoke         text, speaker, language, sample_rate, frames, rtf, greedy
                     reproducibility result from Task 4
open_decisions       anything Tasks 2 through 4 could not settle
```

- [x] **Step 2: Write the porting log**

Create `_porting-log.md` following the structure of
`reports/porting/kokoro/kokoro-v1-0/_porting-log.md`: a dated heading, a bullet
list of what was pinned and measured, then subsections for the license audit,
the stochastic finding, and any open decision. Write what was measured, not what
was expected. Where a number contradicts the family plan, say so explicitly.

- [x] **Step 3: Update the family plan**

In `docs/porting/families/qwen3-tts.md`:

1. Change the Status line to record intake completion and the date, following
   the wording in `docs/porting/families/kokoro.md`.
2. Add a `## Reference Contract` section naming the pinned source revision, the
   pinned weights revision, the measured checkpoint digest, the license finding,
   and the oracle entry point `Qwen3TTSModel.generate_custom_voice`.
3. Strike each of the six Open Questions that Tasks 2 through 4 answered, moving
   the answer into the body. Leave any that remain open, and say why.

- [x] **Step 4: Verify nothing large or ignored is staged**

```bash
git status --short
git diff --cached --stat
```

Expected: only the three files listed above. If any file under
`models/` appears, unstage it — the Global Constraints forbid committing
checkpoint bytes.

- [x] **Step 5: Commit**

```bash
git add reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/intake.json \
        reports/porting/qwen3-tts/qwen3-tts-12hz-0-6b-customvoice/_porting-log.md \
        docs/porting/families/qwen3-tts.md
git commit -m "Add qwen3-tts intake packet for the CustomVoice variant"
```

---

## Completion Criteria

Intake is complete when all of the following hold:

1. `uv run --project scripts/envs/qwen3-tts` can load the pinned checkpoint and
   synthesize on CPU, and the lockfile is committed.
2. Every digest, dimension, speaker name, and timing in `intake.json` was
   measured on this machine, not copied from documentation or from a
   third-party port.
3. The license audit found `license: apache-2.0` with no prose restriction on
   both the variant repository and the shared tokenizer repository.
4. Open questions 1 through 5 in `docs/porting/families/qwen3-tts.md` are
   answered or explicitly restated as still open with a reason. Open question 6
   is answered by the license audit.
5. No checkpoint bytes are committed.
