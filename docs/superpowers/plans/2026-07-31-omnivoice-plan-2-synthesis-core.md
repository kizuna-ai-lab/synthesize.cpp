# OmniVoice Family — Plan 2: Synthesis Core (slices 4–6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Greedy end-to-end synthesis on CPU with exact-token parity against the oracle: harden the oracle so its baselines can be trusted, land the Plan-1 carry-over cleanups whose blast radius grows with time, then build the bidirectional generator graph (single-forward parity against the step-0 probes), the greedy mask-predict decode loop (exact 8×T token-grid equality against the oracle — this family's `structural_exactness`), and the Higgs Audio V2 codec decode to a waveform, all driven by a replay runner + validator harness whose FIRST measured tolerances are committed before support is declared.

**Architecture:** OmniVoice's generator is a Qwen3-0.6B block stack run **bidirectionally, cache-free**, over an 8-codebook × 1025-vocabulary canvas at 25 Hz. Each decode step scores every still-masked canvas position from a conditional and an unconditional forward combined in log-softmax space (CFG scale 2.0), bans the mask id, and commits a scheduled number of positions by flat top-k over the penalized confidences; the committed grid is decoded by the Higgs Audio V2 DAC-style convolutional decoder (hop 960) to 24 kHz mono. Everything discrete — argmax, ranking, commit — runs on host CPU per the discrete-outputs placement rule. Design record: `docs/superpowers/specs/2026-07-30-omnivoice-family-design.md` §5/§6/§8 (slices 4–6, both dated amendments apply); authoritative formulas: `docs/porting/families/omnivoice.md`; debt register consumed by this plan: `docs/superpowers/plans/2026-07-30-omnivoice-plan-2-carryover.md`.

**Tech Stack:** C++17 + GGML/GGUF (submodule, vanilla — zero new operators), Python oracle in the locked `scripts/envs/omnivoice/` uv env, CMake/CTest.

**Follow-on plans (not in this document):** Plan 3 = public sampling path (Gumbel, seed contract) + cloning (internal resampler, HuBERT, RVQ encode, Reference Audio / Description Text profiles, Serialized Profiles); Plan 4 = Port Validation Suite end-to-end, Quantization Profiles, Execution Backends, Adapters, ship.

**Non-goals of Plan 2 (say no explicitly):**

- **The public seam stays STUBBED.** `synth_synthesize`'s omnivoice branch keeps returning `SYNTH_ERR_INTERNAL` / `synthesis.not_implemented` (controller ruling: the public defaults are the *sampled* path, which is Plan 3). `Model::run_synthesis` is reached by the family-internal API and the replay runner only.
- No Gumbel sampling, no seed contract, no `class_temperature > 0` path (carry-over item 15's sampled-path parts land in Plan 3 with the sampler).
- No cloning synthesis: no reference encoder, resampler, HuBERT, RVQ encode, or profile machinery. The two greedy clone golden cases ARE covered — their oracle-dumped reference tokens are replayed through the prompt (a replay input, exactly like qwen3-tts's replayed codes), which exercises the generator and loop without any encode path. Carry-over 13 (semantic-hidden probe semantics) and 14 (quiet-reference unit case + the `ref_rms == 0.0` inverse-gate decision) are Plan-3 decisions; Plan 2 mirrors only the no-reference `peak-normalise-to-0.5` branch its cases exercise.
- No quantization profiles, no CUDA/backends work, no adapter registration, no ship artifacts (Plan 4).
- Carry-over 16 (diff the two `tokenizer.json` pre-tokenizers directly) stays owed: its trigger — the qwen3-tts weights being materialized on this host — does not occur in Plan 2.
- No long-form chunking, and no token-sequence input (superseded by the design spec's 2026-07-31 amendment).

## Global Constraints

- **Worktree/branch:** all work happens on the current worktree branch `omnivoice-plan-2` (based on merged main containing all of Plan 1). Never push; never open a PR — both remain jiangzhuo's call, per repo policy.
- **Exact-token equality is the gate for slices 5–6.** A greedy golden case passes `structural_exactness` only when the port's committed 8×T grid equals the oracle's `codes/grid.i32` **exactly** — it is never a threshold, and the tolerance file never carries a token-grid entry.
- **Tolerances are inputs, not outputs.** `tests/tolerances/omnivoice.json` is populated once, from the first working pair, reviewed, and committed **before** support is declared (docs/port-validation.md discipline). A validator's `--check` refuses to run against an absent cell; no task may weaken a committed threshold to accept its own output.
- **Pinned artifacts this plan compares against:**
  - Package GGUF `models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf`, sha256 `3ecaa5e2f6fbd735296ba1cd60680c90467be22d2140dc4f208fe80111ecb9e5` (post-license-ruling cut; `general.license = "other"`) [superseded 2026-08-03 by `f6d504ffaddcbf32f80f1f6c847f075bbd5d2c7b50fe95a194ceb635772f9fa3`, a metadata-only re-cut fixing a `max_output_frames` unit error; see the porting log's 2026-08-03 entry].
  - Weights revision `WEIGHTS_REV = c5fdb5ccb189668d56333f77ba2629f4cd7535f4`; source revision `OMNI_REV = 468e927ba3716cd8dd86421148dfb3046e9f9d7b` (package 0.2.1).
  - Oracle invocation, always: `uv run --project scripts/envs/omnivoice --locked python …` (F32 on CPU).
  - Oracle artifacts under `build/goldens/omnivoice/<case-id>/` (all 20 cases dumped; raw little-endian f32/i32, no headers; shapes in each case's `metadata.json`).
- **The numbers, stated once** (used verbatim everywhere below): generator 28 layers, hidden 1024, 16 attention / 8 KV heads, head_dim 128, ffn 3072, rms_norm_eps 1e-6, rope_theta 1e6, text vocab 151,676; canvas 8 codebooks × 1025 vocabulary, mask id 1024 (audio-embedding offset for codebook c is c·1025; the stacked tables are [8200 rows, 1024 wide]); codec hop 960, 24 kHz mono, 25 Hz frame rate, upsampling ratios {8,5,4,2,3}, RVQ codebook_dim 64 / codebook_size 1024, concat width 1024 = codec hidden 256 + HuBERT hidden 768, decoder widths 1024→512→256→128→64→32; decoding defaults num_step 32 (fast mode 16), guidance_scale 2.0, t_shift 0.1, layer_penalty_factor 5.0; probe layers {0, 7, 14, 21, 27}; golden suite 20 cases, of which 17 greedy (both temperatures zero) and 3 sampled.
- Weight licenses (unchanged from Plan 1): LM = CC-BY-NC (no version stated upstream); codec = Boson Higgs Audio 2 Community License; code = Apache-2.0. Publication ceiling = Restricted Model Package (ADR 0018). Never label any artifact apache-2.0. Reference port `rockerritesh/omnivoice-tts.cpp` is PolyForm Noncommercial: read-only, zero code reuse; any code adopted from the MIT/Apache ports records its notice in `THIRD_PARTY_NOTICES.md` at adoption time.
- Every C++ slice ends with unit tests registered under the `unit` label AND a clean sanitizer run: `cmake --build build-sanitize --target synthesize-check-unit` (configure per CLAUDE.md if the tree is fresh).
- clang-format: run `scripts/ci/clang-format.sh --fix` **before** `git add` in every commit touching C/C++ — the trap is real: `git add` first and then formatting leaves the unformatted copy staged while the working tree looks clean, and `--check-diff` then fails CI on a commit that looked done. Never format `ggml/`.
- Commit message footer, every commit in this plan:

  ```text
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
  ```

**Three decisions this plan settles (recorded here so no task re-litigates them):**

1. **Two forwards per step, not a padded 2-row batch.** Upstream batches the conditional (row 0) and unconditional (row 1) sequences padded to one length with per-row rectangular attention masks (`_generate_iterative`, `batch_attention_mask[i, :, :c_len, :c_len] = True`; padding attends only to itself). Those masks make the rows mathematically independent: no real position ever attends a pad or the other row, so computing the two branches as two separate full-attention forwards is exactly the same function evaluated without padding waste. The 2-D `codec_transformer_layer` precedent then transfers verbatim, the uncond graph is built at its own (shorter) length, and the probes — which compare only the conditional row — see an identical computation. Parity is against the oracle's dumped values under tolerances, not against its batching layout.
2. **Fresh `GraphRun` per forward, refilled `Persistent` inputs** (the qwen3-tts production pattern), not the spec's "graphs built once per canvas shape and reused". Rebuilding is the measured, known-safe precedent; its cost is captured per step in `setup_seconds` so stage 7 (backends) can revisit reuse with numbers instead of an argument. The spec sentence is an optimization intent, not a correctness requirement; this deviation is recorded here.
3. **The residual output scaling lives inside the family's synthesis path.** Intake proved `peak-normalise-to-0.5` survives every oracle switch for no-reference output; `Model::run_synthesis` applies it after codec decode, faithful to the oracle, and no public normalization control is added in v1. The family doc's open question is closed with exactly this sentence in Task 14.

---

### Task 1: Oracle hardening — pin dumper determinism, verify every pinned input

Carry-over items 1 and 2. Do this BEFORE trusting any parity gate: the exact-token baselines are currently proven reproducible on one machine only, and only the clone reference audio is digest-checked today.

**Files:**
- Modify: `scripts/dump_reference_omnivoice_pytorch.py`

**Interfaces:**
- Produces: per-case `metadata.json` gains an `"environment"` block and the dump report gains `"environment"` + `"verified_inputs"`; consumed by Task 2's full re-dump and by anyone regenerating baselines on another machine.

- [ ] **Step 1: Pin the torch execution configuration**

In `main()`, immediately after the runtime `import torch` (currently right after the `--weights-dir` checks, before `from omnivoice.models.omnivoice import OmniVoice`), insert:

```python
    import torch

    # Exact-token baselines must be reproducible off this machine, not merely on
    # it. The thread pool is pinned because oneDNN/MKL reduction order can move
    # with pool size, and deterministic algorithms are demanded rather than
    # hoped for: an op with no deterministic CPU path aborts the dump instead of
    # quietly varying. set_num_interop_threads must run before any parallel op,
    # which is why this sits directly under the import.
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
```

Add a module-level helper next to `volume_branch()`:

```python
def torch_environment(torch) -> dict:
    """The execution configuration a baseline depends on, recorded per dump."""
    return {
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
    }
```

In `run_case()`, add to the `metadata` dict, directly after the `"probes"` block:

```python
        "environment": torch_environment(torch),
```

In `main()`, add to the `report` dict, after `"module_paths"`:

```python
        "environment": torch_environment(torch),
```

- [ ] **Step 2: Digest-verify the dumper's weight/config/tokenizer inputs**

Add a module-level helper next to `reference_digest()`:

```python
def verify_pinned_inputs(manifest: dict, weights_dir: pathlib.Path) -> list[str]:
    """sha256-verify every weights-repository input the manifest pins.

    Until now only the clone reference audio was checked; the weights, configs
    and tokenizer the dump actually reads were trusted. A parity baseline dumped
    from silently different inputs would be wrong in a way no later gate could
    localise, so a mismatch stops the dump.
    """
    marker = "/resolve/"
    verified = []
    for artifact in manifest["source"]["artifacts"]:
        locator = artifact["locator"]
        if marker not in locator:
            # Source-repository files (the Apache LICENSE) and the clone
            # reference (checked by materialise_reference) are not dump inputs.
            continue
        relative = locator.split(marker, 1)[1].split("/", 1)[1]
        local = weights_dir / relative
        if not local.is_file():
            raise SystemExit(f"{local}: the manifest pins this input and it is missing")
        actual = hashlib.sha256(local.read_bytes()).hexdigest()
        if actual != artifact["sha256"]:
            raise SystemExit(
                f"{local}: sha256 {actual} does not match the manifest's "
                f"{artifact['sha256']}; refusing to dump against unpinned inputs"
            )
        verified.append(relative)
    required = {"model.safetensors", "audio_tokenizer/model.safetensors",
                "config.json", "audio_tokenizer/config.json", "tokenizer.json"}
    missing = required - set(verified)
    if missing:
        raise SystemExit(f"the manifest pins no digest for dump inputs: {sorted(missing)}")
    return verified
```

Call it in `main()` right after the `--weights-dir` is-a-directory check and before the torch import (it needs only hashlib):

```python
    verified_inputs = verify_pinned_inputs(manifest, arguments.weights_dir)
    print(f"verified {len(verified_inputs)} pinned inputs against the manifest", flush=True)
```

and add `"verified_inputs": verified_inputs,` to the `report` dict beside `"environment"`.

- [ ] **Step 3: RED — prove the digest gate bites**

Build a deliberately wrong weights directory out of symlinks (the big files) and one corrupted small file:

```bash
mkdir -p /tmp/omni-fake/audio_tokenizer
ln -sf "$(pwd)/models/omnivoice-0-6b/model.safetensors" /tmp/omni-fake/model.safetensors
ln -sf "$(pwd)/models/omnivoice-0-6b/audio_tokenizer/model.safetensors" /tmp/omni-fake/audio_tokenizer/model.safetensors
cp models/omnivoice-0-6b/config.json /tmp/omni-fake/config.json
cp models/omnivoice-0-6b/audio_tokenizer/config.json /tmp/omni-fake/audio_tokenizer/config.json
cp models/omnivoice-0-6b/audio_tokenizer/LICENSE /tmp/omni-fake/audio_tokenizer/LICENSE
printf '{"corrupted": true}' > /tmp/omni-fake/tokenizer.json
uv run --project scripts/envs/omnivoice --locked python scripts/dump_reference_omnivoice_pytorch.py \
  --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
  --weights-dir /tmp/omni-fake --case omni-short-en; echo "exit=$?"
```

Expected: `tokenizer.json: sha256 … does not match the manifest's …; refusing to dump against unpinned inputs`, nonzero exit, **no torch import, no model load** (fails in under a second).

- [ ] **Step 4: GREEN — one-case re-dump under the pinned configuration**

Record the pre-change binary digests first, then re-dump one greedy case:

```bash
sha256sum build/goldens/omnivoice/omni-short-en/codes/grid.i32 \
          build/goldens/omnivoice/omni-short-en/audio/pcm.f32 | tee /tmp/omni-short-en.before
uv run --project scripts/envs/omnivoice --locked python scripts/dump_reference_omnivoice_pytorch.py \
  --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
  --weights-dir models/omnivoice-0-6b --case omni-short-en
sha256sum build/goldens/omnivoice/omni-short-en/codes/grid.i32 \
          build/goldens/omnivoice/omni-short-en/audio/pcm.f32
python3 -c "import json; m=json.load(open('build/goldens/omnivoice/omni-short-en/metadata.json')); print(m['environment'])"
```

Expected: the two binary digests match the `before` capture (thread pinning is not expected to move CPU GEMM results; if either digest moved, do NOT revert the pinning — the pinned configuration supersedes the unpinned one, note the change in `reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`, and Task 2's full re-dump refreshes every baseline consistently); `metadata.environment` prints `num_threads: 1, num_interop_threads: 1, deterministic_algorithms: True`.

- [ ] **Step 5: Commit**

```bash
git add scripts/dump_reference_omnivoice_pytorch.py
git commit -m "$(cat <<'EOF'
Pin the omnivoice oracle's threads and determinism, verify its pinned inputs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 2: Pin the chunking parameters in the manifest, then one full re-dump with byte-identity evidence

Carry-over items 3 and 10. `omni-long-boundary` (719 frames = 28.76 s) sits just under upstream's default `audio_chunk_threshold` of 30 s; an upstream default change would silently reroute the case through chunked long-form synthesis and change what it means. The dumper already whitelists both keys in `GEN_CONFIG_KEYS`, so pinning them in each case's `oracle.parameters` flows straight into `generate()`.

**Files:**
- Modify: `tests/golden/omnivoice/omnivoice-0-6b.manifest.json` (all 20 cases)
- Modify: `scripts/dump_reference_omnivoice_pytorch.py` (require the new keys; refuse a chunkable suite)
- Modify: `reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md` (re-dump evidence with re-runnable commands)

**Interfaces:**
- Consumes: Task 1's hardened dumper.
- Produces: refreshed `build/goldens/omnivoice/**/metadata.json` (environment + chunk pins) with binary artifacts proven byte-identical — the baselines every later task compares against.

- [ ] **Step 1: Pin the two parameters in every case**

```bash
uv run --project scripts/envs/vits --locked python - <<'EOF'
import json, pathlib
path = pathlib.Path("tests/golden/omnivoice/omnivoice-0-6b.manifest.json")
manifest = json.loads(path.read_text(encoding="utf-8"))
for case in manifest["cases"]:
    parameters = case["oracle"]["parameters"]
    # Upstream defaults at the pinned revision (OmniVoiceGenerationConfig:
    # audio_chunk_duration=15.0, audio_chunk_threshold=30.0). Pinned so a
    # default change upstream cannot silently reroute a case through the
    # chunked long-form path; 30 s = 750 frames = the package ceiling, so no
    # golden case can ever chunk.
    parameters["audio_chunk_duration"] = 15.0
    parameters["audio_chunk_threshold"] = 30.0
path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
EOF
git diff --stat tests/golden/omnivoice/omnivoice-0-6b.manifest.json
```

Inspect `git diff`: exactly 40 added lines (2 per case), nothing else reformatted. If the diff shows wholesale reformatting, the file's on-disk style differs from `indent=2` — match whatever `json.dumps` arguments reproduce the committed formatting before proceeding (compare against `git show HEAD -- tests/golden/omnivoice/omnivoice-0-6b.manifest.json`).

- [ ] **Step 2: Make the dumper require and enforce the pins**

In `load_manifest()`, extend the required-parameters tuple:

```python
        for key in ("num_step", "position_temperature", "class_temperature", "language",
                    "instruct", "postprocess_output",
                    "audio_chunk_duration", "audio_chunk_threshold"):
            require(parameters, key, f"{where}.oracle.parameters")
```

and directly after that loop add the structural guard (add `FRAME_RATE_HZ = 25.0` to the module constants beside `SAMPLES_PER_FRAME`):

```python
        threshold_frames = float(parameters["audio_chunk_threshold"]) * FRAME_RATE_HZ
        if threshold_frames < float(contract["max_output_frames"]):
            raise ManifestError(
                f"{where}: audio_chunk_threshold {parameters['audio_chunk_threshold']} s is "
                f"{threshold_frames:.0f} frames, below max_output_frames "
                f"{contract['max_output_frames']}; a golden case could silently take the "
                "chunked long-form path, which is out of scope"
            )
```

Run the manifest contract test and the dumper's validate-only path:

```bash
ctest --test-dir build --output-on-failure -R synthesize-golden-manifest-contract
python3 scripts/dump_reference_omnivoice_pytorch.py \
  --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json --validate-only
```

Expected: contract test PASS; validate-only prints the planned work and `manifest ok`.

- [ ] **Step 3: Full re-dump, byte-identity proven**

```bash
find build/goldens/omnivoice -name '*.i32' -o -name '*.f32' | grep -v '_smoke\|_pre_redump\|-replay' \
  | sort | xargs sha256sum > /tmp/omnivoice_pre_redump.sha256
uv run --project scripts/envs/omnivoice --locked python scripts/dump_reference_omnivoice_pytorch.py \
  --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
  --weights-dir models/omnivoice-0-6b \
  --report build/goldens/omnivoice/dump-report.json
find build/goldens/omnivoice -name '*.i32' -o -name '*.f32' | grep -v '_smoke\|_pre_redump\|-replay' \
  | sort | xargs sha256sum > /tmp/omnivoice_post_redump.sha256
diff /tmp/omnivoice_pre_redump.sha256 /tmp/omnivoice_post_redump.sha256 && echo BINARIES-UNCHANGED
```

Expected: tens of minutes of CPU, then `BINARIES-UNCHANGED` — greedy cases are RNG-free and the sampled cases are seeded, so with unchanged logits every binary artifact reproduces; only the twenty `metadata.json` files change (environment + the two chunk pins in `decoding`). If any binary moved, Task 1 Step 4's supersession rule applies: the pinned-configuration dump is the new baseline; record exactly which files moved and why in the porting log.

- [ ] **Step 4: Record the evidence re-runnably (carry-over 10)**

Append a dated section to `reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`:

```markdown
## 2026-07-31 — Oracle hardening and re-dump (Plan 2 Tasks 1–2)

The dumper now pins torch to one intra-op and one inter-op thread with
deterministic algorithms demanded, records that configuration in every
metadata.json, and sha256-verifies all five weights-repository inputs against
the manifest before loading anything. audio_chunk_duration/threshold are pinned
at 15.0/30.0 in every case; 30 s equals the 750-frame package ceiling, so no
golden case can take the chunked path, and load_manifest refuses a manifest
where that stops being true.

Full re-dump under the pinned configuration: every `.i32`/`.f32` artifact
byte-identical before and after. Re-runnable evidence commands (from the repo
root; the dumper re-verifies greedy double-run identity in-process as well):

    find build/goldens/omnivoice -name '*.i32' -o -name '*.f32' | sort | xargs sha256sum > before.sha256
    uv run --project scripts/envs/omnivoice --locked python scripts/dump_reference_omnivoice_pytorch.py \
      --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
      --weights-dir models/omnivoice-0-6b --report build/goldens/omnivoice/dump-report.json
    find build/goldens/omnivoice -name '*.i32' -o -name '*.f32' | sort | xargs sha256sum > after.sha256
    diff before.sha256 after.sha256
```

(Adjust the byte-identity sentence if Step 3 found movement, per the supersession rule.)

- [ ] **Step 5: Commit**

```bash
git add tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
        scripts/dump_reference_omnivoice_pytorch.py \
        reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md
git commit -m "$(cat <<'EOF'
Pin the omnivoice chunking parameters and re-prove the oracle baselines

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 3: Register the omnivoice-env Python gate + converter polish batch

Carry-over items 4, 17, and 20. `test_convert_omnivoice.py`'s positive generation-defaults test (`GenerationDefaultsTests.test_defaults_are_read_from_the_upstream_dataclass`) skips forever under the VITS-env discovery because the `omnivoice` package only exists in `scripts/envs/omnivoice`; register a second CTest that runs the module where the import resolves. While both files are open, land the seven-item converter polish batch.

**Files:**
- Modify: `tests/CMakeLists.txt`
- Modify: `scripts/convert-omnivoice.py`
- Modify: `tests/python/test_convert_omnivoice.py`

**Interfaces:**
- Produces: CTest `synthesize-omnivoice-python-unit` (labels `unit;python;omnivoice`) — the gate Tasks 13–14 rely on for python-side changes.

- [ ] **Step 1: RED**

```bash
ctest --test-dir build -R synthesize-omnivoice-python-unit
```

Expected: `No tests were found!!!`

- [ ] **Step 2: Register the omnivoice-env test (and fix the stale comment, item 20)**

In `tests/CMakeLists.txt`, inside the `if(SYNTH_BUILD_PYTHON_TESTS)` block, first extend the comment above `synthesize-vits-python-unit` — append to its existing text:

```cmake
    # The OmniVoice converter tests ride this same discovery; their one
    # package-dependent case skips here and runs for real under
    # synthesize-omnivoice-python-unit below.
```

then add, after the `synthesize-vits-python-unit` registration:

```cmake
    # The OmniVoice converter's positive generation-defaults test needs the
    # upstream package, which exists only in this family's locked environment;
    # under the VITS discovery above it skips forever. Running the module where
    # the import resolves enforces the "defaults come from the pinned package"
    # rule instead of permanently skipping it.
    add_test(
        NAME synthesize-omnivoice-python-unit
        COMMAND ${SYNTH_UV_EXECUTABLE} run
            --project ${CMAKE_SOURCE_DIR}/scripts/envs/omnivoice
            --locked python -m unittest -v
            tests.python.test_convert_omnivoice)
    set_tests_properties(synthesize-omnivoice-python-unit PROPERTIES
        LABELS "unit;python;omnivoice"
        TIMEOUT 600
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
```

- [ ] **Step 3: Converter polish batch (carry-over 17, all seven items)**

Each edit to `scripts/convert-omnivoice.py` unless stated:

1. **`emitted_names()` drops what the converter drops.** In `tests/python/test_convert_omnivoice.py`'s `NameLengthTests.emitted_names`, the list currently keeps the `fc1.*`/`decoder_semantic.*` prefixes and the RVQ EMA buffers the converter never emits (838 names against the report's 798). Align it and pin the count:

   ```python
   def emitted_names(self) -> list[str]:
       """Every name the converter will hand to `shorten_name`."""
       data = inventory()
       names = list(data["generator"])
       names.remove("codebook_layer_offsets")
       for name in data["codec"]:
           if any(name.startswith(prefix) for prefix in convert.DROP_PREFIXES):
               continue
           if name.endswith((".embed_avg", ".cluster_size", ".inited")):
               continue  # RVQ training buffers the converter skips
           if convert.PARAMETRIZATION_INFIX in name:
               if name.endswith(".original1"):
                   continue
               name = convert.folded_weight_name(name)
           names.append("codec." + name)
       return names

   def test_emitted_names_match_the_report_count(self) -> None:
       self.assertEqual(len(self.emitted_names()), 798,
                        "the helper no longer models what the converter emits")
   ```

   (If the EMA suffixes in `tensor-inventory.json` differ from these three, read the converter's `skip_codebook_training_buffers` and use its exact suffix set.)
2. **Config-structure KeyError guards.** Add a helper beside `load_json`:

   ```python
   def require_config(mapping: dict[str, Any], key: str, where: str) -> Any:
       if key not in mapping:
           raise ConverterError(f"{where} carries no {key!r}; the checkpoint layout moved")
       return mapping[key]
   ```

   and route `add_metadata`'s deep accesses through it: `llm = require_config(config, "llm_config", "config.json")`, `require_config(llm, "rope_parameters", "llm_config")["rope_theta"]` → `require_config(require_config(llm, "rope_parameters", "llm_config"), "rope_theta", "rope_parameters")`, and `acoustic = require_config(codec_config, "acoustic_model_config", "audio_tokenizer/config.json")`. A bare `KeyError` out of a moved checkpoint is a stack trace, not a diagnosis.
3. **Weight-norm fold clash assert.** In `fold_weight_norm`, after computing the folded name, before storing:

   ```python
       if folded in tensors:
           raise ConverterError(
               f"{folded} already exists as a plain tensor; folding the parametrized pair "
               "would silently overwrite it"
           )
   ```

   Add the matching test to `WeightNormFoldTests`:

   ```python
   def test_a_fold_that_would_overwrite_a_plain_weight_is_an_error(self) -> None:
       tensors, _ = weight_norm_pair()
       tensors["semantic_model.encoder.pos_conv_embed.conv.weight"] = torch.ones(4)
       with self.assertRaises(convert.ConverterError) as caught:
           convert.fold_weight_norm(tensors, convert.Conversion(), "codec.")
       self.assertIn("overwrite", str(caught.exception))
   ```
4. **Absent-package test discrimination.** In `GenerationDefaultsTests.test_an_absent_package_is_an_error_not_a_fallback_table`, the assertion `assertIn("omnivoice", …)` passes on any message containing the family name. Assert the probed module instead: `self.assertIn("omnivoice_not_installed_anywhere", str(caught.exception))` (and, if `read_generation_defaults`'s message does not currently include the module name, make it: `f"cannot import {module_name}; …"`).
5. **`verify_gguf` compares shapes, not element counts.** Replace the element-count check:

   ```python
        expected_shape = tuple(reversed(output.array.shape))
        if tuple(int(d) for d in tensor.shape) != expected_shape:
            raise ConverterError(
                f"{output.name}: wrote shape {expected_shape}, read back "
                f"{tuple(int(d) for d in tensor.shape)}"
            )
   ```

   (GGUFReader reports ne order — the reverse of numpy's.)
6. **Finiteness check at the conversion boundary.** In `convert_file`, where each tensor's array comes back from `numpy_of`, add:

   ```python
        if not np.isfinite(array).all():
            raise ConverterError(f"{name}: the checkpoint carries non-finite values")
   ```

   with a test in `DtypeTests`:

   ```python
   def test_non_finite_values_are_refused(self) -> None:
       with self.assertRaises(convert.ConverterError) as caught:
           convert.convert_file_from_tensors(  # use the module's existing test seam;
               {"x": torch.tensor([float("nan")])}, "", convert.Conversion())
       self.assertIn("non-finite", str(caught.exception))
   ```

   (Adapt the call to however the existing tests drive `convert_file` — they already exercise drop rules without a real safetensors file; reuse that seam rather than inventing one.)
7. **License copy before the report, not after `verify_gguf`.** In `main()`, move the `licenses = carry_licenses(…)` call to directly **before** `verify_gguf(args.output, conversion.outputs)`. Existence and digest were already verified up front by `verify_pinned_inputs`; this reorder just makes the last possible failure the cheap one.

- [ ] **Step 4: GREEN — both python gates**

```bash
cmake -S . -B build -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON -DSYNTH_BUILD_INTEGRATION_TESTS=OFF
ctest --test-dir build --output-on-failure -R 'synthesize-(vits|omnivoice)-python-unit'
ctest --test-dir build --output-on-failure -R synthesize-omnivoice-python-unit --verbose 2>&1 | grep -A1 defaults_are_read
```

Expected: both PASS; the grep shows `test_defaults_are_read_from_the_upstream_dataclass … ok` — **not** `skipped`. Then re-run the real conversion to prove the polish changed no output byte:

```bash
uv run --project scripts/envs/omnivoice --locked python scripts/convert-omnivoice.py \
  --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json \
  --weights-dir models/omnivoice-0-6b \
  --output models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf
sha256sum models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf
```

Expected: sha256 still `3ecaa5e2f6fbd735296ba1cd60680c90467be22d2140dc4f208fe80111ecb9e5`. Review the regenerated report diff (`git diff reports/convert/omnivoice/`) — bookkeeping only; commit it if it changed. [Superseded 2026-08-03: re-running this exact command against a corrected manifest (`max_output_frames` 750 → 720000) now produces `f6d504ffaddcbf32f80f1f6c847f075bbd5d2c7b50fe95a194ceb635772f9fa3`; see the porting log's 2026-08-03 entry.]

- [ ] **Step 5: Commit**

```bash
git add tests/CMakeLists.txt scripts/convert-omnivoice.py tests/python/test_convert_omnivoice.py reports/convert/omnivoice/
git commit -m "$(cat <<'EOF'
Run the omnivoice converter tests in their own env and land the polish batch

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 4: Uncross the catalog.h/weights.h naming + comment-hygiene batch

Carry-over items 5, 8, 18, 19, and item 15's comment-label half. The qwen3-tts convention is: **weights.h/weights.cpp = HParams + `read_hparams`**, **catalog.h/catalog.cpp = tensor structs + `build_model_weights` + `expected_tensor_count`**. OmniVoice currently has the two headers swapped (HParams in catalog.h, tensor structs in weights.h) while the .cpp roles already match — mechanical to fix now, expensive after Plan 2's five new files start including them.

**Files:**
- Modify: `src/arch/omnivoice/catalog.h`, `src/arch/omnivoice/weights.h` (swap contents)
- Modify: `src/arch/omnivoice/weights.cpp` (include line), `tests/omnivoice_metadata_test.cpp` (include line)
- Modify: `src/arch/qwen3-tts/bpe.h`, `src/bpe-frontend.h` (item 18)
- Modify: `src/arch/omnivoice/frontend-host.cpp`, `tests/omnivoice_frontend_test.cpp` (items 19, 15-label)
- Modify: `scripts/envs/omnivoice/pyproject.toml` (item 8)

**Interfaces:**
- Produces: `arch/omnivoice/weights.h` = HParams + `read_hparams`; `arch/omnivoice/catalog.h` = weight structs + `build_model_weights` + `expected_tensor_count`. Every Plan-2 task after this one includes them under these roles.

- [ ] **Step 1: Swap the header contents**

Move the entire body of `src/arch/omnivoice/catalog.h` (the `QuantizationProfile` enum, `GeneratorParams`, `AudioCanvasParams`, `CodecParams`, `SemanticParams`, `SpecialTokens`, `GenerationDefaults`, `ProfileContract`, `HParams`, and the `read_hparams` declaration, comments included, plus the `struct gguf_context;` forward declaration) into `src/arch/omnivoice/weights.h`, and the entire body of the current `weights.h` (`LinearWeights` through `ModelWeights`, `build_model_weights`, `expected_tensor_count`, comments included, plus the `struct ggml_context; struct ggml_tensor;` forward declarations and the `struct HParams;` forward declaration) into `src/arch/omnivoice/catalog.h`. Both keep `#pragma once`, `#include "synthesize.h"`, and their std includes; adjust each file's include set to what its new content names (`<string>` moves with HParams).

Update the two includes that now point at the wrong role:

- `src/arch/omnivoice/weights.cpp` line 1: `#include "arch/omnivoice/catalog.h"` → `#include "arch/omnivoice/weights.h"`.
- `tests/omnivoice_metadata_test.cpp` line 14: `#include "arch/omnivoice/catalog.h"` → `#include "arch/omnivoice/weights.h"`.

`catalog.cpp`, `model.cpp`, and `omnivoice_catalog_test.cpp` already include both headers — no edit. Verify nothing else references them:

```bash
grep -rn 'arch/omnivoice/\(catalog\|weights\)\.h' src tests
```

Expected: exactly the five includers named above (catalog.cpp ×2, weights.cpp ×1, model.cpp ×2, the two tests ×3).

- [ ] **Step 2: Restore the BPE rationale comments (item 18)**

In `src/bpe-frontend.h`, the `prefix`/`suffix` field comment still reads as qwen3-only ("The reference puts every request inside an assistant turn… See qwen_assistant_turn."). Replace that comment block with:

```cpp
    // Wrapped around the input before tokenizing, when a family's reference
    // does that as part of turning text into ids. qwen3-tts wraps every
    // request in a fixed assistant turn (see synth::qwen3tts::qwen_assistant_turn);
    // omnivoice leaves both empty, because its prompt is assembled by the
    // synthesis path rather than at tokenize time.
    std::string                                  prefix;
    std::string                                  suffix;
```

In `src/arch/qwen3-tts/bpe.h`, above the two `kAssistant*` constants, restore the provenance comment the hoist dropped (verify the original wording with `git log -p --follow -- src/arch/qwen3-tts/bpe.cpp | grep -B4 kAssistantRolePrefixTokens | head -20` and prefer it if it differs):

```cpp
// Counted from the fixed turn above under this package's tokenizer:
// "<|im_start|>assistant\n" tokenizes to three ids, and the trailing
// "<|im_end|>\n<|im_start|>assistant\n" to five. The talker prompt slices at
// exactly these counts, so they are part of the layout, not styling.
```

- [ ] **Step 3: CJK Ext-B boundary confirmation (item 19) + zero-ref-frames label (item 15)**

Verified against the pinned source (`omnivoice/utils/duration.py` at `OMNI_REV`, in the installed env): the classifier reads `if code > 0x20000:` — **strictly greater**, so U+20000 itself falls through to `default` (weight 1.0) while U+20001 is `cjk` (3.0). In `src/arch/omnivoice/frontend-host.cpp`, find the port's `> 0x20000` comparison and set its comment to:

```cpp
    // Upstream tests `code > 0x20000` -- strictly greater, so U+20000 itself is
    // `default` (1.0) and only U+20001 onward is `cjk`. Verified against
    // omnivoice/utils/duration.py at 468e927b; the asymmetry is upstream's, and
    // matching it is the contract.
```

Add to `tests/omnivoice_frontend_test.cpp`'s `check_char_weights` two rows pinning the boundary (U+20000 = `\xF0\xA0\x80\x80`, U+20001 = `\xF0\xA0\x80\x81`):

```cpp
    // The Ext-B boundary is exclusive upstream: U+20000 itself is `default`.
    SYNTH_TEST_CHECK(weight_of("\xF0\xA0\x80\x80") == 1.0);
    SYNTH_TEST_CHECK(weight_of("\xF0\xA0\x80\x81") == 3.0);
```

(Adapt `weight_of` to the file's existing table-driven form — add the two codepoints to its table if that is how the check is written.)

In the same test file, find the duration-estimator case that covers `ref_frames == 0` and label the deliberate divergence:

```cpp
    // Deliberate divergence, recorded in docs/porting/families/omnivoice.md:
    // upstream's estimator divides by zero reference tokens and its max(1, int(...))
    // returns 1; this port treats zero reference frames as "no reference" and
    // uses the anchor pair instead, because a reference with no audio behind it
    // is refused at load and can only mean a caller bug.
```

- [ ] **Step 4: The uv re-lock comment (item 8)**

In `scripts/envs/omnivoice/pyproject.toml`, directly above the `"torch>=2.7,<3",` line:

```toml
    # Re-lock this environment with `uv lock --no-sources`: the pinned upstream
    # package's own pyproject routes torch to a CUDA wheel index that a plain
    # `uv lock` honors, and the oracle contract is CPU F32.
```

- [ ] **Step 5: Green, sanitize, format, commit**

```bash
cmake --build build --target synthesize-check-unit
ctest --test-dir build --output-on-failure -R 'synthesize-omnivoice-(metadata|catalog|frontend)-test'
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add src/arch/omnivoice/catalog.h src/arch/omnivoice/weights.h src/arch/omnivoice/weights.cpp \
        src/bpe-frontend.h src/arch/qwen3-tts/bpe.h src/arch/omnivoice/frontend-host.cpp \
        tests/omnivoice_metadata_test.cpp tests/omnivoice_frontend_test.cpp \
        scripts/envs/omnivoice/pyproject.toml
git commit -m "$(cat <<'EOF'
Uncross the omnivoice catalog/weights headers and restore dropped rationale

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 5: Bounds-check the metadata reader's products + one rule per rejection mutation

Carry-over items 6 and 12. Two products of adversarial-controlled u32 fields reach `int64_t` shape arithmetic in `catalog.cpp` (`attention_head_count * head_dim` at the current lines 276–277, `num_codebooks * vocab_size` at line 301 — a signed overflow is UB), and the ratio product in `weights.cpp`'s `read_codec` can wrap its `uint64_t`. Refuse implausible geometry at read time so the catalog's arithmetic is provably in range.

**Files:**
- Modify: `src/arch/omnivoice/weights.cpp`
- Modify: `tests/omnivoice_metadata_test.cpp`

**Interfaces:**
- Produces: `read_hparams` guarantees `attention_head_count·head_dim`, `key_value_head_count·head_dim`, and `num_codebooks·vocab_size` all fit far inside `int64_t`; every consumer of `HParams` may multiply freely in `int64_t`.

- [ ] **Step 1: RED — three new rejection tests**

In `tests/omnivoice_metadata_test.cpp`'s `run_rejections()` (mirror the file's existing `expect_rejected` form exactly), add:

```cpp
    // Products of these fields feed int64 shape arithmetic in the catalog; a
    // package this large is not a model, it is an overflow attempt.
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_u32(g, "synthesize.omnivoice.generator.attention_head_count", 65536);
                             gguf_set_val_u32(g, "synthesize.omnivoice.generator.key_value_head_count", 65536);
                             gguf_set_val_u32(g, "synthesize.omnivoice.generator.head_dim", 65536);
                         },
                         "attention geometry whose products leave shape arithmetic") == 0);
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             gguf_set_val_u32(g, "synthesize.omnivoice.audio.num_codebooks", 1u << 16);
                             gguf_set_val_u32(g, "synthesize.omnivoice.audio.vocab_size", (1u << 16) + 1);
                             gguf_set_val_u32(g, "synthesize.omnivoice.audio.mask_id", 1u << 16);
                             gguf_set_val_u32(g, "synthesize.omnivoice.codec.codebook_size", 1u << 16);
                         },
                         "a canvas whose embedding table exceeds any real package") == 0);
    // Before the early-exceed check this product could wrap uint64_t; whether
    // it then collided with the hop was luck, not a rule.
    SYNTH_TEST_CHECK(expect_rejected(
                         [](gguf_context * g) {
                             set_i32_array(g, "synthesize.omnivoice.codec.upsampling_ratios",
                                           { 2147483647, 2147483647, 2147483647 });
                         },
                         "upsampling ratios that would wrap the hop product") == 0);
```

(The second mutation keeps `mask_id == vocab_size − 1` and `vocab_size == codebook_size + 1` satisfied so ONLY the new size rule can fire.) Build and run:

```bash
cmake --build build --target synthesize-omnivoice-metadata-test && \
ctest --test-dir build --output-on-failure -R synthesize-omnivoice-metadata-test
```

Expected: FAIL — the first two mutations are accepted today (the third may already fail by luck on the product≠hop rule; the point is making it deterministic).

- [ ] **Step 2: Implement the guards in `weights.cpp`**

File-scope constant next to the other helpers:

```cpp
// Ceiling on any derived tensor dimension. Far past every real model in this
// family's class, and small enough that any product of two guarded values
// stays inside int64_t everywhere the catalog multiplies.
constexpr uint64_t kMaxDimensionProduct = uint64_t(1) << 24;
```

In `read_generator`, after the existing GQA divisibility check:

```cpp
    // Products of these fields feed int64 shape arithmetic in the catalog;
    // guarding the products here means no consumer has to prove overflow
    // freedom case by case.
    const uint64_t attention_inner = uint64_t(generator.attention_head_count) * generator.head_dim;
    const uint64_t kv_inner        = uint64_t(generator.key_value_head_count) * generator.head_dim;
    if (attention_inner > kMaxDimensionProduct || kv_inner > kMaxDimensionProduct ||
        generator.hidden_size > kMaxDimensionProduct || generator.intermediate_size > kMaxDimensionProduct ||
        generator.text_vocab_size > kMaxDimensionProduct) {
        std::fprintf(stderr, "omnivoice: generator geometry is implausibly large (%llu-wide attention)\n",
                     static_cast<unsigned long long>(attention_inner));
        return false;
    }
```

In `read_audio_canvas`, after the mask-id rule:

```cpp
    // The stacked audio tables are [num_codebooks * vocab_size] rows; the
    // catalog multiplies these in int64, so bound the product here.
    if (uint64_t(audio.num_codebooks) * audio.vocab_size > kMaxDimensionProduct) {
        std::fprintf(stderr, "omnivoice: a %u x %u canvas table exceeds any real package\n", audio.num_codebooks,
                     audio.vocab_size);
        return false;
    }
```

In `read_codec`, replace the ratio-product loop body:

```cpp
    uint64_t total = 1;
    for (uint32_t ratio : codec.upsampling_ratios) {
        total *= ratio;
        // Each ratio is a positive i32, so one multiply can raise `total` by at
        // most 2^31: once it exceeds the hop it can never come back, and
        // stopping here is also what keeps the product from wrapping uint64_t.
        if (total > codec.hop_length) {
            break;
        }
    }
    if (total != codec.hop_length) {
```

(the existing `!=` diagnostic and return stay).

- [ ] **Step 3: One rule per mutation (item 12)**

In the same test's codec rejection block, the current `hop_length = 1024` mutation violates both the frame-rate relation and the ratios product, and which diagnostic fires depends on reader order. Replace the ambiguous mutations with single-rule ones and pin the intent in comments:

```cpp
    // One rule per mutation: each of these breaks exactly one reader check, so
    // a reordering of read_codec cannot silently change which rule a test
    // exercises.
    SYNTH_TEST_CHECK(expect_rejected([](gguf_context * g) {
                         gguf_set_val_f32(g, "synthesize.omnivoice.codec.frame_rate_hz", 30.0f);
                     },
                     "hop and frame rate must agree with the sample rate") == 0);
    SYNTH_TEST_CHECK(expect_rejected([](gguf_context * g) {
                         set_i32_array(g, "synthesize.omnivoice.codec.upsampling_ratios", { 8, 5, 4, 2, 2 });
                     },
                     "the upsampling ratios must multiply to the hop") == 0);
```

Keep the existing `codec.sample_rate = 16000` mutation (it isolates the codec-vs-declared-output rule, which the reader checks before the frame relation) but add above it: `// Fires the codec-vs-declared-output rule; the frame relation is checked later and never reached.` Delete or rework the old `hop_length = 1024` mutation into the frame-rate form above so no remaining mutation trips two rules.

- [ ] **Step 4: Green, sanitize, format, commit**

```bash
cmake --build build --target synthesize-check-unit
ctest --test-dir build --output-on-failure -R synthesize-omnivoice-metadata-test
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add src/arch/omnivoice/weights.cpp tests/omnivoice_metadata_test.cpp
git commit -m "$(cat <<'EOF'
Bound the omnivoice metadata reader's products and de-alias its rejections

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 6: Synthetic on-disk package builder + the frontend-arrays refusal test

Carry-over item 7. The refusal branch at `src/arch/omnivoice/model.cpp:151–153` (frontend declared present but `synthesize.omnivoice.frontend.vocab`/`.merges` unreadable → `SYNTH_ERR_GGUF`) has never executed: no test writes a package that reaches it. Build a miniature but fully legal on-disk package (the catalog test's small layout, written through the real GGUF writer) and drive the load path through it.

**Files:**
- Create: `tests/omnivoice_small_layout.h` (extracted from the catalog test)
- Create: `tests/omnivoice_synthetic_package.h` (header-only builder)
- Create: `tests/omnivoice_load_synthetic_test.cpp`
- Modify: `tests/omnivoice_catalog_test.cpp` (include the extracted header, delete the moved code)
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `synth::omnivoice::testing::write_synthetic_package(path, options)` — reusable by any later load-path test (Plan 3's profile loading will want it too).

- [ ] **Step 1: Extract the small layout into a shared header**

Create `tests/omnivoice_small_layout.h` and move — verbatim, marked `inline` where they are functions — `struct Entry`, `add`, `add_linear`, `add_conv`, `add_bare_conv`, `add_transpose_conv`, `add_layer_norm`, `add_snake`, `add_residual_units`, `small_hparams()`, and `expected_entries(const synth::omnivoice::HParams &)` out of `tests/omnivoice_catalog_test.cpp` (they currently sit in that file's anonymous namespace between `make_context()` and `populate()`). Wrap them in `namespace synth::omnivoice::testing { … }`, include `"arch/omnivoice/weights.h"` (HParams — post-Task-4 role) and the std headers they name. In the catalog test, include the new header, add `using namespace synth::omnivoice::testing;` inside its anonymous namespace, and delete the moved definitions. Build the catalog test — it must pass unchanged:

```bash
cmake --build build --target synthesize-omnivoice-catalog-test && \
ctest --test-dir build --output-on-failure -R synthesize-omnivoice-catalog-test
```

- [ ] **Step 2: Write the builder**

`tests/omnivoice_synthetic_package.h`:

```cpp
#pragma once

// Writes a miniature but fully legal omnivoice package to disk: every metadata
// key the reader demands, every tensor the catalog resolves (the small layout
// the catalog test already proves legal), real F32 payloads, and a minimal
// frontend. Options knock out exactly one thing so a test can prove one
// refusal.

#include "omnivoice_small_layout.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace synth::omnivoice::testing {

struct SyntheticPackageOptions {
    bool omit_frontend_vocab  = false;
    bool omit_frontend_merges = false;
};

inline void set_string_array(gguf_context * g, const char * key, const std::vector<std::string> & values) {
    std::vector<const char *> pointers;
    pointers.reserve(values.size());
    for (const std::string & value : values) {
        pointers.push_back(value.c_str());
    }
    gguf_set_arr_str(g, key, pointers.data(), int(pointers.size()));
}

inline void set_i32_array(gguf_context * g, const char * key, const std::vector<int32_t> & values) {
    gguf_set_arr_data(g, key, GGUF_TYPE_INT32, values.data(), values.size());
}

inline bool write_synthetic_package(const std::string & path, const SyntheticPackageOptions & options) {
    const synth::omnivoice::HParams h       = small_hparams();
    const std::vector<Entry>        entries = expected_entries(h);

    // Enough for every small tensor's data plus headers; the largest entry is
    // the [8, 40] text embedding at reduced widths.
    ggml_init_params parameters{};
    parameters.mem_size = 8u * 1024 * 1024;
    parameters.no_alloc = false;
    ggml_context * context = ggml_init(parameters);
    if (context == nullptr) {
        return false;
    }
    gguf_context * gguf = gguf_init_empty();
    if (gguf == nullptr) {
        ggml_free(context);
        return false;
    }

    // --- Metadata: every key read_hparams demands, at the small layout's
    // values. Kept in one place so the builder and small_hparams cannot drift:
    // the values below are small_hparams()'s, transcribed.
    gguf_set_val_str(gguf, "general.architecture", "omnivoice");
    gguf_set_val_u32(gguf, "synthesize.format_version", 1);
    gguf_set_val_str(gguf, "synthesize.model_family", "omnivoice");
    gguf_set_val_str(gguf, "synthesize.model_variant", "synthetic");
    gguf_set_val_str(gguf, "synthesize.quantization.profile", "F32");
    gguf_set_val_u32(gguf, "synthesize.quantization.profile_version", 1);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.architecture_version", 1);

    gguf_set_val_u32(gguf, "synthesize.capabilities.input_flags", 1u << 0);
    gguf_set_val_u32(gguf, "synthesize.capabilities.flags", (1u << 0) | (1u << 1));
    gguf_set_val_u64(gguf, "synthesize.capabilities.max_input_tokens", 64);
    gguf_set_val_u64(gguf, "synthesize.capabilities.max_output_frames", 16);
    gguf_set_val_f32(gguf, "synthesize.capabilities.min_speaking_rate", 0.5f);
    gguf_set_val_f32(gguf, "synthesize.capabilities.max_speaking_rate", 2.0f);
    gguf_set_val_u32(gguf, "synthesize.audio.sample_rate_hz", 150);
    gguf_set_val_u32(gguf, "synthesize.audio.channels", 1);
    gguf_set_val_str(gguf, "synthesize.audio.sample_format", "f32le");
    gguf_set_val_str(gguf, "synthesize.voice.mode", "profile-sources");
    gguf_set_val_bool(gguf, "synthesize.voice.has_package_default", true);
    gguf_set_val_u32(gguf, "synthesize.voice.preset_count", 0);

    gguf_set_val_u32(gguf, "synthesize.omnivoice.generation.num_step", 4);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generation.guidance_scale", 2.0f);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generation.t_shift", 0.1f);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generation.layer_penalty_factor", 5.0f);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generation.position_temperature", 5.0f);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generation.class_temperature", 0.0f);

    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.layer_count", h.generator.layer_count);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.hidden_size", h.generator.hidden_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.attention_head_count", h.generator.attention_head_count);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.key_value_head_count", h.generator.key_value_head_count);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.head_dim", h.generator.head_dim);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.intermediate_size", h.generator.intermediate_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.generator.text_vocab_size", h.generator.text_vocab_size);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generator.rms_norm_eps", h.generator.rms_norm_eps);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.generator.rope_theta", h.generator.rope_theta);
    gguf_set_val_str(gguf, "synthesize.omnivoice.generator.attention", "bidirectional");

    gguf_set_val_u32(gguf, "synthesize.omnivoice.audio.num_codebooks", h.audio.num_codebooks);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.audio.vocab_size", h.audio.vocab_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.audio.mask_id", h.audio.mask_id);

    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.sample_rate", h.codec.sample_rate);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.hop_length", h.codec.hop_length);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.codec.frame_rate_hz", h.codec.frame_rate_hz);
    set_i32_array(gguf, "synthesize.omnivoice.codec.upsampling_ratios", { 2, 3 });
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.decoder_hidden_size", h.codec.decoder_hidden_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.encoder_hidden_size", h.codec.encoder_hidden_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.hidden_size", h.codec.hidden_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.codebook_dim", h.codec.codebook_dim);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.codebook_size", h.codec.codebook_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.codec.semantic_sample_rate", h.codec.semantic_sample_rate);

    gguf_set_val_u32(gguf, "synthesize.omnivoice.semantic.hidden_size", h.semantic.hidden_size);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.semantic.layer_count", h.semantic.layer_count);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.semantic.attention_head_count", h.semantic.attention_head_count);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.semantic.intermediate_size", h.semantic.intermediate_size);
    gguf_set_val_f32(gguf, "synthesize.omnivoice.semantic.layer_norm_eps", h.semantic.layer_norm_eps);
    set_i32_array(gguf, "synthesize.omnivoice.semantic.conv_dim",
                  std::vector<int32_t>(h.semantic.conv_dim.begin(), h.semantic.conv_dim.end()));
    set_i32_array(gguf, "synthesize.omnivoice.semantic.conv_kernel",
                  std::vector<int32_t>(h.semantic.conv_kernel.begin(), h.semantic.conv_kernel.end()));
    set_i32_array(gguf, "synthesize.omnivoice.semantic.conv_stride",
                  std::vector<int32_t>(h.semantic.conv_stride.begin(), h.semantic.conv_stride.end()));

    // Seven markers, eos and pad, all below the small text vocabulary (40).
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.denoise", 30);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.lang_start", 31);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.lang_end", 32);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.instruct_start", 33);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.instruct_end", 34);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.text_start", 35);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.text_end", 36);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.eos", 37);
    gguf_set_val_u32(gguf, "synthesize.omnivoice.token.pad", 38);

    set_string_array(gguf, "synthesize.omnivoice.languages.tags", { "en", "zh", "ja" });

    gguf_set_val_str(gguf, "synthesize.profile.schema", "omnivoice-clone-prompt");
    gguf_set_val_u32(gguf, "synthesize.profile.schema_version", 1);
    gguf_set_val_str(gguf, "synthesize.profile.compatibility_id",
                     "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    gguf_set_val_u32(gguf, "synthesize.reference.target_sample_rate", 150);
    gguf_set_val_u32(gguf, "synthesize.reference.target_channels", 1);
    gguf_set_val_u64(gguf, "synthesize.reference.min_frames_per_clip", 150);
    gguf_set_val_u64(gguf, "synthesize.reference.max_frames_per_clip", 3000);
    gguf_set_val_u64(gguf, "synthesize.reference.max_total_frames", 3000);
    gguf_set_val_u64(gguf, "synthesize.reference.max_reference_count", 1);

    gguf_set_val_bool(gguf, "synthesize.frontend.present", true);
    gguf_set_val_str(gguf, "synthesize.frontend.provider", "synthesize.qwen_bpe");
    gguf_set_val_u32(gguf, "synthesize.frontend.contract_version", 1);
    if (!options.omit_frontend_vocab) {
        // make_bpe_frontend requires only a non-empty dense table; loading does
        // not tokenize, so token text is arbitrary here.
        std::vector<std::string> vocab;
        for (uint32_t index = 0; index < h.generator.text_vocab_size; ++index) {
            vocab.push_back("tok" + std::to_string(index));
        }
        set_string_array(gguf, "synthesize.omnivoice.frontend.vocab", vocab);
    }
    if (!options.omit_frontend_merges) {
        set_string_array(gguf, "synthesize.omnivoice.frontend.merges", { "tok1 tok2" });
    }

    // --- Tensors: the small layout with real F32 payloads.
    for (const Entry & entry : entries) {
        ggml_tensor * tensor =
            ggml_new_tensor(context, GGML_TYPE_F32, int(entry.ne.size()), entry.ne.data());
        if (tensor == nullptr) {
            gguf_free(gguf);
            ggml_free(context);
            return false;
        }
        ggml_set_name(tensor, entry.name.c_str());
        float * data = static_cast<float *>(tensor->data);
        for (int64_t index = 0; index < ggml_nelements(tensor); ++index) {
            data[index] = 0.03125f;  // any finite constant; loading never computes
        }
        gguf_add_tensor(gguf, tensor);
    }

    const bool written = gguf_write_to_file(gguf, path.c_str(), /*only_meta=*/false);
    gguf_free(gguf);
    ggml_free(context);
    return written;
}

}  // namespace synth::omnivoice::testing
```

(If `expected_entries`'s `Entry::ne` field or `small_hparams`'s field spellings differ from the transcription above, the extracted header from Step 1 is authoritative — follow it.)

- [ ] **Step 3: RED — the refusal test**

`tests/omnivoice_load_synthetic_test.cpp`:

```cpp
// Loads a synthetic on-disk package end to end, then knocks out exactly the
// frontend arrays to prove the refusal branch nothing has ever executed:
// frontend declared present, vocab/merges unreadable -> SYNTH_ERR_GGUF.

#include "arch/omnivoice/omnivoice.h"
#include "omnivoice_synthetic_package.h"
#include "test-assert.h"

#include <cstdio>
#include <memory>
#include <string>

namespace {

std::string package_path(const char * directory, const char * name) {
    return std::string(directory) + "/" + name;
}

int check_valid_package_loads(const char * directory) {
    const std::string path = package_path(directory, "synthetic-valid.gguf");
    SYNTH_TEST_CHECK(synth::omnivoice::testing::write_synthetic_package(path, {}));
    std::unique_ptr<synth::omnivoice::Model> model;
    SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(path, model) == SYNTH_OK);
    SYNTH_TEST_CHECK(model != nullptr);
    synth::omnivoice::ModelInfo info;
    SYNTH_TEST_CHECK(model->get_info(info) == SYNTH_OK);
    SYNTH_TEST_CHECK(info.variant == "synthetic");
    SYNTH_TEST_CHECK(model->samples_per_frame() == 6);
    SYNTH_TEST_CHECK(model->text_frontend() != nullptr);
    return 0;
}

int check_missing_frontend_arrays_are_refused(const char * directory) {
    for (int variant = 0; variant < 2; ++variant) {
        synth::omnivoice::testing::SyntheticPackageOptions options;
        options.omit_frontend_vocab  = variant == 0;
        options.omit_frontend_merges = variant == 1;
        const std::string path = package_path(directory, variant == 0 ? "synthetic-no-vocab.gguf"
                                                                      : "synthetic-no-merges.gguf");
        SYNTH_TEST_CHECK(synth::omnivoice::testing::write_synthetic_package(path, options));
        std::unique_ptr<synth::omnivoice::Model> model;
        // The package declares a frontend it does not carry: the load must fail
        // at the arrays, not fall through to a frontend-less model.
        SYNTH_TEST_CHECK(synth::omnivoice::Model::load_cpu(path, model) == SYNTH_ERR_GGUF);
        SYNTH_TEST_CHECK(model == nullptr);
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <scratch-dir>\n", argv[0]);
        return 2;
    }
    int failures = 0;
    failures += check_valid_package_loads(argv[1]);
    failures += check_missing_frontend_arrays_are_refused(argv[1]);
    return failures == 0 ? 0 : 1;
}
```

Register in `tests/CMakeLists.txt` next to the other omnivoice unit tests, following the fixture-dir argument convention `synthesize-omnivoice-model-errors-test` already uses (a scratch dir under `${CMAKE_CURRENT_BINARY_DIR}`):

```cmake
add_executable(synthesize-omnivoice-load-synthetic-test omnivoice_load_synthetic_test.cpp)
target_include_directories(synthesize-omnivoice-load-synthetic-test
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(synthesize-omnivoice-load-synthetic-test PRIVATE synthesize ggml)
add_test(NAME synthesize-omnivoice-load-synthetic-test
    COMMAND synthesize-omnivoice-load-synthetic-test ${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-synthetic)
set_tests_properties(synthesize-omnivoice-load-synthetic-test PROPERTIES LABELS "unit")
synth_register_unit_target(synthesize-omnivoice-load-synthetic-test)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fixtures/omnivoice-synthetic)
```

Build and run first with only Step 1's extraction in place and the builder stubbed out — or simply observe the first real run: if `check_valid_package_loads` fails, the builder and the reader disagree; fix the builder against the reader's diagnostics (`omnivoice: …` lines on stderr say exactly which rule fired), never by loosening the reader. If `make_bpe_frontend` refuses the minimal vocab/merges, read its checks in `src/bpe-frontend.cpp:468` and satisfy them — as of this writing it requires only a non-empty vocab, space-containing merges, and non-negative special ids.

- [ ] **Step 4: Green, sanitize, format, commit**

```bash
cmake --build build --target synthesize-check-unit
ctest --test-dir build --output-on-failure -R 'synthesize-omnivoice-(catalog|load-synthetic)-test'
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add tests/omnivoice_small_layout.h tests/omnivoice_synthetic_package.h \
        tests/omnivoice_load_synthetic_test.cpp tests/omnivoice_catalog_test.cpp tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
Load synthetic omnivoice packages from disk and prove the frontend refusal

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 7: Generator graph — bidirectional layer, embedding merge, full-canvas heads

Slice 4's graph half. The block is qwen3-tts's `codec_transformer_layer` shape (cache-free, no `ggml_cgraph *` parameter, no KV views) **minus** the two `layer_scale` multiplications it has and **plus** the per-head q/k RMS norms the Qwen3 block has — the exact diff the graph-precedent report spells out. The embedding merge is upstream `_prepare_embed_inputs` re-expressed as a concat: text positions take the text embedding of row 0; audio positions take the sum of the eight offset codebook embeddings and *nothing else* (upstream `torch.where`-selects, it does not add the streams).

**Files:**
- Create: `src/arch/omnivoice/generator.h`, `src/arch/omnivoice/generator.cpp`
- Create: `scripts/dump_reference_omnivoice_generator.py`
- Create: `tests/omnivoice_generator_test.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces (namespace `synth::omnivoice`), consumed by Tasks 9–10:
  - `struct AttentionShape { uint32_t hidden_size, attention_head_count, key_value_head_count, head_dim; float rms_norm_eps, rope_theta; }`
  - `ggml_tensor * rms_norm(ggml_context *, ggml_tensor * input, ggml_tensor * weight, float eps);`
  - `ggml_tensor * generator_layer(ggml_context *, ggml_tensor * input, ggml_tensor * position_ids, ggml_tensor * mask, const GeneratorLayerWeights &, const AttentionShape &);` — `mask` may be `nullptr` (fully bidirectional; the only production mode)
  - `ggml_tensor * build_canvas_embedding(ggml_context *, const GeneratorWeights &, ggml_tensor * text_ids, ggml_tensor * audio_ids, uint32_t num_codebooks);` — `text_ids` I32 `[text_count]` or nullptr (uncond branch); `audio_ids` I32 `[audio_count, num_codebooks]` of host-shifted ids or nullptr; at least one non-null; returns `[hidden, text_count + audio_count]`
  - `ggml_tensor * build_generator_forward(ggml_context *, ggml_tensor * embeddings, ggml_tensor * position_ids, ggml_tensor * mask, const GeneratorWeights &, const AttentionShape &, const AudioCanvasParams &, std::vector<ggml_tensor *> * out_layers, ggml_tensor ** out_final);` — returns logits reshaped `[vocab, codebooks, positions]`

- [ ] **Step 1: Write `generator.h`**

```cpp
#pragma once

// catalog.h carries the weight structs and weights.h the HParams-side
// parameter structs (AudioCanvasParams) -- the post-Task-4 roles.
#include "catalog.h"
#include "weights.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace synth::omnivoice {

struct AttentionShape {
    uint32_t hidden_size          = 0;
    uint32_t attention_head_count = 0;
    uint32_t key_value_head_count = 0;
    uint32_t head_dim             = 0;
    float    rms_norm_eps         = 0.0f;
    float    rope_theta           = 0.0f;
};

// RMSNorm with a learned gain, which is what every norm in this family is.
ggml_tensor * rms_norm(ggml_context * context, ggml_tensor * input, ggml_tensor * weight, float eps);

// One Qwen3 block over `positions` tokens, bidirectional and cache-free.
//
// This is deliberately NOT qwen3-tts's decoder_layer: that block writes a KV
// cache and orders the writes through the graph, neither of which exists here
// because no position is ever causal. It is the codec_transformer_layer shape
// with the layer scales removed (a Qwen3 block has none) and the per-head q/k
// norms added (a Qwen3 block has them, at head_dim width, applied before rope).
//
// `input` is [hidden, positions]; `position_ids` is I32 [positions]. `mask` may
// be nullptr -- full bidirectional attention, the only mode synthesis uses --
// or an additive F32 [positions, positions] tensor (0 = may attend, -INF = may
// not), which the unit test uses to prove the mask actually gates.
//
// Returns nullptr rather than aborting on a shape it cannot build.
ggml_tensor * generator_layer(ggml_context *                context,
                              ggml_tensor *                 input,
                              ggml_tensor *                 position_ids,
                              ggml_tensor *                 mask,
                              const GeneratorLayerWeights & weights,
                              const AttentionShape &        shape);

// The canvas embedding: text positions read the text table at row 0's ids;
// audio positions read the stacked audio table once per codebook at the
// host-shifted ids (id + codebook * vocab_size) and sum the eight rows. The
// reference `torch.where`-selects between the two streams, so an audio
// position carries the sum ALONE -- the text stream is never added to it.
//
// `text_ids` is I32 [text_count] or nullptr (the unconditional branch has no
// text region); `audio_ids` is I32 [audio_count, num_codebooks], codebook-major
// rows, or nullptr. At least one must be non-null. Returns [hidden, total].
ggml_tensor * build_canvas_embedding(ggml_context *           context,
                                     const GeneratorWeights & weights,
                                     ggml_tensor *            text_ids,
                                     ggml_tensor *            audio_ids,
                                     uint32_t                 num_codebooks);

// The whole generator: embeddings -> every layer -> final norm -> audio heads
// over EVERY position (a mask-predict model predicts the full canvas at once;
// there is no last-position slice). Returns the logits reshaped to
// [vocab, codebooks, positions]: element (v, c, s) is position s's logit for
// code v of codebook c, matching the head's stacked row order c * vocab + v.
//
// `out_layers`, when non-null, receives each layer's output; `out_final`, when
// non-null, receives the hidden state after the final norm. Port validation
// compares them against the oracle's step-0 probes; nothing else reads them.
ggml_tensor * build_generator_forward(ggml_context *               context,
                                      ggml_tensor *                embeddings,
                                      ggml_tensor *                position_ids,
                                      ggml_tensor *                mask,
                                      const GeneratorWeights &     weights,
                                      const AttentionShape &       shape,
                                      const AudioCanvasParams &    canvas,
                                      std::vector<ggml_tensor *> * out_layers,
                                      ggml_tensor **               out_final);

}  // namespace synth::omnivoice
```

- [ ] **Step 2: Write `generator.cpp`**

The layer body is `codec_transformer_layer` (src/arch/qwen3-tts/codec.cpp:251–333) with these deltas, each marked in the code below: mask optional; per-head q/k norms inserted between the head reshape and rope; no layer scales; plain residual adds.

```cpp
// Graph construction for the mask-predict generator.
//
// The block is the ordinary Qwen3 one -- per-head q/k norms, gated feed-forward
// -- run bidirectionally over the whole canvas. There is no KV cache and no
// causal mask anywhere in this family: every layer is full attention, so the
// cache-free codec_transformer_layer shape from qwen3-tts is the precedent, not
// its decoder_layer. See docs/porting/families/omnivoice.md, "Architecture".

#include "arch/omnivoice/generator.h"

#include "arch/omnivoice/catalog.h"
#include "ggml.h"

#include <cmath>

namespace synth::omnivoice {

ggml_tensor * rms_norm(ggml_context * context, ggml_tensor * input, ggml_tensor * weight, float eps) {
    if (context == nullptr || input == nullptr || weight == nullptr) {
        return nullptr;
    }
    return ggml_mul(context, ggml_rms_norm(context, input, eps), weight);
}

ggml_tensor * generator_layer(ggml_context *                context,
                              ggml_tensor *                 input,
                              ggml_tensor *                 position_ids,
                              ggml_tensor *                 mask,
                              const GeneratorLayerWeights & weights,
                              const AttentionShape &        shape) {
    if (context == nullptr || input == nullptr || position_ids == nullptr ||
        weights.input_layernorm == nullptr || weights.q_proj == nullptr || weights.k_proj == nullptr ||
        weights.v_proj == nullptr || weights.o_proj == nullptr || weights.q_norm == nullptr ||
        weights.k_norm == nullptr || weights.post_attention_layernorm == nullptr || weights.gate_proj == nullptr ||
        weights.up_proj == nullptr || weights.down_proj == nullptr) {
        return nullptr;
    }
    const int64_t hidden    = static_cast<int64_t>(shape.hidden_size);
    const int64_t head_dim  = static_cast<int64_t>(shape.head_dim);
    const int64_t q_heads   = static_cast<int64_t>(shape.attention_head_count);
    const int64_t kv_heads  = static_cast<int64_t>(shape.key_value_head_count);
    const int64_t positions = input->ne[1];
    if (hidden <= 0 || head_dim <= 0 || q_heads <= 0 || kv_heads <= 0 || positions <= 0 || input->ne[0] != hidden ||
        q_heads % kv_heads != 0) {
        return nullptr;
    }
    if (position_ids->type != GGML_TYPE_I32 || position_ids->ne[0] != positions) {
        return nullptr;
    }
    // Delta against codec_transformer_layer: the mask is OPTIONAL. Bidirectional
    // attention over one un-padded sequence needs none, and that is the only
    // mode synthesis uses; a non-null mask exists so a test can prove gating.
    if (mask != nullptr && (mask->type != GGML_TYPE_F32 || mask->ne[0] != positions || mask->ne[1] < positions)) {
        return nullptr;
    }

    ggml_tensor * residual = input;
    ggml_tensor * normed   = rms_norm(context, input, weights.input_layernorm, shape.rms_norm_eps);

    ggml_tensor * q = ggml_mul_mat(context, weights.q_proj, normed);
    ggml_tensor * k = ggml_mul_mat(context, weights.k_proj, normed);
    ggml_tensor * v = ggml_mul_mat(context, weights.v_proj, normed);

    q = ggml_reshape_3d(context, q, head_dim, q_heads, positions);
    k = ggml_reshape_3d(context, k, head_dim, kv_heads, positions);
    v = ggml_reshape_3d(context, v, head_dim, kv_heads, positions);

    // Delta against codec_transformer_layer: Qwen3 normalizes each head of q
    // and k before rope rather than the packed projection; doing it after
    // would rotate an unnormalized vector.
    q = rms_norm(context, q, weights.q_norm, shape.rms_norm_eps);
    k = rms_norm(context, k, weights.k_norm, shape.rms_norm_eps);

    q = ggml_rope_ext(context, q, position_ids, nullptr, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX, 0,
                      shape.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(context, k, position_ids, nullptr, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX, 0,
                      shape.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * q_hd = ggml_cont(context, ggml_permute(context, q, 0, 2, 1, 3));
    ggml_tensor * k_hd = ggml_cont(context, ggml_permute(context, k, 0, 2, 1, 3));
    ggml_tensor * v_hd = ggml_cont(context, ggml_permute(context, v, 0, 2, 1, 3));

    // Grouped-query attention comes out of ggml_mul_mat's own broadcast rather
    // than a materialized repeat of the kv side, exactly as in qwen3-tts.
    ggml_tensor * scores = ggml_mul_mat(context, k_hd, q_hd);
    scores = ggml_soft_max_ext(context, scores, mask, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);

    ggml_tensor * v_t      = ggml_cont(context, ggml_permute(context, v_hd, 1, 0, 2, 3));
    ggml_tensor * attended = ggml_mul_mat(context, v_t, scores);
    attended               = ggml_cont(context, ggml_permute(context, attended, 0, 2, 1, 3));
    attended               = ggml_reshape_2d(context, attended, head_dim * q_heads, positions);

    // Delta against codec_transformer_layer: no layer scales -- a Qwen3 block
    // adds its branches back unscaled.
    ggml_tensor * projected       = ggml_mul_mat(context, weights.o_proj, attended);
    ggml_tensor * after_attention = ggml_add(context, residual, projected);

    ggml_tensor * mlp_in = rms_norm(context, after_attention, weights.post_attention_layernorm, shape.rms_norm_eps);
    ggml_tensor * gate   = ggml_silu(context, ggml_mul_mat(context, weights.gate_proj, mlp_in));
    ggml_tensor * up     = ggml_mul_mat(context, weights.up_proj, mlp_in);
    ggml_tensor * mlp    = ggml_mul_mat(context, weights.down_proj, ggml_mul(context, gate, up));
    return ggml_add(context, after_attention, mlp);
}

ggml_tensor * build_canvas_embedding(ggml_context *           context,
                                     const GeneratorWeights & weights,
                                     ggml_tensor *            text_ids,
                                     ggml_tensor *            audio_ids,
                                     uint32_t                 num_codebooks) {
    if (context == nullptr || weights.text_embedding == nullptr || weights.audio_embeddings == nullptr ||
        num_codebooks == 0 || (text_ids == nullptr && audio_ids == nullptr)) {
        return nullptr;
    }
    if (text_ids != nullptr && (text_ids->type != GGML_TYPE_I32 || text_ids->ne[1] != 1)) {
        return nullptr;
    }
    if (audio_ids != nullptr &&
        (audio_ids->type != GGML_TYPE_I32 || audio_ids->ne[1] != int64_t(num_codebooks))) {
        return nullptr;
    }

    ggml_tensor * text = nullptr;
    if (text_ids != nullptr) {
        text = ggml_get_rows(context, weights.text_embedding, text_ids);
    }

    ggml_tensor * audio = nullptr;
    if (audio_ids != nullptr) {
        // Residual of nothing: the eight codebook embeddings are summed, one
        // get_rows per codebook over the host-shifted ids, mirroring
        // codec_quantizer_decode's level loop. Ascending order matters: float
        // addition is not associative and the reference sums 0..7.
        const int64_t count = audio_ids->ne[0];
        for (uint32_t codebook = 0; codebook < num_codebooks; ++codebook) {
            ggml_tensor * ids =
                ggml_view_1d(context, audio_ids, count, size_t(codebook) * audio_ids->nb[1]);
            ggml_tensor * rows = ggml_get_rows(context, weights.audio_embeddings, ids);
            audio              = audio == nullptr ? rows : ggml_add(context, audio, rows);
        }
    }

    if (text == nullptr) {
        return audio;
    }
    if (audio == nullptr) {
        return text;
    }
    // The reference torch.where-selects per position; regions are contiguous,
    // so selection is concatenation along the sequence axis.
    return ggml_concat(context, text, audio, 1);
}

ggml_tensor * build_generator_forward(ggml_context *               context,
                                      ggml_tensor *                embeddings,
                                      ggml_tensor *                position_ids,
                                      ggml_tensor *                mask,
                                      const GeneratorWeights &     weights,
                                      const AttentionShape &       shape,
                                      const AudioCanvasParams &    canvas,
                                      std::vector<ggml_tensor *> * out_layers,
                                      ggml_tensor **               out_final) {
    if (context == nullptr || embeddings == nullptr || position_ids == nullptr || weights.norm == nullptr ||
        weights.audio_heads == nullptr || weights.layers.empty()) {
        return nullptr;
    }
    if (out_layers != nullptr) {
        out_layers->clear();
        out_layers->reserve(weights.layers.size());
    }
    if (out_final != nullptr) {
        *out_final = nullptr;
    }

    ggml_tensor * hidden = embeddings;
    for (const GeneratorLayerWeights & layer : weights.layers) {
        hidden = generator_layer(context, hidden, position_ids, mask, layer, shape);
        if (hidden == nullptr) {
            return nullptr;
        }
        if (out_layers != nullptr) {
            out_layers->push_back(hidden);
        }
    }
    hidden = rms_norm(context, hidden, weights.norm, shape.rms_norm_eps);
    if (hidden == nullptr) {
        return nullptr;
    }
    if (out_final != nullptr) {
        *out_final = hidden;
    }

    // Every position predicts: [codebooks * vocab, positions], then the split
    // that names the stacked row order c * vocab + v explicitly.
    ggml_tensor * logits = ggml_mul_mat(context, weights.audio_heads, hidden);
    if (logits->ne[0] != int64_t(canvas.num_codebooks) * canvas.vocab_size) {
        return nullptr;
    }
    return ggml_reshape_3d(context, logits, canvas.vocab_size, canvas.num_codebooks, logits->ne[1]);
}

}  // namespace synth::omnivoice
```

Add `arch/omnivoice/generator.cpp` to `add_library(synthesize …)` in `src/CMakeLists.txt` next to the other omnivoice sources.

- [ ] **Step 3: Write the reference script**

`scripts/dump_reference_omnivoice_generator.py` — runs the REAL transformers Qwen3 model (the class `OmniVoice.llm` is) at toy dimensions over LCG-drawn weights and prints the expected arrays to paste into the test. Both sides draw from the same 64-bit LCG in the same order (the qwen3-tts decoder-layer-test technique; the LCG constants are that test's, verbatim).

```python
#!/usr/bin/env python3
"""Reference values for tests/omnivoice_generator_test.cpp.

Runs a one-layer transformers Qwen3Model bidirectionally (4-D all-True boolean
attention mask -- exactly how OmniVoice.forward drives it) over LCG-drawn
weights, plus the embedding merge and the full-canvas heads, and prints the
expected arrays. Fill ORDER is the contract with the C++ test: reordering any
line silently changes every weight.

Usage:
    uv run --project scripts/envs/omnivoice --locked python \
        scripts/dump_reference_omnivoice_generator.py
"""
import torch
import torch.nn.functional as F

SEED = 20260731
HIDDEN = 8
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
INTERMEDIATE = 16
EPS = 1e-6
THETA = 1e6
TEXT_VOCAB = 20
AUDIO_VOCAB = 6          # toy canvas vocabulary; mask id 5
CODEBOOKS = 3
TEXT_POSITIONS = 2
AUDIO_POSITIONS = 3
POSITIONS = TEXT_POSITIONS + AUDIO_POSITIONS

TEXT_IDS = [3, 7]
# Codebook-major [CODEBOOKS][AUDIO_POSITIONS]; 5 is the toy mask id, a real row
# of the audio table like any other.
AUDIO_GRID = [[5, 0, 2], [5, 1, 5], [4, 5, 3]]


class LcgStream:
    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.state >> 40) / 8388608.0 - 1.0

    def fill(self, count, scale, offset):
        return torch.tensor([self.next() * scale + offset for _ in range(count)],
                            dtype=torch.float32)


def dump(name, tensor):
    flat = tensor.reshape(-1).tolist()
    print(f"constexpr float {name}[] = {{")
    for start in range(0, len(flat), 4):
        row = ", ".join(f"{value:.9g}f" for value in flat[start:start + 4])
        print(f"    {row},")
    print("};")


def main():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    config = Qwen3Config(
        hidden_size=HIDDEN, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM, intermediate_size=INTERMEDIATE, rms_norm_eps=EPS,
        rope_theta=THETA, vocab_size=TEXT_VOCAB, num_hidden_layers=1,
        attention_dropout=0.0, attn_implementation="eager",
    )
    model = Qwen3Model(config).eval()

    stream = LcgStream(SEED)
    # The C++ fixture fills its tensors in EXACTLY this order with these
    # scales/offsets (the qwen3-tts convention: projections {0.5, 0}, norm
    # gains {0.25, 1}, tables/inputs {0.5, 0}).
    text_table = stream.fill(TEXT_VOCAB * HIDDEN, 0.5, 0.0).view(TEXT_VOCAB, HIDDEN)
    audio_table = stream.fill(CODEBOOKS * AUDIO_VOCAB * HIDDEN, 0.5, 0.0).view(
        CODEBOOKS * AUDIO_VOCAB, HIDDEN)
    heads_table = stream.fill(CODEBOOKS * AUDIO_VOCAB * HIDDEN, 0.5, 0.0).view(
        CODEBOOKS * AUDIO_VOCAB, HIDDEN)
    layer = model.layers[0]
    assignments = [
        (model.norm.weight, 0.25, 1.0),
        (layer.input_layernorm.weight, 0.25, 1.0),
        (layer.self_attn.q_proj.weight, 0.5, 0.0),
        (layer.self_attn.k_proj.weight, 0.5, 0.0),
        (layer.self_attn.v_proj.weight, 0.5, 0.0),
        (layer.self_attn.o_proj.weight, 0.5, 0.0),
        (layer.self_attn.q_norm.weight, 0.25, 1.0),
        (layer.self_attn.k_norm.weight, 0.25, 1.0),
        (layer.post_attention_layernorm.weight, 0.25, 1.0),
        (layer.mlp.gate_proj.weight, 0.5, 0.0),
        (layer.mlp.up_proj.weight, 0.5, 0.0),
        (layer.mlp.down_proj.weight, 0.5, 0.0),
    ]
    with torch.no_grad():
        for parameter, scale, offset in assignments:
            parameter.copy_(stream.fill(parameter.numel(), scale, offset).view_as(parameter))

    # The embedding merge, exactly _prepare_embed_inputs' arithmetic: text
    # embedding for the text region, the SUM of the shifted codebook rows for
    # the audio region, where-selected -- which over contiguous regions is
    # concatenation.
    text_embed = F.embedding(torch.tensor(TEXT_IDS), text_table)
    grid = torch.tensor(AUDIO_GRID)
    offsets = (torch.arange(CODEBOOKS) * AUDIO_VOCAB).view(CODEBOOKS, 1)
    audio_embed = F.embedding(grid + offsets, audio_table).sum(dim=0)
    merged = torch.cat([text_embed, audio_embed], dim=0)

    mask = torch.ones(1, 1, POSITIONS, POSITIONS, dtype=torch.bool)
    position_ids = torch.arange(POSITIONS).unsqueeze(0)
    with torch.no_grad():
        outputs = model(inputs_embeds=merged.unsqueeze(0), attention_mask=mask,
                        position_ids=position_ids, output_hidden_states=True)
    layer_out = outputs.hidden_states[1][0]      # after layer 0, before the norm
    final = outputs.last_hidden_state[0]         # after the final norm
    logits = F.linear(final, heads_table)        # [POSITIONS, CODEBOOKS * AUDIO_VOCAB]

    dump("kExpectedMerged", merged)
    dump("kExpectedLayerOutput", layer_out)
    dump("kExpectedFinal", final)
    dump("kExpectedLogits", logits)


if __name__ == "__main__":
    main()
```

Run it, eyeball that the arrays are finite and O(1)-magnitude, and paste the four arrays into the test. If the installed transformers rejects a keyword (`attn_implementation` moved between majors), adjust the construction until `model.config._attn_implementation == "eager"` and note the change in the test's provenance comment.

```bash
uv run --project scripts/envs/omnivoice --locked python scripts/dump_reference_omnivoice_generator.py > /tmp/omnivoice_generator_expected.txt
head -8 /tmp/omnivoice_generator_expected.txt
```

- [ ] **Step 4: Write the failing unit test**

`tests/omnivoice_generator_test.cpp`, the qwen3-tts talker-test technique end to end: the same `LcgStream` class verbatim (constants `6364136223846793005ull`/`1442695040888963407ull`, 24-bit numerator over 2^23), a `Fixture` built in a `no_alloc` persistent context committed with `ggml_backend_alloc_ctx_tensors`, the `compute()` helper verbatim from `tests/qwen3_tts_talker_test.cpp` (every read-back output gets `ggml_set_output` + `ggml_build_forward_expand` before `ggml_gallocr_alloc_graph`), the `deviation()` helper, a device sweep over `ggml_backend_dev_count()` with tolerance `1e-4f` on CPU and `5e-3f` on accelerators, and the pasted `kExpected*` arrays with the provenance comment `// values from scripts/dump_reference_omnivoice_generator.py at transformers <version>`.

Fixture tensors, created and LCG-filled in EXACTLY the script's order (toy dims transcribed from the script's constants; the ggml shapes are the torch shapes reversed):

| order | tensor | ggml ne | scale/offset |
|---|---|---|---|
| 1 | t_text_embed | {8, 20} | 0.5, 0.0 |
| 2 | t_audio_embed | {8, 18} | 0.5, 0.0 |
| 3 | t_audio_heads | {8, 18} | 0.5, 0.0 |
| 4 | t_norm | {8} | 0.25, 1.0 |
| 5–15 | the 11 layer weights in the script's `assignments` order | as qwen3's decoder-layer test at these dims | as listed there |

plus I32 inputs `t_text_ids` {2} = {3, 7}, `t_audio_ids` {3, 3} filled codebook-major with the HOST-SHIFTED grid (`AUDIO_GRID[c][s] + c * 6` → row 0: {11, 6, 8}, row 1: {11, 7, 11}, row 2: {16, 17, 15}), `t_positions` {5} = {0..4}.

Checks (each a `check_*` function returning failures, summed in `main`):

- `check_embedding_merge` — `build_canvas_embedding(ctx, weights, t_text_ids, t_audio_ids, 3)` returns `[8, 5]`; computed values match `kExpectedMerged` (flat order s·8+h on both sides).
- `check_forward` — `build_generator_forward(…, &layers, &final)` with `mask = nullptr` returns `[6, 3, 5]`; `layers[0]` matches `kExpectedLayerOutput`, `final` matches `kExpectedFinal`, logits match `kExpectedLogits` (script flat order s·18 + c·6 + v equals the ggml read-back of a contiguous `[6, 3, 5]` tensor).
- `check_uncond_embedding` — `build_canvas_embedding(ctx, weights, nullptr, t_audio_ids, 3)` returns `[8, 3]` equal to the last 3 columns of `kExpectedMerged`.
- `check_mask_gates` — self-consistency, no Python: run `generator_layer` over the 5-position merged input with an explicit F32 mask `[5, 5]` where column 4 (key 4) is `-INFINITY` for queries 0–3 and row 4 attends only itself; separately run the 4-position prefix with `mask = nullptr`. The first four positions of both outputs must agree within `1e-5f` — the mask tensor demonstrably gates attention.
- `check_rejections` — null `q_norm` → nullptr; `position_ids` of the wrong length → nullptr; `audio_ids` with `ne[1] != num_codebooks` → nullptr from `build_canvas_embedding`; both id tensors null → nullptr. (These builders never touch a graph, so nullptr alone is the contract.)

Register:

```cmake
synth_add_unit_test(synthesize-omnivoice-generator-test omnivoice_generator_test.cpp)
```

RED:

```bash
cmake --build build --target synthesize-omnivoice-generator-test
```

Expected: FAIL — `generator.h` symbols undefined until Step 2's files are added; once they compile, the test must pass its numeric checks or the layer diff (per-head norms, no scales, NEOX rope at theta 1e6) is wrong — debug against the script's intermediate values, never by widening the tolerance.

- [ ] **Step 4b: Correct the family doc's embedding-merge sentence**

docs/porting/families/omnivoice.md (Architecture section, the sentence around line 160 reading "the sequence embedding is the text embedding of row 0 **plus** the sum of the eight codebook embeddings") states the opposite of the verified upstream semantics and of the code this task just built. Upstream's `_prepare_embed_inputs` ends in `torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)` — a SELECT: audio positions carry the summed offset codebook embeddings alone, and the text stream computed at those positions is discarded. Rewrite the sentence to state the select semantics (in the doc's voice, one or two sentences), naming `torch.where` so the next reader cannot re-derive the addition. This is the "where-vs-add merge trap" Task 9's debugging list names; the doc is the authority implementers debug against, so it must not point the wrong way.

- [ ] **Step 5: Green, sanitize, format, commit**

```bash
cmake --build build --target synthesize-check-unit
ctest --test-dir build --output-on-failure -R synthesize-omnivoice-generator-test
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add src/arch/omnivoice/generator.h src/arch/omnivoice/generator.cpp src/CMakeLists.txt \
        scripts/dump_reference_omnivoice_generator.py tests/omnivoice_generator_test.cpp tests/CMakeLists.txt \
        docs/porting/families/omnivoice.md
git commit -m "$(cat <<'EOF'
Build the omnivoice bidirectional generator graph with its canvas embedding

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 8: Generator host — prompt grid, timestep schedule, CFG scoring, flat top-k commit

Slice 5's host half, as pure functions with hand-computable fixtures — everything discrete lives here, off the graph, per the discrete-outputs placement rule. Every formula is a transcription of `_generate_iterative` / `_predict_tokens_with_scoring` / `_get_time_steps` at the pinned revision (quoted in `docs/porting/families/omnivoice.md`, "The decode loop"), not a reimplementation of intent.

**Files:**
- Create: `src/arch/omnivoice/generator-host.h`, `src/arch/omnivoice/generator-host.cpp`
- Create: `tests/omnivoice_generator_host_test.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces (namespace `synth::omnivoice`), consumed by Task 9's prompt assembly and Task 10's loop:
  - `struct PromptLayout` + `build_prompt_grid(...)`
  - `shifted_timesteps(num_step, t_shift)` and `commit_schedule(total_mask, num_step, t_shift)`
  - `choose_token(cond, uncond, vocab_size, mask_id, guidance_scale, token&, log_prob&)`
  - `struct MaskedCandidate` + `select_commits(candidates&, budget)`
  - `fill_shifted_audio_ids(grid, row_stride, offset, count, num_codebooks, vocab_size, output&)`

- [ ] **Step 1: Write `generator-host.h`**

```cpp
#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::omnivoice {

// The host side of the mask-predict decode loop: the prompt grid, the commit
// schedule, the guided scoring and the flat top-k commit. None of it touches a
// tensor. Everything here transcribes omnivoice/models/omnivoice.py at the
// pinned revision -- _prepare_inference_inputs, _get_time_steps,
// _generate_iterative and _predict_tokens_with_scoring -- because a wrong rule
// here produces a grid the codec still decodes, and only the exact-token gate
// would ever notice.

// The conditional prompt as an 8-row grid, codebook-major:
// grid[c * total() + s]. Text region first (every row repeats the same ids),
// then the reference audio tokens, then target_frames of the mask id.
struct PromptLayout {
    std::vector<int32_t> grid;
    uint64_t             text_length      = 0;
    uint64_t             reference_frames = 0;
    uint64_t             target_frames    = 0;
    uint32_t             num_codebooks    = 0;

    uint64_t total() const { return text_length + reference_frames + target_frames; }

    uint64_t audio_start() const { return text_length; }

    uint64_t audio_length() const { return reference_frames + target_frames; }
};

// Builds the conditional grid. `text_ids` is row 0's text region (style markers
// plus wrapped text, already tokenized); `reference_tokens` is codebook-major
// [num_codebooks * reference_frames], empty when there is no reference.
// Refuses empty text, zero target frames, a reference stream that is not a
// whole number of codebook rows, or any reference token outside [0, mask_id).
synth_status_t build_prompt_grid(const std::vector<int32_t> & text_ids,
                                 const std::vector<int32_t> & reference_tokens,
                                 uint64_t                     target_frames,
                                 uint32_t                     num_codebooks,
                                 uint32_t                     mask_id,
                                 PromptLayout &               output);

// num_step + 1 points linearly spaced on [0, 1], each shifted by
// t' = t_shift * t / (1 + (t_shift - 1) * t). t_shift 0.1 makes the early
// intervals small: the loop commits little while everything is masked and most
// near the end.
std::vector<double> shifted_timesteps(uint32_t num_step, double t_shift);

// Per-step commit budgets over total_mask = 8 * T positions. Step s commits
// ceil(total_mask * (t'[s+1] - t'[s])) clamped to what remains; the FINAL step
// commits the entire remainder, so the grid is always fully committed after
// num_step steps regardless of rounding. Sums to exactly total_mask.
std::vector<uint64_t> commit_schedule(uint64_t total_mask, uint32_t num_step, double t_shift);

// One masked position's decision from its conditional and unconditional logit
// rows (each vocab_size floats). With guidance_scale != 0 the combination is
// log_softmax(log_pc + s * (log_pc - log_pu)) over log-softmaxed inputs -- the
// double log-softmax is the reference's, not an accident. The mask id is
// banned AFTER the combination; `token` is the argmax over what remains and
// `log_prob` its guided log-probability. `uncond` may be nullptr only when
// guidance_scale == 0 (the reference's `guidance_scale != 0` branch).
void choose_token(const float * cond,
                  const float * uncond,
                  uint32_t      vocab_size,
                  uint32_t      mask_id,
                  float         guidance_scale,
                  int32_t &     token,
                  float &       log_prob);

// One still-masked canvas position, scored. `score` is the guided log-prob
// minus codebook * layer_penalty_factor -- the bias that commits coarse
// codebooks first.
struct MaskedCandidate {
    uint32_t codebook = 0;
    uint64_t frame    = 0;
    int32_t  token    = 0;
    float    score    = 0.0f;
};

// Flat top-k over the candidates: partitions the `budget` highest scores to
// the front and returns how many were kept (min(budget, size)). Ties break on
// (higher score, then lower codebook, then lower frame) -- torch.topk's tie
// order is unspecified, so this port pins its own; an exact tie in F32 scores
// between real logits has never been observed and the exact-token gate would
// surface one instantly.
size_t select_commits(std::vector<MaskedCandidate> & candidates, uint64_t budget);

// Shifts a codebook-major id region into the stacked audio-embedding table's
// row space: output[c * count + i] = grid[c * row_stride + offset + i]
// + c * vocab_size. `output` is sized by the callee.
void fill_shifted_audio_ids(const int32_t *        grid,
                            uint64_t               row_stride,
                            uint64_t               offset,
                            uint64_t               count,
                            uint32_t               num_codebooks,
                            uint32_t               vocab_size,
                            std::vector<int32_t> & output);

}  // namespace synth::omnivoice
```

- [ ] **Step 2: Write `generator-host.cpp`**

```cpp
#include "arch/omnivoice/generator-host.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace synth::omnivoice {

synth_status_t build_prompt_grid(const std::vector<int32_t> & text_ids,
                                 const std::vector<int32_t> & reference_tokens,
                                 uint64_t                     target_frames,
                                 uint32_t                     num_codebooks,
                                 uint32_t                     mask_id,
                                 PromptLayout &               output) {
    output = PromptLayout{};
    if (text_ids.empty() || target_frames == 0 || num_codebooks == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (reference_tokens.size() % num_codebooks != 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    const uint64_t reference_frames = reference_tokens.size() / num_codebooks;
    for (int32_t token : reference_tokens) {
        // A reference token is a committed code; the mask id inside one would
        // mean the encoder emitted a hole, which is a caller bug, not a state.
        if (token < 0 || uint32_t(token) >= mask_id) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }

    output.text_length      = text_ids.size();
    output.reference_frames = reference_frames;
    output.target_frames    = target_frames;
    output.num_codebooks    = num_codebooks;
    const uint64_t total    = output.total();
    output.grid.assign(size_t(num_codebooks) * total, int32_t(mask_id));
    for (uint32_t codebook = 0; codebook < num_codebooks; ++codebook) {
        int32_t * row = output.grid.data() + size_t(codebook) * total;
        // Every row of the codebook dimension repeats the same text ids; the
        // rows differ only where audio tokens live.
        std::copy(text_ids.begin(), text_ids.end(), row);
        const int32_t * reference = reference_tokens.data() + size_t(codebook) * reference_frames;
        std::copy(reference, reference + reference_frames, row + output.text_length);
        // The target region stays at mask_id from the assign above.
    }
    return SYNTH_OK;
}

std::vector<double> shifted_timesteps(uint32_t num_step, double t_shift) {
    std::vector<double> timesteps(size_t(num_step) + 1);
    for (uint32_t index = 0; index <= num_step; ++index) {
        const double t   = num_step == 0 ? 0.0 : double(index) / double(num_step);
        timesteps[index] = t_shift * t / (1.0 + (t_shift - 1.0) * t);
    }
    return timesteps;
}

std::vector<uint64_t> commit_schedule(uint64_t total_mask, uint32_t num_step, double t_shift) {
    const std::vector<double> timesteps = shifted_timesteps(num_step, t_shift);
    std::vector<uint64_t>     schedule(num_step, 0);
    uint64_t                  remaining = total_mask;
    for (uint32_t step = 0; step < num_step; ++step) {
        uint64_t count;
        if (step + 1 == num_step) {
            // The final step commits the entire remainder, so the grid is
            // always fully committed regardless of rounding.
            count = remaining;
        } else {
            const double share = double(total_mask) * (timesteps[step + 1] - timesteps[step]);
            count              = std::min(uint64_t(std::ceil(share)), remaining);
        }
        schedule[step] = count;
        remaining -= count;
    }
    return schedule;
}

namespace {

// Numerically stable log-softmax into `output`. The max subtraction and the
// double accumulator keep the exp sum honest; the argmax downstream depends
// only on monotone shifts, so this is about the confidence value, not the
// winner.
void log_softmax(const float * input, uint32_t count, float * output) {
    float highest = -std::numeric_limits<float>::infinity();
    for (uint32_t index = 0; index < count; ++index) {
        highest = std::max(highest, input[index]);
    }
    double total = 0.0;
    for (uint32_t index = 0; index < count; ++index) {
        total += std::exp(double(input[index]) - highest);
    }
    const float log_total = float(std::log(total));
    for (uint32_t index = 0; index < count; ++index) {
        output[index] = input[index] - highest - log_total;
    }
}

}  // namespace

void choose_token(const float * cond,
                  const float * uncond,
                  uint32_t      vocab_size,
                  uint32_t      mask_id,
                  float         guidance_scale,
                  int32_t &     token,
                  float &       log_prob) {
    thread_local std::vector<float> cond_lp;
    thread_local std::vector<float> uncond_lp;
    thread_local std::vector<float> guided;
    cond_lp.resize(vocab_size);
    guided.resize(vocab_size);

    log_softmax(cond, vocab_size, cond_lp.data());
    if (guidance_scale != 0.0f && uncond != nullptr) {
        uncond_lp.resize(vocab_size);
        log_softmax(uncond, vocab_size, uncond_lp.data());
        // log_softmax(log_pc + s * (log_pc - log_pu)): the second log-softmax
        // is a constant shift per position, which the confidence value (not
        // the argmax) depends on.
        for (uint32_t index = 0; index < vocab_size; ++index) {
            guided[index] = cond_lp[index] + guidance_scale * (cond_lp[index] - uncond_lp[index]);
        }
        log_softmax(guided.data(), vocab_size, guided.data());
    } else {
        std::copy(cond_lp.begin(), cond_lp.end(), guided.begin());
    }

    // The mask id is banned AFTER the combination, so the model can never
    // commit a mask; everything else competes.
    guided[mask_id] = -std::numeric_limits<float>::infinity();

    int32_t best       = 0;
    float   best_value = -std::numeric_limits<float>::infinity();
    for (uint32_t index = 0; index < vocab_size; ++index) {
        if (guided[index] > best_value) {
            best_value = guided[index];
            best       = int32_t(index);
        }
    }
    token    = best;
    log_prob = best_value;
}

size_t select_commits(std::vector<MaskedCandidate> & candidates, uint64_t budget) {
    const size_t keep = size_t(std::min<uint64_t>(budget, candidates.size()));
    const auto   before = [](const MaskedCandidate & left, const MaskedCandidate & right) {
        if (left.score != right.score) {
            return left.score > right.score;
        }
        if (left.codebook != right.codebook) {
            return left.codebook < right.codebook;
        }
        return left.frame < right.frame;
    };
    std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(), before);
    return keep;
}

void fill_shifted_audio_ids(const int32_t *        grid,
                            uint64_t               row_stride,
                            uint64_t               offset,
                            uint64_t               count,
                            uint32_t               num_codebooks,
                            uint32_t               vocab_size,
                            std::vector<int32_t> & output) {
    output.resize(size_t(num_codebooks) * count);
    for (uint32_t codebook = 0; codebook < num_codebooks; ++codebook) {
        const int32_t shift = int32_t(codebook * vocab_size);
        for (uint64_t index = 0; index < count; ++index) {
            output[size_t(codebook) * count + index] =
                grid[size_t(codebook) * row_stride + offset + index] + shift;
        }
    }
}

}  // namespace synth::omnivoice
```

Add `arch/omnivoice/generator-host.cpp` to `src/CMakeLists.txt`.

- [ ] **Step 3: Write the failing unit test**

`tests/omnivoice_generator_host_test.cpp` (standalone `main` + `SYNTH_TEST_CHECK`, no ggml), registered with `synth_add_unit_test(synthesize-omnivoice-generator-host-test omnivoice_generator_host_test.cpp)`. Checks, every expected value hand-computable:

- `check_schedule_hand_fixture` — `commit_schedule(10, 4, 0.5)`: the shifted points are 0, 1/7, 1/3, 3/5, 1 (t' = 0.5t/(1 − 0.5t)); ceil(10·Δ) gives {2, 2, 3}, remainder 3 → assert the schedule equals `{2, 2, 3, 3}` and sums to 10.
- `check_schedule_clamps` — `commit_schedule(2, 4, 1.0)` (identity shift, Δ = 0.25, ceil(0.5) = 1): assert `{1, 1, 0, 0}` — the mid-schedule clamp to the remainder.
- `check_schedule_real_parameters` — `commit_schedule(400, 32, 0.1)` (a 50-frame canvas): sums to exactly 400; every entry ≥ 0; the last entry is the largest (t_shift 0.1 back-loads the budget); `shifted_timesteps(32, 0.1)` starts at 0.0 and ends at exactly 1.0.
- `check_choose_token_guided` — vocab 4, mask id 3, `cond = {ln .5, ln .25, ln .125, ln .125}` (already normalized, so its log-softmax is itself), `uncond = {ln .25, ln .25, ln .25, ln .25}`, scale 2.0. Guided pre-norm is `3·log_pc − 2·log_pu`, whose exps are {2, 1/4, 1/32, 1/32}; after the second log-softmax and the mask ban, argmax is token 0 with log-prob `ln(2 / 2.3125) = −0.1451820098`. Assert token == 0 and `|log_prob − (−0.14518201f)| < 1e-6f`.
- `check_choose_token_bans_mask` — cond with the mask id's logit at +100 (dominant): the argmax must still be a non-mask token.
- `check_choose_token_unguided` — scale 0.0, `uncond = nullptr`: token = argmax of cond alone, log_prob = its plain log-softmax value.
- `check_select_commits` — candidates with scores {5, 4, 4, 3} at (codebook, frame) = (1,0), (0,5), (0,2), (0,0) and budget 2: front two are the score-5 entry then the score-4 entry at (0,2) — the tie inside score 4 breaks on the lower frame; returns 2. Also budget 10 → returns 4.
- `check_prompt_grid` — text {7, 8}, reference (2 codebooks × 2 frames) {10, 11, 20, 21}, target 3, mask 1024: `total() == 7` (2 + 2 + 3); grid row 0 = {7, 8, 10, 11, 1024, 1024, 1024}, row 1 = {7, 8, 20, 21, 1024, 1024, 1024}. Also: empty text → `SYNTH_ERR_INVALID_ARG`; reference size 3 with 2 codebooks → invalid; a reference token equal to 1024 → invalid; zero target frames → invalid.
- `check_shifted_ids` — over `check_prompt_grid`'s layout with vocab_size 1025: `fill_shifted_audio_ids(grid.data(), 7, 2, 5, 2, 1025, out)` yields row 0 {10, 11, 1024, 1024, 1024} and row 1 {20+1025, 21+1025, 1024+1025, 1024+1025, 1024+1025}.

RED:

```bash
cmake --build build --target synthesize-omnivoice-generator-host-test
```

Expected: FAIL until Step 2 lands; then every hand fixture must pass exactly (no tolerances except the one 1e-6 on the guided log-prob).

- [ ] **Step 4: Green, sanitize, format, commit**

```bash
cmake --build build --target synthesize-check-unit
ctest --test-dir build --output-on-failure -R synthesize-omnivoice-generator-host-test
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add src/arch/omnivoice/generator-host.h src/arch/omnivoice/generator-host.cpp src/CMakeLists.txt \
        tests/omnivoice_generator_host_test.cpp tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
Port the omnivoice decode loop's host rules with hand-computable fixtures

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 9: `run_synthesis` probe path + replay runner + validator — single-forward parity (slice 4 gate)

The family synthesis entry, the graph-run machinery, and the Stage-5 harness in its first working form. The gate: the port's **step-0 conditional forward** must reproduce the oracle's `generator/hidden_l{0,7,14,21,27}.f32`, `generator/final.f32` and `generator/logits_step0.f32` on all 20 cases. The comparison vehicle is the real harness — `tests/omnivoice_replay_real.cpp` + `scripts/validate-omnivoice-replay.py` — written once here with its final CLI contract; Tasks 10 and 12 light up the greedy and decode flags it already carries, and Task 14 adds `--check`'s tolerance cell. (This settles the "runner+validator or a focused integration test" question: the harness, so nothing is written twice.)

The runner is an adapter, not a test (the qwen3-tts doctrine): it asserts nothing numeric and reports what it produced; comparisons and thresholds live in the validator and the committed tolerance file.

**Files:**
- Modify: `src/arch/omnivoice/omnivoice.h` (SynthesisRequest/SynthesisOutput/run_synthesis/decode_codes)
- Modify: `src/arch/omnivoice/model.cpp` (GraphRun/Persistent/read_floats + the forward/probe path)
- Create: `tests/omnivoice_replay_real.cpp`
- Create: `scripts/validate-omnivoice-replay.py`
- Modify: `tests/CMakeLists.txt` (integration target registration)

**Interfaces:**
- Produces: `synth::omnivoice::Model::run_synthesis(const SynthesisRequest &, SynthesisOutput &)` and `Model::decode_codes(...)` (declared now; body lands in Task 12); the runner CLI `<model.gguf> <case-dir> <out-dir> <num-step> <run-greedy 0|1> <decode-replay 0|1> <volume peak|none> [probe-layers...]`; the validator CLI with `--require {probes,grid,all}`.
- **The public seam stays stubbed**: `src/synthesize.cpp`'s omnivoice branch is not touched in this task or anywhere in Plan 2.

- [ ] **Step 1: Extend `src/arch/omnivoice/omnivoice.h`**

Add after the `ModelInfo` struct, before `class Model`:

```cpp
// A family-internal synthesis request. Plan 2 reaches this through the replay
// runner only: the public seam stays on the synthesis.not_implemented stub
// until Plan 3 lands the sampled path the public defaults select. The prompt
// arrives as ids (the oracle's input/token_ids.i32, or Plan 3's own frontend
// output) and the canvas length arrives fixed -- the duration estimator's
// output, or the oracle grid's frame count under replay.
struct SynthesisRequest {
    // Row 0's text region: style markers, language/instruct slots and the
    // wrapped text, already tokenized. Every codebook row repeats these ids.
    std::vector<int32_t>  prompt_text_ids;
    // Reference audio tokens, codebook-major [num_codebooks * frames]; empty
    // for auto-voice and voice-design requests. Plan 3's cloning encoder
    // produces these from audio; Plan 2's runner replays the oracle's.
    std::vector<int32_t>  reference_tokens;
    uint64_t              target_frames = 0;
    uint32_t              num_step      = 0;  // 0 = the package's embedded default
    // Stop after the step-0 conditional forward with the probe buffers filled;
    // the sampled golden cases compare only that forward in Plan 2.
    bool                  probe_only    = false;
    int                   threads       = 0;  // 0 = default_synthesis_threads()
    std::vector<uint32_t> probe_layers;       // layer indices probed at step 0
};

struct SynthesisOutput {
    uint64_t             frame_count = 0;
    // The committed grid, codebook-major [num_codebooks * frame_count] --
    // codebook c, frame t at c * frame_count + t, the oracle's codes/grid.i32
    // layout exactly.
    std::vector<int32_t> codes;
    // The decoded waveform with the no-reference volume branch applied
    // (peak-normalise-to-0.5), which is what the oracle returns to its caller.
    std::vector<float>   audio;

    // Step-0 conditional probes, in ggml read-back order; the runner reorders
    // logits into the oracle's [C, S, V] layout on write.
    std::vector<float>              logits_step0;  // [positions][codebooks][vocab]
    std::vector<float>              final_hidden;  // [positions][hidden]
    std::vector<std::vector<float>> layer_hidden;  // one per requested layer

    struct StagePlacement {
        uint64_t nodes             = 0;
        uint64_t accelerator_nodes = 0;
    };

    double         generator_seconds       = 0.0;
    double         generator_setup_seconds = 0.0;
    double         codec_seconds           = 0.0;
    StagePlacement generator_placement;
    StagePlacement codec_placement;
};
```

and inside `class Model`'s public section, after `text_vocab_size()`:

```cpp
    // The greedy mask-predict synthesis path. Deterministic: with the golden
    // parameters it makes no random draw at all, which is what the exact-token
    // gate stands on. The sampled path is Plan 3.
    synth_status_t run_synthesis(const SynthesisRequest & request, SynthesisOutput & output);

    // Decodes a committed grid (codebook-major [num_codebooks * frame_count],
    // values in [0, codebook_size)) to the RAW waveform -- no volume branch,
    // so the replay seam can apply the oracle's branch per case.
    synth_status_t decode_codes(const std::vector<int32_t> & codes,
                                uint64_t                     frame_count,
                                int                          threads,
                                std::vector<float> &         audio);
```

- [ ] **Step 2: Transcribe the run machinery into `model.cpp`**

Into `src/arch/omnivoice/model.cpp`'s anonymous namespace, transcribe **verbatim** from `src/arch/qwen3-tts/model.cpp`: `kSchedulerLeafAllowance` (4096), `now_seconds()`, `class GraphRun` (lines 43–132 there, comments included), `class Persistent` (lines 134–175), and the `read_floats` helper (`output.resize(size_t(ggml_nelements(tensor))); ggml_backend_tensor_get(tensor, output.data(), 0, ggml_nbytes(tensor));`). They are family-agnostic; the one omnivoice-specific note to add above `GraphRun`:

```cpp
// Duplicated from qwen3-tts rather than shared: the third copy is the signal
// to hoist (the BPE rule), and stage 7's graph-reuse question may reshape this
// family's copy anyway. A fresh GraphRun per forward is the measured, known
// pattern; its cost is what setup_seconds exists to expose.
```

Also add the includes: `"arch/omnivoice/generator.h"`, `"arch/omnivoice/generator-host.h"`, `"cpu-parallelism.h"`, `<chrono>`, `<cmath>`, `<cstring>`, `<vector>`.

- [ ] **Step 3: Implement the forward helper and the probe path**

File-local, above `run_synthesis`:

```cpp
// Which probe buffers a forward should fill; empty = no probes.
struct ForwardProbeSinks {
    const std::vector<uint32_t> *     layer_indices = nullptr;
    std::vector<float> *              logits_full   = nullptr;
    std::vector<float> *              final_hidden  = nullptr;
    std::vector<std::vector<float>> * layer_hidden  = nullptr;
};

// One full-canvas forward of one CFG branch on the CPU scheduler. Reads back
// the FULL logits [vocab, codebooks, positions] into `logits`; the caller
// slices the target region (the trailing target_frames positions). text_ids
// is null for the unconditional branch, whose every position is an audio slot.
//
// Takes the plan/weights/hparams pieces rather than Model::Impl: a file-local
// function cannot name a private nested type, and passing the pieces keeps it
// callable from every member without a friend declaration.
synth_status_t generator_branch_forward(const BackendPlan &       plan,
                                        const ModelWeights &      weights,
                                        const HParams &           hparams,
                                        ggml_tensor *             text_ids,
                                        ggml_tensor *             audio_ids,
                                        ggml_tensor *             positions,
                                        int                       threads,
                                        const ForwardProbeSinks & probes,
                                        std::vector<float> &      logits,
                                        SynthesisOutput &         output) {
    const AttentionShape shape{ hparams.generator.hidden_size,     hparams.generator.attention_head_count,
                                hparams.generator.key_value_head_count, hparams.generator.head_dim,
                                hparams.generator.rms_norm_eps,    hparams.generator.rope_theta };
    // Each block is under fifty nodes; the embedding merge and the head add a
    // fixed tail (the qwen3-tts budget formula).
    const size_t nodes = size_t(hparams.generator.layer_count) * 64 + 512;
    GraphRun     run(plan, nodes);
    if (!run.ok()) {
        return SYNTH_ERR_OOM;
    }
    ggml_tensor * embeddings = build_canvas_embedding(run.context(), weights.generator, text_ids, audio_ids,
                                                      hparams.audio.num_codebooks);
    std::vector<ggml_tensor *> layer_tensors;
    ggml_tensor *              final_tensor = nullptr;
    const bool                 probing      = probes.logits_full != nullptr;
    ggml_tensor * logits_tensor = build_generator_forward(run.context(), embeddings, positions, nullptr,
                                                          weights.generator, shape, hparams.audio,
                                                          probing ? &layer_tensors : nullptr,
                                                          probing ? &final_tensor : nullptr);
    if (logits_tensor == nullptr) {
        return SYNTH_ERR_INTERNAL;
    }
    if (probing) {
        // Read-back tensors that are not the graph output must be marked and
        // expanded before allocation, or the allocator reuses their buffers.
        for (ggml_tensor * tensor : layer_tensors) {
            ggml_set_output(tensor);
            ggml_build_forward_expand(run.graph(), tensor);
        }
        ggml_set_output(final_tensor);
        ggml_build_forward_expand(run.graph(), final_tensor);
    }
    const double         started = now_seconds();
    const synth_status_t status  = run.run(logits_tensor, "omnivoice.generator", threads);
    if (status != SYNTH_OK) {
        return status;
    }
    output.generator_seconds += now_seconds() - started;
    output.generator_setup_seconds += run.setup_seconds;
    output.generator_placement.nodes += run.placed_nodes;
    output.generator_placement.accelerator_nodes += run.accelerator_nodes;

    read_floats(logits_tensor, logits);
    if (probing) {
        *probes.logits_full = logits;
        read_floats(final_tensor, *probes.final_hidden);
        probes.layer_hidden->clear();
        for (uint32_t wanted : *probes.layer_indices) {
            if (wanted >= layer_tensors.size()) {
                return SYNTH_ERR_INVALID_ARG;
            }
            std::vector<float> values;
            read_floats(layer_tensors[wanted], values);
            probes.layer_hidden->push_back(std::move(values));
        }
    }
    return SYNTH_OK;
}
```

Then `run_synthesis`, Task 9 scope (validation, prompt, persistent inputs, the step-0 conditional forward with probes; the loop body is Task 10's single insertion point and until then fails loudly):

```cpp
synth_status_t Model::run_synthesis(const SynthesisRequest & request, SynthesisOutput & output) {
    output = SynthesisOutput{};
    Impl &          impl    = *implementation_;
    const HParams & hparams = impl.hparams;

    if (request.prompt_text_ids.empty() || request.target_frames == 0) {
        return SYNTH_ERR_INVALID_ARG;
    }
    if (request.target_frames > hparams.max_output_frames) {
        return SYNTH_ERR_OUTPUT_LIMIT;
    }
    const uint32_t num_step = request.num_step != 0 ? request.num_step : hparams.generation.num_step;
    const float    guidance = hparams.generation.guidance_scale;
    const int      threads  = request.threads > 0 ? request.threads : default_synthesis_threads();

    PromptLayout   prompt;
    synth_status_t status = build_prompt_grid(request.prompt_text_ids, request.reference_tokens,
                                              request.target_frames, hparams.audio.num_codebooks,
                                              hparams.audio.mask_id, prompt);
    if (status != SYNTH_OK) {
        return status;
    }
    const uint64_t total     = prompt.total();
    const uint64_t frames    = prompt.target_frames;
    const uint32_t codebooks = hparams.audio.num_codebooks;
    const uint32_t vocab     = hparams.audio.vocab_size;

    // Every reusable input lives in one persistent buffer; the per-step
    // refills touch only the audio-id tensors.
    Persistent inputs;
    if (!inputs.open(8)) {
        return SYNTH_ERR_OOM;
    }
    ggml_context * ictx          = inputs.context();
    ggml_tensor *  t_text        = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, int64_t(prompt.text_length));
    ggml_tensor *  t_cond_audio  = ggml_new_tensor_2d(ictx, GGML_TYPE_I32, int64_t(prompt.audio_length()),
                                                      int64_t(codebooks));
    ggml_tensor *  t_uncond_audio = ggml_new_tensor_2d(ictx, GGML_TYPE_I32, int64_t(frames), int64_t(codebooks));
    ggml_tensor *  t_cond_pos    = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, int64_t(total));
    ggml_tensor *  t_uncond_pos  = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, int64_t(frames));
    if (!inputs.commit(impl.backend_plan->cpu_backend())) {
        return SYNTH_ERR_OOM;
    }

    ggml_backend_tensor_set(t_text, request.prompt_text_ids.data(), 0, ggml_nbytes(t_text));
    std::vector<int32_t> sequential(size_t(total));
    for (uint64_t index = 0; index < total; ++index) {
        sequential[size_t(index)] = int32_t(index);
    }
    ggml_backend_tensor_set(t_cond_pos, sequential.data(), 0, ggml_nbytes(t_cond_pos));
    // The unconditional branch is its own sequence: positions restart at zero,
    // exactly as the reference's padded batch gives its second row.
    ggml_backend_tensor_set(t_uncond_pos, sequential.data(), 0, ggml_nbytes(t_uncond_pos));

    std::vector<int32_t> shifted;
    fill_shifted_audio_ids(prompt.grid.data(), total, prompt.audio_start(), prompt.audio_length(), codebooks,
                           vocab, shifted);
    ggml_backend_tensor_set(t_cond_audio, shifted.data(), 0, ggml_nbytes(t_cond_audio));

    // --- Step-0 conditional forward, with probes when asked. Its logits are
    // also step 0's conditional half once the loop runs.
    ForwardProbeSinks sinks;
    const bool        probing = !request.probe_layers.empty();
    if (probing) {
        sinks.layer_indices = &request.probe_layers;
        sinks.logits_full   = &output.logits_step0;
        sinks.final_hidden  = &output.final_hidden;
        sinks.layer_hidden  = &output.layer_hidden;
    }
    std::vector<float> cond_logits;
    status = generator_branch_forward(*impl.backend_plan, impl.weights, hparams, t_text, t_cond_audio, t_cond_pos,
                                      threads, sinks, cond_logits, output);
    if (status != SYNTH_OK) {
        return status;
    }
    if (request.probe_only) {
        return SYNTH_OK;
    }

    // === Task 10 replaces everything below this line with the greedy loop. ===
    std::fprintf(stderr, "omnivoice: the greedy decode loop is slice 5 and not built yet\n");
    return SYNTH_ERR_INTERNAL;
}
```

and the Task-12 placeholder body for `decode_codes` (replaced there):

```cpp
synth_status_t Model::decode_codes(const std::vector<int32_t> & codes, uint64_t frame_count, int threads,
                                   std::vector<float> & audio) {
    (void) codes;
    (void) frame_count;
    (void) threads;
    audio.clear();
    std::fprintf(stderr, "omnivoice: codec decode is slice 6 and not built yet\n");
    return SYNTH_ERR_INTERNAL;
}
```

- [ ] **Step 4: Write the replay runner**

`tests/omnivoice_replay_real.cpp`, mirroring `tests/qwen3_tts_replay_real.cpp`'s structure (`read_file`/`read_i32`/`write_f32` helpers verbatim, plus a `write_i32` twin; same exit taxonomy: 2 = usage/artifact IO, 1 = load or synthesis failure, 0 = success; exactly one JSON line on stdout):

```cpp
// Replays the oracle's cases through the port and writes the artifacts the
// Stage 5 validator compares.
//
// This is an adapter, not a test: it asserts nothing numeric and reports what
// it produced. The comparison and its thresholds live in
// scripts/validate-omnivoice-replay.py and tests/tolerances/omnivoice.json.
//
// Two parity channels ride one binary. The step-0 conditional forward's probes
// and the replayed-grid waveform are threshold comparisons; the free-running
// greedy grid is this family's structural_exactness and is compared exactly,
// never under a tolerance.

#include "arch/omnivoice/generator-host.h"
#include "arch/omnivoice/omnivoice.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kCodebooks = 8;
constexpr uint32_t kVocab     = 1025;

bool read_file(const std::string & path, std::vector<char> & bytes) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return false;
    }
    const std::streamsize size = input.tellg();
    input.seekg(0);
    bytes.resize(size_t(size));
    return bool(input.read(bytes.data(), size));
}

bool read_i32(const std::string & path, std::vector<int32_t> & values) {
    std::vector<char> bytes;
    if (!read_file(path, bytes) || bytes.size() % sizeof(int32_t) != 0) {
        return false;
    }
    values.resize(bytes.size() / sizeof(int32_t));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return true;
}

bool write_f32(const std::string & path, const std::vector<float> & values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
    return bool(output);
}

bool write_i32(const std::string & path, const std::vector<int32_t> & values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(int32_t)));
    return bool(output);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 8) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> <case-dir> <out-dir> <num-step> <run-greedy 0|1> "
                     "<decode-replay 0|1> <volume peak|none> [probe-layers...]\n",
                     argv[0]);
        return 2;
    }
    const std::string model_path(argv[1]);
    const std::string case_dir(argv[2]);
    const std::string out_dir(argv[3]);
    const uint32_t    num_step   = uint32_t(std::atoi(argv[4]));
    const bool        run_greedy = argv[5][0] == '1';
    const bool        decode     = argv[6][0] == '1';
    const std::string volume(argv[7]);
    if (volume != "peak" && volume != "none") {
        std::fprintf(stderr, "volume must be peak or none, got %s\n", volume.c_str());
        return 2;
    }

    std::vector<int32_t> text_ids;
    std::vector<int32_t> oracle_grid;
    std::vector<int32_t> oracle_prompt;
    if (!read_i32(case_dir + "/input/token_ids.i32", text_ids) ||
        !read_i32(case_dir + "/codes/grid.i32", oracle_grid) ||
        !read_i32(case_dir + "/input/prompt_grid.i32", oracle_prompt)) {
        std::fprintf(stderr, "cannot read the oracle's artifacts under %s\n", case_dir.c_str());
        return 2;
    }
    if (oracle_grid.empty() || oracle_grid.size() % kCodebooks != 0) {
        std::fprintf(stderr, "codes/grid.i32 holds %zu values, not a whole 8-codebook grid\n", oracle_grid.size());
        return 2;
    }
    const uint64_t frames = oracle_grid.size() / kCodebooks;

    std::vector<int32_t> reference_tokens;  // stays empty when the case has none
    read_i32(case_dir + "/ref/tokens.i32", reference_tokens);

    // The prompt this port would assemble must BE the oracle's step-0 input.
    // Byte equality here is a free structural check on the grid layout before
    // a single forward runs. (The orientation trap is real: a transposed grid
    // decodes to audio rather than to an error.)
    synth::omnivoice::PromptLayout prompt;
    if (synth::omnivoice::build_prompt_grid(text_ids, reference_tokens, frames, kCodebooks, kVocab - 1, prompt) !=
        SYNTH_OK) {
        std::fprintf(stderr, "cannot assemble the prompt grid\n");
        return 2;
    }
    if (prompt.grid.size() != oracle_prompt.size() ||
        std::memcmp(prompt.grid.data(), oracle_prompt.data(), oracle_prompt.size() * sizeof(int32_t)) != 0) {
        std::fprintf(stderr, "assembled prompt grid differs from input/prompt_grid.i32\n");
        return 2;
    }

    std::unique_ptr<synth::omnivoice::Model> model;
    synth_status_t status = synth::omnivoice::Model::load_cpu(model_path, model);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "load -> %d\n", int(status));
        return 1;
    }

    synth::omnivoice::SynthesisRequest request;
    request.prompt_text_ids  = text_ids;
    request.reference_tokens = reference_tokens;
    request.target_frames    = frames;
    request.num_step         = num_step;
    request.probe_only       = !run_greedy;
    request.threads          = 0;
    for (int index = 8; index < argc; ++index) {
        request.probe_layers.push_back(uint32_t(std::atoi(argv[index])));
    }

    synth::omnivoice::SynthesisOutput output;
    const auto                        started = std::chrono::steady_clock::now();
    status                                    = model->run_synthesis(request, output);
    if (status != SYNTH_OK) {
        std::fprintf(stderr, "run_synthesis -> %d\n", int(status));
        return 1;
    }

    // Probes: hidden buffers are already the oracle's [S, hidden] order; the
    // logits are read back position-major and the oracle stores them C-major
    // [C, S, V], so reorder on write.
    bool ok = true;
    if (!request.probe_layers.empty()) {
        const uint64_t positions = output.final_hidden.size() / 1024;
        std::vector<float> reordered(output.logits_step0.size());
        for (uint32_t codebook = 0; codebook < kCodebooks; ++codebook) {
            for (uint64_t position = 0; position < positions; ++position) {
                std::memcpy(reordered.data() + (size_t(codebook) * positions + position) * kVocab,
                            output.logits_step0.data() + (size_t(position) * kCodebooks + codebook) * kVocab,
                            kVocab * sizeof(float));
            }
        }
        ok = write_f32(out_dir + "/logits_step0.f32", reordered) &&
             write_f32(out_dir + "/final.f32", output.final_hidden);
        for (size_t index = 0; index < output.layer_hidden.size() && ok; ++index) {
            ok = write_f32(out_dir + "/hidden_l" + std::to_string(request.probe_layers[index]) + ".f32",
                           output.layer_hidden[index]);
        }
    }
    if (ok && run_greedy) {
        ok = write_i32(out_dir + "/grid.i32", output.codes);
    }

    size_t samples = 0;
    if (ok && decode) {
        std::vector<float> audio;
        status = model->decode_codes(oracle_grid, frames, 0, audio);
        if (status != SYNTH_OK) {
            std::fprintf(stderr, "decode_codes -> %d\n", int(status));
            return 1;
        }
        if (volume == "peak") {
            // The oracle's ungated no-reference branch: audio / peak * 0.5,
            // two float operations per element in numpy's order.
            float peak = 0.0f;
            for (float value : audio) {
                peak = std::fmax(peak, std::fabs(value));
            }
            if (peak > 1e-6f) {
                for (float & value : audio) {
                    value = value / peak * 0.5f;
                }
            }
        }
        samples = audio.size();
        ok      = write_f32(out_dir + "/pcm.f32", audio);
    }
    if (!ok) {
        std::fprintf(stderr, "cannot write under %s\n", out_dir.c_str());
        return 2;
    }

    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::printf("{\"frames\": %llu, \"samples\": %zu, \"probe_layers\": %zu, "
                "\"generator_seconds\": %.4f, \"generator_setup_seconds\": %.4f, \"codec_seconds\": %.4f, "
                "\"placement\": {\"generator\": [%llu, %llu], \"codec\": [%llu, %llu]}, \"wall_seconds\": %.4f}\n",
                (unsigned long long) frames, samples, output.layer_hidden.size(), output.generator_seconds,
                output.generator_setup_seconds, output.codec_seconds,
                (unsigned long long) output.generator_placement.nodes,
                (unsigned long long) output.generator_placement.accelerator_nodes,
                (unsigned long long) output.codec_placement.nodes,
                (unsigned long long) output.codec_placement.accelerator_nodes, wall);
    return 0;
}
```

Register in `tests/CMakeLists.txt` beside the qwen3-tts runners:

```cmake
add_executable(synthesize-omnivoice-replay-real omnivoice_replay_real.cpp)
target_include_directories(synthesize-omnivoice-replay-real PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(synthesize-omnivoice-replay-real PRIVATE synthesize ggml)
synth_register_integration_target(synthesize-omnivoice-replay-real)
```

- [ ] **Step 5: Write the validator**

`scripts/validate-omnivoice-replay.py` — the comparison math is `scripts/validate-qwen3-tts-replay.py`'s (`read_f32`, `cosine`, `compare` transcribed verbatim), plus this family's exact-token arm. Complete script:

```python
#!/usr/bin/env python3
"""Compare the omnivoice port's replay outputs against the oracle dumps.

Deep generator probes gate on cosine similarity, not max-abs; the waveform
gates on both, because a listener hears the waveform and a cosine over it
would hide a constant offset. The token grid is compared EXACTLY: greedy
decoding makes no RNG call, docs/porting/families/omnivoice.md defines
structural_exactness for this family as equality of the 8 x T grid, and no
tolerance file entry exists or ever will for it.

Without --check this script measures; with --check it gates against
tests/tolerances/omnivoice.json (profiles.<PROFILE>.stages.<STAGE>), refusing
to run when the cell is absent -- a threshold the suite writes for itself
proves nothing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

PROBE_LAYERS = (0, 7, 14, 21, 27)
VOLUME_BY_BRANCH = {"peak_normalise_to_0.5": "peak", "none": "none"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=pathlib.Path("tests/golden/omnivoice/omnivoice-0-6b.manifest.json"))
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--runner", type=pathlib.Path,
                        default=pathlib.Path("build/unit/bin/synthesize-omnivoice-replay-real"))
    parser.add_argument("--work", type=pathlib.Path,
                        default=pathlib.Path("build/goldens/omnivoice-replay"))
    parser.add_argument("--report", type=pathlib.Path, default=None)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--require", choices=("probes", "grid", "all"), default="all",
                        help="probes: step-0 forward only; grid: + the greedy free-run; "
                             "all: + the replayed-grid waveform")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--tolerances", type=pathlib.Path,
                        default=pathlib.Path("tests/tolerances/omnivoice.json"))
    parser.add_argument("--profile", default="F32")
    parser.add_argument("--backend", default="CPU")
    parser.add_argument("--stage", default="replay")
    return parser.parse_args(argv)


def read_f32(path: pathlib.Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.float32)


def read_i32(path: pathlib.Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.int32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left)) * float(np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if float(np.abs(left - right).max(initial=0.0)) == 0.0 else 0.0
    return float(np.dot(left, right) / denominator)


def compare(expected: np.ndarray, actual: np.ndarray) -> dict:
    if expected.shape != actual.shape:
        return {"shape_mismatch": [list(expected.shape), list(actual.shape)]}
    difference = np.abs(expected - actual)
    return {
        "elements": int(expected.size),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if expected.size else 0.0,
        "cosine": cosine(expected, actual),
    }


def is_greedy(case: dict) -> bool:
    return float(case["oracle"]["parameters"]["position_temperature"]) == 0.0


def run_case(arguments, case: dict, oracle_root: pathlib.Path) -> dict | None:
    case_id = case["id"]
    oracle = oracle_root / case_id
    if not (oracle / "codes/grid.i32").is_file():
        print(f"{case_id}: no oracle artifacts, skipped")
        return None

    # The probe set is a contract between the manifest, the dumper and this
    # script; drift dumps a file nothing compares or compares one nothing dumps.
    names = {artifact["name"] for artifact in case["expected"]["artifacts"]}
    wanted = {f"generator.hidden_l{layer}" for layer in PROBE_LAYERS}
    probed = {name for name in names if name.startswith("generator.hidden_l")}
    if probed != wanted:
        raise SystemExit(f"{case_id}: manifest probes {sorted(probed)} != validator's {sorted(wanted)}")

    greedy = is_greedy(case)
    run_greedy = greedy and arguments.require in ("grid", "all")
    decode = arguments.require == "all"
    branch = json.loads((oracle / "result.json").read_text(encoding="utf-8"))["volume_branch"]
    if branch not in VOLUME_BY_BRANCH:
        raise SystemExit(f"{case_id}: unsupported volume branch {branch!r}")

    work = arguments.work / case_id
    work.mkdir(parents=True, exist_ok=True)
    command = [str(arguments.runner), str(arguments.model), str(oracle), str(work),
               str(case["oracle"]["parameters"]["num_step"]),
               "1" if run_greedy else "0", "1" if decode else "0",
               VOLUME_BY_BRANCH[branch]] + [str(layer) for layer in PROBE_LAYERS]
    finished = subprocess.run(command, capture_output=True)
    if finished.returncode != 0:
        return {"case": case_id, "status": "runner-failed",
                "stderr": finished.stderr.decode("utf-8", "replace")[:400]}
    stats = None
    for line in reversed(finished.stdout.decode("utf-8", "replace").splitlines()):
        if line.startswith("{"):
            stats = json.loads(line)
            break
    if stats is None:
        return {"case": case_id, "status": "runner-said-nothing"}

    measurements = {}
    for layer in PROBE_LAYERS:
        measurements[f"generator.hidden_l{layer}"] = compare(
            read_f32(oracle / f"generator/hidden_l{layer}.f32"), read_f32(work / f"hidden_l{layer}.f32"))
    measurements["generator.final"] = compare(read_f32(oracle / "generator/final.f32"),
                                              read_f32(work / "final.f32"))
    measurements["generator.logits_step0"] = compare(read_f32(oracle / "generator/logits_step0.f32"),
                                                     read_f32(work / "logits_step0.f32"))
    grid = None
    if run_greedy:
        expected = read_i32(oracle / "codes/grid.i32")
        actual = read_i32(work / "grid.i32")
        mismatches = (int((expected != actual).sum()) if expected.shape == actual.shape
                      else int(expected.size))
        grid = {"elements": int(expected.size), "mismatches": mismatches, "exact": mismatches == 0}
    finite = None
    if decode:
        pcm = read_f32(work / "pcm.f32")
        measurements["audio.pcm"] = compare(read_f32(oracle / "audio/pcm.f32"), pcm)
        finite = bool(np.isfinite(pcm).all())

    return {"case": case_id, "status": "ok", "greedy": greedy, "stats": stats,
            "measurements": measurements, "grid": grid, "finite_pcm": finite}


def main(argv=None) -> int:
    arguments = parse_args(argv)
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    oracle_root = pathlib.Path(manifest["case_artifact_root"])
    cases = [case for case in manifest["cases"]
             if arguments.cases is None or case["id"] in arguments.cases]

    results, failures, structural_failures = [], 0, 0
    worst: dict[str, dict] = {}
    for case in cases:
        result = run_case(arguments, case, oracle_root)
        if result is None:
            continue
        results.append(result)
        if result["status"] != "ok":
            failures += 1
            print(f"{result['case']}: {result['status']}")
            continue
        for name, measurement in result["measurements"].items():
            if "shape_mismatch" in measurement:
                failures += 1
                print(f"{result['case']}: {name} shape mismatch {measurement['shape_mismatch']}")
                continue
            slot = worst.setdefault(name, {"max_abs": 0.0, "min_cosine": 1.0})
            slot["max_abs"] = max(slot["max_abs"], measurement["max_abs"])
            slot["min_cosine"] = min(slot["min_cosine"], measurement["cosine"])
        if result["grid"] is not None and not result["grid"]["exact"]:
            structural_failures += 1
            print(f"{result['case']}: token grid differs at {result['grid']['mismatches']} "
                  f"of {result['grid']['elements']} positions")
        if result["finite_pcm"] is False:
            failures += 1
            print(f"{result['case']}: non-finite PCM")
        # Plan 2 is CPU-only: a node on an accelerator is a placement bug.
        placement = result["stats"]["placement"]
        if placement["generator"][1] != 0 or placement["codec"][1] != 0:
            failures += 1
            print(f"{result['case']}: nodes left the CPU: {placement}")

    print(f"\n{'probe':32} {'max_abs':>12} {'min_cosine':>12}")
    for name in sorted(worst):
        print(f"{name:32} {worst[name]['max_abs']:12.6g} {worst[name]['min_cosine']:12.8f}")
    exact = [r for r in results if r.get("grid") is not None]
    if exact:
        good = sum(1 for r in exact if r["grid"]["exact"])
        print(f"token grids exact: {good}/{len(exact)}")

    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps({
            "schema": "synthesize-validation-report-v1", "family": "omnivoice",
            "variant": manifest["variant"], "suite_version": manifest["suite_version"],
            "phase": "oracle_replay", "profile": arguments.profile,
            "backend": arguments.backend.upper(), "require": arguments.require,
            "cases": results, "worst": worst,
        }, indent=2) + "\n", encoding="utf-8")

    if failures or structural_failures:
        return 1
    if not arguments.check:
        return 0

    # A tolerance is a reviewed number in a committed file. Refusing to run
    # rather than inventing one is the point.
    tolerances = json.loads(arguments.tolerances.read_text(encoding="utf-8"))
    cell = tolerances.get("profiles", {}).get(arguments.profile, {})
    if arguments.backend.upper() != "CPU":
        cell = cell.get("backends", {}).get(arguments.backend.upper(), {})
    stage = cell.get("stages", {}).get(arguments.stage)
    if not stage:
        print(f"\ntolerance cell {arguments.profile}/{arguments.backend}/{arguments.stage} "
              f"is not recorded in {arguments.tolerances}")
        return 1
    breaches = 0
    for name, observed in sorted(worst.items()):
        limits = stage["probes"].get(name)
        if limits is None:
            print(f"{name}: no tolerance recorded")
            breaches += 1
            continue
        if "max_abs" in limits and observed["max_abs"] > limits["max_abs"]:
            print(f"{name}: max_abs {observed['max_abs']:.6g} breaches {limits['max_abs']:.6g}")
            breaches += 1
        if "min_cosine" in limits and observed["min_cosine"] < limits["min_cosine"]:
            print(f"{name}: cosine {observed['min_cosine']:.8f} breaches {limits['min_cosine']:.8f}")
            breaches += 1
    if breaches:
        return 1
    print(f"\nall probes within the {arguments.profile}/{arguments.backend}/{arguments.stage} tolerances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: RED, then the slice-4 gate**

```bash
cmake --build build --target synthesize-omnivoice-replay-real     # RED first: fails until Steps 1-3 compile
cmake --build build --target synthesize-check-unit                # whole unit gate stays green
uv run --project scripts/envs/omnivoice --locked python scripts/validate-omnivoice-replay.py \
  --model models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  --runner build/unit/bin/synthesize-omnivoice-replay-real \
  --require probes --report build/goldens/omnivoice-replay/step0-report.json
```

(If the runner binary lands somewhere other than `build/unit/bin/`, read the path CMake actually used — `find build -name 'synthesize-omnivoice-replay-real'` — and pass it via `--runner`; fix the script's default to the real location.)

Expected: all 20 cases run; the worst-table prints seven probe rows; every `min_cosine` ≥ 0.999 and `generator`/`codec` accelerator nodes 0. Because oracle and port are both F32-on-CPU, expect cosines near 0.9999999 and max_abs in the 1e-4-and-below range on the hidden probes — if any cosine is below 0.999 the forward is WRONG (embedding merge, rope mode/theta, per-head norms, or the where-vs-add merge trap), and the fix is systematic debugging against the probe that first diverges (l0 bad = embedding/rope; l0 good but final bad = a later layer or the norm; hidden good but logits bad = the head reshape/reorder). Never proceed to Task 10 with a failing probe. Record the worst table in `_porting-log.md` (dated "slice 4 — single-forward parity").

- [ ] **Step 7: Green, sanitize, format, commit**

```bash
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add src/arch/omnivoice/omnivoice.h src/arch/omnivoice/model.cpp \
        tests/omnivoice_replay_real.cpp tests/CMakeLists.txt scripts/validate-omnivoice-replay.py \
        reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md
git commit -m "$(cat <<'EOF'
Run the omnivoice step-0 forward through the replay harness to probe parity

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 10: The greedy mask-predict decode loop — exact token-grid equality (slice 5 gate)

The loop replaces Task 9's guarded early return in `run_synthesis`. Everything numeric it needs already exists and is unit-tested (Task 8); this task is composition plus the gate: **all 17 greedy golden cases must reproduce the oracle's 8×T grid exactly**. No new unit test is added here — the loop's parts are the Task 8 fixtures, and its whole is gated by exact equality, which no synthetic test can strengthen.

**Files:**
- Modify: `src/arch/omnivoice/model.cpp` (the loop)

**Interfaces:**
- Consumes: `commit_schedule`, `choose_token`, `select_commits`, `fill_shifted_audio_ids` (Task 8); `generator_branch_forward` (Task 9).
- Produces: `run_synthesis` fills `output.codes` (codebook-major, the oracle's layout) and `output.frame_count`; audio stays empty until Task 12.

- [ ] **Step 1: RED**

```bash
build/unit/bin/synthesize-omnivoice-replay-real \
  models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  build/goldens/omnivoice/omni-short-en /tmp 32 1 0 peak 0 7 14 21 27; echo "exit=$?"
```

Expected: `omnivoice: the greedy decode loop is slice 5 and not built yet`, `run_synthesis -> …`, exit 1.

- [ ] **Step 2: Replace the marker block in `run_synthesis`**

Delete the two lines under `// === Task 10 replaces everything below this line …` and insert:

```cpp
    // The canonical canvas: the target region's committed state, codebook-major
    // [codebooks * frames], all mask at step 0. The reference writes committed
    // tokens back into both the conditional and unconditional rows; here both
    // branches' id tensors are refilled from this one grid, so there is a
    // single source of truth instead of two copies to keep in step.
    std::vector<int32_t> canvas(size_t(codebooks) * frames, int32_t(hparams.audio.mask_id));

    const std::vector<uint64_t> schedule =
        commit_schedule(uint64_t(codebooks) * frames, num_step, double(hparams.generation.t_shift));

    std::vector<float>           uncond_logits;
    std::vector<int32_t>         uncond_shifted;
    std::vector<MaskedCandidate> candidates;
    // One position's logits in the read-back buffer: codebooks * vocab floats,
    // position-major (position s starts at s * row).
    const uint64_t          row = uint64_t(codebooks) * vocab;
    const ForwardProbeSinks no_probes;

    for (uint32_t step = 0; step < num_step; ++step) {
        if (step > 0) {
            // The reference tokens never move; only the target region follows
            // the canvas. Refill and rerun the conditional branch (step 0's ran
            // above, with the probes).
            for (uint32_t codebook = 0; codebook < codebooks; ++codebook) {
                int32_t * target_row = prompt.grid.data() + size_t(codebook) * total + prompt.audio_start() +
                                       prompt.reference_frames;
                std::memcpy(target_row, canvas.data() + size_t(codebook) * frames,
                            size_t(frames) * sizeof(int32_t));
            }
            fill_shifted_audio_ids(prompt.grid.data(), total, prompt.audio_start(), prompt.audio_length(),
                                   codebooks, vocab, shifted);
            ggml_backend_tensor_set(t_cond_audio, shifted.data(), 0, ggml_nbytes(t_cond_audio));
            status = generator_branch_forward(*impl.backend_plan, impl.weights, hparams, t_text, t_cond_audio,
                                              t_cond_pos, threads, no_probes, cond_logits, output);
            if (status != SYNTH_OK) {
                return status;
            }
        }
        if (guidance != 0.0f) {
            // The unconditional branch carries the target region only -- no
            // style markers, no text, no reference audio.
            fill_shifted_audio_ids(canvas.data(), frames, 0, frames, codebooks, vocab, uncond_shifted);
            ggml_backend_tensor_set(t_uncond_audio, uncond_shifted.data(), 0, ggml_nbytes(t_uncond_audio));
            status = generator_branch_forward(*impl.backend_plan, impl.weights, hparams, nullptr, t_uncond_audio,
                                              t_uncond_pos, threads, no_probes, uncond_logits, output);
            if (status != SYNTH_OK) {
                return status;
            }
        }

        const uint64_t budget = schedule[step];
        if (budget == 0) {
            continue;
        }
        candidates.clear();
        for (uint32_t codebook = 0; codebook < codebooks; ++codebook) {
            for (uint64_t frame = 0; frame < frames; ++frame) {
                if (canvas[size_t(codebook) * frames + frame] != int32_t(hparams.audio.mask_id)) {
                    continue;  // committed in an earlier step; cannot be revisited
                }
                const size_t cond_offset =
                    size_t(total - frames + frame) * row + size_t(codebook) * vocab;
                const size_t    uncond_offset = size_t(frame) * row + size_t(codebook) * vocab;
                MaskedCandidate candidate;
                candidate.codebook = codebook;
                candidate.frame    = frame;
                float log_prob     = 0.0f;
                choose_token(cond_logits.data() + cond_offset,
                             guidance != 0.0f ? uncond_logits.data() + uncond_offset : nullptr, vocab,
                             hparams.audio.mask_id, guidance, candidate.token, log_prob);
                // The layer penalty biases commitment toward the coarse
                // codebooks first. (audio_codebook_weights is training-loss
                // weighting and plays NO part here -- see the family doc.)
                candidate.score = log_prob - float(codebook) * hparams.generation.layer_penalty_factor;
                candidates.push_back(candidate);
            }
        }
        const size_t committed = select_commits(candidates, budget);
        for (size_t index = 0; index < committed; ++index) {
            const MaskedCandidate & choice = candidates[index];
            canvas[size_t(choice.codebook) * frames + choice.frame] = choice.token;
        }
    }

    for (int32_t token : canvas) {
        if (token == int32_t(hparams.audio.mask_id)) {
            std::fprintf(stderr, "omnivoice: a mask survived the schedule; the loop is wrong\n");
            return SYNTH_ERR_INTERNAL;
        }
    }
    output.frame_count = frames;
    output.codes       = std::move(canvas);
    // Task 12 attaches the codec decode and the volume branch here; until then
    // the committed grid is the product and output.audio stays empty.
    return SYNTH_OK;
```

- [ ] **Step 3: GREEN — the exact-token gate, smoke then sweep**

Smoke on the shortest case first (one case ≈ 64 CPU forwards over 66 positions):

```bash
cmake --build build --target synthesize-check-unit
uv run --project scripts/envs/omnivoice --locked python scripts/validate-omnivoice-replay.py \
  --model models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  --runner build/unit/bin/synthesize-omnivoice-replay-real \
  --require grid --cases omni-short-en
```

Expected: `token grids exact: 1/1`, exit 0. If the grid mismatches: this is where the port lives or dies, so debug systematically, in this order — (a) mismatch count near 100%: the canvas layout or logits indexing is transposed (the orientation trap); (b) mismatches concentrated in late steps: the schedule disagrees — print both schedules (`commit_schedule` vs the oracle's, recomputable from `_get_time_steps`); (c) a handful of scattered flips: F32 arithmetic landed a near-tie differently — inspect the flipped positions' score gaps by re-running with prints; a genuine near-tie would be the first ever observed and goes to the porting log before any threshold discussion. Never relax the gate.

Then the full greedy sweep (17 cases; the two clone canvases are S=471, so expect 1–2 hours on CPU):

```bash
uv run --project scripts/envs/omnivoice --locked python scripts/validate-omnivoice-replay.py \
  --model models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  --runner build/unit/bin/synthesize-omnivoice-replay-real \
  --require grid --report build/goldens/omnivoice-replay/grid-report.json
```

Expected: `token grids exact: 17/17`, all probe rows still within Task 9's observations, exit 0. Record the 17/17 line and per-case wall clocks (from the report's `stats`) in `_porting-log.md` (dated "slice 5 — exact token grids").

- [ ] **Step 4: Sanitize, format, commit**

```bash
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add src/arch/omnivoice/model.cpp reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md
git commit -m "$(cat <<'EOF'
Commit the omnivoice greedy decode loop at exact token-grid parity

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 11: Codec decode graph — RVQ dequant, fc2, DAC decoder (slice 6, graph half)

The Higgs Audio V2 decode path, transcribed from the pinned transformers implementation (`transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py` + `models/dac/modeling_dac.py`, both installed in the oracle env): `decode(codes)` = RVQ sum (per level: `F.embedding` then a biased `project_out` Linear) → `fc2` (concat 1024 → acoustic 256) → the DAC decoder with the two Higgs adjustments (`output_padding = stride % 2` on every ConvTranspose1d; the final `Tanh` removed). Unlike qwen3-tts's codec, **nothing here is causal**: DAC pads symmetrically (`conv k7 pad 3`, res-unit `conv k7 dilation d pad 3d`, `conv_t k=2s pad ⌈s/2⌉`), and the Snake is alpha-only with a `+1e-9` guard (`x + sin²(αx)/(α+1e-9)`, alpha stored linearly `[1, C, 1]`) — kokoro's helper pattern plus the guard.

**Files:**
- Create: `src/arch/omnivoice/codec.h`, `src/arch/omnivoice/codec.cpp`
- Create: `src/arch/omnivoice/codec-host.h`, `src/arch/omnivoice/codec-host.cpp`
- Create: `scripts/dump_reference_omnivoice_codec.py`
- Create: `tests/omnivoice_codec_test.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Produces (namespace `synth::omnivoice`), consumed by Task 12:
  - `ggml_tensor * codec_snake(ggml_context *, ggml_tensor * input, const SnakeWeights &);`
  - `ggml_tensor * codec_conv1d(ggml_context *, ggml_tensor * input, const Conv1dWeights &, int dilation, int padding);`
  - `ggml_tensor * codec_transpose_conv1d(ggml_context *, ggml_tensor * input, const Conv1dWeights &, int stride, int padding, int output_padding);`
  - `ggml_tensor * codec_rvq_decode(ggml_context *, const std::vector<RvqQuantizerWeights> &, ggml_tensor * codes);`
  - `ggml_tensor * build_codec_decoder(ggml_context *, ggml_tensor * codes, const ModelWeights &, const HParams &);` — codes I32 `[frames, num_codebooks]` level-major rows; returns the raw 1-D waveform `[frames * hop]`
  - `codec-host.h`: `synth_status_t validate_code_grid(const std::vector<int32_t> &, uint64_t frame_count, uint32_t num_codebooks, uint32_t codebook_size);` and `void apply_no_reference_volume(std::vector<float> & audio);`

- [ ] **Step 1: Write `codec.h`**

```cpp
#pragma once

// catalog.h carries the weight structs and weights.h the HParams the decoder's
// geometry reads -- the post-Task-4 roles.
#include "catalog.h"
#include "weights.h"

#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace synth::omnivoice {

// The Higgs Audio V2 decode path. Layout is channel-major [channels, length]
// throughout, the qwen3-tts codec convention. NOTHING here is causal: DAC pads
// symmetrically, so there is no keep-prefix crop anywhere -- the recurring
// hazard of the qwen3 codec does not exist in this one, and porting its crops
// here would shift the waveform in time and fail nothing.

// Plain Snake, alpha only: x + sin^2(alpha * x) / (alpha + 1e-9). Alpha is
// stored linearly in a [1, channels, 1] tensor (not as a logarithm -- the
// qwen3 SnakeBeta contrast), and the 1e-9 guard is the reference's own
// (transformers Snake1d), part of the function rather than a courtesy.
ggml_tensor * codec_snake(ggml_context * context, ggml_tensor * input, const SnakeWeights & weights);

// Symmetric-padding convolution via im2col + matmul (ggml_conv_1d's CPU path
// asserts an F16 kernel and this family's are F32). With DAC's odd kernels and
// pad = dilation * (kernel - 1) / 2 the output length equals the input length.
ggml_tensor * codec_conv1d(ggml_context *        context,
                           ggml_tensor *         input,
                           const Conv1dWeights & weights,
                           int                   dilation,
                           int                   padding);

// Transposed convolution via mul_mat + ggml_col2im_1d (the VITS recipe). Torch
// semantics: out = (L-1)*stride - 2*padding + kernel + output_padding. The
// scatter is computed unpadded and the torch padding becomes a view crop of
// `padding` rows from the left with `padding - output_padding` from the right,
// which is what makes Higgs's output_padding = stride % 2 representable
// without a new operator. Requires 0 <= output_padding <= padding.
ggml_tensor * codec_transpose_conv1d(ggml_context *        context,
                                     ggml_tensor *         input,
                                     const Conv1dWeights & weights,
                                     int                   stride,
                                     int                   padding,
                                     int                   output_padding);

// Residual dequantization: every level looks its codes up in its own codebook
// and projects them out through a biased Linear; the levels are summed in
// ascending order (float addition is not associative and the reference sums
// 0..7). `codes` is I32 [frames, num_quantizers], level-major rows.
ggml_tensor * codec_rvq_decode(ggml_context *                           context,
                               const std::vector<RvqQuantizerWeights> & quantizers,
                               ggml_tensor *                            codes);

// The whole decode: RVQ sum -> fc2 (concat width -> acoustic width) -> DAC
// conv1 -> per-ratio blocks (snake, transposed conv, three residual units at
// dilations 1/3/9) -> snake -> conv2 to mono. Higgs removes DAC's final tanh,
// so the raw convolution output IS the waveform: no clamp, no activation.
// Returns a 1-D [frames * hop] tensor.
ggml_tensor * build_codec_decoder(ggml_context *       context,
                                  ggml_tensor *        codes,
                                  const ModelWeights & weights,
                                  const HParams &      hparams);

}  // namespace synth::omnivoice
```

- [ ] **Step 2: Write `codec.cpp`**

```cpp
#include "arch/omnivoice/codec.h"

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/weights.h"
#include "ggml.h"

#include <cstddef>

namespace synth::omnivoice {

namespace {

// The DAC residual stack's fixed dilations, three units per block.
constexpr int kDacDilations[3] = { 1, 3, 9 };

bool bound(const Conv1dWeights & weights) {
    return weights.weight != nullptr && weights.bias != nullptr;
}

// A bias over [channels, length] is per channel, so it broadcasts along the
// length rather than across it.
ggml_tensor * add_channel_bias(ggml_context * context, ggml_tensor * signal, ggml_tensor * bias) {
    return ggml_add(context, signal, ggml_reshape_2d(context, bias, bias->ne[0], 1));
}

}  // namespace

ggml_tensor * codec_snake(ggml_context * context, ggml_tensor * input, const SnakeWeights & weights) {
    if (context == nullptr || input == nullptr || weights.alpha == nullptr) {
        return nullptr;
    }
    // Alpha is stored [1, channels, 1]; one value per channel.
    if (weights.alpha->ne[1] != input->ne[0]) {
        return nullptr;
    }
    ggml_tensor * per_channel = ggml_reshape_1d(context, weights.alpha, weights.alpha->ne[1]);
    ggml_tensor * scaled      = ggml_mul(context, input, per_channel);
    ggml_tensor * sine        = ggml_sin(context, scaled);
    ggml_tensor * squared     = ggml_mul(context, sine, sine);
    // The reference divides by alpha plus a fixed guard (Snake1d's 1e-9). It is
    // part of the function, not a numerical courtesy.
    ggml_tensor * guard = ggml_scale_bias(context, per_channel, 1.0f, 1e-9f);
    return ggml_add(context, input, ggml_div(context, squared, guard));
}

ggml_tensor * codec_conv1d(ggml_context *        context,
                           ggml_tensor *         input,
                           const Conv1dWeights & weights,
                           int                   dilation,
                           int                   padding) {
    if (context == nullptr || input == nullptr || !bound(weights) || dilation <= 0 || padding < 0) {
        return nullptr;
    }
    const int64_t kernel       = weights.weight->ne[0];
    const int64_t in_channels  = weights.weight->ne[1];
    const int64_t out_channels = weights.weight->ne[2];
    const int64_t length       = input->ne[1];
    if (kernel <= 0 || length <= 0 || in_channels != input->ne[0] || out_channels <= 0) {
        return nullptr;
    }
    ggml_tensor * time_major = ggml_cont(context, ggml_transpose(context, input));
    ggml_tensor * columns =
        ggml_im2col(context, weights.weight, time_major, 1, 0, padding, 0, dilation, 0, false, weights.weight->type);
    ggml_tensor * kernel_2d = ggml_reshape_2d(context, weights.weight, kernel * in_channels, out_channels);
    ggml_tensor * signal =
        ggml_mul_mat(context, kernel_2d, ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[1]));
    return add_channel_bias(context, signal, weights.bias);
}

ggml_tensor * codec_transpose_conv1d(ggml_context *        context,
                                     ggml_tensor *         input,
                                     const Conv1dWeights & weights,
                                     int                   stride,
                                     int                   padding,
                                     int                   output_padding) {
    if (context == nullptr || input == nullptr || !bound(weights) || stride <= 0 || padding < 0 ||
        output_padding < 0 || output_padding > padding) {
        return nullptr;
    }
    // ConvTranspose1d stores [in, out, kernel], so ggml reports [kernel, out, in].
    const int64_t kernel       = weights.weight->ne[0];
    const int64_t out_channels = weights.weight->ne[1];
    const int64_t in_channels  = weights.weight->ne[2];
    const int64_t length       = input->ne[1];
    if (kernel < stride || out_channels <= 0 || in_channels != input->ne[0] || length <= 0 ||
        weights.weight->ne[3] != 1) {
        return nullptr;
    }
    const int64_t out_length = (length - 1) * stride + kernel - 2 * int64_t(padding) + output_padding;
    if (out_length < 1) {
        return nullptr;
    }
    // Merging the kernel's leading pair yields exactly the out_channels-major,
    // kernel-minor column order col2im_1d scatters back; the transpose then
    // puts the input channels first so the matrix multiply reduces over them.
    ggml_tensor * columns_weight = ggml_reshape_2d(context, weights.weight, kernel * out_channels, in_channels);
    columns_weight               = ggml_cont(context, ggml_transpose(context, columns_weight));
    ggml_tensor * contiguous     = ggml_is_contiguous(input) ? input : ggml_cont(context, input);
    ggml_tensor * columns        = ggml_mul_mat(context, columns_weight, contiguous);
    // The FULL scatter, [(length-1)*stride + kernel, out_channels]; the torch
    // padding crops `padding` from the left and `padding - output_padding`
    // from the right of it.
    ggml_tensor * wide = ggml_col2im_1d(context, columns, stride, static_cast<int>(out_channels), 0);
    ggml_tensor * cropped =
        ggml_view_2d(context, wide, out_length, out_channels, wide->nb[1], size_t(padding) * wide->nb[0]);
    cropped = ggml_cont(context, cropped);
    cropped = ggml_add(context, cropped, ggml_reshape_2d(context, weights.bias, 1, out_channels));
    return ggml_cont(context, ggml_transpose(context, cropped));
}

ggml_tensor * codec_rvq_decode(ggml_context *                           context,
                               const std::vector<RvqQuantizerWeights> & quantizers,
                               ggml_tensor *                            codes) {
    if (context == nullptr || codes == nullptr || codes->type != GGML_TYPE_I32 || quantizers.empty()) {
        return nullptr;
    }
    if (codes->ne[1] != int64_t(quantizers.size()) || codes->ne[0] <= 0) {
        return nullptr;
    }
    ggml_tensor * total = nullptr;
    for (size_t level = 0; level < quantizers.size(); ++level) {
        const RvqQuantizerWeights & quantizer = quantizers[level];
        if (quantizer.codebook == nullptr || quantizer.output_proj.weight == nullptr ||
            quantizer.output_proj.bias == nullptr) {
            return nullptr;
        }
        ggml_tensor * ids  = ggml_view_1d(context, codes, codes->ne[0], size_t(level) * codes->nb[1]);
        ggml_tensor * rows = ggml_get_rows(context, quantizer.codebook, ids);
        // Unlike most RVQ ports the projection is a biased Linear, not a
        // kernel-one convolution.
        ggml_tensor * projected = ggml_mul_mat(context, quantizer.output_proj.weight, rows);
        projected               = add_channel_bias(context, projected, quantizer.output_proj.bias);
        total                   = total == nullptr ? projected : ggml_add(context, total, projected);
    }
    return total;
}

ggml_tensor * build_codec_decoder(ggml_context *       context,
                                  ggml_tensor *        codes,
                                  const ModelWeights & weights,
                                  const HParams &      hparams) {
    if (context == nullptr || codes == nullptr || codes->type != GGML_TYPE_I32 ||
        codes->ne[1] != int64_t(hparams.audio.num_codebooks) || codes->ne[0] <= 0 ||
        weights.fc2.weight == nullptr || weights.fc2.bias == nullptr) {
        return nullptr;
    }
    const int64_t frames = codes->ne[0];

    ggml_tensor * latent = codec_rvq_decode(context, weights.quantizers, codes);
    if (latent == nullptr) {
        return nullptr;
    }
    // fc2 brings the dequantized concat-width latent down to the acoustic
    // width the DAC decoder consumes.
    ggml_tensor * acoustic = ggml_mul_mat(context, weights.fc2.weight, latent);
    acoustic               = add_channel_bias(context, acoustic, weights.fc2.bias);

    const AcousticDecoderWeights & decoder = weights.acoustic_decoder;
    ggml_tensor * hidden = codec_conv1d(context, acoustic, decoder.conv1, 1, 3);  // kernel 7, pad 3
    if (hidden == nullptr) {
        return nullptr;
    }
    for (size_t block_index = 0; block_index < decoder.blocks.size(); ++block_index) {
        const AcousticDecoderBlock & block  = decoder.blocks[block_index];
        const int                    stride = int(hparams.codec.upsampling_ratios[block_index]);
        hidden                              = codec_snake(context, hidden, block.snake1);
        // kernel 2*stride, padding ceil(stride/2), and Higgs's adjustment:
        // output_padding = stride % 2, which keeps every stage at exactly
        // length * stride.
        hidden = codec_transpose_conv1d(context, hidden, block.conv_t1, stride, (stride + 1) / 2, stride % 2);
        if (hidden == nullptr) {
            return nullptr;
        }
        for (size_t unit_index = 0; unit_index < block.res_units.size(); ++unit_index) {
            const DacResidualUnit & unit     = block.res_units[unit_index];
            const int               dilation = kDacDilations[unit_index];
            ggml_tensor * branch             = codec_snake(context, hidden, unit.snake1);
            branch = codec_conv1d(context, branch, unit.conv1, dilation, 3 * dilation);  // kernel 7
            branch = codec_snake(context, branch, unit.snake2);
            branch = codec_conv1d(context, branch, unit.conv2, 1, 0);  // kernel 1
            if (branch == nullptr) {
                return nullptr;
            }
            hidden = ggml_add(context, hidden, branch);
        }
    }
    hidden = codec_snake(context, hidden, decoder.snake1);
    ggml_tensor * wave = codec_conv1d(context, hidden, decoder.conv2, 1, 3);  // kernel 7 -> mono
    if (wave == nullptr || wave->ne[0] != 1 || wave->ne[1] != frames * int64_t(hparams.codec.hop_length)) {
        return nullptr;
    }
    // Higgs replaces DAC's final tanh with identity: the raw output is the
    // waveform. No clamp -- qwen3-tts clamps because ITS reference does.
    return ggml_reshape_1d(context, ggml_cont(context, ggml_transpose(context, wave)), wave->ne[1]);
}

}  // namespace synth::omnivoice
```

(Note the final reshape: `wave` is channel-major `[1, samples]`; the transpose+cont gives `[samples, 1]` contiguous, reshaped to 1-D. If `wave` is already contiguous with ne0 == 1, `ggml_reshape_1d(context, wave, wave->ne[1])` is equivalent — prefer whichever the shape checker accepts, and keep the sample-count assertion.)

- [ ] **Step 3: Write `codec-host.{h,cpp}`**

`codec-host.h`:

```cpp
#pragma once

#include "synthesize.h"

#include <cstdint>
#include <vector>

namespace synth::omnivoice {

// Host-side guards around the codec, per the discrete-outputs rule: the grid
// is discrete, so its validation never rides the graph.

// Every value must be a real code in [0, codebook_size): the mask id or
// anything past the table means the loop (or a replay input) is broken, and
// get_rows would read a row that exists but means nothing.
synth_status_t validate_code_grid(const std::vector<int32_t> & codes,
                                  uint64_t                     frame_count,
                                  uint32_t                     num_codebooks,
                                  uint32_t                     codebook_size);

// The ungated no-reference volume branch measured at intake: when the peak
// exceeds 1e-6, every sample becomes sample / peak * 0.5 -- two float
// operations per element, numpy's order, so the oracle's bytes reproduce.
// Applies to auto-voice and voice-design output; the clone branches are
// Plan 3's.
void apply_no_reference_volume(std::vector<float> & audio);

}  // namespace synth::omnivoice
```

`codec-host.cpp`:

```cpp
#include "arch/omnivoice/codec-host.h"

#include <cmath>

namespace synth::omnivoice {

synth_status_t validate_code_grid(const std::vector<int32_t> & codes,
                                  uint64_t                     frame_count,
                                  uint32_t                     num_codebooks,
                                  uint32_t                     codebook_size) {
    if (frame_count == 0 || num_codebooks == 0 ||
        codes.size() != size_t(num_codebooks) * frame_count) {
        return SYNTH_ERR_INVALID_ARG;
    }
    for (int32_t code : codes) {
        if (code < 0 || uint32_t(code) >= codebook_size) {
            return SYNTH_ERR_INVALID_ARG;
        }
    }
    return SYNTH_OK;
}

void apply_no_reference_volume(std::vector<float> & audio) {
    float peak = 0.0f;
    for (float value : audio) {
        peak = std::fmax(peak, std::fabs(value));
    }
    if (peak <= 1e-6f) {
        return;
    }
    for (float & value : audio) {
        value = value / peak * 0.5f;
    }
}

}  // namespace synth::omnivoice
```

Add `arch/omnivoice/codec.cpp` and `arch/omnivoice/codec-host.cpp` to `src/CMakeLists.txt`.

- [ ] **Step 4: Write the reference script**

`scripts/dump_reference_omnivoice_codec.py` — the real transformers `DacDecoder` with the real Higgs adjustment applied, plus the RVQ/fc2 arithmetic composed from the same primitives the reference classes use (`F.embedding`, `F.linear`), at toy dimensions: concat 4, acoustic hidden 4, decoder hidden 8, ratios [2, 3] (hop 6), 2 quantizers, codebook_dim 3, codebook_size 5, T = 4 frames → 24 samples. Fill order is the contract; the C++ fixture mirrors it line for line.

```python
#!/usr/bin/env python3
"""Reference values for tests/omnivoice_codec_test.cpp.

Composes the Higgs decode path at toy dimensions from the REAL classes: the
transformers DacDecoder with HiggsAudioV2TokenizerModel._adjust_dac_decoder
applied (output_padding = stride % 2, tanh removed), and the RVQ/fc2 stages as
the exact F.embedding / F.linear arithmetic the reference modules perform.

Usage:
    uv run --project scripts/envs/omnivoice --locked python \
        scripts/dump_reference_omnivoice_codec.py
"""
import torch
import torch.nn.functional as F

SEED = 20260732
QUANTIZERS = 2
CODEBOOK_SIZE = 5
CODEBOOK_DIM = 3
CONCAT = 4          # RVQ width = fc2 input
ACOUSTIC = 4        # fc2 output = decoder input channels
DECODER_HIDDEN = 8
RATIOS = [2, 3]     # toy hop = 6
FRAMES = 4
CODES = [[0, 3, 1, 4], [2, 2, 0, 1]]  # level-major [QUANTIZERS][FRAMES]


class LcgStream:
    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.state >> 40) / 8388608.0 - 1.0

    def fill(self, count, scale, offset):
        return torch.tensor([self.next() * scale + offset for _ in range(count)],
                            dtype=torch.float32)


def dump(name, tensor):
    flat = tensor.reshape(-1).tolist()
    print(f"constexpr float {name}[] = {{")
    for start in range(0, len(flat), 4):
        row = ", ".join(f"{value:.9g}f" for value in flat[start:start + 4])
        print(f"    {row},")
    print("};")


def main():
    from transformers.models.dac.configuration_dac import DacConfig
    from transformers.models.dac.modeling_dac import DacDecoder
    from transformers.models.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer import (
        HiggsAudioV2TokenizerModel,
    )

    config = DacConfig(hidden_size=ACOUSTIC, decoder_hidden_size=DECODER_HIDDEN,
                       upsampling_ratios=RATIOS)
    decoder = DacDecoder(config).eval()
    HiggsAudioV2TokenizerModel._adjust_dac_decoder(decoder)

    stream = LcgStream(SEED)
    # RVQ + fc2 weights first, then the decoder's, module by module in forward
    # order. Weights {0.25, 0}, biases {0.1, 0}, snake alphas {0.25, 1}.
    codebooks, out_weights, out_biases = [], [], []
    for _ in range(QUANTIZERS):
        codebooks.append(stream.fill(CODEBOOK_SIZE * CODEBOOK_DIM, 0.25, 0.0)
                         .view(CODEBOOK_SIZE, CODEBOOK_DIM))
        out_weights.append(stream.fill(CONCAT * CODEBOOK_DIM, 0.25, 0.0).view(CONCAT, CODEBOOK_DIM))
        out_biases.append(stream.fill(CONCAT, 0.1, 0.0))
    fc2_weight = stream.fill(ACOUSTIC * CONCAT, 0.25, 0.0).view(ACOUSTIC, CONCAT)
    fc2_bias = stream.fill(ACOUSTIC, 0.1, 0.0)

    assignments = [(decoder.conv1.weight, 0.25, 0.0), (decoder.conv1.bias, 0.1, 0.0)]
    for block in decoder.block:
        assignments += [(block.snake1.alpha, 0.25, 1.0),
                        (block.conv_t1.weight, 0.25, 0.0), (block.conv_t1.bias, 0.1, 0.0)]
        for unit in (block.res_unit1, block.res_unit2, block.res_unit3):
            assignments += [(unit.snake1.alpha, 0.25, 1.0),
                            (unit.conv1.weight, 0.25, 0.0), (unit.conv1.bias, 0.1, 0.0),
                            (unit.snake2.alpha, 0.25, 1.0),
                            (unit.conv2.weight, 0.25, 0.0), (unit.conv2.bias, 0.1, 0.0)]
    assignments += [(decoder.snake1.alpha, 0.25, 1.0),
                    (decoder.conv2.weight, 0.25, 0.0), (decoder.conv2.bias, 0.1, 0.0)]
    with torch.no_grad():
        for parameter, scale, offset in assignments:
            parameter.copy_(stream.fill(parameter.numel(), scale, offset).view_as(parameter))

    codes = torch.tensor(CODES)
    latent = torch.zeros(CONCAT, FRAMES)
    for level in range(QUANTIZERS):
        rows = F.embedding(codes[level], codebooks[level])          # [FRAMES, DIM]
        latent = latent + F.linear(rows, out_weights[level], out_biases[level]).T
    acoustic = F.linear(latent.T, fc2_weight, fc2_bias).T           # [ACOUSTIC, FRAMES]
    with torch.no_grad():
        wave = decoder(acoustic.unsqueeze(0))[0, 0]                 # [FRAMES * 6]

    # Dump in the C++ read-back order: channel-major tensors flatten
    # frame-major with channels contiguous per frame -> transpose first.
    dump("kExpectedLatent", latent.T)
    dump("kExpectedAcoustic", acoustic.T)
    dump("kExpectedWave", wave)


if __name__ == "__main__":
    main()
```

Run it; assert the wave has exactly 24 values and everything is finite. `_adjust_dac_decoder` is a `@staticmethod`, callable without instantiating the Higgs model.

```bash
uv run --project scripts/envs/omnivoice --locked python scripts/dump_reference_omnivoice_codec.py > /tmp/omnivoice_codec_expected.txt
grep -c "constexpr" /tmp/omnivoice_codec_expected.txt   # expect 3
```

- [ ] **Step 5: Write the failing unit test**

`tests/omnivoice_codec_test.cpp` — the same technique as Task 7's test (LcgStream verbatim, persistent fixture context, `compute()` with `ggml_set_output` on intermediates, device sweep, 1e-4f CPU / 5e-3f accelerators). Fixture: build the toy `HParams` (audio.num_codebooks 2, vocab 6, mask 5; codec hidden_size 4, decoder_hidden 8, ratios {2,3}, hop 6, codebook_dim 3, codebook_size 5; semantic.hidden_size 0 — unused here) and a partial `ModelWeights` holding only quantizers/fc2/acoustic_decoder, with tensors created and filled in EXACTLY the script's order and scales (weights {0.25, 0}, biases {0.1, 0}, snake alphas {0.25, 1}); shapes are the torch shapes reversed (conv `[k, in, out]`, transposed conv `[k, out, in]`, snake `[1, C, 1]`, Linear `[in, out]`). Codes tensor I32 `[4, 2]` filled level-major from `CODES`. Checks:

- `check_rvq` — `codec_rvq_decode` returns `[4, 4]` matching `kExpectedLatent`.
- `check_decode` — `build_codec_decoder` returns 1-D `[24]`; read back the RVQ and fc2 intermediates too (`ggml_set_output` on them) and match `kExpectedLatent` / `kExpectedAcoustic` / `kExpectedWave`. This closes the two Higgs adjustments: with output_padding wrong, stride-3's stage produces 23 or 25 samples and the builder's length assertion already fails; with a tanh/clamp left in, the wave values diverge.
- `check_rejections` — codes typed F32 → nullptr; codes `ne[1]` ≠ quantizer count → nullptr; `output_padding > padding` → nullptr from `codec_transpose_conv1d`; a null snake alpha → nullptr.
- `check_host_guards` (no ggml) — `validate_code_grid`: accepts the fixture codes; rejects a 1024-style out-of-range value (use 5 with codebook_size 5), a negative, and a wrong element count. `apply_no_reference_volume`: `{0.2f, -0.4f}` → `{0.25f, -0.5f}` exactly; an all-zero vector is untouched.

Register `synth_add_unit_test(synthesize-omnivoice-codec-test omnivoice_codec_test.cpp)`. RED first (`cmake --build build --target synthesize-omnivoice-codec-test` fails until the sources land), then green.

- [ ] **Step 6: Green, sanitize, format, commit**

```bash
cmake --build build --target synthesize-check-unit
ctest --test-dir build --output-on-failure -R synthesize-omnivoice-codec-test
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add src/arch/omnivoice/codec.h src/arch/omnivoice/codec.cpp \
        src/arch/omnivoice/codec-host.h src/arch/omnivoice/codec-host.cpp src/CMakeLists.txt \
        scripts/dump_reference_omnivoice_codec.py tests/omnivoice_codec_test.cpp tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
Build the Higgs Audio V2 decode graph for omnivoice

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 12: `decode_codes` + the volume branch — end-to-end greedy waveform (slice 6 gate)

Wire the codec into the model: `decode_codes` (raw waveform, the qwen3-tts one-shot pattern), the no-reference volume branch inside `run_synthesis` (decision 3 of this plan), and the runner's decode flag lights up. The gate: on all 20 cases, the oracle's grid replayed through this codec reproduces `audio/pcm.f32` at F32-vs-F32 tightness, and the 17 greedy grids stay exact.

**Files:**
- Modify: `src/arch/omnivoice/model.cpp` (decode_codes body; run_synthesis tail; includes)
- Modify: `tests/omnivoice_replay_real.cpp` (use the shared volume helper)

**Interfaces:**
- Consumes: `build_codec_decoder`, `validate_code_grid`, `apply_no_reference_volume` (Task 11).
- Produces: `Model::decode_codes` for the replay seam; `run_synthesis` output gains `audio` (scaled) beside `codes`.

- [ ] **Step 1: Replace the `decode_codes` placeholder**

In `model.cpp` (add `#include "arch/omnivoice/codec.h"` and `#include "arch/omnivoice/codec-host.h"`):

```cpp
synth_status_t Model::decode_codes(const std::vector<int32_t> & codes, uint64_t frame_count, int threads,
                                   std::vector<float> & audio) {
    audio.clear();
    Impl &          impl    = *implementation_;
    const HParams & hparams = impl.hparams;
    const uint32_t  groups  = hparams.audio.num_codebooks;

    synth_status_t status = validate_code_grid(codes, frame_count, groups, hparams.codec.codebook_size);
    if (status != SYNTH_OK) {
        return status;
    }

    Persistent inputs;
    if (!inputs.open(2)) {
        return SYNTH_ERR_OOM;
    }
    ggml_tensor * t_codes =
        ggml_new_tensor_2d(inputs.context(), GGML_TYPE_I32, int64_t(frame_count), int64_t(groups));
    if (!inputs.commit(impl.backend_plan->cpu_backend())) {
        return SYNTH_ERR_OOM;
    }
    // The committed grid is codebook-major [c * frames + t], which IS the
    // level-major row layout the quantizer's per-level views read -- no
    // transpose, unlike qwen3-tts's frame-major stream.
    ggml_backend_tensor_set(t_codes, codes.data(), 0, ggml_nbytes(t_codes));

    // One pass over the whole stream; the node count does not grow with the
    // frame count (the qwen3-tts codec budget).
    GraphRun run(*impl.backend_plan, 8192);
    if (!run.ok()) {
        return SYNTH_ERR_OOM;
    }
    ggml_tensor * wave = build_codec_decoder(run.context(), t_codes, impl.weights, hparams);
    if (wave == nullptr) {
        return SYNTH_ERR_INTERNAL;
    }
    const double started = now_seconds();
    status = run.run(wave, "omnivoice.codec", threads > 0 ? threads : default_synthesis_threads());
    if (status != SYNTH_OK) {
        return status;
    }
    codec_seconds_           = now_seconds() - started;
    codec_placed_nodes_      = run.placed_nodes;
    codec_accelerator_nodes_ = run.accelerator_nodes;
    read_floats(wave, audio);
    if (audio.size() != size_t(frame_count) * hparams.codec.hop_length) {
        std::fprintf(stderr, "omnivoice: the codec produced %zu samples for %llu frames\n", audio.size(),
                     (unsigned long long) frame_count);
        return SYNTH_ERR_INTERNAL;
    }
    return SYNTH_OK;
}
```

The thread policy is the one expression inside `run.run` (`threads > 0 ? threads : default_synthesis_threads()`). The `codec_seconds_` scratch follows the qwen3-tts precedent exactly: declare in the anonymous namespace,

```cpp
// The replay seam calls decode_codes free-standing, so its timing and
// placement land in thread_local scratch and run_synthesis copies them into
// its output -- the qwen3-tts pattern verbatim.
thread_local double   codec_seconds_           = 0.0;
thread_local uint64_t codec_placed_nodes_      = 0;
thread_local uint64_t codec_accelerator_nodes_ = 0;
```

and have the runner's decode path pick them up implicitly through `run_synthesis` only; the free-standing runner call reports codec placement via its own JSON (`output.codec_placement` stays zero there — the runner prints the placement it got from the greedy path when it ran, and the validator's CPU-only rule tolerates zeros).

- [ ] **Step 2: Attach decode + volume to `run_synthesis`**

Replace Task 10's closing comment (`// Task 12 attaches the codec decode …`) with:

```cpp
    status = decode_codes(output.codes, frames, threads, output.audio);
    if (status != SYNTH_OK) {
        return status;
    }
    output.codec_seconds            = codec_seconds_;
    output.codec_placement.nodes    = codec_placed_nodes_;
    output.codec_placement.accelerator_nodes = codec_accelerator_nodes_;
    // Decision 3 of the plan: the residual output scaling lives HERE, inside
    // the family's synthesis path, faithful to the oracle. Auto-voice and
    // voice-design requests take the no-reference branch; the clone branches
    // arrive with Plan 3's reference handling.
    apply_no_reference_volume(output.audio);
    return SYNTH_OK;
```

- [ ] **Step 3: One formula, one home — the runner's volume path**

In `tests/omnivoice_replay_real.cpp`, add `#include "arch/omnivoice/codec-host.h"` and replace the inline peak-normalization block inside `if (ok && decode)` with:

```cpp
        if (volume == "peak") {
            synth::omnivoice::apply_no_reference_volume(audio);
        }
```

- [ ] **Step 4: GREEN — the slice-6 gate**

```bash
cmake --build build --target synthesize-check-unit
uv run --project scripts/envs/omnivoice --locked python scripts/validate-omnivoice-replay.py \
  --model models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  --runner build/unit/bin/synthesize-omnivoice-replay-real \
  --require all --cases omni-short-en omni-clone-en omni-sampled-seed-zero
```

Expected: three cases pass; `audio.pcm` appears in the worst-table with min_cosine ≥ 0.999 and small max_abs (F32 against F32: expect ≥ 0.9999); `token grids exact: 2/2` (the sampled case free-runs nothing). If the waveform cosine is high but max_abs is a constant factor, a volume branch is applied where it should not be (or vice versa — check the case's `volume_branch` in the oracle's result.json against the validator's mapping). If the waveform is noise at cosine ≈ 0, the grid orientation trap fired inside decode — recheck the level-major comment in Step 1. Then the full sweep:

```bash
uv run --project scripts/envs/omnivoice --locked python scripts/validate-omnivoice-replay.py \
  --model models/omnivoice-0-6b/omnivoice-0-6b-F32.gguf \
  --runner build/unit/bin/synthesize-omnivoice-replay-real \
  --require all --report build/goldens/omnivoice-replay/full-report.json
```

Expected: 20/20 cases ok, 17/17 exact, all placement on CPU, exit 0. Listen once (`ffplay -f f32le -ar 24000 -ac 1 build/goldens/omnivoice-replay/omni-short-en/pcm.f32`): intelligible speech, no clicks. Record the worst-table and the wall clocks in `_porting-log.md` (dated "slice 6 — end-to-end greedy waveform").

- [ ] **Step 5: Sanitize, format, commit**

```bash
cmake --build build-sanitize --target synthesize-check-unit
scripts/ci/clang-format.sh --fix
git add src/arch/omnivoice/model.cpp tests/omnivoice_replay_real.cpp \
        reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md
git commit -m "$(cat <<'EOF'
Decode omnivoice grids to waveforms and close the greedy end-to-end path

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 13: Harness hardening — the runner/validator debt batch

Carry-over items 9 and 11 (the parts not already true). Verified during planning: the probe-layers-vs-manifest assertion and stale-artifact clearing already exist in the dumper (`load_manifest`'s probe check; `write_case` clears the case dir) — confirm, don't re-implement. What remains: clone digests must resolve in `load_manifest` (a `ManifestError` currently escapes mid-dump), artifact `format` must agree with the writer that produces it, the tokenizer dumper must assert `preprocess_prompt=false` on clone cases, `urlopen` needs a timeout and a bad-cache hint, and the tolerance/manifest `case_count` cross-check lands in `test_golden_manifests.py` (which also exposes the qwen3-tts file's 18-vs-20 inconsistency).

**Files:**
- Modify: `scripts/dump_reference_omnivoice_pytorch.py`
- Modify: `scripts/dump_reference_omnivoice_tokenizer.py`
- Modify: `tests/python/test_golden_manifests.py`
- Modify: `tests/tolerances/qwen3-tts.json` (case_count 18 → 20)

**Interfaces:**
- Produces: the cross-check every future family's tolerance file must satisfy; the dumper's malformed-manifest failures all become `error: …` + exit 1 before any model loads.

- [ ] **Step 1: RED — the case_count cross-check**

Add to `tests/python/test_golden_manifests.py`, following the file's existing manifest-iteration pattern (mirror `test_tolerance_file_is_committed`'s loop and `subTest` style):

```python
    def test_tolerance_case_count_matches_manifest(self):
        """A tolerance file describing N cases must mean the manifest's N.

        The qwen3-tts file said 18 while its manifest had grown to 20 -- an
        honest historical number that read as a current claim. case_count is
        bookkeeping about the suite, so it tracks the suite.
        """
        for path in self.manifest_paths():
            with self.subTest(manifest=path.name):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                tolerance_path = REPO_ROOT / manifest["tolerance_file"]
                tolerance = json.loads(tolerance_path.read_text(encoding="utf-8"))
                if "case_count" not in tolerance:
                    continue
                self.assertEqual(
                    tolerance["case_count"], len(manifest["cases"]),
                    f"{manifest['tolerance_file']}: case_count disagrees with {path.name}")
```

(Use the module's actual helper for enumerating manifests — if there is no `manifest_paths()`, inline the same glob `test_tolerance_file_is_committed` uses.) Run:

```bash
ctest --test-dir build --output-on-failure -R synthesize-vits-python-unit
```

Expected: FAIL — `tests/tolerances/qwen3-tts.json: case_count disagrees` (18 vs 20).

- [ ] **Step 2: Fix the qwen3-tts bookkeeping**

In `tests/tolerances/qwen3-tts.json`, change the top-level `"case_count": 18` to `20`, and append one sentence to the top-level `note`: `"\n\ncase_count corrected to 20 on 2026-07-31 when the manifest cross-check landed; the per-stage 'cases' fields record each sweep's size at measurement time and are deliberately unchanged."` The per-stage `"cases"` entries are historical sweep records — do NOT touch them or any threshold. Re-run Step 1's command: PASS.

- [ ] **Step 3: Dumper hardening (items 9's live parts)**

In `scripts/dump_reference_omnivoice_pytorch.py`:

1. **Clone digests resolve at load time.** In `load_manifest`'s per-case loop, where `reference_input` is validated, add:

   ```python
        if reference_input is not None:
            # Resolving the digest HERE turns an unpinned reference into an
            # `error:` + exit 1 before any model loads, instead of a
            # ManifestError escaping mid-dump with three cases already written.
            reference_digest(manifest, reference_input["artifact"])
            if parameters.get("preprocess_prompt") is not False:
                raise ManifestError(
                    f"{where}: a clone case must pin preprocess_prompt=false; anything else "
                    "lets silence stripping into a parity baseline"
                )
   ```

   (`reference_digest` already raises `ManifestError` for a missing entry or digest; move its definition above `load_manifest` if ordering demands.)
2. **Writer/format agreement.** In `write_case`, before the writing loop:

   ```python
    writer_formats = {write_i32: "i32le", write_f32: "f32le", write_json: "json"}
    for name, (writer, _payload) in produced.items():
        declared = expected[name]["format"]
        if writer_formats[writer] != declared:
            raise SystemExit(
                f"{case_id}: {name} is declared {declared!r} but the dump would write "
                f"{writer_formats[writer]!r}; the manifest and the writer registry drifted"
            )
   ```
3. **Timeout + bad-cache hint.** In `materialise_reference`: `urllib.request.urlopen(locator, timeout=60)`, and extend the digest-mismatch `SystemExit` message with: `" If a stale cached file is the cause, delete it and re-run to re-fetch."`

In `scripts/dump_reference_omnivoice_tokenizer.py`: where the script iterates manifest cases, add the same clone assertion (adapted to its error style — it exits via `SystemExit` on pinned-expectation mismatches):

```python
        reference_input = case["input"].get("reference")
        if reference_input is not None and case["oracle"]["parameters"].get("preprocess_prompt") is not False:
            raise SystemExit(
                f"{case['id']}: a clone case must pin preprocess_prompt=false; the tokenizer "
                "cases would otherwise describe a prompt the oracle never builds"
            )
```

- [ ] **Step 4: GREEN**

```bash
python3 scripts/dump_reference_omnivoice_pytorch.py \
  --manifest tests/golden/omnivoice/omnivoice-0-6b.manifest.json --validate-only
uv run --project scripts/envs/omnivoice --locked python scripts/dump_reference_omnivoice_tokenizer.py \
  --weights-dir models/omnivoice-0-6b --output /tmp/tokenizer-cases-check.json
ctest --test-dir build --output-on-failure -R 'synthesize-(vits|omnivoice)-python-unit|synthesize-golden-manifest-contract'
```

Expected: validate-only OK (the committed manifest already pins `preprocess_prompt: false` on both clone cases — this is a tripwire, not a change); the tokenizer dumper completes and its output matches the committed expectations; all three ctest entries PASS. Also prove the new load_manifest guard fires: run the pytorch dumper's `--validate-only` against a copy of the manifest with one clone case's `preprocess_prompt` deleted — expect `error: … must pin preprocess_prompt=false`, exit 1.

- [ ] **Step 5: Commit**

```bash
git add scripts/dump_reference_omnivoice_pytorch.py scripts/dump_reference_omnivoice_tokenizer.py \
        tests/python/test_golden_manifests.py tests/tolerances/qwen3-tts.json
git commit -m "$(cat <<'EOF'
Harden the omnivoice harness edges and cross-check tolerance case counts

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

### Task 14: Measure and commit the first tolerances; register the golden gate; close the docs

The Port Validation discipline verbatim: "Tolerance values are measured from the first working reference and implementation, reviewed, and then committed before support is declared." Task 12's full-sweep report IS the first working pair's measurement; this task turns it into the committed grid, registers the CTest gate that enforces it, and writes the family record.

**Files:**
- Modify: `tests/tolerances/omnivoice.json` (pending skeleton → committed grid)
- Modify: `tests/CMakeLists.txt` (the golden gate)
- Modify: `docs/porting/families/omnivoice.md`, `reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md`

**Interfaces:**
- Produces: the `profiles.F32.stages.replay` cell `validate-omnivoice-replay.py --check` gates on; CTest `synthesize-omnivoice-replay-golden`.

- [ ] **Step 1: Derive the thresholds from the measurement**

From `build/goldens/omnivoice-replay/full-report.json`'s `worst` block (regenerate via Task 12 Step 4's sweep command if stale), compute each probe's threshold as **five times the measured deviation**: `min_cosine = 1 − 5·(1 − observed_min_cosine)` (round DOWN to 6 significant decimals — rounding up would tighten past the rule), and for `audio.pcm` additionally `max_abs = 5 × observed_max_abs` rounded UP to one significant figure with a floor of `1e-5`. Helper to print the grid (values then reviewed by hand, not pasted blind):

```bash
uv run --project scripts/envs/vits --locked python - <<'EOF'
import json, math
worst = json.load(open("build/goldens/omnivoice-replay/full-report.json"))["worst"]
for name in sorted(worst):
    observed = worst[name]
    floor = math.floor((1 - 5 * (1 - observed["min_cosine"])) * 1e6) / 1e6
    line = f'"{name}": {{ "min_cosine": {floor}, ' \
           f'"observed_min_cosine": {observed["min_cosine"]}, ' \
           f'"observed_max_abs": {observed["max_abs"]}'
    if name == "audio.pcm":
        exponent = math.floor(math.log10(max(observed["max_abs"] * 5, 1e-5)))
        line += f', "max_abs": {math.ceil(observed["max_abs"] * 5 / 10**exponent) * 10**exponent}'
    print(line + " },")
EOF
```

- [ ] **Step 2: Rewrite `tests/tolerances/omnivoice.json`**

Replace the pending skeleton wholesale (delete the legacy empty top-level `"stages": {}` — with a `profiles` key present, `tests/python/test_tolerance_coverage.py` takes the populated path and the legacy key would be dead weight):

```json
{
  "schema": "synthesize-tolerances-v1",
  "family": "omnivoice",
  "variant": "omnivoice-0-6b",
  "suite_version": 1,
  "status": "thresholds-committed-and-enforced",
  "reference_stage": "source-f32-oracle-vs-f32-cpu",
  "case_count": 20,
  "note": "Measured 2026-07-31 from the first working reference and implementation across all twenty Golden cases, then reviewed and committed. A tolerance is an input to validation, not an output of it.\n\nThe oracle and the port both run F32 on CPU, so these thresholds carry neither a dtype nor a device difference -- only an implementation one -- and are accordingly tighter than qwen3-tts's. Every threshold is five times the measured deviation in (1 - cosine); the waveform additionally gates on max-abs, because a listener hears the waveform and a cosine over it would hide a constant offset.\n\nOne threshold this file will never carry: the token grid. structural_exactness for this family is EXACT equality of the committed 8 x T grid against the oracle's (17 greedy cases, measured 17/17 at commit time); the validator fails any inexact grid unconditionally, before and independent of --check.\n\nThe grid is keyed on Quantization Profile, Execution Backend and stage, the layout tests/python/test_tolerance_coverage.py checks; F32 on CPU is the only measured cell, because Plan 2 is CPU-only and quantization is expected to be HARDER here than in any shipped family (a reference port measured token agreement collapsing to ~7% under an F16 generator; every future profile re-passes the exact-token gate or is not shipped).",
  "profiles": {
    "F32": {
      "stages": {
        "replay": {
          "phase": "oracle_replay",
          "backend": "CPU",
          "cases": 20,
          "description": "The oracle's grids replayed through the codec and the step-0 conditional forward compared against its probes; the greedy free-run grids compared exactly, outside this file.",
          "probes": {
            "generator.hidden_l0": { "min_cosine": 0.0, "observed_min_cosine": 0.0, "observed_max_abs": 0.0 },
            "generator.hidden_l7": { "min_cosine": 0.0, "observed_min_cosine": 0.0, "observed_max_abs": 0.0 },
            "generator.hidden_l14": { "min_cosine": 0.0, "observed_min_cosine": 0.0, "observed_max_abs": 0.0 },
            "generator.hidden_l21": { "min_cosine": 0.0, "observed_min_cosine": 0.0, "observed_max_abs": 0.0 },
            "generator.hidden_l27": { "min_cosine": 0.0, "observed_min_cosine": 0.0, "observed_max_abs": 0.0 },
            "generator.final": { "min_cosine": 0.0, "observed_min_cosine": 0.0, "observed_max_abs": 0.0 },
            "generator.logits_step0": { "min_cosine": 0.0, "observed_min_cosine": 0.0, "observed_max_abs": 0.0 },
            "audio.pcm": { "min_cosine": 0.0, "max_abs": 0.0, "observed_min_cosine": 0.0, "observed_max_abs": 0.0 }
          }
        }
      }
    }
  }
}
```

**Every `0.0` above is a slot for Step 1's measured/derived value — filling them from the report is this step; committing any literal zero is a failure.** Then:

```bash
ctest --test-dir build --output-on-failure -R synthesize-vits-python-unit   # tolerance-coverage + manifest tests
```

Expected: PASS — with `profiles` populated, `test_tolerance_coverage`'s populated path requires the validator set `{replay}` (from `scripts/validate-omnivoice-*.py`) to equal the measured stage set `{replay}`: true, because Plan 2 ships exactly one validator. The `case_count` cross-check (Task 13) sees 20 == 20.

- [ ] **Step 3: Register the golden gate**

In `tests/CMakeLists.txt`, beside the qwen3-tts golden blocks (mirror their sentinel + env-run pattern):

```cmake
    # OmniVoice's replay validator runs with --check, so it is a gate against
    # the committed tolerances rather than a measurement -- and its exact-token
    # arm fails any greedy case whose grid is not the oracle's, tolerance file
    # or no. The golden sentinel is the oracle payload, deliberately not
    # committed.
    set(_synth_omnivoice_golden_sentinel
        "${CMAKE_SOURCE_DIR}/build/goldens/omnivoice/omni-short-en/codes/grid.i32")
    if(EXISTS "${SYNTH_OMNIVOICE_TEST_MODEL}" AND EXISTS "${_synth_omnivoice_golden_sentinel}")
        add_test(
            NAME synthesize-omnivoice-replay-golden
            COMMAND ${SYNTH_UV_EXECUTABLE} run
                --project ${CMAKE_SOURCE_DIR}/scripts/envs/omnivoice
                --locked python ${CMAKE_SOURCE_DIR}/scripts/validate-omnivoice-replay.py
                --manifest ${CMAKE_SOURCE_DIR}/tests/golden/omnivoice/omnivoice-0-6b.manifest.json
                --model ${SYNTH_OMNIVOICE_TEST_MODEL}
                --runner $<TARGET_FILE:synthesize-omnivoice-replay-real>
                --check --profile F32 --backend CPU --stage replay)
        set_tests_properties(synthesize-omnivoice-replay-golden PROPERTIES
            LABELS "integration;omnivoice;golden"
            TIMEOUT 14400
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()
```

(TIMEOUT 14400: the 17 greedy free-runs include two 471-position clone canvases at 64 forwards each; Task 10 measured the sweep in hours, not minutes.) Reconfigure and run the gate once end to end:

```bash
cmake -S . -B build -DSYNTH_BUILD_TESTS=ON -DSYNTH_BUILD_PYTHON_TESTS=ON -DSYNTH_BUILD_INTEGRATION_TESTS=ON
cmake --build build --target synthesize-check-unit synthesize-omnivoice-replay-real
ctest --test-dir build --output-on-failure -R synthesize-omnivoice-replay-golden
```

Expected: PASS, ending in `all probes within the F32/CPU/replay tolerances`.

- [ ] **Step 4: Close the docs**

`docs/porting/families/omnivoice.md`:

1. Status line → `Status: Confirmed 2026-07-31. Intake through the greedy synthesis core (slices 4–6) done: single-forward parity, exact token grids 17/17, replay waveform under committed tolerances (tests/tolerances/omnivoice.json). Public sampling, cloning, and the full validation suite have not started.`
2. Close the "Where the residual output scaling lives" Open Question — replace its paragraph with: `**Where the residual output scaling lives — decided 2026-07-31 (Plan 2).** Inside the family's synthesis path: Model::run_synthesis applies the no-reference peak-normalise-to-0.5 branch after codec decode, faithful to the oracle, and no public normalisation control exists in v1. The replay seam's decode_codes returns the raw waveform so validation can apply the oracle's branch per case. The quiet-reference and zero-reference branches are decided with cloning (Plan 3).`
3. Carry-over 15's greedy-relevant record lines — add to the Reference Contract section: `class_temperature's upstream default is 0.0: the public sampled path draws positions, not classes, unless a caller raises it.` Add to the Architecture section's voice-modes paragraph: `Auto-voice means the voice follows the synthesis seed: with no profile selected, nothing but the sampled path's draws picks the speaker.` Add to the Delivery/limits section: `The loader refuses a package without embedded generation defaults — the converter's "second copy" doctrine is enforced at load, not merely at conversion.`
4. Port Validation Fit section: append `Measured 2026-07-31: all 17 greedy cases reproduce the oracle's grid exactly on CPU F32; the replay-stage thresholds are committed in tests/tolerances/omnivoice.json with reference_stage source-f32-oracle-vs-f32-cpu.`

Append the closing section to `_porting-log.md` (dated; slices 4–6 evidence: worst-table, 17/17, gate registered and green, tolerance-derivation rule used).

- [ ] **Step 5: Final gates + commit**

```bash
cmake --build build --target synthesize-check-unit
cmake --build build-sanitize --target synthesize-check-unit
ctest --test-dir build --output-on-failure -L unit
scripts/ci/clang-format.sh --fix
git add tests/tolerances/omnivoice.json tests/CMakeLists.txt \
        docs/porting/families/omnivoice.md reports/porting/omnivoice/omnivoice-0-6b/_porting-log.md
git commit -m "$(cat <<'EOF'
Commit the first omnivoice replay tolerances and register the golden gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NFjueGrsKXs44rfZ7DunQK
EOF
)"
```

---

## Plan-Level Completion Checklist

- [ ] All 14 tasks committed on `omnivoice-plan-2`; `git log --oneline` reads as the 14 task commits (plus this plan's own commit).
- [ ] `cmake --build build --target synthesize-check-unit` and the sanitizer build both green; `ctest -L unit` green.
- [ ] `ctest -R synthesize-omnivoice-python-unit` green with the defaults test RUNNING (not skipped); `ctest -R synthesize-golden-manifest-contract` and `-R synthesize-vits-python-unit` green.
- [ ] `ctest -R synthesize-omnivoice-replay-golden` green: 17/17 exact token grids, every probe inside `tests/tolerances/omnivoice.json`, all nodes on CPU.
- [ ] The package GGUF still hashes `3ecaa5e2f6fbd735296ba1cd60680c90467be22d2140dc4f208fe80111ecb9e5` [superseded 2026-08-03 by `f6d504ffaddcbf32f80f1f6c847f075bbd5d2c7b50fe95a194ceb635772f9fa3`, a metadata-only re-cut; see the porting log's 2026-08-03 entry]; every oracle binary artifact byte-identical through the Task 2 re-dump (or the recorded supersession).
- [ ] `synth_synthesize`'s omnivoice branch still returns `synthesis.not_implemented` — the public seam was NOT opened.
- [ ] `_porting-log.md` narrates slices 4–6 with dates, worst-tables, and wall clocks; the family doc's status line, scaling decision, and item-15 record lines are in.
- [ ] Report to jiangzhuo with the evidence above; pushing the branch or opening any PR remains jiangzhuo's call.
