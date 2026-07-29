# qwen3-tts-12hz-0-6b-customvoice Porting Log

## 2026-07-27 — Intake and oracle smoke

- Pinned `QwenLM/Qwen3-TTS` at commit
  `022e286b98fbec7e1e916cb940cdf532cd9f488e` (package version 0.1.1, not on
  PyPI, so the environment pins it by git revision) and the weights repository
  `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` at revision
  `85e237c12c027371202489a0ec509ded67b5e4b5`.
- Downloaded 13 files totalling 2,498,388,392 bytes into the ignored local
  model cache: the 1,811,626,576-byte talker `model.safetensors`, the
  682,293,092-byte `speech_tokenizer/model.safetensors`, and the BPE vocabulary
  and configuration files. No training corpus was downloaded. Every file's
  SHA-256 is recorded in `intake.json` under `weights.files`.
- Locked the project-owned CPU reference environment in
  `scripts/envs/qwen3-tts/uv.lock` (Python 3.12.13, Torch 2.13.0,
  transformers 4.57.3, accelerate 1.12.0, numpy 1.26.4).
- Inventoried both checkpoints: 402 tensors and 905,788,672 parameters in the
  talker file, 496 tensors and 170,557,441 in the codec file.
- Ran the upstream reference on CPU in F32 with eager attention. It produces a
  finite 24 kHz waveform and is reproducible across processes under fully
  greedy decoding.

### The environment pin in the plan was internally inconsistent

The intake plan pinned `huggingface-hub==0.30.2` alongside
`transformers==4.57.3`. Those cannot coexist — 4.57.3 requires
`huggingface-hub>=0.34.0,<1.0` — so resolution failed outright. The 0.30.2 pin
is correct for the `kokoro` and `vits` environments, which sit on transformers
4.51.3, and had been carried across without re-checking. Pinned to 0.36.2, the
version the resolver selects for this set.

Two import-time warnings are expected rather than defects. `sox: not found` is
the plan's deliberate exclusion of a package that only wraps a system binary
this path never calls. `flash-attn is not installed` is required rather than
merely tolerated: `docs/port-validation.md` Phase 1 wants the CPU reference, and
the oracle passes `attn_implementation="eager"`.

### License audit — clear, and audited at the pinned revision

Both the source and the weights carry an explicit Apache-2.0 grant. The pinned
repository ships `LICENSE` (11,343 bytes, Apache 2.0 text) at `022e286b`, and
the model card declares `license: apache-2.0` in frontmatter. The only line in
either card matching *licence*, *non-commercial*, *cc-by* or *restrict* is the
frontmatter declaration itself. The `Qwen/Qwen3-TTS-Tokenizer-12Hz` card agrees.

Two deliberate departures from the plan's procedure, both in the direction of
finding a blocker earlier or more reliably.

The gate is written as Step 4 and the download as Step 2. Run in that order a
blocking restriction would surface only after pulling 2.5 GB, and the gate's
answer cannot depend on having the bytes, so it was run first.

More importantly, the plan's audit fetches each card from `main` while the plan
pins revision `85e237c1`. Those are not necessarily the same bytes, and auditing
one while shipping the other is the shape of mistake that removed OmniVoice from
consideration — there a port's README stood in for the model card. The audit was
therefore repeated against the downloaded card, which agrees.

Alibaba does not disclose the training corpora. The Apache-2.0 grant on the
checkpoint is the basis relied on, the same basis on which Kokoro was accepted.

### The declared multimodal RoPE collapses exactly

`talker_config.rope_scaling` declares `mrope_section [24, 20, 20]` — summing to
64, which is `head_dim / 2` — with `interleaved: true`. The machinery is real but
never exercised. Every path that builds `position_ids` in
`modeling_qwen3_tts.py` produces three identical rows, including
`get_rope_index`, whose body is an attention-mask cumsum followed by
`expand(3,-1,-1)` and whose docstring about temporal, height and width video
positions is inherited from Qwen2-VL. This model has no vision branch.

`apply_interleaved_rope` starts from `x[0].clone()` and overwrites strided
slices with rows 1 and 2, so equal rows make every write store the value it
replaces. Checked directly rather than argued: with three identical rows the
output is `torch.equal` to plain 1-D RoPE; with three different rows it differs
by 5.5 at these shapes.

Stage 4 implements ordinary RoPE and records the reason, because the
configuration will keep saying otherwise.

### `speech_tokenizer/config.json` disagrees with its own weights

The configuration declares `codebook_dim: 512` and
`semantic_codebook_size: 4096`. Every codebook tensor is `(2048, 256)`, and no
tensor anywhere in the file has 4096 entries — the only 4096s are the ConvNeXt
pointwise widths in `decoder.upsample`. Only `codebook_size: 2048` matches.

A converter that sizes the codebooks from this configuration allocates the wrong
tables and reports nothing. Size from the tensors.

### The RVQ codebooks are EMA accumulators, and the two sides disagree on names

The codebooks must be reconstructed at convert time as
`embedding_sum / clip(cluster_usage, 1e-5)[:, None]`. Reading the reference port
established the rule; the checkpoint adds something that reading did not show:

| Side | Fields | Codebooks |
| --- | --- | --- |
| `decoder.quantizer.*.vq.layers.N._codebook` | `embedding_sum`, `cluster_usage` | 16 |
| `encoder.quantizer.*.layers.N.codebook` | `embed_sum`, `cluster_usage` | 32 |

A conversion rule keyed on `embedding_sum` silently skips every encoder
codebook, and one keyed on `embed_sum` skips every decoder codebook. Encoder
codebooks also carry an `initialized` flag of shape `(1,)`, which is not a
weight.

### Speakers are token ids, so there is no speaker encoder

`get_supported_speakers()` returns nine names, and each is a token id in the
codec vocabulary between 2861 and 3066 rather than a row in an embedding table.
The tensor inventory confirms the consequence: there are no ECAPA-TDNN tensors
in either file. The family plan's claim that CustomVoice carries no speaker
encoder is correct.

`spk_is_dialect` is not a boolean but a language name. `eric` maps to
`sichuan_dialect` and `dylan` to `beijing_dialect`; both appear in
`codec_language_id` and neither appears in the public language list, so a
dialect is reachable only by selecting its speaker.

Three language counts are all correct and easy to confuse: the model card
frontmatter lists 10 codes, `get_supported_languages()` returns 11 including
`auto`, and `codec_language_id` holds 12 including the two dialects.

### Greedy decoding needs two switches, and the plan named one

This is the finding that most nearly became a false record.

The plan states that `do_sample=False` selects greedy decoding on both the
talker and the sub-talker. It does not. `generate_custom_voice` takes a separate
`subtalker_dosample`, which defaults to `True`, so the code predictor samples
under a nominally greedy call. Two such runs produced 69,120 against 65,280
frames and diverged at sample 20 — inside the first frame.

Recorded as measured, that is exactly the material finding the plan warns
about: a greedy oracle not reproducible against itself, which would force the
stage-5 replay seam to capture more than a code sequence. It would have been
false, and it would have reshaped the validation design around a problem that
does not exist.

With `do_sample=False` **and** `subtalker_dosample=False`, two runs in separate
processes are bit-identical: 97,920 frames each, `max_abs_diff` exactly 0.0,
peak agreeing to the last digit at 0.5742930173873901. The replay seam needs
only the captured code sequence the family plan assumed.

Any script that captures this oracle must set both switches. Setting one is
silently non-deterministic, which is a worse failure mode than an error.

Unseeded sampling behaves as expected and sets the stochastic capability: two
runs gave 51,840 and 76,800 frames, differing by 0.70 over the common prefix.

### Real-time factor 9.4 to 9.7, and it is a finding about the family

Greedy on CPU, F32, eager attention, `aiden` in `english`: 4.08 seconds of
24 kHz audio in 39.6 and 38.4 seconds of wall time, on the aarch64 GB10 host
with the GPU unused.

That number is not merely a note about the oracle. Every sampled token
conditions the next, so `docs/backends.md`'s discrete-output rule holds the
autoregressive core on CPU on every Execution Backend, and the CPU figure is
therefore **not** automatically the CUDA figure. Upstream deploys this model on
CUDA with bfloat16 and FlashAttention 2 -- the only device guidance its card
gives -- so the figure above is a floor set by this project's CPU-oracle rule
rather than a measurement of the model as its authors run it.

Whether CUDA helps is then a policy question. `docs/backends.md`'s
discrete-output rule would hold the autoregressive core on CPU on every backend,
and applied as written it makes CUDA nearly worthless here: Kokoro paid 95
percent of synthesis time to hold two stages of seven and VITS 29 percent to
hold one graph, while this family would hold the loop itself. But port
validation replays the captured codes, so the sampler does not run in the graph
being compared and cannot affect stage 5. The rule bites only on the public
request path, where the real question is whether the same text and seed must
yield the same tokens on every backend -- cheap to promise for VITS and Kokoro,
very expensive to promise here. Stage 7 decides it.

Roughly ten times slower than real time on this host is the honest planning
number for a Stage 1 CPU claim. The 1.7B voice-design rung has not been
measured and must not inherit this figure.

### Stage 1 claims Chunked Audio Delivery

`CONTEXT.md` defines Native Streaming Synthesis as a *validated* capability to
produce usable audio incrementally, so architecture alone cannot earn the claim.
The entry point used here returns a complete waveform, and upstream is explicit
that its own flag does not change that: `non_streaming_mode` "currently only
simulates streaming text input when set to `false`, rather than enabling true
streaming input or streaming generation".

The stronger claim remains reachable and its cost is known — carrying causal
convolution left context, transposed convolution overlap carry, and the
transformer sliding-window KV ring across frames — and it needs its own
validated evidence at a later stage.

### Open decisions carried into stage 2

- Whether SnakeBeta's `alpha` and `beta` exponentials can be folded once at
  load. The reference applies `exp()` on every forward; deciding this needs the
  forward rather than the shapes, and it is a stage-3 question.
- The 1.7B rung's real-time factor, which must be measured rather than inferred.
- The code predictor cache-reset mitigation, unmeasured here and a stage-4
  implementation concern.
- Whether the codec encoder half — 225 tensors — ships in the Model Package at
  all. Stage 1 selects preset speakers by token id and needs only the decoder
  side.

## 2026-07-29 — Listening audit: no obvious regression

The project owner listened to the comparison page and reported no problem. The
page offered eight replayed cases blind against the PyTorch reference --
switching sides keeps the playhead, so the same instant is heard twice -- plus
five natively-sampled clips on CPU and CUDA, which is the path replay does not
cover.

Recorded for exactly what it is: **one listener, informally, no rated
comparison, no panel, no score.** Under `docs/model-porting.md` this is a
Listening Audit reporting `no_obvious_regression`, and it does not move the
Validation Level. The package stays `port_validated`.

What it establishes is the thing a cosine cannot: that agreement measured at
0.999427 is not hiding an audible disagreement. That is worth having before
shipping, and this port shipped before asking for it -- the packages were
published and the pull request opened while this step was still outstanding. It
happened because the owner asked why it had been skipped, not because the
process caught it.

The published README still reports `quality_evaluation: not_run`, which was
accurate when written. Updating it is a separate outward act and has not been
made.

## 2026-07-29 — Truncation found by ear, fixed, and confirmed by ear

The project owner listened to the Q8_MIXED comparison and reported that long
inputs stopped mid-sentence -- on both profiles, at points that moved with the
seed. That ruled out quantization immediately and pointed at the sampler.

The cause: the checkpoint ships `repetition_penalty: 1.05` in
`generation_config.json` and this port never implemented it, while hardcoding the
other three values at figures that happened to match. The converter now carries
all seven decoding values into the package and the loader requires them.

Confirmed by listening after the fix: the two reported lines, two seeds, both
profiles, all reaching their final words. One listener, informally.

Two things this leaves on the record. The defect was found by a person listening,
not by the suite -- the replay seam that makes validation deterministic also makes
`select_code` the one stage eighteen Golden cases never run. And every committed
tolerance was within range before and after the fix, to the digit, which is
exactly what a suite that does not sample would report.

The published packages predate this and no longer load: they carry no sampling
metadata, and the loader refuses rather than guessing. Re-uploading is a separate
outward act and has not been made.

## 2026-07-29 — Q8_MIXED listening audit: no obvious regression

The project owner listened to six natively-sampled pairs against F16 -- English
medium and long, Chinese, Japanese, a second Voice, and a dialect speaker -- and
reported no problem.

Native sampling rather than A/B, and that is structural: on the replay path the
Q8 and F16 waveforms are byte-identical, verified by comparing the files. Replay
supplies the oracle's codes and the codec that renders them is F32 in both
packages, so nothing downstream of the draw can differ. What quantization moves
is upstream of it -- the talker's logits fall from cosine 0.9994 to 0.9956 -- and
only a run that draws its own codes can show what that costs.

One observation the listener cleared: Q8 came out shorter than F16 in five of the
six pairs, by 8 to 13 percent, with the sixth 2 percent longer. Six clips at one
seed each is thin evidence for a systematic claim, and the pass says it is not
audible. Worth revisiting if a larger set ever runs.

**This pass had to be run three times, and both invalidations were caught by the
listener rather than by anything automated.** The first used a sampler missing
the checkpoint's repetition penalty -- the defect that truncated long inputs. The
second was correct at synthesis and wrong at presentation: the page embedded
audio capped at ten seconds, so the two long cases were cut off in the player
while their reported durations said otherwise. A listening page that trims its
own audio cannot answer the question it asks, and the disclosure sat in a footer.

Recorded for what it is: one listener, informally, six cases.

## 2026-07-29 — Republished, with Q8_MIXED

Uploaded to `jiangzhuo9357/qwen3-tts-12hz-0-6b-customvoice-gguf`: BF16 and F16
re-cut, Q8_MIXED new, and the regenerated card. Digests verified equal to the
validated local files.

The re-cut was not optional. The published packages carried no sampling metadata
and the loader now requires it rather than guessing, so they had stopped loading
entirely -- the same shape as the stale VITS packages in July, and for the same
underlying reason: a package published before a contract changed keeps serving
until someone checks.

The card grew a `listening_audit` field. `quality_evaluation` stays `not_run`,
which is accurate -- ADR 0017's automated grid has never run and is not scheduled
-- and the audit is a separate, weaker claim that the generator had no way to
express. Kokoro's card has the same gap: it was listened to on 2026-07-26 and its
published card still says nothing about it. Closing that is a separate upload and
has not been made.

## 2026-07-29 — Both validators registered with CTest

`synthesize-qwen3-tts-replay-golden` and `synthesize-qwen3-tts-public-request`
now register under `integration;qwen3-tts` from `SYNTH_QWEN3_TTS_TEST_MODEL`,
which defaults to the BF16 package because that is this family's source profile.

Both were run: replay golden passes in 1003 seconds over eighteen cases, public
request in 40. Neither appears under the `unit` label, checked rather than
assumed.

The replay validator registers with `--check`, which is the point of the slice.
Until now `tests/tolerances/qwen3-tts.json` was a record nothing enforced -- the
numbers were measured, committed, and never compared against again except by
hand. Proved it can fail: with every `min_cosine` raised to 0.9999999 the
validator exits 1 and names each probe that fell short; with the committed file
it exits 0. A gate never seen to fail is not known to be a gate.

The two have different prerequisites and that is deliberate. Replay needs the
uncommitted oracle payload and is not registered without it. The public phase
needs only the package, because it asserts relations between runs of this port
rather than agreement with the reference.

## 2026-07-29 — Stage 7's owed items, and twenty cases

Three gaps closed in parallel.

**BF16 on CUDA**, which had never been swept because the accelerator work ran
under F16. Eighteen cases, all ok. Every talker probe is bit-for-bit equal to the
CPU entry and only `audio.pcm` moved, 0.999427 to 0.999404 -- the placement
proving itself rather than being asserted. Placement was also checked directly:
`nvidia-smi` showed the runner holding 606 MiB on the GB10 while it ran.

**Repeated-run and resource cleanup.** The investigation found more than the
task: `docs/backends.md` gate 6 has never been satisfied in code by any family.
All four manifests declare `resource_cleanup` per case and nothing reads the
field; `backend_placement` is the same. VITS's evidence was twenty repeated
processes, which cannot detect an in-process leak.

No leak found. CPU over thirty cycles: post-free RSS oscillates 47.7-65.5 MB with
no trend, minimum at cycle 12, cycle 29 below cycle 3, and the 1.4 GB of weights
returns every cycle. CUDA: +7.6 MB and +5.7 MB on the first two cycles then
0-156 kB, plateauing, with the last six identical. LeakSanitizer clean over three
cycles, verified armed against a control that leaks 4096 bytes.
`tests/qwen3_tts_public_cleanup_test.cpp` is registered under
`integration;abi;qwen3-tts`, runs CPU and CUDA, and drops to three cycles under
`SYNTH_SANITIZE` because LeakSanitizer decides the same question at allocation
granularity. It passes in 63 seconds.

**Twenty Golden cases**, up from eighteen: `qwen3-longer-english` at 18.7 seconds
and `qwen3-longer-chinese` at 21.3, against a previous longest of 9.3. Oracle run
at the pinned revision. All twenty pass the committed tolerances with `--check`
and **no worst-case figure moved** -- both new cases are better than the suite
worst on every probe. Accumulated error does not grow with length here.

## 2026-07-29 — Retracting the public-CUDA placement claim

Asked to make the public seam place only the codec on CUDA and keep sampling on
the CPU, the first step was checking that it did not already. It does.

The claim being acted on was mine: that `SYNTH_BACKEND_CUDA` at the public seam
moved the sampled path onto the accelerator, evidenced by one case drawing ten
frames on CUDA against eleven on CPU. **That comparison had two variables in it.**
It used a Release CPU build against the CUDA preset, and the cleanup
investigation had incidentally shown that the same CPU code gives a different PCM
digest under `-O3` than under `-O2` -- float contraction changes with optimization
level, and a changed logit changes a draw.

Re-run with a single binary, `--backend cpu` against `--backend cuda`, five cases
across English, Chinese and Japanese and three Voices: identical frame counts in
every one. Only the waveform bytes differ, which is the codec's arithmetic and
the same signature the BF16-on-CUDA sweep records.

No code change was made because none was needed. The claim is retracted in the
family doc and in the tolerance file's note, both of which stated it as fact.

The lesson is not about CUDA. A measurement taken across two builds cannot
attribute a difference to the thing being varied, and this one was carried into a
committed tolerance file, a family document and a pull request description before
anyone re-ran it with one variable.
