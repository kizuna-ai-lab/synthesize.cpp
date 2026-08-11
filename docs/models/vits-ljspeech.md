# VITS LJSpeech

Status: F32, F16, and Q8_MIXED are `port_validated`. Quality evaluation has not
been run. The packages are published in
[`jiangzhuo9357/vits-ljspeech-gguf`](https://huggingface.co/jiangzhuo9357/vits-ljspeech-gguf)
at revision `53273eea758782f86ffd79a830ffb4ccde533acf`.

## Package

The package is converted from the official single-speaker VITS LJSpeech
checkpoint at pinned upstream revision
`2e561ba58618d021b5b8323d3765880f7e0ecfdb`. It has one unnamed fixed
package-default Voice, produces 22,050 Hz mono F32 PCM, and does not accept a
preset Voice identifier. It accepts UTF-8 phonemes through the built-in
`synthesize.symbol_map` frontend or exact token IDs. Raw-text G2P is not included.

| Profile | Bytes | Tensor storage | SHA-256 |
| --- | ---: | --- | --- |
| F32 | 113,245,056 | 460 F32 | `bd17e44c7c2d761d33c1527059bd3921f9d3fa9d46bd73b3e020d746a8b7db7b` |
| F16 | 75,778,368 | 346 F32 + 114 F16 | `8683788b4dec7b81bf86f1cca024ed790724e491e2df40f77f1fc0bc614f860f` |
| Q8_MIXED | 58,215,040 | 346 F32 + 114 Q8_0 | `f751607f7514fa2aa1dca3ba46738b081c6b6c6b9579bfd22b1b80016551a3eb` |

The public C ABI is profile-independent. C++, Rust, and Python callers load a
local GGUF through the same model interface. VITS tensor policy, packed Q8
matrix layout, and execution details remain private to the architecture module.

## Port validation

Seven graph stages and 12 deterministic cases run on DGX Spark CPU and NVIDIA
GB10 CUDA 13.3. The RTX 4070 SUPER CUDA 13.3 host also ran them for the packages
cut on 2026-07-23; it was not available for the 2026-07-27 re-cut. Duration structure
is exact in every case. Two of the seven stages — the text encoder and the
duration predictor — run on CPU on every Execution Backend under
`docs/backends.md`'s discrete-outputs rule, and the other five run on CUDA. No
node falls back: the hold is a separate CPU scheduler over mirrored weights, not
a mixed graph. (Corrected 2026-08-12: this read "Every CUDA placement record has
one split and zero executable CPU fallback nodes", which was measured on
2026-07-22 — `docs/backends.md`'s five-stage checkpoint — four days before
`df1351e` moved the duration path onto CPU. The claim was true about fallback
and false about full GPU execution, which `docs/backends.md` forbids conflating.)

| Profile | CPU max PCM drift | GB10 CUDA |
| --- | ---: | ---: |
| F32 | 0.00029484 | 0.01902072 |
| F16 | 0.01533963 | 0.02761611 |
| Q8_MIXED | 0.36558404 | 0.37650996 |

Re-measured on 2026-07-27 against the re-cut F16 and Q8_MIXED packages, on the
DGX Spark host only. The RTX 4070 SUPER column is dropped rather than carried
forward: that host was not available, so its figures would describe packages
that no longer exist. The earlier GB10 column is not comparable either --- its
F32 and F16 entries, 0.20982037 and 0.20995625, disagree by an order of
magnitude with the same stage's CUDA record in `tests/tolerances/vits.json`, and
that discrepancy predates this change and was not reconstructed.

These measurements prove functional execution and record numerical drift; they
are not perceptual-quality thresholds. Naturalness and intelligibility remain
unevaluated.

## Reproduction

Generate and verify the model card:

```bash
uv run scripts/hf_cards/generate.py scripts/hf_cards/vits-ljspeech.yaml
uv run scripts/hf_cards/generate.py scripts/hf_cards/vits-ljspeech.yaml --check
```

Run the locked Python tests:

```bash
uv run --project scripts/envs/vits --locked \
  python -m unittest discover -s tests/python -p 'test_*.py'
```

The published flat directory was created with:

```bash
hf repos create jiangzhuo9357/vits-ljspeech-gguf \
  --type model --public --exist-ok
hf upload jiangzhuo9357/vits-ljspeech-gguf models/vits-ljspeech . \
  --commit-message "Publish VITS LJSpeech F32, F16, and Q8_MIXED"
```

The upstream source is MIT licensed. The author-published checkpoint does not
state separate weight terms. Under the maintainer's explicit project policy,
that absence is treated as no additional redistribution restriction and is
disclosed in the generated model card.
