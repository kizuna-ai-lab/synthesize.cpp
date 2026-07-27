# VITS VCTK

Status: F32, F16, and Q8_MIXED are `port_validated`. Quality evaluation has not
been run. The packages are published in
[`jiangzhuo9357/vits-vctk-gguf`](https://huggingface.co/jiangzhuo9357/vits-vctk-gguf)
at revision `eb27a41f7a01dbe3ec88dabcd834844e9383a5e7`.

## Package

The package is converted from the official 109-speaker VITS VCTK checkpoint at
the pinned upstream revision
`2e561ba58618d021b5b8323d3765880f7e0ecfdb`. It exposes preset Voice IDs
`speaker-000` through `speaker-108`, produces 22,050 Hz mono F32 PCM, and requires
the caller to select a Voice explicitly. It accepts UTF-8 phonemes through the
built-in `synthesize.symbol_map` frontend or exact token IDs. Raw-text G2P is not
included.

| Profile | Bytes | Tensor storage | SHA-256 |
| --- | ---: | --- | --- |
| F32 | 120,407,552 | 473 F32 | `b9e69b257cc600679a45e4197614f36ec678156180e2d366f7e1dcb6668b2be0` |
| F16 | 79,533,024 | 354 F32 + 119 F16 | `0b3c4e067c6fdd736edb90b7e1d11f6f480e5a832620740178a3119a4ef0913c` |
| Q8_MIXED | 60,372,192 | 354 F32 + 119 Q8_0 | `9f6caab60c66cdfa2379a1f1b93dbbd886161cd6f1115431fbe964d198b698e3` |

The public C ABI is profile-independent. C++, Rust, and Python callers load a
local GGUF through the same model interface. VITS tensor policy, packed Q8
matrix layout, and execution details remain private to the architecture module.

## Port validation

Seven graph stages and 12 deterministic cases run on DGX Spark CPU and NVIDIA
GB10 CUDA 13.3. The RTX 4070 SUPER CUDA 13.3 host also ran them for the packages
cut on 2026-07-23; it was not available for the 2026-07-27 re-cut. Duration structure
is exact in every case. Both CUDA placement records contain zero executable CPU
fallback nodes.

| Profile | CPU max PCM drift | GB10 CUDA |
| --- | ---: | ---: |
| F32 | 0.00023869 | 0.03104201 |
| F16 | 0.07394360 | 0.21655512 |
| Q8_MIXED | 0.77769499 | 0.84563246 |

Re-measured on 2026-07-27 against the re-cut F16 and Q8_MIXED packages, on the
DGX Spark host only. The RTX 4070 SUPER column is dropped rather than carried
forward: that host was not available, so its figures would describe packages
that no longer exist.

These measurements prove functional execution and record numerical drift; they
are not perceptual-quality thresholds. Naturalness, intelligibility, and speaker
similarity remain unevaluated.

## Reproduction

Generate and verify the model card:

```bash
uv run scripts/hf_cards/generate.py scripts/hf_cards/vits-vctk.yaml
uv run scripts/hf_cards/generate.py scripts/hf_cards/vits-vctk.yaml --check
```

Run the locked Python tests:

```bash
uv run --project scripts/envs/vits --locked \
  python -m unittest discover -s tests/python -p 'test_*.py'
```

The published flat directory was created with:

```bash
hf repos create jiangzhuo9357/vits-vctk-gguf \
  --type model --public --exist-ok
hf upload jiangzhuo9357/vits-vctk-gguf models/vits-vctk . \
  --commit-message "Publish VITS VCTK F32, F16, and Q8_MIXED"
```

The upstream source is MIT licensed. The author-published checkpoint does not
state separate weight terms. Under the maintainer's explicit project policy,
that absence is treated as no additional redistribution restriction and is
disclosed in the generated model card.
