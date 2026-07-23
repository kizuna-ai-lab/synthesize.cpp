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
| F16 | 74,208,224 | 350 F32 + 123 F16 | `ff11efb1106834efb3609647e68642b48a58dbbdbabbc776d3afd83cf46085af` |
| Q8_MIXED | 55,047,392 | 350 F32 + 4 F16 + 119 Q8_0 | `149438d3a6c817ca6e4ab803a67207a41f62097cc8f586520355610801fb0542` |

The public C ABI is profile-independent. C++, Rust, and Python callers load a
local GGUF through the same model interface. VITS tensor policy, packed Q8
matrix layout, and execution details remain private to the architecture module.

## Port validation

Seven graph stages and 12 deterministic cases run on one-thread DGX Spark CPU,
NVIDIA GB10 CUDA 13.3, and NVIDIA RTX 4070 SUPER CUDA 13.3. Duration structure
is exact in every case. Both CUDA placement records contain zero executable CPU
fallback nodes.

| Profile | CPU max PCM drift | GB10 CUDA | RTX 4070 SUPER CUDA |
| --- | ---: | ---: | ---: |
| F32 | 0.0002596639 | 0.025091962 | 0.0183914602 |
| F16 | 0.06752773 | 0.24737186 | 0.19204060 |
| Q8_MIXED | 0.77982019 | 0.57462588 | 0.62477511 |

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
