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
| F16 | 70,453,568 | 342 F32 + 118 F16 | `5fc428ba97416cc164055f509af1b9e120b089bd0bab792ad674c862411d7bca` |
| Q8_MIXED | 52,890,240 | 342 F32 + 4 F16 + 114 Q8_0 | `df95091f975e78088c4908c3f2adfc381cfa234c3f520ddd3ffe10160ff12ba1` |

The public C ABI is profile-independent. C++, Rust, and Python callers load a
local GGUF through the same model interface. VITS tensor policy, packed Q8
matrix layout, and execution details remain private to the architecture module.

## Port validation

Seven graph stages and 12 deterministic cases run on one-thread DGX Spark CPU,
NVIDIA GB10 CUDA 13.3, and NVIDIA RTX 4070 SUPER CUDA 13.3. Duration structure
is exact in every case. Every CUDA placement record has one split and zero
executable CPU fallback nodes.

| Profile | CPU max PCM drift | GB10 CUDA | RTX 4070 SUPER CUDA |
| --- | ---: | ---: | ---: |
| F32 | 0.0002223924 | 0.20982037 | 0.2027478628 |
| F16 | 0.01712550 | 0.20995625 | 0.20820463 |
| Q8_MIXED | 0.36679696 | 0.32943400 | 0.34679114 |

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
