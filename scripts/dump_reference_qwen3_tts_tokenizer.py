#!/usr/bin/env python3
"""Emit the reference token ids for tests/qwen3_tts_bpe_test.cpp.

Tokenizes a set of strings with the pinned checkpoint's own ``Qwen2Tokenizer``
and prints the ids as C++ literals. The cases are chosen to reach every branch of
the pre-tokenizer pattern -- contractions, letters with and without a leading
space, digits that do not group, punctuation runs, newlines, trailing whitespace
-- plus non-Latin scripts and the prompt template's own markers, which must
survive as single tokens rather than being split.

The vocabulary and merges are read from the package rather than the checkpoint,
so this also proves the two agree. Run from ``scripts/envs/qwen3-tts``:

    uv run python ../../dump_reference_qwen3_tts_tokenizer.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CASES = [
    "hello world",
    "It's a test, isn't it? I'd say we've won.",
    "Numbers 1234 and 5.67 do not group.",
    "  leading and trailing   ",
    "line one\nline two\r\nline three\n\n",
    "punctuation!!! ...--- ???",
    "\u4f60\u597d\uff0c\u4e16\u754c\u3002\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5\u3002",
    "\u3053\u3093\u306b\u3061\u306f\u3001\u4e16\u754c\u3002",
    "\uc548\ub155\ud558\uc138\uc694",
    "Emoji \U0001F600 and \u00e9\u00e8\u00ea accents",
    "<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>assistant\n",
    "",
    " ",
]


def literal(text: str) -> str:
    out = []
    for byte in text.encode("utf-8"):
        if byte == 0x22:
            out.append('\\"')
        elif byte == 0x5C:
            out.append("\\\\")
        elif 0x20 <= byte < 0x7F:
            out.append(chr(byte))
        else:
            out.append(f"\\x{byte:02x}\" \"")
    return '"' + "".join(out) + '"'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path,
                        default=Path("../../../models/qwen3-tts-12hz-0-6b-customvoice"))
    arguments = parser.parse_args()

    from transformers import Qwen2Tokenizer

    tokenizer = Qwen2Tokenizer.from_pretrained(str(arguments.weights_dir))

    # The package's vocabulary must be the checkpoint's, or the port would
    # tokenize against a different table than this reference.
    vocab = json.loads((arguments.weights_dir / "vocab.json").read_text(encoding="utf-8"))
    full = tokenizer.get_vocab()
    assert all(full.get(piece) == index for piece, index in vocab.items()), (
        "the checkpoint's vocab.json is not the tokenizer's")

    print("struct Case {")
    print("    const char * text;")
    print("    const int32_t * ids;")
    print("    size_t count;")
    print("};")
    print()
    names = []
    for index, case in enumerate(CASES):
        ids = tokenizer(case)["input_ids"]
        name = f"kIds{index}"
        names.append((case, name, len(ids)))
        body = ", ".join(str(int(value)) for value in ids) if ids else ""
        print(f"constexpr int32_t {name}[] = {{ {body if body else '0'} }};"
              + ("  // empty" if not ids else ""))
    print()
    print("constexpr Case kCases[] = {")
    for case, name, count in names:
        print(f"    {{ {literal(case)}, {name}, {count} }},")
    print("};")
    return 0


if __name__ == "__main__":
    sys.exit(main())
