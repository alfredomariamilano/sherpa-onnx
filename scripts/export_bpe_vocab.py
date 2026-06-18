#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# You can install sentencepiece via:
#
#  pip install sentencepiece
#
# Due to an issue reported in
# https://github.com/google/sentencepiece/pull/642#issuecomment-857972030
#
# Please install a version >=0.1.96

import argparse
import codecs
import struct
import sys
from typing import List, Tuple

try:
    import sentencepiece as spm
    _HAS_SENTENCEPIECE = True
except ImportError:
    _HAS_SENTENCEPIECE = False


# ---------------------------------------------------------------------------
# Minimal protobuf parser used as a fallback when sentencepiece is not
# installed. SentencePiece model files are serialized ModelProto messages; the
# repeated ``pieces`` field (field 1) contains the token/score pairs we need.
# ---------------------------------------------------------------------------

def _decode_varint(data: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _parse_piece(payload: bytes) -> Tuple[str, float, int]:
    piece = ""
    score = 0.0
    ptype = 0
    p = 0
    while p < len(payload):
        tag = payload[p]
        p += 1
        field = tag >> 3
        wire = tag & 0x7
        if field == 1 and wire == 2:  # string piece
            length, p = _decode_varint(payload, p)
            piece = payload[p:p + length].decode("utf-8")
            p += length
        elif field == 2 and wire == 5:  # float score
            score = struct.unpack("<f", payload[p:p + 4])[0]
            p += 4
        elif field == 3 and wire == 0:  # int32 type
            ptype, p = _decode_varint(payload, p)
        elif wire == 0:
            _, p = _decode_varint(payload, p)
        elif wire == 2:
            length, p = _decode_varint(payload, p)
            p += length
        elif wire == 5:
            p += 4
        elif wire == 1:
            p += 8
        else:
            raise ValueError(f"Unexpected wire type {wire} at offset {p}")
    return piece, score, ptype


def _extract_vocab_without_sentencepiece(model_file: str) -> List[Tuple[str, float]]:
    with open(model_file, "rb") as f:
        data = f.read()

    pos = 0
    entries = []
    while pos < len(data):
        tag = data[pos]
        pos += 1
        field = tag >> 3
        wire = tag & 0x7
        if field == 1 and wire == 2:  # repeated SentencePiece pieces
            length, pos = _decode_varint(data, pos)
            payload = data[pos:pos + length]
            pos += length
            piece, score, _ = _parse_piece(payload)
            entries.append((piece, score))
        elif wire == 0:
            _, pos = _decode_varint(data, pos)
        elif wire == 2:
            length, pos = _decode_varint(data, pos)
            pos += length
        elif wire == 5:
            pos += 4
        elif wire == 1:
            pos += 8
        else:
            raise ValueError(f"Unexpected wire type {wire} at offset {pos}")

    if not entries:
        raise RuntimeError("No BPE entries found; is this a valid SentencePiece model?")
    return entries


def _extract_vocab_with_sentencepiece(model_file: str) -> List[Tuple[str, float]]:
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    vocab_size = sp.get_piece_size()
    return [(sp.id_to_piece(i), sp.get_score(i)) for i in range(vocab_size)]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bpe-model",
        type=str,
        required=True,
        help="The path to the bpe model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output vocab path. Defaults to <bpe-model> with .model replaced by .vocab.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    model_file = args.bpe_model

    vocab_file = args.output
    if not vocab_file:
        vocab_file = model_file.replace(".model", ".vocab")

    if _HAS_SENTENCEPIECE:
        print("Using installed sentencepiece package to export vocab.")
        entries = _extract_vocab_with_sentencepiece(model_file)
    else:
        print("sentencepiece not installed; falling back to built-in protobuf parser.")
        entries = _extract_vocab_without_sentencepiece(model_file)

    with codecs.open(vocab_file, "w", "utf-8") as vfile:
        for piece, score in entries:
            vfile.write(f"{piece}\t{score}\n")

    print(f"Vocabulary file is written to {vocab_file}")


if __name__ == "__main__":
    main()
