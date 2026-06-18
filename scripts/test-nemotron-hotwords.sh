#!/usr/bin/env bash
# scripts/test-nemotron-hotwords.sh
#
# End-to-end test for streaming Nemotron hotword/context biasing.
#
# Usage:
#   ./scripts/test-nemotron-hotwords.sh [model_dir]
#
# If model_dir is omitted, $HOME/Downloads is used. The script expects
# encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx, tokens.txt,
# and tokenizer.model in that directory. It will download the test WAV
# file if it is missing and generate bpe.vocab from tokenizer.model.

set -e

MODEL_DIR="${1:-$HOME/Downloads}"
WAV_DIR="$MODEL_DIR/test_wavs"
WAV="$WAV_DIR/en.wav"
HOTWORDS="$MODEL_DIR/hotwords.txt"
BPE_VOCAB="$MODEL_DIR/bpe.vocab"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build-shared"
BIN="$BUILD_DIR/bin/test-nemotron-hotwords-cxx-api"

MODEL_BASE="https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/resolve/main"
WAV_URL="$MODEL_BASE/test_wavs/en.wav"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: required file not found: $1"
    echo "Please place encoder/decoder/joiner/tokens.txt/tokenizer.model in $MODEL_DIR"
    exit 1
  fi
}

download_if_missing() {
  local file="$1"
  local url="$2"
  if [[ ! -f "$file" ]]; then
    echo "Downloading $(basename "$file")..."
    wget -c -q -O "$file" "$url"
    echo "Saved $file"
  fi
}

echo "=== Nemotron streaming hotword test ==="
echo "Model directory: $MODEL_DIR"

download_if_missing "$MODEL_DIR/encoder.int8.onnx" "$MODEL_BASE/encoder.int8.onnx"
download_if_missing "$MODEL_DIR/decoder.int8.onnx" "$MODEL_BASE/decoder.int8.onnx"
download_if_missing "$MODEL_DIR/joiner.int8.onnx" "$MODEL_BASE/joiner.int8.onnx"
download_if_missing "$MODEL_DIR/tokens.txt" "$MODEL_BASE/tokens.txt"

require_file "$MODEL_DIR/tokenizer.model"

if [[ ! -f "$WAV" ]]; then
  echo "Downloading test WAV file..."
  mkdir -p "$WAV_DIR"
  wget -q -O "$WAV" "$WAV_URL"
  echo "Saved $WAV"
fi

if [[ ! -f "$BPE_VOCAB" || "$MODEL_DIR/tokenizer.model" -nt "$BPE_VOCAB" ]]; then
  echo "Generating $BPE_VOCAB from tokenizer.model..."
  python3 "$REPO_ROOT/scripts/export_bpe_vocab.py" \
      --bpe-model "$MODEL_DIR/tokenizer.model" --output "$BPE_VOCAB"
fi

cat > "$HOTWORDS" <<'EOF'
tribal chief
pieces of gold
EOF
echo "Hotwords written to $HOTWORDS"

if [[ ! -f "$BIN" ]]; then
  echo "Building $BIN ..."
  mkdir -p "$BUILD_DIR"
  cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DBUILD_SHARED_LIBS=ON
  cmake --build "$BUILD_DIR" --target test-nemotron-hotwords-cxx-api -j"$(nproc)"
fi

echo ""
echo "=== Running modified_beam_search with hotwords ==="

export LD_LIBRARY_PATH="$BUILD_DIR/lib:$BUILD_DIR/_deps/onnxruntime-src/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$BIN" "$MODEL_DIR" "$WAV" "$HOTWORDS" "$BPE_VOCAB" en
