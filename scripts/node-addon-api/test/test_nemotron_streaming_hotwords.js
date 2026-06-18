// Copyright (c)  2026  Xiaomi Corporation
//
// Test streaming Nemotron ASR with hotword/context biasing through the
// sherpa-onnx Node.js addon.
//
// Usage:
//   node test/test_nemotron_streaming_hotwords.js [model_dir]
//
// model_dir should contain:
//   encoder.int8.onnx  decoder.int8.onnx  joiner.int8.onnx
//   tokens.txt         tokenizer.model
//
// The script will create bpe.vocab and hotwords.txt in model_dir if they do
// not already exist, then decode test_wavs/en.wav.

const fs = require('fs');
const path = require('path');
const {performance} = require('perf_hooks');

const sherpa_onnx = require('../lib/sherpa-onnx.js');

const modelDir = process.argv[2] || path.join(require('os').homedir(), 'Downloads');
const wavDir = path.join(modelDir, 'test_wavs');
const wavFilename = path.join(wavDir, 'en.wav');
const hotwordsFilename = path.join(modelDir, 'hotwords.txt');
const bpeVocabFilename = path.join(modelDir, 'bpe.vocab');
const tokenizerFilename = path.join(modelDir, 'tokenizer.model');

function ensureFile(filename, content) {
  if (!fs.existsSync(filename)) {
    fs.writeFileSync(filename, content);
    console.log(`Created ${filename}`);
  }
}

function ensureBpeVocab() {
  if (fs.existsSync(bpeVocabFilename) &&
      fs.statSync(bpeVocabFilename).mtime >= fs.statSync(tokenizerFilename).mtime) {
    return;
  }
  const exportScript = path.join(__dirname, '..', '..', 'export_bpe_vocab.py');
  const {execFileSync} = require('child_process');
  execFileSync('python3', [exportScript, '--bpe-model', tokenizerFilename,
                           '--output', bpeVocabFilename],
               {stdio: 'inherit'});
}

function ensureWav() {
  if (fs.existsSync(wavFilename)) {
    return;
  }
  const url =
      'https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/resolve/main/test_wavs/en.wav';
  fs.mkdirSync(wavDir, {recursive: true});
  const {execFileSync} = require('child_process');
  console.log(`Downloading ${wavFilename}...`);
  execFileSync('wget', ['-q', '-O', wavFilename, url], {stdio: 'inherit'});
}

const requiredFiles = [
  'encoder.int8.onnx',
  'decoder.int8.onnx',
  'joiner.int8.onnx',
  'tokens.txt',
  'tokenizer.model',
];

for (const f of requiredFiles) {
  const p = path.join(modelDir, f);
  if (!fs.existsSync(p)) {
    console.error(`ERROR: required file not found: ${p}`);
    process.exit(1);
  }
}

ensureBpeVocab();
ensureFile(hotwordsFilename, 'tribal chief\npieces of gold\n');
ensureWav();

const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  },
  'modelConfig': {
    'transducer': {
      'encoder': path.join(modelDir, 'encoder.int8.onnx'),
      'decoder': path.join(modelDir, 'decoder.int8.onnx'),
      'joiner': path.join(modelDir, 'joiner.int8.onnx'),
    },
    'tokens': path.join(modelDir, 'tokens.txt'),
    'numThreads': 1,
    'provider': 'cpu',
    'debug': 0,
    'modelType': '',
    'modelingUnit': 'bpe',
    'bpeVocab': bpeVocabFilename,
  },
  'decodingMethod': 'modified_beam_search',
  'maxActivePaths': 4,
  'hotwordsFile': hotwordsFilename,
  'hotwordsScore': 1.5,
};

console.log('Creating recognizer...');
const recognizer = new sherpa_onnx.OnlineRecognizer(config);

const wave = sherpa_onnx.readWave(wavFilename);
if (!wave) {
  console.error(`Failed to read ${wavFilename}`);
  process.exit(1);
}

const stream = recognizer.createStream();
stream.acceptWaveform({samples: wave.samples, sampleRate: wave.sampleRate});

const tailPadding = new Float32Array(wave.sampleRate * 0.4);
stream.acceptWaveform({samples: tailPadding, sampleRate: wave.sampleRate});

console.log('Decoding...');
const start = performance.now();
while (recognizer.isReady(stream)) {
  recognizer.decode(stream);
}
const result = recognizer.getResult(stream);
const stop = performance.now();

const elapsedSeconds = (stop - start) / 1000;
const duration = wave.samples.length / wave.sampleRate;
const rtf = elapsedSeconds / duration;

console.log('text:', result.text);
console.log('Duration:', duration.toFixed(3), 'seconds');
console.log('Elapsed:', elapsedSeconds.toFixed(3), 'seconds');
console.log('RTF:', rtf.toFixed(3));
