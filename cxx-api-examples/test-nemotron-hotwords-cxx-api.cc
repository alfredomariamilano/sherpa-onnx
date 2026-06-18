// cxx-api-examples/test-nemotron-hotwords-cxx-api.cc
// Copyright (c)  2026  Xiaomi Corporation
//
// This file tests hotword/context biasing with streaming Nemotron
// using sherpa-onnx's C++ API.
//
// Usage:
//   ./test-nemotron-hotwords-cxx-api \
//       <model_dir> <wav_file> <hotwords_file> <bpe_vocab> [language]
//
// <model_dir> should contain:
//   encoder.int8.onnx  decoder.int8.onnx  joiner.int8.onnx  tokens.txt
//
// Example hotwords file (one phrase per line):
//   NEMOTRON
//   SHERPA ONNX

#include <chrono>  // NOLINT
#include <cstdio>
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

static void PrintUsage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " <model_dir> <wav_file> <hotwords_file> <bpe_vocab>"
               " [language]\n"
            << "\n"
            << "  model_dir: directory containing encoder/decoder/joiner/tokens\n"
            << "  wav_file: 16kHz mono PCM WAV file\n"
            << "  hotwords_file: one hotword phrase per line\n"
            << "  bpe_vocab: SentencePiece model file, e.g. tokenizer.model\n"
            << "  language: optional language code (default: en)\n";
}

static std::string JoinPath(const std::string &dir,
                            const std::string &name) {
  if (dir.empty()) return name;
  if (dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

int32_t main(int32_t argc, char *argv[]) {
  using namespace sherpa_onnx::cxx;  // NOLINT

  if (argc < 5 || argc > 6) {
    PrintUsage(argv[0]);
    return -1;
  }

  const std::string model_dir = argv[1];
  const std::string wave_filename = argv[2];
  const std::string hotwords_file = argv[3];
  const std::string bpe_vocab = argv[4];
  const std::string language = argc >= 6 ? argv[5] : "en";

  OnlineRecognizerConfig config;

  config.decoding_method = "modified_beam_search";
  config.max_active_paths = 4;
  config.hotwords_file = hotwords_file;
  config.hotwords_score = 1.5f;

  config.model_config.transducer.encoder =
      JoinPath(model_dir, "encoder.int8.onnx");
  config.model_config.transducer.decoder =
      JoinPath(model_dir, "decoder.int8.onnx");
  config.model_config.transducer.joiner =
      JoinPath(model_dir, "joiner.int8.onnx");
  config.model_config.tokens = JoinPath(model_dir, "tokens.txt");

  config.model_config.modeling_unit = "bpe";
  config.model_config.bpe_vocab = bpe_vocab;
  config.model_config.num_threads = 1;

  std::cout << "Loading model (modeling_unit=" << config.model_config.modeling_unit
            << ", bpe_vocab=" << config.model_config.bpe_vocab << ")\n";
  OnlineRecognizer recognizer = OnlineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  std::cout << "Start recognition (hotwords: " << hotwords_file << ")\n";
  const auto begin = std::chrono::steady_clock::now();

  OnlineStream stream = recognizer.CreateStream();
  stream.SetOption("language", language.c_str());
  stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                        wave.samples.size());
  stream.InputFinished();

  while (recognizer.IsReady(&stream)) {
    recognizer.Decode(&stream);
  }

  OnlineRecognizerResult result = recognizer.GetResult(&stream);

  const auto end = std::chrono::steady_clock::now();
  const float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = wave.samples.size() / static_cast<float>(wave.sample_rate);
  float rtf = elapsed_seconds / duration;

  std::cout << "text: " << result.text << "\n";
  printf("Number of threads: %d\n", config.model_config.num_threads);
  printf("Duration: %.3fs\n", duration);
  printf("Elapsed seconds: %.3fs\n", elapsed_seconds);
  printf("(Real time factor) RTF = %.3f / %.3f = %.3f\n", elapsed_seconds,
         duration, rtf);

  return 0;
}
