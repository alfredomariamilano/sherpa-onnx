// cxx-api-examples/streaming-nemotron-cxx-api.cc
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates how to use streaming Nemotron
// with sherpa-onnx's C++ API.
//
// clang-format off
//
// wget https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/resolve/main/encoder.int8.onnx
// wget https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/resolve/main/decoder.int8.onnx
// wget https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/resolve/main/joiner.int8.onnx
// wget https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/resolve/main/tokens.txt
// wget https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/resolve/main/test_wavs/en.wav
//
// clang-format on

#include <chrono>  // NOLINT
#include <cstdio>
#include <iostream>
#include <string>

#include "sherpa-onnx/c-api/cxx-api.h"

int32_t main(int32_t argc, char *argv[]) {
  using namespace sherpa_onnx::cxx;  // NOLINT
  OnlineRecognizerConfig config;

  // Optional hotwords file. Pass it as the first argument, e.g.
  // ./streaming-nemotron-cxx-api hotwords.txt
  std::string hotwords_file;
  if (argc >= 2) {
    hotwords_file = argv[1];
  }

  config.decoding_method = "modified_beam_search";
  config.max_active_paths = 4;
  config.hotwords_file = hotwords_file;
  config.hotwords_score = 1.5f;

  config.model_config.transducer.encoder =
      "./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/"
      "encoder.int8.onnx";

  config.model_config.transducer.decoder =
      "./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/"
      "decoder.int8.onnx";

  config.model_config.transducer.joiner =
      "./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/"
      "joiner.int8.onnx";

  config.model_config.tokens =
      "./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/"
      "tokens.txt";

  config.model_config.num_threads = 1;

  std::cout << "Loading model\n";
  OnlineRecognizer recognizer = OnlineRecognizer::Create(config);
  if (!recognizer.Get()) {
    std::cerr << "Please check your config\n";
    return -1;
  }
  std::cout << "Loading model done\n";

  std::string wave_filename =
      "./sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/"
      "test_wavs/en.wav";
  Wave wave = ReadWave(wave_filename);
  if (wave.samples.empty()) {
    std::cerr << "Failed to read: '" << wave_filename << "'\n";
    return -1;
  }

  std::cout << "Start recognition\n";
  const auto begin = std::chrono::steady_clock::now();

  OnlineStream stream = recognizer.CreateStream();
  // Multilingual Nemotron models use the generic stream option "language".
  // For example: stream.SetOption("language", "ja");
  // Empty/unset means auto. English-only Nemotron ignores this option.
  stream.SetOption("language", "en");
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
