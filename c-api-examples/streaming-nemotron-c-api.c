// c-api-examples/streaming-nemotron-c-api.c
//
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates how to use streaming Nemotron with sherpa-onnx's C
// API.
// clang-format off
//
// wget https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/resolve/main/encoder.int8.onnx
// wget https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/resolve/main/decoder.int8.onnx
// wget https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/resolve/main/joiner.int8.onnx
// wget https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/resolve/main/tokens.txt
// wget https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/resolve/main/test_wavs/en.wav
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main(int32_t argc, char *argv[]) {
  const char *wav_filename =
      "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/"
      "test_wavs/en.wav";
  const char *encoder_filename =
      "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/"
      "encoder.int8.onnx";
  const char *decoder_filename =
      "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/"
      "decoder.int8.onnx";
  const char *joiner_filename =
      "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/"
      "joiner.int8.onnx";
  const char *tokens_filename =
      "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11/"
      "tokens.txt";
  const char *provider = "cpu";

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  // Optional hotwords file. Pass it as the first command-line argument, e.g.
  // ./streaming-nemotron-c-api hotwords.txt [bpe_vocab]
  const char *hotwords_file = "";
  const char *bpe_vocab = "";
  if (argc >= 2) {
    hotwords_file = argv[1];
  }
  if (argc >= 3) {
    bpe_vocab = argv[2];
  }

  // Recognizer config
  SherpaOnnxOnlineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.decoding_method = "modified_beam_search";
  recognizer_config.max_active_paths = 4;
  recognizer_config.hotwords_file = hotwords_file;
  recognizer_config.hotwords_score = 1.5f;
  recognizer_config.model_config.debug = 1;
  recognizer_config.model_config.num_threads = 1;
  recognizer_config.model_config.provider = provider;
  recognizer_config.model_config.tokens = tokens_filename;
  recognizer_config.model_config.transducer.encoder = encoder_filename;
  recognizer_config.model_config.transducer.decoder = decoder_filename;
  recognizer_config.model_config.transducer.joiner = joiner_filename;
  if (strlen(bpe_vocab)) {
    recognizer_config.model_config.modeling_unit = "bpe";
    recognizer_config.model_config.bpe_vocab = bpe_vocab;
  }
  recognizer_config.enable_endpoint = 1;

  fprintf(stderr, "decoding method: %s\n", recognizer_config.decoding_method);
  fprintf(stderr, "max active paths: %d\n", recognizer_config.max_active_paths);
  fprintf(stderr, "hotwords file: %s\n",
          strlen(hotwords_file) ? hotwords_file : "(none)");
  fprintf(stderr, "hotwords score: %.2f\n", recognizer_config.hotwords_score);
  fprintf(stderr, "bpe vocab: %s\n",
          strlen(bpe_vocab) ? bpe_vocab : "(none)");

  const SherpaOnnxOnlineRecognizer *recognizer =
      SherpaOnnxCreateOnlineRecognizer(&recognizer_config);

  if (recognizer == NULL) {
    fprintf(stderr, "Please check your config!\n");
    SherpaOnnxFreeWave(wave);
    return -1;
  }

  const SherpaOnnxOnlineStream *stream =
      SherpaOnnxCreateOnlineStream(recognizer);
  // Multilingual Nemotron models use the generic stream option "language".
  // For example: SherpaOnnxOnlineStreamSetOption(stream, "language", "ja");
  // Empty/unset means auto. English-only Nemotron ignores this option.
  SherpaOnnxOnlineStreamSetOption(stream, "language", "en");

  const SherpaOnnxDisplay *display = SherpaOnnxCreateDisplay(50);
  int32_t segment_id = 0;

// simulate streaming. You can choose an arbitrary N
#define N 3200

  fprintf(stderr, "sample rate: %d, num samples: %d, duration: %.2f s\n",
          wave->sample_rate, wave->num_samples,
          (float)wave->num_samples / wave->sample_rate);

  int32_t k = 0;
  while (k < wave->num_samples) {
    int32_t start = k;
    int32_t end =
        (start + N > wave->num_samples) ? wave->num_samples : (start + N);
    k += N;

    SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate,
                                         wave->samples + start, end - start);
    while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
      SherpaOnnxDecodeOnlineStream(recognizer, stream);
    }

    const SherpaOnnxOnlineRecognizerResult *r =
        SherpaOnnxGetOnlineStreamResult(recognizer, stream);

    if (strlen(r->text)) {
      SherpaOnnxPrint(display, segment_id, r->text);
    }

    if (SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream)) {
      if (strlen(r->text)) {
        ++segment_id;
      }
      SherpaOnnxOnlineStreamReset(recognizer, stream);
    }

    SherpaOnnxDestroyOnlineRecognizerResult(r);
  }

  // add some tail padding
  float tail_paddings[4800] = {0};  // 0.3 seconds at 16 kHz sample rate
  SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate, tail_paddings,
                                       4800);

  SherpaOnnxFreeWave(wave);

  SherpaOnnxOnlineStreamInputFinished(stream);
  while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
    SherpaOnnxDecodeOnlineStream(recognizer, stream);
  }

  const SherpaOnnxOnlineRecognizerResult *r =
      SherpaOnnxGetOnlineStreamResult(recognizer, stream);

  if (strlen(r->text)) {
    SherpaOnnxPrint(display, segment_id, r->text);
  }

  SherpaOnnxDestroyOnlineRecognizerResult(r);

  SherpaOnnxDestroyDisplay(display);
  SherpaOnnxDestroyOnlineStream(stream);
  SherpaOnnxDestroyOnlineRecognizer(recognizer);
  fprintf(stderr, "\n");

  return 0;
}
