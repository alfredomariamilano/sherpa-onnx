// sherpa-onnx/csrc/online-transducer-modified-beam-search-nemo-decoder.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_NEMO_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_NEMO_DECODER_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/online-transducer-nemo-model.h"

namespace sherpa_onnx {

class OnlineStream;

// Beam state that persists across streaming chunks for the NeMo transducer.
struct OnlineTransducerNeMoBeamSearchState {
  struct Hypothesis {
    // Decoded token IDs so far.
    std::vector<int64_t> ys;

    // Frame index (after subsampling) where each token was emitted.
    std::vector<int32_t> timestamps;

    // Log probability for each emitted token.
    std::vector<float> ys_probs;

    // Context-graph score for each emitted token.
    std::vector<float> context_scores;

    // Per-hypothesis decoder (LSTM) states. CopyableOrtValue deep-copies on
    // copy so hypotheses can be safely branched.
    std::vector<CopyableOrtValue> decoder_states;

    // Current context-graph state for hotword biasing.
    const ContextState *context_state = nullptr;

    // Accumulated log probability.
    float log_prob = 0.0f;

    // Absolute frame offset this hypothesis is currently decoding.
    int32_t frame_offset = 0;

    // Number of non-blank symbols emitted at the current frame.
    int32_t num_symbols = 0;

    // Number of trailing blank frames.
    int32_t num_trailing_blanks = 0;

    // Allocator used to clone decoder states.
    OrtAllocator *allocator = nullptr;

    Hypothesis() = default;
    Hypothesis(const Hypothesis &) = default;
    Hypothesis &operator=(const Hypothesis &) = default;
    Hypothesis(Hypothesis &&) = default;
    Hypothesis &operator=(Hypothesis &&) = default;
  };

  std::vector<Hypothesis> cur_hyps;
};

class OnlineTransducerModifiedBeamSearchNeMoDecoder {
 public:
  OnlineTransducerModifiedBeamSearchNeMoDecoder(OnlineTransducerNeMoModel *model,
                                                int32_t max_active_paths,
                                                int32_t unk_id,
                                                float blank_penalty,
                                                float hotwords_score)
      : model_(model),
        max_active_paths_(max_active_paths),
        unk_id_(unk_id),
        blank_penalty_(blank_penalty),
        hotwords_score_(hotwords_score) {}

  // @param encoder_out A 3-D tensor of shape (B, T, encoder_dim).
  // @param ss A list of OnlineStreams. n is the batch size.
  void Decode(Ort::Value encoder_out, OnlineStream **ss, int32_t n) const;

 private:
  OnlineTransducerNeMoModel *model_;  // Not owned
  int32_t max_active_paths_;
  int32_t unk_id_;
  float blank_penalty_;
  float hotwords_score_;
  int32_t max_symbols_per_frame_ = 10;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODIFIED_BEAM_SEARCH_NEMO_DECODER_H_
