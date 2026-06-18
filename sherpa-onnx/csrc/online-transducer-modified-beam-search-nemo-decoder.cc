// sherpa-onnx/csrc/online-transducer-modified-beam-search-nemo-decoder.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-transducer-modified-beam-search-nemo-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

void OnlineTransducerModifiedBeamSearchNeMoDecoder::Decode(
    Ort::Value encoder_out, OnlineStream **ss, int32_t n) const {
  auto shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t num_frames = static_cast<int32_t>(shape[1]);
  int32_t encoder_dim = static_cast<int32_t>(shape[2]);

  if (batch_size != n) {
    SHERPA_ONNX_LOGE(
        "Size mismatch! encoder_out.size(0) %d != number of streams %d",
        batch_size, n);
    SHERPA_ONNX_EXIT(-1);
  }

  int32_t vocab_size = model_->VocabSize();
  int32_t blank_id = vocab_size - 1;  // NeMo models use blank as the last id

  OrtAllocator *allocator = model_->Allocator();
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  const float *encoder_data = encoder_out.GetTensorData<float>();

  for (int32_t b = 0; b != batch_size; ++b) {
    OnlineStream *stream = ss[b];

    auto state = stream->GetNeMoBeamSearchState();
    if (!state) {
      state = std::make_shared<OnlineTransducerNeMoBeamSearchState>();
      stream->SetNeMoBeamSearchState(state);
    }

    auto &cur_hyps = state->cur_hyps;

    int32_t start_frame = stream->GetResult().frame_offset;
    int32_t end_frame = start_frame + num_frames;

    // Initialize the beam on the first chunk.
    if (cur_hyps.empty()) {
      OnlineTransducerNeMoBeamSearchState::Hypothesis blank_hyp;
      blank_hyp.log_prob = 0.0f;
      blank_hyp.frame_offset = start_frame;
      blank_hyp.num_symbols = 0;
      blank_hyp.num_trailing_blanks = 0;
      blank_hyp.allocator = allocator;
      blank_hyp.decoder_states = Convert(model_->GetDecoderInitStates());

      auto context_graph = stream->GetContextGraph();
      if (context_graph != nullptr) {
        blank_hyp.context_state = context_graph->Root();
      }

      cur_hyps.push_back(std::move(blank_hyp));
    }

    const float *this_encoder =
        encoder_data + b * num_frames * encoder_dim;

    // Frame-synchronous beam expansion. Hypotheses may be at different absolute
    // frame offsets, so we always expand the one(s) at the smallest offset.
    while (true) {
      int32_t min_frame = end_frame;
      for (const auto &hyp : cur_hyps) {
        if (hyp.frame_offset < min_frame) {
          min_frame = hyp.frame_offset;
        }
      }

      if (min_frame >= end_frame) {
        break;  // all hypotheses have consumed the current chunk
      }

      std::vector<std::pair<float, OnlineTransducerNeMoBeamSearchState::Hypothesis>>
          all_candidates;

      for (const auto &hyp : cur_hyps) {
        if (hyp.frame_offset > min_frame) {
          // This hypothesis is already ahead; keep it for the next frame.
          all_candidates.emplace_back(hyp.log_prob, hyp);
          continue;
        }

        if (hyp.num_symbols >= max_symbols_per_frame_) {
          // Force a blank to advance the frame.
          auto new_hyp = hyp;
          new_hyp.num_symbols = 0;
          new_hyp.num_trailing_blanks += 1;
          new_hyp.frame_offset += 1;
          all_candidates.emplace_back(new_hyp.log_prob, std::move(new_hyp));
          continue;
        }

        // Decoder input: the last emitted token, or blank for an empty history.
        int32_t last_token = hyp.ys.empty()
                                 ? blank_id
                                 : static_cast<int32_t>(hyp.ys.back());

        std::array<int64_t, 2> decoder_input_shape{1, 1};
        std::vector<int32_t> decoder_input_data{last_token};

        Ort::Value decoder_input = Ort::Value::CreateTensor<int32_t>(
            memory_info, decoder_input_data.data(), 1,
            decoder_input_shape.data(), decoder_input_shape.size());

        auto decoder_states = Convert(hyp.decoder_states);
        auto decoder_result =
            model_->RunDecoder(std::move(decoder_input),
                               std::move(decoder_states));

        // Encoder output for the current frame.
        std::array<int64_t, 3> encoder_frame_shape{1, encoder_dim, 1};
        const float *frame_data =
            this_encoder + (min_frame - start_frame) * encoder_dim;

        Ort::Value encoder_frame = Ort::Value::CreateTensor(
            memory_info, const_cast<float *>(frame_data), encoder_dim,
            encoder_frame_shape.data(), encoder_frame_shape.size());

        Ort::Value logit = model_->RunJoiner(View(&encoder_frame),
                                             View(&decoder_result.first));

        float *p_logit = logit.GetTensorMutableData<float>();

        if (blank_penalty_ > 0.0f) {
          p_logit[blank_id] -= blank_penalty_;
        }

        LogSoftmax(p_logit, vocab_size, 1);

        // Context boosting: prefer tokens that continue a hotword prefix.
        if (hyp.context_state != nullptr) {
          for (const auto &pair : hyp.context_state->next) {
            int32_t token_id = pair.first;
            if (token_id >= 0 && token_id < vocab_size) {
              p_logit[token_id] += hotwords_score_;
            }
          }
        }

        auto top_k_tokens = TopkIndex(p_logit, vocab_size, max_active_paths_);

        // Decoder states for a non-blank emission are the same for every
        // candidate token, so convert them once and copy per candidate.
        auto next_decoder_states = Convert(std::move(decoder_result.second));

        for (int32_t token : top_k_tokens) {
          auto new_hyp = hyp;

          if (token == blank_id || token == unk_id_) {
            // Blank / unk: decoder state does not change, advance one frame.
            new_hyp.num_trailing_blanks = hyp.num_trailing_blanks + 1;
            new_hyp.frame_offset = hyp.frame_offset + 1;
            new_hyp.num_symbols = 0;
            new_hyp.log_prob = p_logit[token] + hyp.log_prob;
          } else {
            // Non-blank: extend the hypothesis and stay on the same frame.
            new_hyp.ys.push_back(token);
            new_hyp.timestamps.push_back(min_frame);
            new_hyp.ys_probs.push_back(p_logit[token]);
            new_hyp.decoder_states = next_decoder_states;
            new_hyp.frame_offset = hyp.frame_offset;
            new_hyp.num_symbols = hyp.num_symbols + 1;
            new_hyp.num_trailing_blanks = 0;

            float context_score = 0.0f;
            auto context_graph = stream->GetContextGraph();
            if (context_graph != nullptr) {
              auto context_res = context_graph->ForwardOneStep(
                  new_hyp.context_state, token, false /*strict mode*/);
              context_score = std::get<0>(context_res);
              new_hyp.context_state = std::get<1>(context_res);
            }

            new_hyp.log_prob =
                p_logit[token] + hyp.log_prob + context_score;
            new_hyp.context_scores.push_back(context_score);
          }

          all_candidates.emplace_back(new_hyp.log_prob, std::move(new_hyp));
        }  // for (int32_t token : top_k_tokens)
      }    // for (const auto &hyp : cur_hyps)

      if (all_candidates.empty()) {
        break;
      }

      int32_t keep = std::min(
          max_active_paths_, static_cast<int32_t>(all_candidates.size()));

      std::partial_sort(
          all_candidates.begin(),
          all_candidates.begin() + keep, all_candidates.end(),
          [](const auto &a, const auto &b) { return a.first > b.first; });

      cur_hyps.clear();
      cur_hyps.reserve(keep);
      for (int32_t k = 0; k != keep; ++k) {
        cur_hyps.push_back(std::move(all_candidates[k].second));
      }
    }  // while (true)

    // Commit the most probable hypothesis to the stream result.
    auto best_it =
        std::max_element(cur_hyps.begin(), cur_hyps.end(),
                         [](const auto &a, const auto &b) {
                           return a.log_prob < b.log_prob;
                         });

    auto &result = stream->GetResult();
    if (best_it != cur_hyps.end()) {
      result.tokens = best_it->ys;
      result.timestamps = best_it->timestamps;
      result.ys_probs = best_it->ys_probs;
      result.context_scores = best_it->context_scores;
      result.num_trailing_blanks = best_it->num_trailing_blanks;
    }

    result.frame_offset += num_frames;
  }  // for (int32_t b = 0; b != batch_size; ++b)
}

}  // namespace sherpa_onnx
