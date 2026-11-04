// Sampler chain for the think35 protocol, in vLLM's order: presence penalty, temperature, top-k, top-p, draw.
//
// Presence penalty covers generated tokens only (vLLM v0.33.0, layers/utils.py apply_penalties uses output_mask).
// llama-server also feeds prompt tokens into the window, which would penalize \boxed from the first token.
// This chain sees a token only when the caller accepts it, so callers must never accept prompt tokens.
#pragma once

#include "llama.h"

#include <cstdint>

namespace protocol {

constexpr int32_t MAX_NEW_TOKENS = 32768;
constexpr float PRESENCE_PENALTY = 1.5f;
constexpr float TEMPERATURE = 0.6f;
constexpr int32_t TOP_K = 20;
constexpr float TOP_P = 0.95f;
constexpr size_t TOP_P_MIN_KEEP = 1;
constexpr float NO_REPEAT_PENALTY = 1.0f;
constexpr float NO_FREQUENCY_PENALTY = 0.0f;

// Without the final draw, so tests can inspect the distribution.
inline llama_sampler * make_filter_chain(int32_t n_vocab) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    // The window holds every generated token, as vLLM's output_mask does.
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(
        n_vocab, MAX_NEW_TOKENS, NO_REPEAT_PENALTY, NO_FREQUENCY_PENALTY, PRESENCE_PENALTY));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(TEMPERATURE));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(TOP_K));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(TOP_P, TOP_P_MIN_KEEP));
    return chain;
}

inline llama_sampler * make_chain(int32_t n_vocab, uint32_t seed) {
    llama_sampler * chain = make_filter_chain(n_vocab);
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    return chain;
}

} // namespace protocol
