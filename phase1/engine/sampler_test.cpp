// Checks the sampler chain against values computed by hand from vLLM's formulas. Needs no model.
#include "protocol_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace {

constexpr int32_t TEST_VOCAB = 100;
int failures = 0;

void expect(bool ok, const char * what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    failures += ok ? 0 : 1;
}

// Survivors of the filter chain, as {token: probability}.
std::map<llama_token, float> filter(llama_sampler * chain, const std::vector<float> & logits) {
    std::vector<llama_token_data> data;
    for (size_t i = 0; i < logits.size(); ++i) {
        data.push_back({(llama_token) i, logits[i], 0.0f});
    }
    llama_token_data_array arr = {data.data(), data.size(), -1, false};
    llama_sampler_apply(chain, &arr);

    std::map<llama_token, float> kept;
    float total = 0.0f;
    for (size_t i = 0; i < arr.size; ++i) {
        kept[arr.data[i].id] = arr.data[i].logit;
        total += std::exp(arr.data[i].logit);
    }
    for (auto & kv : kept) {
        kv.second = std::exp(kv.second) / total;
    }
    return kept;
}

bool near(float a, float b) { return std::fabs(a - b) < 1e-3f; }

void test_unaccepted_tokens_are_not_penalized() {
    llama_sampler * chain = protocol::make_filter_chain(TEST_VOCAB);
    auto p = filter(chain, {1.0f, 1.0f});
    expect(near(p[0], 0.5f) && near(p[1], 0.5f), "a token that was never accepted, such as a prompt token, has no penalty");
    llama_sampler_free(chain);
}

void test_penalty_applies_before_temperature() {
    llama_sampler * chain = protocol::make_filter_chain(TEST_VOCAB);
    llama_sampler_accept(chain, 0);
    // Logits {0, 0}: the penalty makes token 0 -1.5, temperature makes it -2.5, so odds are exp(-2.5).
    auto p = filter(chain, {0.0f, 0.0f});
    const float expected = std::exp(-2.5f) / (1.0f + std::exp(-2.5f));
    expect(near(p[0], expected), "an accepted token loses 1.5 logits before temperature 0.6");
    llama_sampler_free(chain);
}

void test_repeat_count_does_not_matter() {
    llama_sampler * chain = protocol::make_filter_chain(TEST_VOCAB);
    for (int i = 0; i < 5; ++i) {
        llama_sampler_accept(chain, 0);
    }
    auto p = filter(chain, {0.0f, 0.0f});
    const float expected = std::exp(-2.5f) / (1.0f + std::exp(-2.5f));
    expect(near(p[0], expected), "presence penalty does not grow with repeats");
    llama_sampler_free(chain);
}

void test_top_k_removes_tokens_outside_the_top_twenty() {
    llama_sampler * chain = protocol::make_filter_chain(TEST_VOCAB);
    // Slowly falling logits, so only top-k can remove tokens 20 and up. Of any 20 survivors the largest 19 hold at
    // least 95%, so top-p 0.95 then drops the smallest. vLLM's cumulative rule gives the same 19.
    std::vector<float> logits;
    for (int i = 0; i < 40; ++i) {
        logits.push_back(5.0f - 0.001f * (float) i);
    }
    auto p = filter(chain, logits);
    expect(p.size() == 19, "top-k then top-p keeps 19 of 40 near-equal tokens");
    expect(p.rbegin()->first == 18, "the survivors are the largest tokens");
    llama_sampler_free(chain);
}

void test_top_p_uses_tempered_probabilities() {
    llama_sampler * chain = protocol::make_filter_chain(TEST_VOCAB);
    // With logits {3, 0, -1}, the first token has 0.992 after temperature 0.6, so it alone passes 0.95.
    // Without temperature it has 0.934 and would not.
    auto p = filter(chain, {3.0f, 0.0f, -1.0f});
    expect(p.size() == 1 && p.count(0) == 1, "top-p runs after temperature");
    llama_sampler_free(chain);
}

void test_top_p_keeps_the_crossing_token() {
    llama_sampler * chain = protocol::make_filter_chain(TEST_VOCAB);
    // Equal tempered probabilities of 1/3: vLLM removes a token only while the ascending sum stays at or below 0.05.
    auto p = filter(chain, {1.0f, 1.0f, 1.0f});
    expect(p.size() == 3, "top-p keeps every token needed to reach 0.95");
    llama_sampler_free(chain);
}

} // namespace

int main() {
    test_unaccepted_tokens_are_not_penalized();
    test_penalty_applies_before_temperature();
    test_repeat_count_does_not_matter();
    test_top_k_removes_tokens_outside_the_top_twenty();
    test_top_p_uses_tempered_probabilities();
    test_top_p_keeps_the_crossing_token();
    std::printf("%d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
