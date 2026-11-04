// Write the logits after the last token, for comparison with the reference implementation.
//
//   logits <model.gguf> <ids.txt> <out.f32> <threads> <n_batched> [n_ctx chunk]
//
// The first n_batched tokens are decoded in chunks with logits only after the last one, and the rest one at a time with
// logits each time, as generation does. By default they are one chunk, in a context just big enough for the input.
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 6 && argc != 8) {
        std::fprintf(stderr, "usage: %s <model.gguf> <ids.txt> <out.f32> <threads> <n_batched> [n_ctx chunk]\n", argv[0]);
        return 2;
    }
    std::setlocale(LC_NUMERIC, "C");
    std::vector<llama_token> ids;
    {
        std::ifstream in(argv[2]);
        llama_token id;
        while (in >> id) {
            ids.push_back(id);
        }
    }
    const int threads = std::atoi(argv[4]);
    const size_t n_batched = std::max(std::atoi(argv[5]), 1);
    const uint32_t n_ctx = argc == 8 ? (uint32_t) std::atoi(argv[6]) : (uint32_t) ids.size() + 16;
    const uint32_t chunk = argc == 8 ? (uint32_t) std::atoi(argv[7]) : n_ctx;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    // GPU port: RSI_GPU_LAYERS offloads layers (e.g. 99 = all); unset keeps the verified CPU path.
    mp.n_gpu_layers = std::getenv("RSI_GPU_LAYERS") ? std::atoi(std::getenv("RSI_GPU_LAYERS")) : 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = chunk;
    cp.n_ubatch = chunk;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // as in generate.cpp
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        return 1;
    }
    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    size_t done = 0;
    while (done < ids.size()) {
        const size_t count = done < n_batched ? std::min<size_t>({chunk, n_batched - done, ids.size() - done}) : 1;
        llama_batch_ext_clear(batch);
        for (size_t i = done; i < done + count; ++i) {
            const int32_t idx = llama_batch_ext_add_token(batch, 0, ids[i]);
            const llama_pos pos = (llama_pos) i;
            llama_batch_ext_set_pos(batch, idx, &pos);
        }
        done += count;
        llama_batch_ext_set_output_logits(batch, (int32_t) count - 1, done >= std::min(n_batched, ids.size()));
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
            std::fprintf(stderr, "decode failed at %zu\n", done);
            return 1;
        }
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, -1);
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char *>(logits), n_vocab * sizeof(float));

    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
