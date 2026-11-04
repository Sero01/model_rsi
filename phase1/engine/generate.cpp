// Generate one sample with the think35 protocol, resumable at token level.
//
//   generate <model.gguf> <prompt_ids.txt> <tokens.txt> <seed> <threads> [max_new_tokens]
//
// tokens.txt receives one generated token id per line. If it already has tokens, they are replayed through the model
// and the penalty window, and generation continues. On success the last stdout line is JSON with the finish reason.
#include "protocol_sampler.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#endif

namespace {

// Without flash attention, each batched token reserves attention scores over the whole context, so keep batches small.
constexpr int32_t BATCH_SIZE = 32;
constexpr int32_t PROGRESS_EVERY = 1000;
constexpr int EXIT_FILE_IN_USE = 3;
// vLLM stops at the tokenizer's eos and at generation_config's eos.
constexpr llama_token STOP_TOKENS[] = {248046 /* <|im_end|> */, 248044 /* <|endoftext|> */};

std::vector<llama_token> read_ids(const std::string & path) {
    std::vector<llama_token> ids;
    std::ifstream in(path);
    llama_token id;
    while (in >> id) {
        ids.push_back(id);
    }
    return ids;
}

bool is_stop(llama_token id) {
    for (llama_token stop : STOP_TOKENS) {
        if (id == stop) {
            return true;
        }
    }
    return false;
}

// Decode tokens starting at pos, asking for logits only after the last one.
bool decode(llama_context * ctx, llama_batch_ext * batch, const std::vector<llama_token> & tokens, llama_pos pos) {
    for (size_t start = 0; start < tokens.size(); start += BATCH_SIZE) {
        const size_t end = std::min(tokens.size(), start + BATCH_SIZE);
        llama_batch_ext_clear(batch);
        for (size_t i = start; i < end; ++i) {
            const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
            const llama_pos p = pos + (llama_pos) i;
            llama_batch_ext_set_pos(batch, idx, &p);
        }
        llama_batch_ext_set_output_logits(batch, (int32_t) (end - start) - 1, end == tokens.size());
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s <model.gguf> <prompt_ids.txt> <tokens.txt> <seed> <threads> [max_new_tokens]\n", argv[0]);
        return 2;
    }
    std::setlocale(LC_NUMERIC, "C");
    const std::string model_path = argv[1], prompt_path = argv[2], tokens_path = argv[3];
    const uint32_t seed = (uint32_t) std::strtoul(argv[4], nullptr, 10);
    const int32_t threads = std::atoi(argv[5]);
    const int32_t max_new = argc > 6 ? std::atoi(argv[6]) : protocol::MAX_NEW_TOKENS;

    // Two engines appending to one file would interleave tokens without any error. The lock lasts until this process exits.
#ifdef _WIN32
    // Windows port: Windows locks are mandatory, so lock one byte far past EOF; ordinary appends are never blocked.
    HANDLE lock_h = CreateFileA(tokens_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    OVERLAPPED lock_ov{};
    lock_ov.OffsetHigh = 0x7FFFFFFF;
    if (lock_h == INVALID_HANDLE_VALUE ||
        !LockFileEx(lock_h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &lock_ov)) {
#else
    const int lock_fd = open(tokens_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
#endif
        std::fprintf(stderr, "another engine holds %s\n", tokens_path.c_str());
        return EXIT_FILE_IN_USE;
    }

    const std::vector<llama_token> prompt = read_ids(prompt_path);
    std::vector<llama_token> generated = read_ids(tokens_path);
    if (prompt.empty()) {
        std::fprintf(stderr, "empty prompt: %s\n", prompt_path.c_str());
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    // GPU port: RSI_GPU_LAYERS offloads layers (e.g. 99 = all); unset keeps the verified CPU path.
    mp.n_gpu_layers = std::getenv("RSI_GPU_LAYERS") ? std::atoi(std::getenv("RSI_GPU_LAYERS")) : 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (model == nullptr) {
        std::fprintf(stderr, "cannot load %s\n", model_path.c_str());
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) prompt.size() + (uint32_t) max_new;
    cp.n_batch = BATCH_SIZE;
    cp.n_ubatch = BATCH_SIZE;
    // Flash attention moves top log-probs by up to 0.03 against transformers. Without it they agree to 0.00002, at the same speed.
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        std::fprintf(stderr, "cannot create context\n");
        return 1;
    }
    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    // A replayed sample continues on its own random stream. Exact samples are not reproducible across engines anyway.
    llama_sampler * chain = protocol::make_chain(llama_vocab_n_tokens(llama_model_get_vocab(model)), seed + (uint32_t) generated.size());

    const int64_t t_start = llama_time_us();
    std::vector<llama_token> context = prompt;
    context.insert(context.end(), generated.begin(), generated.end());
    for (llama_token id : generated) {
        llama_sampler_accept(chain, id);
    }
    const bool already_stopped = !generated.empty() && is_stop(generated.back());
    if (!already_stopped && (int32_t) generated.size() < max_new && !decode(ctx, batch, context, 0)) {
        std::fprintf(stderr, "decode failed while replaying\n");
        return 1;
    }
    const size_t n_replayed = generated.size();
    const int64_t t_gen = llama_time_us();
    std::fprintf(stderr, "prompt %zu, replayed %zu, %.1f s\n", prompt.size(), n_replayed, (t_gen - t_start) / 1e6);

    std::ofstream out(tokens_path, std::ios::app | std::ios::binary);  // binary: plain \n line ends, as on Linux
    int64_t t_window = t_gen;
    std::vector<llama_token> one(1);
    bool stopped = already_stopped;
    while (!stopped && (int32_t) generated.size() < max_new) {
        // llama_sampler_sample accepts the token into the chain, so the penalty window stays current.
        const llama_token id = llama_sampler_sample(chain, ctx, -1);
        generated.push_back(id);
        out << id << '\n' << std::flush;
        if (!out) {
            std::fprintf(stderr, "cannot write %s\n", tokens_path.c_str());
            return 1;
        }
        stopped = is_stop(id);
        if (generated.size() % PROGRESS_EVERY == 0) {
            const int64_t now = llama_time_us();
            std::fprintf(stderr, "ctx %zu: %.2f tok/s\n", prompt.size() + generated.size(), PROGRESS_EVERY / ((now - t_window) / 1e6));
            t_window = now;
        }
        if (!stopped && (int32_t) generated.size() < max_new) {
            one[0] = id;
            if (!decode(ctx, batch, one, (llama_pos) (prompt.size() + generated.size() - 1))) {
                std::fprintf(stderr, "decode failed at %zu\n", generated.size());
                return 1;
            }
        }
    }

    const double seconds = (llama_time_us() - t_gen) / 1e6;
    std::printf("{\"finish\": \"%s\", \"tokens\": %zu, \"new_tokens\": %zu, \"seconds\": %.1f}\n",
                stopped ? "stop" : "length", generated.size(), generated.size() - n_replayed, seconds);

    llama_batch_ext_free(batch);
    llama_sampler_free(chain);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
