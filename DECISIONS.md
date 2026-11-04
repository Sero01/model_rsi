# Decisions

## Phase 1, 2026-11-04

The user delegated the handover's open questions ("Take all decisions optimally"). Evidence for each is in `RESULTS.md`.

### Local run design

| Question | Decision | Why |
|---|---|---|
| Base, edited or both | Both, edited first within each problem | The contrast between them is the strongest check on the pipeline. Edited runs first in each pair because it is shorter and is the claim under test, so a run stopped mid-pair loses the longer base sample. |
| Samples per problem | 1 | 4 samples would take about 21 days. The sample tests the pipeline and cannot confirm the published figures at any sample count we could afford. |
| Token cap | Keep 32,768 | A lower cut would turn slow correct answers into failures and break the comparison with published cut share and accuracy. |
| Thinking | On | User decision in the handover. |
| Threads | 4 | Measured over 120 tokens: 4 threads 41 s, 6 threads 42 s, 8 threads 82 s. The laptop has 4 physical cores. |

### Engine

- **llama.cpp b12099 (`bee2332`) as a library, with our own driver.** vLLM does not support the MX250. llama.cpp's `qwen35` architecture implements the model, and its sampler chain can match the protocol.
- **Not `llama-server`.** It feeds prompt tokens into the penalty window (`server-context.cpp:485`). vLLM penalizes output tokens only (`layers/utils.py`, `output_mask`), so the server would penalize tokens such as `\boxed` that appear in the prompt.
- **Sampler order:** presence penalty, temperature, top-k, top-p, then draw, matching vLLM v0.33.0. Stop tokens are 248046 and 248044.
- **F32 GGUF, no quantization.** F32 represents every BF16 weight exactly. `verify_gguf.py` checks that the edited tensors cast back to MIT's hashes.
- **Our own GGUF writer, `phase1/make_gguf.py`.** The auto-mode classifier refused to run the clone's converter, including inside the sandbox. We read the converter and wrote the same mapping on PyPI `gguf`. The tensor hashes and logit comparison verify the result.
- **Flash attention off.** With it on, top log-probs differ from `transformers` by up to 0.03 when decoding one token at a time. With it off they agree to 0.00002 at the same speed and memory. We isolated the error to the flash path but did not trace it inside ggml.
- **Batch size 32.** Without flash attention, each batched token reserves attention scores over the whole context. At 32 the peak is 3.7 GB. The only cost is slower replay on resume.

### Known differences from MIT's run

- **Arithmetic.** vLLM evaluated in BF16 arithmetic. We use F32, and the CPU has no AVX512-BF16. On one 2,000-token input, `transformers` in BF16 and F32 differ by up to 0.078 in top log-probs, with the same top 20. So our distributions match F32 `transformers`, not MIT's exact numbers.
- **Samples.** Random streams differ, so bit-exact samples were never expected.
- **Sample count.** We take 1 sample per problem. MIT took 4.

### Publishing, 2026-11-04

The user asked for a public repo. These stay out of it:

- **`results/phase1/problems.jsonl`:** it holds MATH-500 problem text, and MIT's license says third-party benchmark content stays under its own license, which we have not confirmed. `prepare_problems.py` regenerates the file byte for byte.
- **Engine logs and `prompt.txt`:** the logs contain home directory paths, and the prompts are the same problem text as token ids. `run_local.py` rewrites `prompt.txt` on each run.
- **Personal identity:** commits use the `Sero` name and the GitHub noreply address, set per repo.

Vendored code in `phase1/vendor/` keeps MIT's notice in `phase1/vendor/LICENSE`. The top-level `README.md` gives the CC BY 4.0 attribution for published figures. The repo itself has no license yet, which is the user's call.

The plan and handover are public too. This adds nothing to what MIT already published, but Phase 2 editor runs must have no network access, or a search could reach the answer.

### Sandbox

- **Landlock plus seccomp, `phase1/sandbox.py`, instead of bubblewrap.** bubblewrap cannot run here, because `apparmor_restrict_unprivileged_userns=1` and there is no bwrap profile. We did not change system settings.
- **For Phase 2:** its proposed isolation relies on bubblewrap. That design must be revisited for this laptop, or run elsewhere.

### Grader

- **`mathqa.py` vendored verbatim, with a stub `mcqa.py`.** `mcqa` is unpublished and only loop diagnostics use it.
- **Grader check covers 200 texts and 500 `pred_first` values per checkpoint.** `evaluate.py:197` keeps 20 texts per shard, despite the README saying every problem.
- **Extraction behaviour.** `mathqa.extract` falls back to the last `\boxed{}` in the thought when there is no `</think>`. A sample cut at the cap can therefore still score. We keep this because it is what produced the published scores.

## Phase 1 on Windows and a local GPU, 2026-11-04

A collaborator ran Phase 1 on a new machine: Windows 11 with a Core Ultra 7 255H, 23 GB RAM and an RTX 5050 Laptop GPU with 8 GB. They asked for it to run natively on Windows. The commands are in `phase1/WINDOWS.md`.

| Question | Decision | Why |
|---|---|---|
| WSL or native | Native | WSL was not installed, and the user asked for Windows. The changes are small and guarded by platform checks, so the Linux path is unchanged. |
| Isolation | None on Windows | Landlock and seccomp are Linux-only. The engine is llama.cpp at the pinned official tag. Phase 2 editor isolation still needs its own design. |
| Grader timeouts | Off on Windows | Windows has no `signal.alarm`, and math-verify's fallback crashes under `-I`. This is the same no-timeout path `mathqa.py` already uses off the main thread. `check_grader.py` finds 0 disagreements. |
| CPU or GPU | GPU, with all layers offloaded and `NVIDIA_TF32_OVERRIDE=0` | GPU decode measured 73 tok/s against 9.5 for the best CPU setting. The run is local, so it costs nothing and needs no rental approval. With TF32 off, logits match `transformers` to 0.00006, so it is held to the same verification bar as the CPU engine. |
| One or two GPU processes | One | One engine takes about 4.1 GB at full context, and the card has 8 GB. |
| CPU threads | 4 | Measured over 300 tokens: 4 threads 9.46 tok/s, 8 threads 8.17, 12 threads 6.90, 16 threads 6.19. |
| Windows CPU run | Stopped after the GPU engine was verified | It was redundant. Its partial output is kept in `results/phase1/local_windows/`. |

A local GPU is still not the H200 and vLLM setup, so these scores do not count toward the exit gate. The GPU and CPU engines are numerically equivalent to F32 `transformers`, not to MIT's BF16 arithmetic.
