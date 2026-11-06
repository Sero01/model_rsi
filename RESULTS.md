# Results

## Phase 1, 2026-11-04

**Status:** not complete. The user stopped the local run early, so there are no scores. The exit gate needs full GPU runs, which have not been approved.

### Verified

| Check | Result |
|---|---|
| Edit rebuilt from `edit.npz` | SHA-256 `fd1e42ac…737b553`, identical to the LFS hash of the published full checkpoints. All 488 tensors compared, and only the 2 manifest tensors differ. |
| Grader | No disagreements on 200 published texts and 500 `pred_first` values per checkpoint. |
| GGUFs | 320 tensors compared. Only `blk.15.ffn_down.weight` and `blk.19.ffn_down.weight` differ, by 65,260 and 64,928 entries. Cast to BF16, they match MIT's base and edited hashes. |
| Logits against F32 `transformers` | Top-20 log-prob error at most 0.00002, on both models, on 73- and 2,100-token inputs, batched, stepped and in `generate`'s configuration. Top 20 identical. |
| Logits at long context, base model | At most 0.00055 out to 32,000 tokens. See below. |
| Engine, sandbox and sampler tests | 4 of 4, 13 of 13 and 7 of 7 pass. |

### Engine error, resolved

With flash attention on, stepped decoding differed from `transformers` by a top-20 log-prob error that levelled off, so it does not grow with context:

| Tokens decoded one at a time | 0 | 1 | 10 | 50 | 200 | 500 | 1,000 | 1,950 |
|---|---|---|---|---|---|---|---|---|
| Error | 0.00081 | 0.00773 | 0.01855 | 0.02510 | 0.02851 | 0.02739 | 0.02793 | 0.02637 |

With flash attention off, the error at 0, 1 and 50 stepped tokens is 0.00001, 0.00001 and 0.00002. For scale, BF16 against F32 arithmetic in `transformers` differs by 0.07774 on the same 2,000-token input.

### Logits at long context

Base model, in `generate`'s context and chunk size, on a published transcript, against `transformers` in cached chunks of the same size:

| Context tokens | 4,096 | 8,000 | 16,000 | 32,000 |
|---|---|---|---|---|
| Top-20 log-prob error | 0.00002 | 0.00003 | 0.00011 | 0.00055 |

The top 20 and the top token are the same at every length. The error grows with context, faster than linearly at the end. We did not trace the cause, but it stays under the 0.001 limit and is about 140 times smaller than the 0.078 gap between BF16 and F32 arithmetic. One transcript was tested. The edited model was not run at this length, because it shares the engine path and differs only in two tensors.

### Speed and memory

4-core laptop CPU with 7.4 GB RAM, base model, 4 threads, nice 10, flash attention off, batch 32, over 100 new tokens at each context length:

| Context tokens | tok/s | Seconds per token | Peak RSS |
|---|---|---|---|
| 1,000 | 2.96 | 0.34 | 2.95 GB |
| 16,000 | 2.37 | 0.42 | 3.31 GB |
| 32,000 | 1.82 | 0.55 | 3.70 GB |

These are short measurements, so sustained heat may lower them.

Replay on resume took 1,012 s for 32,000 tokens at batch 32, against 754 s at batch 512 with flash attention on.

### Projected time

One sample that reaches the cap takes about 3.95 hours. The projection below combines measured speed with the published cut shares and mean tokens:

| 20 problems at 1 sample | Hours |
|---|---|
| Edited | about 50 |
| Base | about 74 |
| Both | about 124, or 5.2 days |

Four samples per problem would take about 21 days.

### What the run can show

Published scores on the same 20 problems, over 4 samples each, are base 4/80 = 5.0% and edited 37/80 = 46.2%. At 20 problems the clustered standard error near 48% is about 9 to 11 points. So the run tests the pipeline and cannot confirm or reject the published figures.

### Local run results

None. The run started at 2026-11-04T10:10:40Z and stopped after about 9 minutes, at 1,605 tokens of the first sample (problem 0, edited). The tokens are saved in `results/phase1/local/edited/p00_s0.work/`, and the run can resume from them.

### Fresh-context review

A reviewer read the pipeline against llama.cpp, vLLM and MIT's code while the run was going, and found nothing that invalidates samples. Sampler parity, stop tokens, positions, context sizing, grading, scoring and the published figures all checked out.

Fixed after the review, with `phase1/test_generate.py` failing first and passing now (4 of 4):

- Two engines on one token file interleaved tokens with no error. The engine now locks the file.
- Nothing checked that the token file matched the engine's count. Failed writes are now fatal, and grading checks the count.

The third gap is closed too: logits now match at the real context size, past the 2,048-token boundary and through 32-token chunks, as shown above.

The reviewer also noted that BF16 against F32 arithmetic could shift how often the model enters loops over 30,000 tokens. A cut share that differs from MIT's on CPU may therefore not mean a pipeline bug.

## Phase 1 status, 2026-11-06

Closed without its exit gate, by user decision (`DECISIONS.md`, "Trajectory change"). The 20-problem GPU run below is the last Phase 1 measurement. There is no full MATH-500 result.

## Phase 1 on Windows and a local GPU, 2026-11-04

**Status:** the 20-problem local run is complete, on a new machine: Windows 11 with a Core Ultra 7 255H, 23 GB RAM and an RTX 5050 Laptop GPU (8 GB). These are the first scores this project has produced. This is a small local GPU, so they do not count toward the exit gate. Setup is in `phase1/WINDOWS.md`, and the choices are in `DECISIONS.md`.

### Verification repeated on Windows

| Check | CPU engine (MinGW GCC 16.2) | GPU engine (MSVC and CUDA 13.1, sm_120) |
|---|---|---|
| Downloads and pins | All SHA-256 values and commits match `phase1/README.md` | Same inputs |
| Rebuilt edit | SHA-256 `fd1e42ac…737b553`. All 488 tensors compared, and only the 2 manifest tensors differ, with the expected entry, column and row counts. | Same file |
| `verify_edit.py` relative change | Off by 1.04e-6 against a 1e-6 tolerance on layer 19. In float64 the true value is 0.0810628, and both the manifest's 0.0810612 and our float32 0.0810601 are float32 approximations of it. The bits are identical. | Same file |
| Grader | 0 disagreements | Same grader |
| GGUFs | OK, only `blk.15` and `blk.19` differ | Same files |
| Sampler test | 7 of 7 | 7 of 7 |
| `test_generate.py` | 3 of 3, with write failure skipped because Windows has no `RLIMIT_FSIZE` | 3 of 3, same skip |
| Logits against `transformers`, 12 cases | 12 of 12, max 0.00002 | 12 of 12, max 0.00006, with `NVIDIA_TF32_OVERRIDE=0` |
| Logits with llama.cpp's default TF32 | Not applicable | Base model fails 4 of 6, up to 0.0106 |

### Speed

| Setup | tok/s |
|---|---|
| CPU, 4 threads, short context | 9.46. 8, 12 and 16 threads gave 8.17, 6.90 and 6.19. |
| CPU, two engines at once | about 5.3 each, about 11.6 in total |
| GPU, by context | 72.2 at 0 to 4k, 60.2 at 8k to 12k, 50.0 at 20k to 24k, 43.5 at 32k |
| GPU, whole run | base 51.6 over 3.4 h, edited 52.6 over 2.5 h |

The GPU run held all 25 layers, the KV cache (774 MiB F32) and the recurrent state on the card, using about 4.1 GB of VRAM. Only the 970 MiB token-embedding table stayed in host RAM. Samples that hit the cap took 573 to 852 s. The last few were slower, probably from heat or other load on the laptop.

### Local run results, GPU

MATH-500 `rows[0::25]`, 20 problems, 1 sample each, protocol think35, seed `problem * 1000`. From `results/phase1/local_gpu_summary.txt`:

| | Accuracy | Clustered SE | Published on these 20 | Published on all 500 | Cut at cap | Published cut | Mean tokens | Published mean |
|---|---|---|---|---|---|---|---|---|
| Base | 1/20 = 5.0% | 5.0 | 4/80 = 5.0% | 7.9% | 95.0% | 93.0% | 31,483 | 30,868 |
| Edited | 9/20 = 45.0% | 11.4 | 37/80 = 46.2% | 48.1% | 45.0% | 33.05% | 23,911 | 22,037 |

Per problem, against MIT's 4 published samples:

- **Edited:** correlation r = 0.82 with MIT's per-problem accuracy. It agrees with MIT's majority on 17 of 20 problems.
- **Base:** it solves too few problems for a correlation to mean anything (r = -0.11). It agrees with MIT's majority on 19 of 20. Our one correct base answer, problem index 5, was 0 of 4 in MIT's samples.

Base accuracy, cut share and length all match. Edited accuracy is within 1.2 points of MIT's score on the same problems, and its mean length is within 9%. Edited cut share is 45% against 33%. That gap is 2.4 of 20 samples, well inside the noise of 20 single samples, but it is the largest deviation. The reviewer's BF16-versus-F32 loop concern could also explain it.

The edit raises accuracy from 5% to 45% and lowers cut share from 95% to 45% on these problems. That is the published effect in direction and size. With 20 problems the standard error is about 11 points, so this checks the pipeline and is not a replication. The exit gate still needs the full benchmark, with 4 samples, on a vLLM-class GPU.

### Local run, Windows CPU, partial

Stopped by design when the GPU engine passed its checks. Edited problem 0 finished on CPU in 5,859 tokens and was wrong. The GPU sample of the same problem was right, and MIT had 4 of 4, so this is sampling variance. Base problem 0 stopped at 7,720 tokens and edited problem 1 at 2,339. All of it is in `results/phase1/local_windows/`.
