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
