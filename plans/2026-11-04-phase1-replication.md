# Plan — Phase 1 replication
Date: 2026-11-04

## Goal
Reproduce MATH-500 base 7.9% and edited 48.1% with the published edit and protocol. The exit gate is in `WSP_RESEARCH_PROMPT.md`, with the additions below.

**Status, 2026-11-06:** closed without the exit gate, by user decision. Phase 2 continues in `plans/2026-11-06-phase2-editor.md`.

**Status, 2026-11-04:** the laptop steps are done, and the user stopped the local sample run early. The GPU steps have not been approved. `HANDOVER.md` set the laptop-first order and dropped the thinking-off control, and this plan now reflects both.

## Layout

```
external/lm_understand-logs/   clone at a pinned commit, untrusted, gitignored
models/                        base, text-only and edited checkpoints, gitignored
phase1/
  README.md                    how to rerun
  requirements.lock
  vendor/                      apply_edit.py, st_numpy.py, mathqa.py, merge.py, copied after reading; mcqa.py is our stub
  engine/                      llama.cpp driver: generate, logits, sampler_test
  verify_edit.py
  check_grader.py
  prepare_problems.py          selects the 20 local problems
  make_gguf.py, verify_gguf.py F32 GGUF writer and its check
  check_logits.py              compares the engine with transformers
  sandbox.py, test_sandbox.py  Landlock and seccomp sandbox
  run_local.py, score_local.py resumable CPU run and its summary
  eval_math.py                 GPU run, not written until the GPU steps are approved
results/phase1/                scores, per-problem data, sample texts
RESULTS.md
DECISIONS.md
```

## Dependencies
Laptop: `phase1/requirements.lock`, and llama.cpp `b12099` @ `bee23322f1baa3d2dd16b8339bed6aff4a8eb663`.

GPU, if approved: Python 3.12, `vllm==0.33.0`, `transformers==5.20.2`, `math-verify==0.9.0`, `latex2sympy2_extended==1.11.0`, `antlr4-python3-runtime==4.13.2`. The vLLM and transformers pins come from the independent re-run. I found no vLLM version pinned in MIT's repo. IFEval needs the official checkers at a pinned commit, because MIT's repo leaves them out.

## Steps

Laptop:

- [x] `git init`, `.gitignore`, clone the logs repo at a recorded commit, and read `mathqa.py`, `st_numpy.py` and `merge.py`.
- [x] Download `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`.
- [x] Check the two base tensors against `base_sha256` in `runs/p50c/edit/manifest.json`.
- [x] Rebuild with `apply_edit.py` and check `edited_sha256`.
- [x] Compare every tensor, base against edited. Require exactly the 2 named tensors to differ, 65,260 + 64,928 = 130,188 differing entries, 64 touched columns each (131,072 entries), and relative changes of 0.10636 and 0.08106.
- [x] Compare tensor hashes against `RikiyaT/...` and `ML-Intern-lab/...` and report any mismatch.
- [x] Regrade MIT's published first samples with our grader, base and p50c, and require agreement with `per_problem.json`. The logs keep 200 texts and 500 `pred_first` values per checkpoint, so that is what was checked.
- [x] Check that our rendered prompt matches the evaluation prompt format in the logs.
- [x] Build the CPU engine and F32 GGUFs, and check them against MIT's hashes and `transformers` logits, out to 32,000 tokens.
- [x] Measure speed: 2.96 tok/s at 1,000 tokens of context, 1.82 at 32,000. A capped sample takes about 3.95 hours, so 20 problems at 1 sample on both models would take about 5.2 days.
- [x] Run 20 problems locally. The laptop CPU run was stopped at 1,605 tokens. It was completed on 2026-11-04 on a Windows RTX 5050: base 1/20, edited 9/20. See `RESULTS.md`.

GPU, not approved:

- [ ] Smoke test on shard 0/10 of MATH-500, base and edited, on the text-only export and on the multimodal release. Published base shard: 14/200 correct, 190/200 cut, 6.27M tokens. Record tokens per second.
- [ ] Stop and report before the full run if the smoke test misses, or if measured speed puts Phase 1 above the agreed spend.
- [ ] Full MATH-500, base and edited.
- [ ] GSM8K and IFEval if budget allows, on the agreed spend.
- [ ] Optional stronger-penalty control, reported separately.
- [ ] Update `RESULTS.md` and `DECISIONS.md`, commit.

## Exit gate additions
Keep the plan's 3-point accuracy bands and add:

- Cut share within 3 points of 93.0 and 33.05.
- Mean tokens within 5% of 30,868 and 22,037.
- Cluster-bootstrap CIs by problem.
- Per-problem correlation with MIT's published per-problem scores.

Accuracy alone can pass with a wrong setup. Cut share and length catch template and stop-token faults far sooner.

## Cost
About 247M tokens for everything, 106M for MATH-500 alone. The H200 re-run measured 14.2k tokens/s, so about 5 hours there. A 24 GB card will be slower by an amount I can't predict, because concurrency is limited by cache memory at these lengths. The smoke test measures it.

## Known Risks / Unknowns
- MIT did not publish its engine version, so a small accuracy offset may have no findable cause.
- The text-only export may not load in vLLM 0.33.0. The release is the fallback, and it is known to work.
- The re-run's 51.0% is 2.9 points above the post. A result near the edge of the gate is plausible without any fault.

## Decision Points
- Go-ahead, GPU and spend cap.
- Whether to resume the laptop sample run.

## Why a project subagent does not isolate the Phase 2 editor
- Its Read and Bash tools run on the host, outside any container its code uses.
- Project subagents inherit project context such as `CLAUDE.md`. Allowlists do not stop Bash from reaching the network.
- It ends with the parent session, and a run lasts many hours.

Proposed replacement, to be approved in Phase 2: copy MIT's design. A driver calls `claude -p --tools "" --strict-mcp-config` once per turn, resuming the session, from an empty directory with a separate config directory holding only the OAuth token. The model has no tools. The driver runs each returned Python block in a sandbox with no network, sends back the output, and keeps the budget and transcript on disk. `--bare` is unavailable because it refuses OAuth. A canary test must show that no user or project context reaches the model.

MIT's sandbox is bubblewrap, which cannot run on this laptop because of `apparmor_restrict_unprivileged_userns=1`. `phase1/sandbox.py` uses Landlock and seccomp instead. Whether that is strong enough for an editor run is a Phase 2 decision.

## Outcome
Not complete. See the status at the top and `RESULTS.md`.
