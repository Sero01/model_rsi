# Handover: continue Phase 1 on a new machine

Updated 2026-11-04. Experimenter-only, because it names the known edit. Never put this file in an editor workspace.

Where this file disagrees with older notes, this file wins.

## Read first

1. `WSP_RESEARCH_PROMPT.md`, sections 0, 1a, 2 and the Phase 1 part of section 4.
2. `RESULTS.md` and `DECISIONS.md`, for what was measured and why each choice was made.
3. `plans/2026-11-04-phase1-replication.md`, for the steps still open.
4. `tasks/phase1-ledger.md`, for the rulings made so far.

## Current state

- **2026-11-06: Phase 1 is closed without its exit gate, by user decision.** Phase 2 is active: build the editor, skip its scored runs, then Phase 3. Read `plans/2026-11-06-phase2-editor.md` and the 2026-11-06 section of `DECISIONS.md` first.
- **Verified:** the rebuilt edit, the grader, the GGUFs, and logits against `transformers` out to 32,000 tokens.
- **Local run:** the laptop CPU run was stopped at 1,605 tokens. The 20-problem run was completed on 2026-11-04 on a Windows RTX 5050, with base 1/20 and edited 9/20. See `RESULTS.md` and `phase1/WINDOWS.md`.
- **Exit gate:** waived by the user. No full MATH-500 run will be made.

## User decisions

1. **Thinking stays on.** Do not propose a thinking-off control.
2. **Do not rent a GPU or spend money without asking.** The plan's GPU steps have no approved spend.
3. **Records:** update `memory/`, `RESULTS.md` and `DECISIONS.md` after every major step, without being asked.
4. **The repo is public.** See the rules in `CLAUDE.md`.

## Open questions for the user

- What hardware the new machine has, if it isn't already clear from checking.
- Whether to run the full MATH-500 evaluation on a GPU, on which GPU, and with what spend cap. The plan estimates about 106M tokens for MATH-500.
- Whether to resume the 20-problem CPU sample. It would take about 5.2 days on the old laptop, so measure speed first. A real GPU makes it unnecessary.

## Setup on a new machine

1. Clone the repo and set the commit identity:
   ```bash
   git clone https://github.com/Sero01/model_rsi && cd model_rsi
   git config user.name Sero
   git config user.email 102875414+Sero01@users.noreply.github.com
   ```
2. Follow `phase1/README.md` from Setup through Tests. It rebuilds everything gitignored: packages, the logs repo, the dataset, both checkpoints, `results/phase1/problems.jsonl`, the engine and the GGUFs. Every step has a pin, hash or check to compare against.
3. Run `phase1/test_sandbox.py` first. `sandbox.py` needs Landlock, so check that the new kernel supports it. Decide on the Phase 2 isolation only when Phase 2 starts: bubblewrap could not run on the old laptop, but it may run here.
4. Record the machine's hardware in `user/preferences.md`, and re-measure speed. The old figures are for that laptop only.

If the machine has a GPU that vLLM supports, ask the user about the GPU run before investing more in the CPU pipeline. `phase1/eval_math.py` has not been written.

## Review items

The fresh-context review's Important findings are fixed, and `RESULTS.md` has the evidence.

Minor items, fix when convenient:

- On resume, cut anything after the last newline in `tokens.txt`, and reject non-numeric lines.
- `new_tokens` and `seconds` in a record cover only the last process, so speed after a resume reads wrong, and `score_local.py` divides by zero if every record shows 0 seconds.
- `score_local.py` compares with whole-benchmark cut share and mean tokens without saying so in its output.
- `sampler_test.cpp` never tests a positive logit under the penalty.

A killed `run_local.py` still leaves `generate` running. A restart then stops with an error instead of corrupting the sample, but you still have to kill the old engine.

## Protocol

From `exp8_math/README.md` and `code/evaluate.py`, protocol `think35`:

- Chat template with `enable_thinking=True`.
- User message is the problem, a newline, then `Please reason step by step, and put your final answer within \boxed{}.`
- temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, max 32,768 new tokens, seed 0.
- Grade the last `\boxed{}` with math-verify 0.9.0.

## Reference numbers

| MATH-500 | Accuracy | Clustered SE | Cut at cap | Mean tokens |
|---|---|---|---|---|
| Base, MIT | 7.90% | 0.97 | 93.0% | 30,868 |
| Edited, MIT | 48.10% | 1.90 | 33.05% | 22,037 |
| Edited, independent re-run on the multimodal release, vLLM 0.33.0, H200 | 51.0% | not computed | 31.6% | 21,704 |

With 20 problems the clustered SE near 48% is about 9 to 11 points, so a local sample cannot confirm or reject these. It tests the pipeline. Scores from a CPU or small GPU are not comparable to the paper and do not count toward the exit gate.

## Do not

- Do not run the editor before Gates A and B in `plans/2026-11-06-phase2-editor.md` are approved.
- Do not rent a GPU or spend money without asking.
- Do not put raw weights in any prompt.
- Do not add the edit's location, its failure mode or reference scores to `CLAUDE.md`.
- Do not commit anything `CLAUDE.md` lists as off limits for this public repo.
