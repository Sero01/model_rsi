# SDD ledger — plan: plans/2026-11-04-phase1-replication.md
Moved from the gitignored .superpowers workspace so it travels with the repo.
Task list comes from HANDOVER.md steps 1-10, which override the plan.
HANDOVER step 1: git init + .gitignore done (no commit until end of phase).
HANDOVER steps 2-6 done: logs repo @4d09ba6, model downloaded, edit verified, grader 0 disagreements.
Ruling: mathqa.py vendored verbatim with a stub mcqa.py — mcqa is unpublished and used only for loop diagnostics — cost if wrong: none for grading, regrade agreed.
Ruling: grader check uses 200 texts plus 500 pred_first per checkpoint — evaluate.py keeps only 20 texts per shard despite README — cost if wrong: extraction checked on 200 problems only.
Step 7 BLOCKED: classifier denied install from external clone ("Code from External"). Build and converter need user permission.
Step 7 unblocked after user said Proceed: torch from CPU index, gguf from PyPI, llama.cpp b12099 built through cmake inside the Landlock sandbox.
Ruling: own GGUF writer (phase1/make_gguf.py) instead of the clone's converter — classifier denied running the clone's converter — cost if wrong: caught by verify_gguf hashes and the logit comparison.
Ruling: phase1/sandbox.py (Landlock + seccomp) instead of bubblewrap — bwrap cannot run on this laptop (apparmor_restrict_unprivileged_userns=1) — cost if wrong: Phase 2's isolation design must not assume bwrap either.
Ruling: llama.cpp library with our own sampler chain, not llama-server — llama-server accepts prompt tokens into the penalty window, vLLM penalizes output tokens only — cost if wrong: own C++ driver, covered by sampler_test.
Stepped-decode error measured: 0.00773 at 1 step, 0.025 at 50, level at about 0.027 from 200 to 1950 steps; batched 0.001. Cause pending diag2 (flash attention, BF16 yardstick).
Stepped-decode cause found: flash attention. Disabled in generate and logits; check_logits threshold now 0.001; both models pass at 0.00002 or better.
Ruling: flash attention off, batch 32 — matches transformers at equal speed and memory — cost if wrong: replay on resume is slower (17 min at 32K against 13).
Ruling: both models, 1 sample, cap 32768, edited first, 4 threads — see DECISIONS.md — cost if wrong: about 5.2 days of laptop time, stoppable and resumable at any point.
Ruling: no GPU script written — GPU steps are unapproved and it could not be tested here — cost if wrong: written when approved.
Step 9 started 2026-11-04T10:10:40Z. Fresh-context review dispatched while the first sample runs.
User stopped the local run at 1,605 tokens of edited p00_s0, and approved updating the stale plan and memory files. Both updated. GPU question still unanswered.
Final: fresh-context review done (no Critical). Important items deferred to the new machine because the run is stopped: engine file lock, token-count check, long-context logit comparison. Listed in HANDOVER.md.
Ruling: review's 3-hour replay estimate and 2x slowdown not adopted — measured replay was 17 min at 32K and speed fell 1.6x at 32K — cost if wrong: projection is optimistic, which the first long sample would show.
Ruling: first commit before the phase ends — user asked to push so the project can move machines — cost if wrong: none.
Final: fixed second engine corrupting a sample — test_generate second_engine RED→GREEN, suite 4/4, sandbox and sampler tests green.
Final: fixed unnoticed write failure and token count mismatch — test_generate write_failure and grade RED→GREEN, suite 4/4.
Final: logit check extended to generate's context and chunk size and past the 2,048 boundary; check_logits_long.py added to 32,000 tokens on the base model only, since both models share the same engine path. Results pending.
Final: logit checks pass: 12 of 12 at generate's settings on both models, and at 4,096, 8,000, 16,000 and 32,000 tokens on base (max 0.00055). Fix pass complete. Minors deferred, listed in HANDOVER.md.
Windows/GPU 2026-11-04: port to native Windows (see phase1/WINDOWS.md). GPU engine verified 12/12 with NVIDIA_TF32_OVERRIDE=0, and it fails with TF32 on. 20-problem GPU run complete: base 1/20, edited 9/20; published on the same problems 4/80 and 37/80. Exit gate still open.
