# Phase 1 local run state, 2026-11-04

- **Phase:** 1, not complete. Do not start Phase 2.
- **Local run:** stopped by the user at 1,605 tokens of the first sample, with no scores and no process running. To resume, use the command in `phase1/README.md`. A full run takes about 5.2 days.
- **Delegated decisions:** the user said "Take all decisions optimally". They are recorded in `DECISIONS.md`. The user also approved updating the stale plan and memory files, and that is done.
- **Repo:** public at `github.com/Sero01/model_rsi`, pushed at the user's request so the project can resume on a new machine. `HANDOVER.md` has the setup.
- **New rule from the user:** update `memory/`, `RESULTS.md` and `DECISIONS.md` without being asked after every major step. It is in `CLAUDE.md`.
- **Review:** finished with no sample-invalidating bugs. The file lock and token count fixes are done, with `phase1/test_generate.py` passing. The long-context logit comparison passes out to 32,000 tokens, with a maximum error of 0.00055, which grows with context. Only minor review items remain, listed in `HANDOVER.md`.

## Pending from the user

- Whether to approve a GPU for the full runs. The exit gate cannot be met without one, and the handover forbids renting one without asking.
- Whether to resume the local run.
- Exit gate confirmation, which is the user's alone.
