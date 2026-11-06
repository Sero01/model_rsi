# Plan: Phase 2, the blind editor (built, not scored)
Date: 2026-11-06. Experimenter-only. Never put this file in an editor workspace.

## Goal
Build the editing setup that Phase 3 reuses: an isolated, resumable editor run with a forward-pass budget it cannot change. Prove that it works without scoring any edit. The reasons for skipping the scored runs are in `DECISIONS.md`, "Trajectory change, 2026-11-06".

**Exit gate (amended):** the setup runs end to end on the RTX 5050 machine, the canary test passes, and one unscored smoke run on Qwen3.5-0.8B has a clean leak audit. Nothing is scored on MATH-500.

## Approval gates
The research plan requires two approvals before any editor run:

- **Gate A:** the isolation design (below) and `EDITOR_TASK.md`, shown to the user before building the sandbox and before the first run.
- **Gate B:** the canary test result, shown before the smoke run.

## Machines
- **ZBook (CPU only):** code, unit tests, a stub editor, and CPU-engine forward passes for testing the plumbing.
- **RTX 5050 machine (Windows 11, 8 GB):** model server, canary test and smoke run.

## Architecture (proposed, for Gate A)
This is the design from `plans/2026-11-04-phase1-replication.md`, "Why a project subagent does not isolate the Phase 2 editor", moved to Windows.

```
driver (host, phase2/driver.py)
 ├─ claude -p --tools "" --strict-mcp-config --resume <session>
 │    cwd: empty directory; CLAUDE_CONFIG_DIR: holds only the OAuth credential
 │    The model has no tools. It replies with text and Python blocks.
 ├─ executor: runs each Python block inside the sandbox, returns capped output
 ├─ model server: run_model requests, with the budget counter kept on the host
 └─ run state on disk: session id, transcript, budget, edits (resumable)

sandbox (no network)
 ├─ workspace, read-only: base model tensors, tool library, EDITOR_TASK.md, practice problems
 ├─ scratch, read-write
 └─ outbox: run_model requests and submit_edit diffs, picked up by the driver
```

- **Sandbox, recommended: Windows Sandbox** with `<Networking>Disable</Networking>`, read-only mapped folders for the workspace, and one read-write scratch folder. It ships with Windows 11 Pro, isolates at the hypervisor level, and needs no install beyond enabling the feature (admin, one reboot). It is not enabled on the ZBook. Whether the RTX 5050 machine runs Pro is still to be checked.
  - Fallback 1: WSL2 with bubblewrap, which is MIT's own design. It needs WSL installed, and bwrap must actually start there.
  - Fallback 2: Docker Desktop with `--network none`.
- **run_model never runs inside the sandbox.** Sandboxed code writes a request to the outbox. The driver checks the budget, applies the current diff to the base model, runs the CUDA engine and writes the results back. The counter lives in host-only run state, so the editor cannot change it.
- **Weights never reach the model.** The executor caps each output (proposed: 4,000 characters) and refuses outputs that are mostly long runs of numbers. This enforces "summaries only" mechanically.
- **Generation:** the verified llama.cpp CUDA engine, F32, `NVIDIA_TF32_OVERRIDE=0`, with the GGUF rebuilt from base plus diff for each pass. Rebuild time is to be measured.
- **Teacher-forced passes that return activations:** `transformers` on the GPU in F32 (0.8B is about 3.2 GB). This needs a CUDA torch build that supports sm_120. Pin it.

## Steps
1. **Read Astra's setup in the MIT logs.** Clone `RikiyaT/lm_understand-logs` at `4d09ba6` into `external/`. Extract Astra's original task wording, tool list, output limits, what counted as a forward pass, and whether Astra could run its own CPU forward passes. Record these in `DECISIONS.md`.
2. **Write `EDITOR_TASK.md`** from Astra's wording plus a description of the tools only. It must not mention the paper, Astra, layers, `down_proj`, loops or repetition, or any number. **Gate A:** show it and the isolation design to the user.
3. **Practice problems.** Take them from the MATH train split. MATH-500 is drawn from the test split, so they are disjoint, but check for duplicate problem text anyway. Use the same chat template and sampling protocol as Phase 1.
4. **Build and test on the ZBook**, test-first, with a stub editor that sends scripted replies instead of calling Claude:
   - The budget decrements once per pass, and a request past the limit is refused.
   - Sandboxed code cannot reach the network, read outside its mounts, or write to the budget or run state.
   - A killed run resumes with the same session, budget and edits.
   - A submitted diff round-trips: apply it to base, and the hash matches.
   - The output cap and the number filter trigger.
5. **Canary test (Gate B).** Plant unique strings in the user-level and project `CLAUDE.md`, a memory file, an environment variable and the regular config directory. Start the editor model as the driver does, ask it to report any instructions or context it has and to try listing files, and confirm that no canary appears and that no tool call succeeds. Also confirm that no plugin or hook output, such as the SessionStart hook on this machine, reaches it.
6. **On the RTX 5050:** measure batched generation throughput with parallel sequences, and GGUF rebuild time. Set the pass size so that one pass takes a bounded time, and record the deviation from Astra's 512 generations or 1,024 texts.
7. **Unscored smoke run** on Qwen3.5-0.8B with a small budget, proposed at 5 passes. Audit the transcript for leaks: searches, reads outside the workspace, or mentions of the paper. Keep the submitted diff but do not score it on MATH-500. Record where it edited, for the experimenter only.
8. **Records and commit:** `phase2/README.md`, `RESULTS.md`, `DECISIONS.md`, `memory/` and the ledger. Commit at the end of the phase.

## Layout
```
phase2/
  README.md
  driver.py          turn loop around claude -p, run state, resume
  executor/          sandbox launcher and in-sandbox runner
  server.py          run_model, budget, diff application, engine calls
  tools/             the editor's library, mounted read-only
  canary_test.py
  tests/
EDITOR_TASK.md       written in step 2, shown at Gate A
runs/                run state and transcripts, gitignored
```

## Open questions
- **Where the driver runs.** The editor uses Sero's Claude Pro login. Simplest: run everything on the RTX 5050 machine with Sero logged in there. Otherwise the driver runs on the ZBook and sends run_model requests to the 5050 over the network.
- **Windows edition on the 5050 machine**, for Windows Sandbox.
- **Phase 3 benchmark** that replaces MATH-500 for baselines. This does not block Phase 2.
- **Untracking the `*.work/tokens.txt` files** that were added in the merge.
