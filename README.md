# Weight-space editing of small models

A multi-phase research project on editing small models' weights directly, without any training. Phase 1 replicates the MATH-500 edit on Qwen3.5-0.8B from "Sparks of Weight-Space Programming" (MIT CSAIL).

**Status:** Phase 1 is in progress. The published edit, grader and engine are verified. A 20-problem local run on a Windows laptop GPU scored base 1/20 and edited 9/20, against published 4/80 and 37/80 on the same problems. The exit gate still needs the full benchmark. See `RESULTS.md`. Native Windows and CUDA setup is in `phase1/WINDOWS.md`.

## Layout

| Path | Contents |
|---|---|
| `WSP_RESEARCH_PROMPT.md` | Goals, phases and exit gates |
| `HANDOVER.md` | Current state and how to continue |
| `RESULTS.md`, `DECISIONS.md` | Outcomes and choices |
| `phase1/` | Verification scripts, CPU engine and run driver. `phase1/README.md` has the commands. |
| `plans/`, `memory/`, `tasks/` | Plans, working notes and progress |

## Blinding

The research plan and several other files here state the answer to the editing task, so anyone who has read them is an experimenter. Never give this repo's contents to a model being tested as an editor.

## Attribution

This builds on the logs, edit and code in [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), by the authors of "LLMs can understand and write neural network weights". Published figures quoted here come from that repo under CC BY 4.0. The code in `phase1/vendor/` comes from it under the MIT License, in `phase1/vendor/LICENSE`, except `mcqa.py`, which is our stub.

Benchmark problems are not included. `phase1/prepare_problems.py` regenerates them from the pinned dataset.
