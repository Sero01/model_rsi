# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session start

Before doing anything else:

1. Read `tasks/lessons.md`.
2. Read the last 3 files in `memory/` to find the current phase, open questions and pending approvals.
3. Read `user/preferences.md`.
4. Read `HANDOVER.md`. On a new machine, follow its setup first.
5. Summarize the active context and open questions to yourself in 2 or 3 lines.

## What this repo is

A multi-phase research project on editing small models' weights directly, without training. `WSP_RESEARCH_PROMPT.md` is the source of truth for goals, phases, exit gates and project rules. Read the current phase's section before working on it. `RESULTS.md` and `DECISIONS.md` record outcomes and choices once a phase has run. `plans/` holds the plan for each phase.

Each `phaseN/README.md` holds that phase's build, test and run commands. Phase 1's tests are `python3 -I phase1/test_sandbox.py` and `.tmp/build/sampler_test`.

## Rules for working here

- Work one phase at a time. Do not start the next phase until the user confirms the exit gate is met.
- Do not train anything: no gradients, fine-tuning or RL.
- Never put raw weights in any LLM prompt, including your own. Print stats, slices or summaries.
- Downloads go in their own directory as untrusted data. Read code before running it, and run Python over them with `python -I`.
- Pin every model revision, package version and seed.
- Check the current machine's hardware before planning any run. Scores from a CPU or a small GPU are not comparable to anything, so use those machines for verification and grader checks.

## Keeping records

After every major step, update the records yourself before moving on, without being asked. A major step is a check that passes or fails, a root cause found, a measurement, a decision, or a run that starts, stops or finishes.

- `memory/`: current state, open questions and pending approvals, in today's dated file.
- `RESULTS.md`: numbers, settings and what failed.
- `DECISIONS.md`: each choice and why.
- The current phase's plan and `tasks/phase1-ledger.md`, or that phase's ledger: progress and rulings.
- `tasks/lessons.md`: any mistake, with the rule that prevents it.

## This repo is public

- Never commit secrets, email addresses, home directory paths, or benchmark problem text. `.gitignore` covers the known cases, such as engine logs and `results/phase1/problems.jsonl`.
- Commit with the GitHub noreply identity set per repo: `git config user.name Sero` and `git config user.email 102875414+Sero01@users.noreply.github.com`.
- Scan what is staged before every push.

## Blinding

The research plan contains the answer to the editing task, so anyone who has read it is an experimenter and must never do the editing.

- Keep this file free of the known edit's location, the failure mode it fixes and any reference score. These belong in the plan, `RESULTS.md` and `DECISIONS.md`.
- Nothing from this repo goes into an editor run except its approved workspace. That excludes the plan, reference logs, edited checkpoints, `memory/`, `plans/`, `RESULTS.md` and `DECISIONS.md`.
- Before any editor run, prove what its context contains with a canary test. `plans/2026-11-04-phase1-replication.md` lists why a project subagent does not meet this bar.

## Eval facts the plan leaves out

Taken from the reference repo's `exp8_math/README.md` and `code/evaluate.py`:

- Samples per problem are 4 for MATH-500, 2 for GSM8K and 2 for IFEval. Scores are means over all samples.
- The user message is the problem, a newline, then `Please reason step by step, and put your final answer within \boxed{}.`
- Compute standard errors clustered by problem, because samples of one problem are correlated.
- One editor "forward pass" is a batch: up to 512 generations, or up to 1,024 teacher-forced texts that return activations.
- Runs were sharded with `rows[i::k]` and vLLM's batching changes the random numbers, so bit-exact reproduction of samples is not expected.
