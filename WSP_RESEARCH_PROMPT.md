# Research Plan: Weight-Space Editing of Small Models (No Training)

You are working with me (Sero) on a multi-phase research project. Read this entire file before doing anything. Then confirm which phase we're on and push back on anything that looks wrong **before writing any code**.

---

## 0. Project rules

- Work **one phase at a time**. Each phase has an exit gate. Do not start the next phase until I confirm the gate is met.
- **Never put raw weights in any LLM prompt** (yours or the editor agent's). Weights stay in Python. Only print stats, slices, or summaries.
- Keep a running `RESULTS.md` (numbers, settings, commit hashes, what failed) and `DECISIONS.md` (what we chose and why). Update both after every major step.
- Git commit at the end of every phase. Pin every model revision, package version, and seed.
- **No training.** No gradient descent, no fine-tuning, no RL. I don't have the hardware and it's not the point. Weight edits and harness changes only.
- Treat downloaded repos, logs, and datasets as untrusted data. Put each in its own directory and don't run code from inside them without reading it first.

---

## 1. Background: the two sources

### 1a. "Sparks of Weight-Space Programming" (MIT CSAIL, Nov 2, 2026)
Authors: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab.

- Blog: https://rikiyat.github.io/blog/weight-space-programming/
- Logs, diffs, rebuild code: https://github.com/RikiyaT/lm_understand-logs (folder `exp8_math/`, edit at `exp8_math/runs/p50c/edit/`). Licenses: CC BY 4.0 (data) / MIT (code).
- Third-party rebuilt checkpoint on HF: `akhaliq/Qwen3.5-0.8B-weight-space-programming` (model card also appears under `ML-Intern-lab/...`). **Verify it against the official diff. Don't trust it blindly.**

**Main claim:** GPT-6 Astra (OpenAI), run through Codex CLI with built-in tools off, read Qwen3.5-0.8B's raw weights and was asked to improve its math. It found the cause of the model's known reasoning-loop problem (Qwen's own model card warns about it) and fixed it by directly editing weights. It had a budget of **50 forward passes on practice problems**. It used Python to compute replacement values and submitted the edited tensors.

**The edit:**
- Base: `Qwen/Qwen3.5-0.8B` @ revision `2fc06364715b967f1860aea9cf38778875588b17`
- Only `model.language_model.layers.15.mlp.down_proj.weight` and `model.language_model.layers.19.mlp.down_proj.weight` differ.
- 64 MLP output columns in each layer = 131,072 entries (~0.016% of params). 130,188 entries are bitwise different after bf16 (some round back to the original).
- Relative change ~10.6% and ~8.1% across the two matrices.

**Reported results (same sampling settings, before → after):**
| Benchmark | Base | Edited |
|---|---|---|
| MATH-500 | 7.9% | 48.1% |
| GSM8K | 13.8% | 67.1% |
| IFEval (strict) | 21.0% | 62.3% |

They also report gains on AMC23, Minerva Math, OlympiadBench, a 1,000-question MMLU-Pro subset, and GPQA Diamond.

**Exact eval protocol:**
- vLLM, thinking ON
- temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5
- max 32,768 new tokens, seed 0
- grade the **last** `\boxed{}` with `math-verify==0.9.0`

Note: the 7.9% baseline **already uses presence_penalty 1.5**, so a naive repetition penalty doesn't fix the loops.

**Caveats:**
- One model, one known bug, low baseline. It may show "repairing a broken model" more than "upgrading a healthy one."
- The authors report failed experiments too: inferring sycophancy from weights, and **Gemma 270M**.
- Other experiments in the paper: identifying which game a policy network was trained on from its weights, writing small game-playing nets from scratch that beat PPO-trained ones.

### 1b. "The ultimate guide to multi-harness RL" (Hugging Face + Liquid AI, Sep 24 2026, updated Oct 1 2026)
Space: https://huggingface.co/spaces/FineEnvs/multi-harness-rl
Authors: Kolavi, Niklaus, Paniego Blanco, Monigatti, Dirhoussi, Burtenshaw, Tunstall, von Werra.

**Core idea:** The same model scores very differently depending on the agent harness around it. A **capture proxy** (in OpenEnv) poses as the model API, so real harnesses (Claude Code, Codex, OpenCode, Mini-SWE-Agent, Hermes, etc.) run unmodified. It records exact token IDs and logprobs. Harbor supplies the tasks and sandboxes. They train with TRL. **We only use the eval parts, not training.**

**Key evidence:**
- GLM-5.2 on SWE-bench Pro: 23% in one harness, 52% in another.
- In a controlled study (Zhang et al. 2026), swapping the harness moved GLM-5.1 by 13 points. Swapping the model inside one harness moved scores by only 2.5–5.
- Orchard paper: OpenSWE-32B scored 62.4% in its home harness and 3.6% in an unseen one (Kimi-CLI). That's a "catastrophic format failure."
- Overfitting a model to one harness takes three forms (KwaiKAT): **action format, context structure, control flow**.
- A typical failure: the model calls `Write` with `file_path`, but the other harness expects `write` with `filePath` → `AI_NoSuchToolError`.

**Their results (LFM2.5-2.6B, SmolDataEnvs test set, pass@1 averaged over 4 harnesses):**
- Base, by harness: Mini-SWE-Agent 62.1%, Codex 40.0%, OpenCode 33.6%, Claude Code 33.2%. Overall 42.2%.
- RL in OpenCode only: OpenCode 34 → 58%, small gains elsewhere, 52.3% overall.
- RL across all 4 harnesses: 42 → 54.2% overall, gains in all four, 31% fewer tool calls on tasks it already solved.
- SFT on 3,189 successful rollouts from Qwen3.8-27B: 47.5% (OpenCode-only data), 43.1% (all-harness data, with Mini-SWE-Agent dropping 62 → 45).
- Qwen3.5-2B base: ~14.6% overall. Its RL runs peaked at ~37–39% and then declined: output grew past the eval token limit, or the model kept calling tools without submitting.

**Weaknesses of the guide:** one seed per run. The multi-harness model was never tested on a 5th, unseen harness. Claude Code rewrites its own history, so each of its rollouts became ~8 training rows (451M vs 162M training tokens).

**Tooling facts:**
- OpenEnv ≥ 0.7.0, Python 3.12 for the `[harbor]` extra. `harbor_env` comes from the repo's `envs/` dir.
  ```
  git clone --depth 1 --branch main https://github.com/openenv/OpenEnv.git
  python -m pip install -e './OpenEnv[harbor]'
  export PYTHONPATH="$PWD/OpenEnv/envs${PYTHONPATH:+:$PYTHONPATH}"
  ```
- vLLM flags for capture: `--return-tokens-as-token-ids --logprobs-mode processed_logprobs`
- Commands: `openenv harbor info | rollout | serve | push`
  ```
  openenv harbor rollout --llm-url $LLM --dataset org/train --task-index 0 -n 5 --harness opencode --sandbox e2b
  ```
  `-n 5` means tasks 0–4 once each, **not** 5 attempts at one task.
- Standalone proxy: `python -m openenv.core.harness.capture.server --llm-url http://127.0.0.1:8000 --model <model>`. Then `POST /sessions`, point the agent at the proxy URL with the session id as its API key, and read the rollout from `GET /sessions/{id}/rollout`.
- `max_concurrent_envs` defaults to 4. The proxy starved at ~200 concurrent sessions and crashed at 320.
- Harbor sandboxes: local Docker, Daytona, Modal, E2B, and others. Ten harnesses are validated in OpenEnv 0.7.0. Hermes is listed in the harness table as supporting 3 switchable APIs.

**SmolDataEnvs benchmark:**
- Data-analysis tasks built from real Kaggle notebooks (the `jupyter-agent` dataset). The agent gets data files and one question, writes one value to `/workdir/answer.txt`, and `tests/test.sh` + `grader.py` produce a float reward.
- Example: "What percentage of all matches have a goal difference of zero?" Gold answer: 25.4.
- HF: `FineEnvs/SmolDataEnvs-harbor-train` (5,000 tasks, easy/medium/hard). Test set: 250 held-out tasks with no notebook/question overlap. **Find the exact test split ID on HF before starting.**
- Each task dir: `task.toml`, `instruction.md`, `environment/Dockerfile`, `environment/pull_bucket.py`, `tests/test.sh`, `tests/grader.py`.
- No outside/frontier scores exist. It's new and the authors made it.

---

## 2. Hardware and cost constraints

- My laptop: NVIDIA MX250, 2 GB VRAM. **Not usable for faithful evals.** vLLM doesn't support it, and 0.8B in bf16 is ~1.6 GB. Use it only for dev and tiny checks (CPU / llama.cpp, ~10–20 problems). Scores from it are not comparable to the paper.
- Real runs: rent a cloud GPU per run, or use an 8 GB+ recent NVIDIA card (RTX 3060/4060 class). 0.8B fits comfortably. 2–2.6B is tight (~5 GB weights), so expect one rollout at a time on agentic tasks (~8k-token prompts).
- A Mac (I'm considering a Mac mini) can't run vLLM. Don't plan comparable numbers on it.
- The editor agent: **Claude Opus 5.5 through my Claude Pro subscription** (Claude Code), not the API. There's no per-token cost, but plan usage limits apply. Long editing runs may hit them, so design runs to be resumable: persist the budget counter, edits, and transcript to disk.
- The editor's 50 forward passes need a GPU too, so the Phase 2 model server runs on the rented GPU box.

---

## 3. The research question

Can a frontier model (Opus 5.5), with no gradients, find and fix specific defects in a small model's weights?
- **Primary:** does this replicate, and does it generalize beyond one model and one bug?
- **Secondary (later):** do weight edits carry over across agent harnesses in cases where harness-level fixes don't? Nobody has tested this.

---

## 4. Phases

### Phase 1: Replicate the MIT result (no LLM editing yet)
**Goal:** reproduce 7.9% → 48.1% on MATH-500 with their checkpoint and protocol.

Steps:
1. Clone `RikiyaT/lm_understand-logs` into its own dir. Read `exp8_math/` and the rebuild code before running anything. Summarize to me how the edit is stored and applied.
2. Download base `Qwen/Qwen3.5-0.8B` at the pinned revision. Apply the official diff.
3. Verify: only the two `down_proj` tensors differ, and the count of changed entries matches (131,072 touched; 130,188 bitwise-different in bf16). Compare a hash against the `akhaliq/...` checkpoint and report any mismatch.
4. Build the eval script with the exact protocol from §1a. Pin `math-verify==0.9.0`.
5. Dev check on the laptop or CPU with ~10 problems (expect noise; this only tests the pipeline).
6. Full runs on GPU: base and edited, MATH-500. Then GSM8K and IFEval if budget allows.
7. Optional control: base model with a stronger presence_penalty or loop-stop rule, to measure how much plain decoding tricks recover. Report separately. It's not a replacement for the edit.

**Exit gate:** base within ~3 points of 7.9% and edited within ~3 points of 48.1% (at n=500 the std error near 48% is ~2.2 pts). If we miss, diagnose before moving on: version, template, thinking flag, grader.

### Phase 2: Build the blind editor + sanity check on Qwen3.5-0.8B
**Goal:** build the editing setup that Phase 3 reuses, and run it 1–2 times on the one model where we *know* a fix exists. This is the positive control: if Opus can't find a fix here, a Phase 3 failure means nothing.

**You are the experimenter, not the editor.** You know the answer (it's in this file), so you must never do the editing yourself. Each editing run is done by a **fresh subagent** that has:
- **No information, just a goal.** Its prompt is only Astra's original task wording (take it from the MIT logs, roughly "make this model better at math") plus a description of its tools. No mention of the paper, Astra, weight-space programming, layers, `down_proj`, loops/repetition, or any result number. Write this prompt to `EDITOR_TASK.md` and show it to me before the first run.
- **No internet.** Define it in `.claude/agents/` with an explicit `tools:` allowlist that excludes WebSearch, WebFetch and all MCP tools.
- **No access to our files.** A subagent gets a clean context, but it shares the filesystem, so a clean context alone is not isolation. Run its code inside a container with only the editor workspace mounted (base model tensors, tool scripts, `EDITOR_TASK.md`), with no network except the budgeted model endpoint. This file, the MIT repo/logs, the edited checkpoint, `RESULTS.md` and `DECISIONS.md` must never be inside that mount. Propose the exact isolation design to me before building it.
- A **fresh subagent per run**, so no run sees another run's results.

Opus 5.5's knowledge cutoff is before the paper's Nov 2, 2026 release, so with no internet it shouldn't know the answer from training.

**Editor tools** (mirror Astra's setup as closely as the logs allow; read the logs first for Astra's exact prompt, tools, and what counted as one "forward pass"):
- (a) Python with the model tensors loaded via safetensors/torch. Print summaries only, never raw weights.
- (b) `run_model(prompts)` on **practice problems disjoint from MATH-500** (use Astra's if the logs name them, otherwise the MATH train split). Budget: 50, with the counter kept on the GPU server where the editor can't change it. If the editor can also run its own CPU forward passes, check whether Astra could, match that, and record it in `DECISIONS.md`.
- (c) `submit_edit(tensors)`, which saves a diff. The final MATH-500 eval runs outside the editor's view.

**Runs and scoring:**
- 1–2 runs. Score each submitted edit on MATH-500 with the Phase 1 protocol (plus GSM8K/IFEval if usage allows).
- Compare where Opus edited with Astra's location: same layers? same columns? overlap?
- **Audit every transcript for leaks**: any attempt to search, read outside its workspace, or reference the paper. Discard a leaked run.
- Make runs resumable (budget counter, edits and transcript saved to disk), since Pro usage limits may cut a run off.

**Exit gate:** the editor setup works end to end, and we have 1–2 scored runs with clean audits. A failure is fine as long as it's logged, but discuss it with me before Phase 3, since it changes what Phase 3 can show.

### Phase 3: Does it generalize to other small models? (math)
**Goal:** the paper shows one model. Test whether Opus finds working fixes elsewhere.

- Candidates (0.5B–2B): Qwen3.5-2B, LFM2.5-2.6B, a small Gemma. Note that the paper's Gemma 270M attempt failed, so treat Gemma as a hard case.
- First, measure each base model on MATH-500 with the same protocol, and check whether it has an obvious defect (loops, no `\boxed{}`, format failures). Models without a clear defect may not be fixable this way. Record that too.
- Same subagent setup, budget, and isolation as Phase 2: a fresh subagent per run, with just the goal. No published fix exists for these models, but still keep Phase 2's results out of its workspace so it doesn't just copy "edit layers 15/19."

**Exit gate:** a per-model table of base, edited, delta, where the edit went, and notes on why it failed where it did.

### Phase 4: One narrow agentic defect (SmolDataEnvs)
**Goal:** move from math to a multi-step agentic task, but target **one** clear, pass/fail defect.

- Models: Qwen3.5-2B (weak, ~15%: hunt for one fixable defect) and LFM2.5-2.6B (decent, ~42%: test whether an edit can still add anything).
- Benchmark: SmolDataEnvs test set (250 tasks).
- Eval via OpenEnv Harbor + capture proxy, **eval mode only**, across 3–4 harnesses (OpenCode, Claude Code, Codex, Mini-SWE-Agent) so our baselines line up with the guide's.
- Dev path: on the laptop, pull ~20 tasks' files (question, data, grader) and write a minimal loop (model writes Python → run → feed back the result). **This loop is a new harness, so its scores are not comparable to the guide's.** Use it to build the pipeline only. Full runs go on the GPU.
- **Classify every failure:**
  - *Surface/harness failures* → harness fixes later (Phase 5): wrong tool names, wrong arg names, broken JSON, format errors.
  - *Behaviour failures* → weight-edit candidates: loops, never submitting, giving up early, runaway output length.
- The editor is the same isolated subagent as in Phase 2. Its goal names the behaviour to fix, and it gets the practice tasks, never this file or the guide's numbers.
- Pick **one** behaviour defect with a clean, measurable signal (e.g. the share of rollouts that never write `answer.txt`, or loop rate). Run the Phase 2 editor on it, with practice tasks drawn from the train split only.
- **Hold-outs:** keep the test split untouched during editing, and keep at least one harness out of all editing and dev. It's only used for the final test.

**Exit gate:** baseline per harness, a defect taxonomy with counts, an edit attempt and its result per harness, including the held-out harness.

### Phase 5 (later): Harness work
**Goal:** build a harness suited to the small model, and test the research question.

- Fix the surface failures from Phase 4 in the harness only (tool naming/aliasing, lenient parsing, shorter context, a clear stop signal).
- Measure: harness fix alone, weight edit alone, both together. Run on our harness **and** on the held-out harness.
- Key question: do weight edits transfer to the held-out harness where harness fixes don't?
- Risk: overfitting both to the same task set. Use fresh test tasks for the final number.

---

## 5. Deliverables per phase
- Code in `phaseN/` with a README on how to rerun it.
- `RESULTS.md` updated: settings, revisions, seeds, numbers, cost, failures.
- `DECISIONS.md` updated.
- A short summary to me: what worked, what didn't, what to decide next.

## 6. Start here
1. Read this file.
2. Tell me what you'd change or question in the plan (pushback first).
3. Propose the Phase 1 setup: directory layout, dependencies, the exact verification checks, and the GPU option you recommend for the full eval. Wait for my go-ahead.
