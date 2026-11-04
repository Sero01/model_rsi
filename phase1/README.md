# Phase 1: replication

Rebuilds the published edit on Qwen3.5-0.8B, then samples MATH-500 on CPU to check the pipeline. CPU scores are not comparable to the published ones and do not count toward the exit gate. `DECISIONS.md` explains the choices and `RESULTS.md` holds the outcomes.

Run everything from the repo root, with `R=$PWD`. `external/`, `models/` and `results/phase1/problems.jsonl` are gitignored, so a fresh clone rebuilds them with the steps below. That needs about 11 GB of disk, of which the two GGUFs and two checkpoints take 9 GB.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r phase1/requirements.lock \
    --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
source .venv/bin/activate

git clone https://github.com/RikiyaT/lm_understand-logs external/lm_understand-logs
git -C external/lm_understand-logs checkout 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f
git clone --depth 1 --branch b12099 https://github.com/ggml-org/llama.cpp external/llama.cpp

hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir models/Qwen3.5-0.8B
hf download HuggingFaceH4/MATH-500 test.jsonl --repo-type dataset \
    --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be --local-dir external/datasets/MATH-500
```

The lock file pins the CPU build of torch. Treat `external/` as untrusted data: read code before running it, and run Python over it with `python -I`.

Pins:

| Input | Pin |
|---|---|
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |
| `RikiyaT/lm_understand-logs`, in `external/lm_understand-logs/` | `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` |
| `HuggingFaceH4/MATH-500` `test.jsonl`, in `external/datasets/MATH-500/` | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| llama.cpp `b12099`, in `external/llama.cpp/` | `bee23322f1baa3d2dd16b8339bed6aff4a8eb663` |

Expected SHA-256 values:

| File | SHA-256 |
|---|---|
| `external/datasets/MATH-500/test.jsonl` | `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132` |
| `models/Qwen3.5-0.8B/*.safetensors` | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| `models/Qwen3.5-0.8B-p50c/*.safetensors` | `fd1e42acdd118ac1b300d37500fd2885e75f5a09de3a33010d802bf7f737b553` |

## Verify the weights and grader

```bash
ST=model.safetensors-00001-of-00001.safetensors
EXP=external/lm_understand-logs/exp8_math
mkdir -p models/Qwen3.5-0.8B-p50c
python -I phase1/vendor/apply_edit.py models/Qwen3.5-0.8B/$ST $EXP/runs/p50c/edit models/Qwen3.5-0.8B-p50c/$ST
for f in chat_template.jinja config.json merges.txt model.safetensors.index.json tokenizer_config.json tokenizer.json vocab.json; do
    cp models/Qwen3.5-0.8B/$f models/Qwen3.5-0.8B-p50c/
done

python -I phase1/verify_edit.py models/Qwen3.5-0.8B/$ST models/Qwen3.5-0.8B-p50c/$ST $EXP/runs/p50c/edit/manifest.json
python -I phase1/check_grader.py $EXP external/datasets/MATH-500/test.jsonl
mkdir -p results/phase1
python -I phase1/prepare_problems.py models/Qwen3.5-0.8B external/datasets/MATH-500/test.jsonl $EXP results/phase1/problems.jsonl
```

`prepare_problems.py` should print 4/80 for the base model and 37/80 for the edited one.

## Build the engine and GGUFs

The build runs inside the sandbox.

```bash
cmake -S phase1/engine -B .tmp/build -DLLAMA_SRC=$R/external/llama.cpp
python3 -I phase1/sandbox.py --rw .tmp/build --ro phase1/engine --ro external/llama.cpp --ro .venv -- \
    bash -c "export PATH=$R/.venv/bin:\$PATH; cmake --build $R/.tmp/build --target generate logits sampler_test -j2"

mkdir -p models/gguf
python -I phase1/make_gguf.py models/Qwen3.5-0.8B models/Qwen3.5-0.8B/$ST models/gguf/base-f32.gguf
python -I phase1/make_gguf.py models/Qwen3.5-0.8B-p50c models/Qwen3.5-0.8B-p50c/$ST models/gguf/p50c-f32.gguf
python -I phase1/verify_gguf.py models/gguf/base-f32.gguf models/gguf/p50c-f32.gguf $EXP/runs/p50c/edit/manifest.json
```

## Tests

```bash
python3 -I phase1/test_sandbox.py
.tmp/build/sampler_test
mkdir -p .tmp/logits_check
python -I phase1/check_logits.py $R/models/Qwen3.5-0.8B $R/models/gguf/base-f32.gguf $R/.tmp/build/logits results/phase1/problems.jsonl \
    $EXP/eval/unedited/parts/math500_s1of10/math500_first_samples.jsonl.gz $R/.tmp/logits_check
python -I phase1/check_logits.py $R/models/Qwen3.5-0.8B-p50c $R/models/gguf/p50c-f32.gguf $R/.tmp/build/logits results/phase1/problems.jsonl \
    $EXP/eval/p50c/parts/math500_s1of10/math500_first_samples.jsonl.gz $R/.tmp/logits_check
```

`check_logits.py` needs about 6 GB of RAM, so do not run it during a generation. Stored output is in `results/phase1/logits_*.txt`.

## Run and score

```bash
setsid nohup .venv/bin/python -I phase1/run_local.py \
    --models edited=models/gguf/p50c-f32.gguf,base=models/gguf/base-f32.gguf \
    --tokenizer models/Qwen3.5-0.8B --problems results/phase1/problems.jsonl \
    --engine .tmp/build/generate --out results/phase1/local --samples 1 \
    >> results/phase1/local/run.log 2>&1 < /dev/null &

.venv/bin/python -I phase1/score_local.py results/phase1/local
```

To stop the run, kill `run_local.py` and then `generate` by PID, and confirm `generate` is gone before restarting. Two engines on one token file would corrupt the sample without any error (see `HANDOVER.md`). `pkill -f` can match your own shell, so avoid it.

To resume, run the same command again: finished samples are skipped and the unfinished one replays its saved tokens. Replay took 17 minutes at 32,000 tokens on the laptop.
