"""Run the local MATH-500 sample on CPU with our llama.cpp engine. Safe to stop and rerun at any time.

    python -I phase1/run_local.py --models base=<base.gguf>,edited=<edited.gguf> --tokenizer <model dir> \
        --problems results/phase1/problems.jsonl --engine <generate binary> --out results/phase1/local --samples 1

Runs model by model within each problem, so a partial run still gives paired results. Each sample keeps a token file,
which generate resumes from, and a JSON record once finished. Finished samples are skipped on rerun.
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import time

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))
import mathqa as Q  # noqa: E402

SANDBOX = os.path.join(HERE, "sandbox.py")
BASE_SEED = 0  # the protocol's seed
SEEDS_PER_PROBLEM = 1000
THREADS = 4
NICE = 10


def sample_seed(problem_index, sample):
    return BASE_SEED + problem_index * SEEDS_PER_PROBLEM + sample


def run_sample(args, gguf_path, problem, problem_index, sample, work):
    """Generate one sample. Returns the record of the finished generation."""
    os.makedirs(work, exist_ok=True)
    prompt_path, tokens_path = os.path.join(work, "prompt.txt"), os.path.join(work, "tokens.txt")
    with open(prompt_path, "w") as f:
        f.write("\n".join(map(str, problem["prompt_ids"])))
    cmd = ["nice", "-n", str(NICE), sys.executable, "-I", SANDBOX, "--rw", work, "--ro", gguf_path, "--ro", args.engine,
           "--", args.engine, gguf_path, prompt_path, tokens_path, str(sample_seed(problem_index, sample)), str(THREADS),
           str(args.max_new_tokens)]
    started = time.time()
    with open(os.path.join(work, "engine.log"), "a") as log:
        done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True)
    if done.returncode != 0:
        raise SystemExit(f"generate failed for {work}, see engine.log")
    result = json.loads(done.stdout.strip().splitlines()[-1])
    result["wall_seconds"] = round(time.time() - started, 1)
    return result


def grade(tok, work, answer):
    with open(os.path.join(work, "tokens.txt")) as f:
        ids = [int(line) for line in f if line.strip()]
    text = tok.decode(ids, skip_special_tokens=True)
    pred = Q.extract(text)[0]
    return text, pred, bool(Q.is_correct(pred, answer))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="label=path,label=path")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    args = parser.parse_args()
    args.engine = os.path.abspath(args.engine)
    args.out = os.path.abspath(args.out)

    models = dict(item.split("=", 1) for item in args.models.split(","))
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.problems) as f:
        problems = [json.loads(line) for line in f][:args.max_problems]

    for sample in range(args.samples):
        for index, problem in enumerate(problems):
            for label, gguf_path in models.items():
                name = f"p{index:02d}_s{sample}"
                record_path = os.path.join(args.out, label, f"{name}.json")
                if os.path.exists(record_path):
                    continue
                work = os.path.join(args.out, label, name + ".work")
                result = run_sample(args, os.path.abspath(gguf_path), problem, index, sample, work)
                text, pred, correct = grade(tok, work, problem["answer"])
                with gzip.open(os.path.join(args.out, label, f"{name}.txt.gz"), "wt") as f:
                    f.write(text)
                record = {"id": problem["id"], "index": index, "sample": sample, "label": label, "answer": problem["answer"],
                          "pred": pred, "correct": correct, "seed": sample_seed(index, sample),
                          "published_correct": problem["published"]["unedited" if label == "base" else "p50c"], **result}
                with open(record_path + ".tmp", "w") as f:
                    json.dump(record, f, indent=1)
                os.replace(record_path + ".tmp", record_path)
                print(f"{label} {name}: {result['finish']}, {result['tokens']} tokens, {result['seconds']:.0f} s, "
                      f"correct={correct}", flush=True)


if __name__ == "__main__":
    main()
