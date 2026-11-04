"""Select the local sample, render its prompts, and attach MIT's published per-problem scores.

    python -I phase1/prepare_problems.py <model dir> <math500.jsonl> <exp8_math dir> <out.jsonl>

Selection is rows[0::25] of MATH-500, 20 problems. Prompts use the protocol's chat template with thinking on.
"""
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import mathqa as Q  # noqa: E402

STRIDE = 25
SHARDS = 10
CHECKPOINTS = ("unedited", "p50c")
GENERATION_PROMPT = "<|im_start|>assistant\n<think>\n"


def published_scores(logs, name):
    scores = {}
    for i in range(SHARDS):
        path = os.path.join(logs, "eval", name, "parts", f"math500_s{i}of{SHARDS}", "math500_per_problem.json")
        scores.update({r["id"]: r["correct"] for r in json.load(open(path))})
    return scores


def main(model_dir, dataset, logs, out):
    tok = AutoTokenizer.from_pretrained(model_dir)
    rows = [json.loads(line) for line in open(dataset)][::STRIDE]
    published = {name: published_scores(logs, name) for name in CHECKPOINTS}
    with open(out, "w") as f:
        for row in rows:
            messages = [{"role": "user", "content": Q.build_prompt(row["problem"])}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            if not prompt.endswith(GENERATION_PROMPT) or Q.INSTRUCTION not in prompt:
                raise SystemExit(f"{row['unique_id']}: unexpected rendered prompt ending: {prompt[-60:]!r}")
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f.write(json.dumps({"id": row["unique_id"], "answer": row["answer"], "subject": row["subject"],
                                "level": row["level"], "prompt": prompt, "prompt_ids": ids,
                                "published": {n: published[n][row["unique_id"]] for n in CHECKPOINTS}}) + "\n")
    print(f"{len(rows)} problems -> {out}")
    print("sample prompt:", repr(prompt[:60]), "...", repr(prompt[-90:]))
    for n in CHECKPOINTS:
        correct = [sum(published[n][r["unique_id"]]) for r in rows]
        print(f"published {n}: {sum(correct)}/{4 * len(rows)} = {100 * sum(correct) / (4 * len(rows)):.1f}% on this sample")


if __name__ == "__main__":
    main(*sys.argv[1:5])
