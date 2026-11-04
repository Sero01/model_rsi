"""Regrade MIT's published MATH-500 samples with the vendored grader and require agreement with their per-problem data.

    python -I phase1/check_grader.py <exp8_math dir> <math500.jsonl>

Two checks per checkpoint:
  text:  the published text of a first sample -> extract + grade == published pred_first and correct[0].
         evaluate.py keeps text for only the first 20 problems of each shard, so this covers 200 problems.
  preds: the published pred_first of all 500 problems, graded against the dataset's gold == published correct[0].
Exits non-zero on any disagreement.
"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import mathqa as Q  # noqa: E402

CHECKPOINTS = ("unedited", "p50c")
SHARDS = 10


def check_checkpoint(logs, name, gold):
    bad = []
    n_text = n_preds = 0
    for i in range(SHARDS):
        part = os.path.join(logs, "eval", name, "parts", f"math500_s{i}of{SHARDS}")
        per = {r["id"]: r for r in json.load(open(os.path.join(part, "math500_per_problem.json")))}
        with gzip.open(os.path.join(part, "math500_first_samples.jsonl.gz"), "rt") as f:
            for row in map(json.loads, f):
                pub = per[row["id"]]
                pred = Q.extract(row["text"])[0]
                ok = int(Q.is_correct(pred, row["answer"]))
                n_text += 1
                if pred != pub["pred_first"] or ok != pub["correct"][0]:
                    bad.append(("text", row["id"], pred, pub["pred_first"], ok, pub["correct"][0]))
        for pid, pub in per.items():
            ok = int(Q.is_correct(pub["pred_first"], gold[pid]))
            n_preds += 1
            if ok != pub["correct"][0]:
                bad.append(("preds", pid, pub["pred_first"], gold[pid], ok, pub["correct"][0]))
    print(f"{name}: {n_text} texts and {n_preds} predictions regraded, {len(bad)} disagreements")
    for b in bad:
        print("  ", b)
    return bad


def main(logs, dataset):
    gold = {r["unique_id"]: r["answer"] for r in map(json.loads, open(dataset))}
    bad = [b for name in CHECKPOINTS for b in check_checkpoint(logs, name, gold)]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(*sys.argv[1:3])
