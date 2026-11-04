"""Summarize the local run per model and compare it with MIT's published figures on the same problems.

    python -I phase1/score_local.py <results dir> > results/phase1/local_summary.txt

Standard errors are clustered by problem. Published cut share and token counts exist only per shard, so those are
compared with the whole-benchmark figures.
"""
import glob
import json
import math
import os
import sys

PUBLISHED = {"base": {"acc": 7.90, "cut": 93.0, "tokens": 30868}, "edited": {"acc": 48.10, "cut": 33.05, "tokens": 22037}}


def clustered_se(by_problem):
    """Standard error of the mean of problem means. Treats problems as the independent units."""
    means = [sum(v) / len(v) for v in by_problem.values()]
    n = len(means)
    if n < 2:
        return float("nan")
    mean = sum(means) / n
    return math.sqrt(sum((m - mean) ** 2 for m in means) / (n - 1) / n)


def summarize(label, records):
    by_problem = {}
    for r in records:
        by_problem.setdefault(r["id"], []).append(int(r["correct"]))
    n = len(records)
    new_tokens = sum(r["new_tokens"] for r in records)
    seconds = sum(r["seconds"] for r in records)
    acc = 100 * sum(r["correct"] for r in records) / n
    published_same = [c for r in records for c in r["published_correct"]]
    pub = PUBLISHED[label]
    print(f"{label}: {n} samples over {len(by_problem)} problems")
    print(f"  accuracy        {acc:.1f}% (clustered SE {100 * clustered_se(by_problem):.1f})")
    print(f"  published       {pub['acc']:.1f}% on all 500, {100 * sum(published_same) / len(published_same):.1f}% on these problems")
    print(f"  cut at cap      {100 * sum(r['finish'] == 'length' for r in records) / n:.1f}% (published {pub['cut']:.1f}%)")
    print(f"  mean tokens     {sum(r['tokens'] for r in records) / n:,.0f} (published {pub['tokens']:,})")
    print(f"  speed           {new_tokens / seconds:.2f} tok/s over {seconds / 3600:.1f} h")


def main(results):
    for label in PUBLISHED:
        paths = sorted(glob.glob(os.path.join(results, label, "p*_s*.json")))
        if paths:
            summarize(label, [json.load(open(p)) for p in paths])


if __name__ == "__main__":
    main(sys.argv[1])
