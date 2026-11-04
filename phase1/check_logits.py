"""Compare our llama.cpp engine's logits with transformers, the reference implementation, on real prompts.

    python -I phase1/check_logits.py <model dir> <model.gguf> <logits binary> <problems.jsonl> <published samples.jsonl.gz> <work dir>

Cases are a short prompt, and a prompt plus the start of a published answer. Each runs batched and with the last tokens
decoded one at a time, as generation does. Exits non-zero if the distribution the sampler would see differs materially.
"""
import gzip
import json
import os
import subprocess
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SANDBOX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox.py")
LONG_LENGTH = 2000
STEPPED_TOKENS = 50
THREADS = 4
TOP_K = 20
MAX_TOP_LOGPROB_ERROR = 0.001  # measured 0.00002 at most; flash attention, which we disable, gives 0.03
MIN_TOP_K_OVERLAP = 19


def llama_logits(binary, gguf_path, ids, n_batched, work):
    ids_path, out_path = os.path.join(work, "ids.txt"), os.path.join(work, "out.f32")
    with open(ids_path, "w") as f:
        f.write("\n".join(map(str, ids)))
    cmd = [sys.executable, "-I", SANDBOX, "--rw", work, "--ro", gguf_path, "--ro", binary, "--",
           binary, gguf_path, ids_path, out_path, str(THREADS), str(n_batched)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"logits failed:\n{done.stderr[-2000:]}")
    return np.fromfile(out_path, dtype=np.float32)


def compare(name, reference, candidate):
    ref = torch.log_softmax(torch.from_numpy(reference), dim=-1)
    can = torch.log_softmax(torch.from_numpy(candidate), dim=-1)
    top_ref = torch.topk(ref, TOP_K).indices
    top_can = torch.topk(can, TOP_K).indices
    overlap = len(set(top_ref.tolist()) & set(top_can.tolist()))
    error = float((ref[top_ref] - can[top_ref]).abs().max())
    ok = overlap >= MIN_TOP_K_OVERLAP and error <= MAX_TOP_LOGPROB_ERROR
    print(f"{'PASS' if ok else 'FAIL'}: {name}: top-{TOP_K} overlap {overlap}, max log-prob error {error:.5f}, "
          f"same argmax {bool(top_ref[0] == top_can[0])}", flush=True)
    return ok


def main(model_dir, gguf_path, binary, problems, samples, work):
    tok = AutoTokenizer.from_pretrained(model_dir)
    with open(problems) as f:
        first = json.loads(f.readline())
    with gzip.open(samples, "rt") as f:
        text = json.loads(f.readline())["text"]
    short = first["prompt_ids"]
    long_ids = (short + tok(text, add_special_tokens=False)["input_ids"])[:LONG_LENGTH]

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    print("reference class:", type(model).__name__, flush=True)

    ok = True
    for name, ids in (("short", short), ("long", long_ids)):
        with torch.no_grad():
            reference = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].numpy().copy()
        print(f"{name}: {len(ids)} tokens, reference done", flush=True)
        stepped = max(1, len(ids) - STEPPED_TOKENS)
        ok &= compare(f"{name} batched", reference, llama_logits(binary, gguf_path, ids, len(ids), work))
        ok &= compare(f"{name} last tokens stepped", reference, llama_logits(binary, gguf_path, ids, stepped, work))
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:7])
