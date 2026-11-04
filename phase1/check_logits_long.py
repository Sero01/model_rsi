"""Compare our engine's logits with transformers at long context, in generate's configuration.

    python -I phase1/check_logits_long.py <model dir> <model.gguf> <logits binary> <problems.jsonl> <published samples.jsonl.gz> <work dir>

A single forward pass at this length does not fit in RAM, so the reference feeds a published transcript through
transformers in cached chunks, which agreed with a full pass to 0.00001 at 2,000 tokens. Exits non-zero on any failure.
"""
import gc
import gzip
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_logits as C  # noqa: E402

CHECKPOINTS = (4096, 8000, 16000, 32000)
LAST_STEPPED = 50


def long_transcript(tok, samples, prompt):
    """The prompt plus the first published answer long enough for the last checkpoint."""
    with gzip.open(samples, "rt") as f:
        for line in f:
            ids = prompt + tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"]
            if len(ids) >= CHECKPOINTS[-1]:
                return ids
    raise SystemExit("no published answer is long enough")


def reference_logits(model_dir, ids):
    """Logits after each checkpoint, feeding chunks that end exactly on the checkpoints."""
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    logits, cache, done, started = {}, None, 0, time.time()
    with torch.no_grad():
        for checkpoint in CHECKPOINTS:
            while done < checkpoint:
                end = min(done + C.CHUNK, checkpoint)
                out = model(input_ids=torch.tensor([ids[done:end]]), past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache, done = out.past_key_values, end
            logits[checkpoint] = out.logits[0, -1].numpy().copy()
            print(f"reference at {checkpoint}: {(time.time() - started) / 3600:.2f} h", flush=True)
    del model, cache, out
    gc.collect()
    return logits


def main(model_dir, gguf_path, binary, problems, samples, work):
    tok = AutoTokenizer.from_pretrained(model_dir)
    with open(problems) as f:
        prompt = json.loads(f.readline())["prompt_ids"]
    ids = long_transcript(tok, samples, prompt)
    print(f"transcript: {len(ids)} tokens", flush=True)
    reference = reference_logits(model_dir, ids)

    context = (len(prompt) + C.MAX_NEW_TOKENS, C.CHUNK)
    ok = True
    for checkpoint in CHECKPOINTS:
        candidate = C.llama_logits(binary, gguf_path, ids[:checkpoint], checkpoint - LAST_STEPPED, work, context)
        ok &= C.compare(f"{checkpoint} tokens", reference[checkpoint], candidate)
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:7])
