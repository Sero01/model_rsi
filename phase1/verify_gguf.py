"""Verify the base and edited GGUFs against each other and against MIT's manifest.

    python -I phase1/verify_gguf.py <base.gguf> <edited.gguf> <manifest.json>

Requires exactly the manifest's tensors to differ, and those tensors, cast back to BF16, to match MIT's base and edited
hashes. down_proj has no value transform in the conversion, so these are the same bits MIT evaluated.
Exits non-zero on any mismatch. Prints counts only, never weights.
"""
import hashlib
import json
import os
import re
import sys

import gguf
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import st_numpy as S  # noqa: E402


def gguf_tensor_name(hf_name):
    layer = re.search(r"layers\.(\d+)\.mlp\.down_proj\.weight$", hf_name).group(1)
    return f"blk.{layer}.ffn_down.weight"


def bf16_sha(values):
    return hashlib.sha256(np.ascontiguousarray(S.bf16_bits(values)).tobytes()).hexdigest()


def main(base_path, edited_path, manifest_path):
    specs = {gguf_tensor_name(t["name"]): t for t in json.load(open(manifest_path))["tensors"]}
    base = {t.name: t for t in gguf.GGUFReader(base_path).tensors}
    edit = {t.name: t for t in gguf.GGUFReader(edited_path).tensors}
    problems = []
    if set(base) != set(edit):
        problems.append("tensor names differ")
    changed = sorted(n for n in base if n in edit and not np.array_equal(base[n].data, edit[n].data))
    print(f"{len(base)} tensors compared, {len(changed)} differ: {changed}")
    if set(changed) != set(specs):
        problems.append(f"changed tensors differ from manifest: {sorted(specs)}")
    for name, spec in specs.items():
        b, e = np.asarray(base[name].data), np.asarray(edit[name].data)
        n_diff = int((b != e).sum())
        print(f"{name}: shape {list(b.shape)}, {n_diff:,} entries differ")
        if list(b.shape) != spec["shape"]:
            problems.append(f"{name}: shape")
        if n_diff != spec["changed_entries"]:
            problems.append(f"{name}: changed entries {n_diff}")
        if bf16_sha(b) != spec["base_sha256"]:
            problems.append(f"{name}: base hash")
        if bf16_sha(e) != spec["edited_sha256"]:
            problems.append(f"{name}: edited hash")
    for p in problems:
        print("MISMATCH", p)
    print("OK" if not problems else "FAILED")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main(*sys.argv[1:4])
