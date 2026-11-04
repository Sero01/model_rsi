"""Verify a rebuilt edited checkpoint against the base model and MIT's manifest.

    python -I phase1/verify_edit.py <base.safetensors> <edited.safetensors> <manifest.json>

Compares every tensor's bits, then requires the manifest's tensors, hashes, entry counts, column counts and relative
changes. Exits non-zero on any mismatch. Prints counts and norms only, never weights.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import st_numpy as S  # noqa: E402

TOUCHED_COLUMNS = 64
REL_TOLERANCE = 1e-6


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def read_tensor(f, data_start, info):
    a, b = info["data_offsets"]
    f.seek(data_start + a)
    return f.read(b - a)


def changed_tensors(base_path, edited_path):
    """Names whose bytes differ. Reads one tensor at a time, so memory stays near the largest tensor."""
    h_base, off_base = S.read_header(base_path)
    h_edit, off_edit = S.read_header(edited_path)
    h_base.pop("__metadata__", None)
    h_edit.pop("__metadata__", None)
    if set(h_base) != set(h_edit):
        raise SystemExit(f"tensor names differ: {sorted(set(h_base) ^ set(h_edit))[:10]}")
    changed = []
    with open(base_path, "rb") as fb, open(edited_path, "rb") as fe:
        for name, info in h_base.items():
            other = h_edit[name]
            if (info["dtype"], info["shape"]) != (other["dtype"], other["shape"]):
                raise SystemExit(f"{name}: dtype or shape differs")
            if read_tensor(fb, off_base, info) != read_tensor(fe, off_edit, other):
                changed.append(name)
    return len(h_base), changed


def check_tensor(spec, base_bits, edit_bits):
    """Problems found in one edited tensor, plus a printable summary."""
    diff = base_bits != edit_bits
    base32, edit32 = S.bf16_to_f32(base_bits), S.bf16_to_f32(edit_bits)
    rel = float(np.linalg.norm(edit32 - base32) / np.linalg.norm(base32))
    n_entries, n_cols, n_rows = int(diff.sum()), int(diff.any(axis=0).sum()), int(diff.any(axis=1).sum())
    problems = []
    if sha(base_bits) != spec["base_sha256"]:
        problems.append("base_sha256")
    if sha(edit_bits) != spec["edited_sha256"]:
        problems.append("edited_sha256")
    if n_entries != spec["changed_entries"]:
        problems.append(f"changed_entries {n_entries}")
    if n_cols != TOUCHED_COLUMNS or n_cols != spec["changed_columns"]:
        problems.append(f"changed_columns {n_cols}")
    if n_rows != spec["changed_rows"]:
        problems.append(f"changed_rows {n_rows}")
    if abs(rel - spec["relative_change"]) > REL_TOLERANCE:
        problems.append(f"relative_change {rel}")
    summary = (f"{spec['name']}: {n_entries:,} entries, {n_cols} columns ({n_cols * base_bits.shape[0]:,} touched), "
               f"{n_rows} rows, relative change {rel:.5f}")
    return problems, summary


def main(base_path, edited_path, manifest_path):
    specs = {t["name"]: t for t in json.load(open(manifest_path))["tensors"]}
    n_tensors, changed = changed_tensors(base_path, edited_path)
    print(f"{n_tensors} tensors compared, {len(changed)} differ")
    problems = []
    if set(changed) != set(specs):
        problems.append(f"changed tensors {changed} != manifest {sorted(specs)}")
    base = S.load(base_path, set(specs))
    edit = S.load(edited_path, set(specs))
    total = touched = 0
    for name, spec in specs.items():
        tensor_problems, summary = check_tensor(spec, base[name][1], edit[name][1])
        print(summary)
        problems += [f"{name}: {p}" for p in tensor_problems]
        total += int((base[name][1] != edit[name][1]).sum())
        touched += TOUCHED_COLUMNS * base[name][1].shape[0]
    print(f"total: {total:,} entries differ, {touched:,} touched")
    for p in problems:
        print("MISMATCH", p)
    print("OK" if not problems else "FAILED")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main(*sys.argv[1:4])
