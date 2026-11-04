# Lessons

| # | Mistake | Rule | Context |
|---|---------|------|---------|
| 1 | `pkill -f <pattern>` matched its own shell and killed the command partway through. | Stop processes by exact PID. | Stopping the local run. |
| 2 | A background job kept running after `run_local.py` was killed, because `generate` is not tied to its parent. | Confirm `generate` has exited before restarting. Two engines on one token file corrupt the sample without any error. | Found in review. The fix is listed in `HANDOVER.md`. |
| 3 | A sandboxed command with a relative binary path printed nothing, because `sandbox.py` changes into its first `--rw` directory. | Use absolute paths inside `sandbox.py`. | Thread count test. |
| 4 | The first logit threshold (0.02) was a guess. | Set thresholds from a measured yardstick, and chase any deviation to its cause. | Flash attention moved top log-probs by 0.03. With it off, the error is 0.00002. |
| 5 | The handover's 20 tok/s was a placeholder, and the real speed was 2 to 3 tok/s. | Measure speed before fixing a sample design. | CPU projection went from hours to 5.2 days. |
| 6 | Checks that stop at 2,000 tokens say little about a sample that runs to 32,768. | Test at the real context size, and in the real batching path. | Open item from the review. |
| 7 | A new script failed on `import check_logits`, because `python -I` leaves the script's directory off the import path. | Scripts run with `-I` must add their own directory to `sys.path`. | `check_logits_long.py`. |
| 8 | Going public raises checks beyond secrets. | Before any public push, also check commit identity, paths in logs, and redistribution rights for data and vendored code. | First push of this repo. |
