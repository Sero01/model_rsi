# Phase 1 Windows and local GPU run, 2026-11-04

- **Machine:** Windows 11, Core Ultra 7 255H (16 cores), 23 GB RAM, RTX 5050 Laptop GPU (8 GB, sm_120), driver CUDA 13.1. Run by a collaborator.
- **Done:** a native Windows port, in `phase1/WINDOWS.md`, and a CUDA engine with TF32 off. Every verification passed. The 20-problem local run is complete on the GPU: base 1/20 against 4/80 published, edited 9/20 against 37/80 published. The numbers are in `RESULTS.md`.
- **Not done:** the exit gate. It needs the full MATH-500 with 4 samples on a vLLM-class GPU, which is still unapproved. On this laptop GPU, the full protocol would take about 22 days.
- **Open:** the edited cut share was 45% against 33% published, which is within noise at n = 20. Watch it in the full run.
