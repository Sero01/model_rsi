# Phase 1 on Windows, CPU or NVIDIA GPU

Port of `phase1/README.md` to native Windows 11, with no WSL. The steps and pins are the same, and so are the checks and their expected output. Only the toolchain and the isolation change. Tested on a Core Ultra 7 255H with 23 GB RAM and an RTX 5050 Laptop GPU (8 GB). Results are in `RESULTS.md`.

## What differs from Linux

| Linux | Windows | Why |
|---|---|---|
| `python -I` | `python -I -X utf8` | Windows defaults to cp1252, which breaks `make_gguf.py`. `-I` ignores `PYTHONUTF8`, so the flag is required. |
| `sandbox.py` (Landlock and seccomp) | none | Landlock is Linux-only. `run_local.py` and `check_logits.py` call the engine directly on Windows. llama.cpp comes from the pinned official tag. |
| `nice -n 10` | `BELOW_NORMAL_PRIORITY_CLASS` | Set by `run_local.py`. |
| `flock` on `tokens.txt` | `LockFileEx` on one byte past EOF | Windows locks are mandatory, so locking past EOF never blocks appends. `test_generate.py` still checks that a second engine is refused. |
| math-verify `signal.alarm` timeouts | no timeouts | Windows has no `SIGALRM`, and math-verify's process-spawn fallback crashes under `-I`. `mathqa.py` already graded without timeouts off the main thread, and `check_grader.py` still finds 0 disagreements. |
| `test_generate.py` write-failure case | skipped | There is no `RLIMIT_FSIZE` to stage it. |
| `test_sandbox.py` | not applicable | Linux-only. |

The engine also has an opt-in GPU path. `RSI_GPU_LAYERS=99` offloads every layer. When it is unset, the engine keeps the verified CPU path (`n_gpu_layers = 0`).

## Setup

PowerShell, from the repo root. Downloads and clones are the same as in `phase1/README.md`, with `.venv\Scripts\python.exe` and `.venv\Scripts\hf.exe`.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r phase1/requirements.lock --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
$PY = ".venv\Scripts\python.exe"
```

Every `python -I phase1/...` command in `phase1/README.md` then runs as `& $PY -I -X utf8 phase1/...`.

## CPU engine (MinGW-w64 GCC)

```powershell
winget install --id BrechtSanders.WinLibs.POSIX.UCRT
```

GCC's linker fails when its install path contains a space, as in `C:\Users\First Last\...`. In that case, copy `...\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_*\mingw64` to a path without spaces and use that copy:

```powershell
$env:PATH = "C:\path\without\spaces\mingw64\bin;$env:PATH"
cmake -S phase1/engine -B .tmp/build -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ "-DLLAMA_SRC=$((Get-Location).Path -replace '\\','/')/external/llama.cpp"
cmake --build .tmp/build --target generate logits sampler_test -j 14
```

Copy `libgcc_s_seh-1.dll`, `libstdc++-6.dll`, `libwinpthread-1.dll`, `libgomp-1.dll` and `libdl.dll` from `mingw64\bin` into `.tmp\build` so the executables run without MinGW on `PATH`.

## GPU engine (MSVC and CUDA)

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --quiet --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install --id Nvidia.CUDA --version 13.1
cmd /c phase1\windows\build_cuda.bat 120
```

Pick a CUDA toolkit no newer than the driver's `CUDA Version` in `nvidia-smi`. The argument is the GPU's compute capability without the dot. Find it with `nvidia-smi --query-gpu=compute_cap --format=csv`.

Every GPU command needs this environment:

```powershell
$env:PATH = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64;$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;$env:PATH"
$env:RSI_GPU_LAYERS = "99"
$env:NVIDIA_TF32_OVERRIDE = "0"   # required, see below
```

**TF32 must be off.** llama.cpp sets cuBLAS to TF32 tensor-core math (`ggml-cuda/common.cuh`), which keeps 10 mantissa bits in batched matmuls. With it on, `check_logits.py` fails 4 of 6 cases on the base model, with errors up to 0.0106 against the 0.001 limit. With `NVIDIA_TF32_OVERRIDE=0`, both models pass 12 of 12, with errors of at most 0.00006.

## Checks

```powershell
.tmp\build_cuda\sampler_test.exe
& $PY -I -X utf8 phase1/test_generate.py models/gguf/base-f32.gguf .tmp/build_cuda/generate.exe .tmp/test_generate
& $PY -I -X utf8 phase1/check_logits.py <abs model dir> <abs gguf> <abs .tmp/build_cuda/logits.exe> results/phase1/problems.jsonl <published samples .jsonl.gz> <abs work dir>
```

For the CPU engine, use `.tmp\build` and leave out the three GPU environment variables.

## Run and score

The two engines should not share an 8 GB card: one engine takes about 4.1 GB at full context. Run one process with both models, in the GPU environment above:

```powershell
Start-Process -FilePath $PY -WindowStyle Hidden -RedirectStandardOutput results\phase1\local_gpu\run.log -RedirectStandardError results\phase1\local_gpu\run.err `
  -ArgumentList "-I -X utf8 phase1/run_local.py --models edited=models/gguf/p50c-f32.gguf,base=models/gguf/base-f32.gguf --tokenizer models/Qwen3.5-0.8B --problems results/phase1/problems.jsonl --engine .tmp/build_cuda/generate.exe --out results/phase1/local_gpu --samples 1"
& $PY -I -X utf8 phase1/score_local.py results/phase1/local_gpu
```

On CPU, 4 threads was fastest on the 255H, because more threads ran slower. Set `RSI_THREADS` to change it on Windows. To stop a run, stop the `run_local.py` processes by PID first, then `generate.exe`.
