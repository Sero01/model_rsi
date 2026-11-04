@echo off
rem Build generate, logits and sampler_test with MSVC and CUDA. Run from the repo root.
rem Needs Visual Studio 2022 Build Tools (VC workload) and CUDA Toolkit 13.1. Arch 120 is Blackwell (RTX 50xx);
rem pass another compute capability as the first argument, e.g. build_cuda.bat 89 for Ada.
setlocal
set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=120"
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -products * -latest -property installationPath`) do set "VS=%%i"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if "%CUDA_PATH%"=="" set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cmake -S phase1/engine -B .tmp/build_cuda -G Ninja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=%ARCH% -DLLAMA_SRC=%CD:\=/%/external/llama.cpp || exit /b 1
cmake --build .tmp/build_cuda --target generate logits sampler_test -j 12 || exit /b 1
echo BUILD_DONE
