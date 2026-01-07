@echo off
setlocal enabledelayedexpansion

echo [1/5] Checking prerequisites...

:: Find candle-flash-attn checkout directory (search in git checkouts)
set "FLASH_ATTN_DIR="
for /d %%A in ("%USERPROFILE%\.cargo\git\checkouts\candle-*") do (
    for /d %%B in ("%%A\*") do (
        if exist "%%B\candle-flash-attn\kernels" (
            set "FLASH_ATTN_DIR=%%B\candle-flash-attn"
        )
    )
)

if not defined FLASH_ATTN_DIR (
    echo ERROR: candle-flash-attn not found in cargo git checkouts.
    echo Run 'cargo check --lib --features flash-attn' first to download dependencies.
    exit /b 1
)

echo Found candle-flash-attn at: %FLASH_ATTN_DIR%

:: Check if CUTLASS already exists
if exist "%FLASH_ATTN_DIR%\kernels\cutlass\include\cute\tensor.hpp" (
    echo [2/5] CUTLASS already installed. Skipping clone.
    goto :build
)

echo [2/5] Cloning CUTLASS v3.5.1 into %FLASH_ATTN_DIR%\kernels\cutlass...

:: Clone CUTLASS (v3.5.1 is compatible with Flash Attention 2)
cd /d "%FLASH_ATTN_DIR%\kernels"
if exist cutlass rmdir /s /q cutlass

git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git cutlass
if errorlevel 1 (
    echo ERROR: Failed to clone CUTLASS
    exit /b 1
)

echo [3/5] CUTLASS cloned successfully.

:build
echo [4/5] Building with Flash Attention...

cd /d "%~dp0"

:: Initialize VS environment if not already done
where cl >nul 2>&1
if errorlevel 1 (
    echo Initializing Visual Studio 2022 environment...
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

:: Set CUDA path
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "PATH=%CUDA_PATH%\bin;%PATH%"

:: Build
cargo build --lib --features flash-attn

if errorlevel 1 (
    echo.
    echo [5/5] Build failed. This is expected on Windows - nvcc cannot link directly.
    echo Creating static library manually...
    
    for /d %%D in (target\debug\build\candle-flash-attn-*) do (
        if exist "%%D\out\*.o" (
            lib /NOLOGO /OUT:"%%D\out\libflashattention.a" "%%D\out\*.o"
            echo Created: %%D\out\libflashattention.a
        )
    )
    
    echo Retrying build...
    cargo build --lib --features flash-attn
)

echo.
echo [5/5] Build completed!
endlocal
