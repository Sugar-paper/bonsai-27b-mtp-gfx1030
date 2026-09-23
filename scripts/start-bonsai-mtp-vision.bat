@echo off
setlocal EnableExtensions

rem ===========================================================================
rem  Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0 + OFFICIAL VISION TOWER
rem  Radeon RX 6900 XT (gfx1030 / RDNA2 / 16 GB), Windows + cmd
rem ===========================================================================
rem  Same runtime as start-bonsai-mtp.bat plus the Bonsai mmproj (Qwen-VL class):
rem    --mmproj <proj.gguf> --no-mmproj-offload --image-min-tokens 1024
rem
rem  --no-mmproj-offload keeps the vision tower in host RAM (saves ~0.6 GB VRAM;
rem  the projector's own loader warns that Qwen-VL wants >=1024 image tokens for
rem  grounding accuracy, hence IMGMINTOK).
rem
rem  Cold start note: the FIRST image request after loading costs ~2 minutes
rem  (projector warm-up + kernel compile on the CPU-side tower). Warm repeats
rem  are ~2 s. This is expected, not a hang.
rem
rem  Usage:
rem     start-bonsai-mtp-vision.bat              (ctx 262144, port 11234)
rem     start-bonsai-mtp-vision.bat --check
rem     set NOVISION=1 ^& start-bonsai-mtp-vision.bat   (text only)
rem     set IMGMINTOK=0 ^& start-bonsai-mtp-vision.bat  (cheaper images, weaker grounding)
rem ===========================================================================

set "ROOT=%~dp0.."
set "RUNTIME=%ROOT%\runtime"
set "HIPCOMPAT=%ROOT%\hip-compat"
set "TEMPLATE=%ROOT%\chat-templates\abl-27b-reasoning-compat.jinja"
set "MODEL=%ROOT%\model\Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf"
set "MMPROJ=%ROOT%\model\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
set "ROCM=%ProgramFiles%\AMD\ROCm\7.1"

if not defined CTX       set "CTX=262144"
if not defined PORT      set "PORT=11234"
if not defined EFFORT    set "EFFORT=low"
if not defined NMTP      set "NMTP=3"
if not defined THREADS   set "THREADS=12"
if not defined BUDGET    set "BUDGET=36864"
if not defined GENRES    set "GENRES=16384"
if not defined IMGMINTOK set "IMGMINTOK=1024"
if not defined NOVISION  set "NOVISION=0"
if not defined MMPROJGPU set "MMPROJGPU=0"
if not defined HOSTBIND  set "HOSTBIND=127.0.0.1"
if not defined ALIAS     set "ALIAS=bonsai-27b-mtp-vision"

set "CHECK=0"
if /i "%~1"=="--check" set "CHECK=1"

set "HIP_VISIBLE_DEVICES="
set "ROCR_VISIBLE_DEVICES="
set "HSA_OVERRIDE_GFX_VERSION="
set "HIP_PATH=%ROCM%\"
set "HIP_PATH_64=%ROCM%\"
set "ROCBLAS_TENSILE_LIBPATH=%RUNTIME%\rocblas\library"
set "HIPBLASLT_TENSILE_LIBPATH=%RUNTIME%\hipblaslt\library"
set "PATH=%HIPCOMPAT%;%RUNTIME%;%ROCM%\bin;%RUNTIME%\rocblas;%RUNTIME%\hipblaslt;%PATH%"

if not exist "%RUNTIME%\llama-kvmem-server.exe" ( echo [ERROR] runtime not found: %RUNTIME%\llama-kvmem-server.exe & goto fail )
if not exist "%MODEL%" ( echo [ERROR] model not found: %MODEL% & echo         see README.md for the download links & goto fail )
if not exist "%TEMPLATE%" ( echo [WARN] compat chat template missing & set "TEMPLATE=" )
if "%NOVISION%"=="1" goto vision_ready
if not exist "%MMPROJ%" ( echo [warn] projector not found ^(%MMPROJ%^) - continuing text-only. & set "NOVISION=1" )
:vision_ready

set "VFLAGS="
if "%NOVISION%"=="0" set "VFLAGS=--mmproj "%MMPROJ%" --no-mmproj-offload"
if "%NOVISION%"=="0" if "%MMPROJGPU%"=="1" set "VFLAGS=--mmproj "%MMPROJ%""
if "%NOVISION%"=="0" if not "%IMGMINTOK%"=="0" set "VFLAGS=%VFLAGS% --image-min-tokens %IMGMINTOK%"

echo ===========================================================================
echo  Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0 + vision  ^|  %CTX% ctx, MTP x%NMTP%
echo ===========================================================================
echo  runtime : %RUNTIME%
echo  model   : %MODEL%
if "%NOVISION%"=="1" ( echo  vision  : OFF ) else ( echo  vision  : ON   mmproj=%MMPROJ% )
echo  think   : on, effort=%EFFORT%   http://%HOSTBIND%:%PORT%
echo  stop    : Ctrl+C
echo ===========================================================================

if "%CHECK%"=="1" (
    echo [check] "%RUNTIME%\llama-kvmem-server.exe" -m "%MODEL%" --alias %ALIAS% %VFLAGS% -ngl 999 -c %CTX% --kv-dtype q8_0 -np 1 --host %HOSTBIND% --port %PORT% --threads %THREADS% --jinja --chat-template-file "%TEMPLATE%" --chat-template-kwargs "{\"enable_thinking\":true}" --reasoning-effort %EFFORT% --spec-type draft-mtp --spec-draft-n-max %NMTP% --kvmem-budget %BUDGET% --kvmem-gen-reserve %GENRES% --kvmem-method retrieval --kvmem-mtp-state snapshots
    goto done
)

if "%TEMPLATE%"=="" goto launch_plain

"%RUNTIME%\llama-kvmem-server.exe" ^
  -m "%MODEL%" --alias %ALIAS% %VFLAGS% ^
  -ngl 999 -c %CTX% --kv-dtype q8_0 -np 1 ^
  --host %HOSTBIND% --port %PORT% --threads %THREADS% ^
  --jinja --chat-template-file "%TEMPLATE%" ^
  --chat-template-kwargs "{\"enable_thinking\":true}" ^
  --reasoning-effort %EFFORT% ^
  --spec-type draft-mtp --spec-draft-n-max %NMTP% ^
  --kvmem-budget %BUDGET% --kvmem-gen-reserve %GENRES% ^
  --kvmem-method retrieval --kvmem-mtp-state snapshots
goto done

:launch_plain
"%RUNTIME%\llama-kvmem-server.exe" ^
  -m "%MODEL%" --alias %ALIAS% %VFLAGS% ^
  -ngl 999 -c %CTX% --kv-dtype q8_0 -np 1 ^
  --host %HOSTBIND% --port %PORT% --threads %THREADS% ^
  --reasoning-effort %EFFORT% ^
  --spec-type draft-mtp --spec-draft-n-max %NMTP% ^
  --kvmem-budget %BUDGET% --kvmem-gen-reserve %GENRES% ^
  --kvmem-method retrieval --kvmem-mtp-state snapshots

:done
echo.
if "%CHECK%"=="1" exit /b 0
echo [server exited with code %ERRORLEVEL%]
pause
exit /b %ERRORLEVEL%

:fail
if "%CHECK%"=="1" exit /b 1
pause
exit /b 1
