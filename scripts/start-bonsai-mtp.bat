@echo off
setlocal EnableExtensions

rem ===========================================================================
rem  Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0  --  256K + MTP + VISION, Radeon RX 6900 XT
rem  (gfx1030 / RDNA2 / 16 GB)  Windows + cmd launcher
rem ===========================================================================
rem  Layout expected (as shipped in the release zip):
rem     <release>\runtime\        llama-kvmem-server.exe + dlls (+ rocblas, hipblaslt)
rem     <release>\hip-compat\     the b1233 gfx103X HIP runtime dlls
rem     <release>\chat-templates\abl-27b-reasoning-compat.jinja
rem     <release>\scripts\       this file
rem     <release>\model\         the .gguf files downloaded from Hugging Face
rem
rem  Vision: ON by default - the stock mmproj ships in model\ next to the weights.
rem  The FIRST image request after load costs ~2 minutes (CPU-side projector
rem  warm-up + kernel compile); warm repeats are ~2 s. NOVISION=1 disables.
rem
rem  Why the runtime is llama-kvmem-server and not llama-server:
rem     KVMem is the only way 262144 context fits in 16 GB. See docs/FUSION.md.
rem
rem  Usage:
rem     start-bonsai-mtp.bat                  defaults: ctx 262144, vision ON
rem     start-bonsai-mtp.bat --check          print the resolved command only
rem     set NOVISION=1 ^& start-bonsai-mtp.bat  text-only
rem     set CTX=65536 ^& start-bonsai-mtp.bat  smaller context (faster load)
rem     set EFFORT=medium ^& start-bonsai-mtp.bat
rem ===========================================================================

set "ROOT=%~dp0.."
set "RUNTIME=%ROOT%\runtime"
set "HIPCOMPAT=%ROOT%\hip-compat"
set "TEMPLATE=%ROOT%\chat-templates\abl-27b-reasoning-compat.jinja"
set "MODEL=%ROOT%\model\Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf"
set "ROCM=%ProgramFiles%\AMD\ROCm\7.1"

rem --- vision: the stock mmproj ships in model\ next to the weights (ON by
rem --- default; missing file -> automatic text-only fallback; NOVISION=1 disables).
if not defined MMPROJ set "MMPROJ=%ROOT%\model\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
if not defined IMGMINTOK set "IMGMINTOK=1024"
if not defined NOVISION set "NOVISION=0"

if not defined CTX     set "CTX=262144"
if not defined PORT    set "PORT=11234"
if not defined EFFORT  set "EFFORT=low"
if not defined NMTP    set "NMTP=3"
if not defined THREADS set "THREADS=12"
if not defined BUDGET  set "BUDGET=98304"
if not defined GENRES  set "GENRES=16384"
if not defined HOSTBIND set "HOSTBIND=127.0.0.1"
if not defined ALIAS   set "ALIAS=bonsai-27b-mtp"

set "CHECK=0"
if /i "%~1"=="--check" set "CHECK=1"

rem --- HIP: the compatibility runtime must come FIRST in PATH -------------
rem  System ROCm 7.1 does not enumerate gfx1030 on this host; the b1233 gfx103X
rem  runtime does. Without it the server aborts before loading the model.
set "HIP_VISIBLE_DEVICES="
set "ROCR_VISIBLE_DEVICES="
set "HSA_OVERRIDE_GFX_VERSION="
set "HIP_PATH=%ROCM%\"
set "HIP_PATH_64=%ROCM%\"
set "ROCBLAS_TENSILE_LIBPATH=%RUNTIME%\rocblas\library"
set "HIPBLASLT_TENSILE_LIBPATH=%RUNTIME%\hipblaslt\library"
set "PATH=%HIPCOMPAT%;%RUNTIME%;%ROCM%\bin;%RUNTIME%\rocblas;%RUNTIME%\hipblaslt;%PATH%"

if not exist "%RUNTIME%\llama-kvmem-server.exe" ( echo [ERROR] runtime not found: %RUNTIME%\llama-kvmem-server.exe ^(extract the runtime zip^) & goto fail )
if not exist "%MODEL%" ( echo [ERROR] model not found: %MODEL% & echo         Download it from Hugging Face - see README.md & goto fail )
if not exist "%TEMPLATE%" ( echo [WARN] compat chat template missing - using the model's own template & set "TEMPLATE=" )
if not "%NOVISION%"=="0" goto vision_flags_done
if not exist "%MMPROJ%" ( echo [warn] projector not found ^(%MMPROJ%^) - continuing text-only. & set "MMPROJ=" )
:vision_flags_done

set "VFLAGS="
if not "%NOVISION%"=="0" goto vision_flags_done2
if defined MMPROJ set "VFLAGS=--mmproj "%MMPROJ%" --no-mmproj-offload"
if defined MMPROJ if not "%IMGMINTOK%"=="0" set "VFLAGS=%VFLAGS% --image-min-tokens %IMGMINTOK%"
:vision_flags_done2

echo ===========================================================================
echo  Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0  ^|  %CTX% ctx, MTP x%NMTP%, KVMem on
echo ===========================================================================
echo  runtime : %RUNTIME%
echo  model   : %MODEL%
if not "%NOVISION%"=="0" echo  vision  : OFF ^(NOVISION set^)
if "%NOVISION%"=="0" if defined MMPROJ echo  vision  : ON   mmproj=%MMPROJ%
if "%NOVISION%"=="0" if not defined MMPROJ echo  vision  : OFF ^(projector not found^)
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
