@echo off
setlocal EnableExtensions

rem ===========================================================================
rem  Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP  --  UNCENSORED variant
rem  256K + MTP on a Radeon RX 6900 XT (gfx1030 / RDNA2 / 16 GB)
rem ===========================================================================
rem  Same runtime, same KVMem recipe as start-bonsai-mtp.bat; only the weights
rem  and the alias differ. This pack is the abliterated (refusal-removed)
rem  sibling of the Q8_0 pack:
rem
rem    packs                 blocks  MTP   vision projector
rem    Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0      65   yes   yes (official mmproj)
rem    Ternary-Bonsai-2-27B-Abliterated-...      65   yes   NO projector ships
rem
rem  Measured on this host (same flags as the Q8_0 pack, q8_0 KV + KVMem):
rem    262144 ctx / 147,725 tok prompt : pp 225.5 t/s, gen 41.7 t/s, accept 40.5%
rem     65536 ctx /  44,414 tok prompt : pp 256.6 t/s, gen 40.7 t/s, accept 40.5%
rem     32768 ctx /   1,620 tok prompt : 38.8 t/s baseline, 41.6 (MTP 3), 42.5 (MTP 2)
rem
rem  Usage:
rem     start-bonsai-abliterated.bat [--check]
rem     set CTX=65536 ^& start-bonsai-abliterated.bat
rem ===========================================================================

set "ROOT=%~dp0.."
set "RUNTIME=%ROOT%\runtime"
set "HIPCOMPAT=%ROOT%\hip-compat"
set "TEMPLATE=%ROOT%\chat-templates\abl-27b-reasoning-compat.jinja"
set "MODEL=%ROOT%\model\Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf"
set "ROCM=%ProgramFiles%\AMD\ROCm\7.1"

if not defined CTX     set "CTX=262144"
if not defined PORT    set "PORT=11234"
if not defined EFFORT  set "EFFORT=low"
if not defined NMTP    set "NMTP=3"
if not defined THREADS set "THREADS=12"
if not defined BUDGET  set "BUDGET=36864"
if not defined GENRES  set "GENRES=16384"
if not defined HOSTBIND set "HOSTBIND=127.0.0.1"
if not defined ALIAS   set "ALIAS=bonsai-27b-abliterated-mtp"

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
if not exist "%MODEL%" ( echo [ERROR] model not found: %MODEL% & echo         Download it from Hugging Face - see README.md section 2 & goto fail )
if not exist "%TEMPLATE%" ( echo [WARN] compat chat template missing & set "TEMPLATE=" )

echo ===========================================================================
echo  Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP  ^|  %CTX% ctx, MTP x%NMTP%
echo ===========================================================================
echo  runtime : %RUNTIME%
echo  model   : %MODEL%
echo  think   : on, effort=%EFFORT%   http://%HOSTBIND%:%PORT%
echo  note    : UNCENSORED pack - no vision projector ships with it
echo  stop    : Ctrl+C
echo ===========================================================================

if "%CHECK%"=="1" (
    echo [check] "%RUNTIME%\llama-kvmem-server.exe" -m "%MODEL%" --alias %ALIAS% -ngl 999 -c %CTX% --kv-dtype q8_0 -np 1 --host %HOSTBIND% --port %PORT% --threads %THREADS% --jinja --chat-template-file "%TEMPLATE%" --chat-template-kwargs "{\"enable_thinking\":true}" --reasoning-effort %EFFORT% --spec-type draft-mtp --spec-draft-n-max %NMTP% --kvmem-budget %BUDGET% --kvmem-gen-reserve %GENRES% --kvmem-method retrieval --kvmem-mtp-state snapshots
    goto done
)

if "%TEMPLATE%"=="" goto launch_plain

"%RUNTIME%\llama-kvmem-server.exe" ^
  -m "%MODEL%" --alias %ALIAS% ^
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
  -m "%MODEL%" --alias %ALIAS% ^
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
