#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launcher for Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0 on a Radeon RX 6900 XT (gfx1030).

Self-contained: it locates the release layout relative to its own file, sets the
HIP environment the gfx1030 build needs, starts llama-kvmem-server, waits for
/health and then either stays in the foreground or (with --hold) keeps hosting it.

Layout expected (as shipped):
    <release>/runtime/            llama-kvmem-server.exe + dlls (+ rocblas, hipblaslt)
    <release>/hip-compat/         b1233 gfx103X HIP runtime dlls
    <release>/chat-templates/     abl-27b-reasoning-compat.jinja
    <release>/model/              the .gguf files from Hugging Face
    <release>/scripts/            this file

Examples
    python start-bonsai-mtp.py                     # 256K ctx, MTP, thinking on/low
    python start-bonsai-mtp.py --check             # resolve + validate, do not launch
    python start-bonsai-mtp.py --ctx 65536 --port 8080
    python start-bonsai-mtp.py --vision            # add the mmproj vision tower
    python start-bonsai-mtp.py --effort xhigh      # longer thinking
"""
import argparse
import atexit
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
HIPCOMPAT = ROOT / "hip-compat"
TEMPLATE = ROOT / "chat-templates" / "abl-27b-reasoning-compat.jinja"
MODELDIR = ROOT / "model"
DEFAULT_MODEL = MODELDIR / "Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf"
DEFAULT_MMPROJ = MODELDIR / "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
SERVER = RUNTIME / "llama-kvmem-server.exe" if os.name == "nt" else RUNTIME / "llama-kvmem-server"
ROCM = pathlib.Path(os.environ.get("ROCM_PATH", r"C:\Program Files\AMD\ROCm\7.1"))


def mk_env(threads_unused=None):
    """compat HIP first: on gfx1030 the system HIP runtime must be shadowed."""
    e = {k.upper(): v for k, v in os.environ.items()}
    e.pop("PATH", None)
    for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION"):
        e.pop(k, None)
    e["HIP_PATH"] = str(ROCM) + os.sep
    e["HIP_PATH_64"] = e["HIP_PATH"]
    e["ROCBLAS_TENSILE_LIBPATH"] = str(RUNTIME / "rocblas" / "library")
    e["HIPBLASLT_TENSILE_LIBPATH"] = str(RUNTIME / "hipblaslt" / "library")
    e["PATH"] = os.pathsep.join([str(HIPCOMPAT), str(RUNTIME), str(ROCM / "bin"),
                                 str(RUNTIME / "rocblas"), str(RUNTIME / "hipblaslt"),
                                 e.get("PATH", "")])
    return e


def health(port, timeout=4):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def preflight(args):
    problems = []
    if not SERVER.is_file():
        problems.append(f"runtime missing: {SERVER}  (extract the runtime zip)")
    if not args.model.is_file():
        problems.append(f"model missing: {args.model}  (download from Hugging Face, see README)")
    if not (RUNTIME / "rocblas" / "library").is_dir():
        problems.append(f"tensor libs missing: {RUNTIME / 'rocblas'}  (extract the tensile zip)")
    if not HIPCOMPAT.is_dir() or not any(HIPCOMPAT.glob("amdhip64*")):
        problems.append(f"HIP compat runtime missing: {HIPCOMPAT}  (extract the hip-compat zip)")
    if args.vision and not args.mmproj.is_file():
        problems.append(f"projector missing: {args.mmproj}")
    return problems


def build_argv(args):
    alias = args.alias or ("bonsai-27b-mtp-vision" if args.vision else "bonsai-27b-mtp")
    argv = [str(SERVER), "-m", str(args.model), "--alias", alias,
            "-ngl", "999", "-c", str(args.ctx), "--kv-dtype", "q8_0", "-np", "1",
            "--host", args.host, "--port", str(args.port), "--threads", str(args.threads),
            "--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.n_mtp),
            "--kvmem-mtp-state", "snapshots"]
    if args.vision:
        argv += ["--mmproj", str(args.mmproj)]
        if not args.mmproj_on_gpu:
            argv += ["--no-mmproj-offload"]
        if args.image_min_tokens:
            argv += ["--image-min-tokens", str(args.image_min_tokens)]
    if not args.no_kvmem:
        argv += ["--kvmem-budget", "36864", "--kvmem-gen-reserve", "16384",
                 "--kvmem-method", "retrieval"]
    if not args.no_think:
        argv += ["--chat-template-kwargs", '{"enable_thinking":true}',
                 "--reasoning-effort", args.effort]
    if TEMPLATE.is_file():
        argv += ["--jinja", "--chat-template-file", str(TEMPLATE)]
    return argv


def tie_lifetime_to_launcher(proc):
    """On Windows, put the server in a job object with kill-on-close.

    Without this the server survives a hard kill of the launcher (TerminateProcess
    never runs atexit) and keeps VRAM allocated. Nested jobs are supported since
    Windows 8, so this also works when the launcher itself runs inside a job.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                        ("WriteOperationCount", ctypes.c_ulonglong),
                        ("OtherOperationCount", ctypes.c_ulonglong),
                        ("ReadTransferCount", ctypes.c_ulonglong),
                        ("WriteTransferCount", ctypes.c_ulonglong),
                        ("OtherTransferCount", ctypes.c_ulonglong)]

        class BASIC_LIMITS(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXT_LIMITS(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC_LIMITS),
                        ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9
        k32 = ctypes.windll.kernel32
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return
        info = EXT_LIMITS()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                    ctypes.byref(info), ctypes.sizeof(info))
        k32.AssignProcessToJobObject(job, int(proc._handle))
    except Exception as exc:  # never let this break the launch
        print(f"[launch] note: could not attach job object ({exc}); "
              f"if you hard-kill this launcher, stop the server manually")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    ap.add_argument("--mmproj", type=pathlib.Path, default=DEFAULT_MMPROJ)
    ap.add_argument("--vision", action="store_true", help="enable the vision tower")
    ap.add_argument("--mmproj-on-gpu", action="store_true")
    ap.add_argument("--image-min-tokens", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--port", type=int, default=11234)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--n-mtp", type=int, default=3, help="speculative draft depth (1-3)")
    ap.add_argument("--effort", default="low", choices=("xhigh", "medium", "low"))
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--no-kvmem", action="store_true")
    ap.add_argument("--alias", default=None)
    ap.add_argument("--check", action="store_true", help="validate and print, do not launch")
    args = ap.parse_args()

    problems = preflight(args)
    if problems:
        print("[preflight] not ready:")
        for p in problems:
            print("   -", p)
        if args.check:
            return 2
        print("[preflight] aborting")
        return 2

    argv = build_argv(args)
    print("[launch]", " ".join(f'"{a}"' if " " in a else a for a in argv))
    if args.check:
        print("[check] all prerequisites present; server not started")
        return 0

    proc = subprocess.Popen(argv, cwd=str(RUNTIME), env=mk_env(), stdin=subprocess.DEVNULL)
    tie_lifetime_to_launcher(proc)

    def shutdown(*_):
        """Never leave the server orphaned holding VRAM (Ctrl+C, kill, exception)."""
        if proc.poll() is None:
            print("\n[launch] stopping server ...")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(shutdown)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: sys.exit(0))
        except (ValueError, OSError):
            pass

    t0 = time.time()
    try:
        while time.time() - t0 < 300:
            if proc.poll() is not None:
                print(f"[launch] server exited rc={proc.returncode}")
                return proc.returncode
            if health(args.port):
                print(f"[launch] ready in {time.time()-t0:.0f}s -> "
                      f"http://{args.host}:{args.port}/v1/chat/completions")
                return proc.wait()
            time.sleep(2)
        print("[launch] timed out waiting for /health")
        return 3
    finally:
        shutdown()


if __name__ == "__main__":
    sys.exit(main())
