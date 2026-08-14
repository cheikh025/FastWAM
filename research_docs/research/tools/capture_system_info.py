#!/usr/bin/env python3
"""Capture a reproducibility-oriented system snapshot without secrets.

Usage:
    python research/tools/capture_system_info.py --output research/progress/system_exp_0001.json

Run this from the same activated environment used for training when possible.
"""
from __future__ import annotations
import argparse, json, os, platform, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd):
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        return {"ok": p.returncode == 0, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def read_first_existing(paths):
    for p in paths:
        try:
            t = Path(p).read_text(errors="replace")
            if t:
                return t
        except Exception:
            pass
    return None


def cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text(errors='replace').splitlines():
            if line.lower().startswith('model name'):
                return line.split(':',1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def mem_total():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemTotal:'):
                kb = int(line.split()[1]); return {"bytes": kb*1024, "gib": round(kb/1024/1024, 2)}
    except Exception:
        pass
    return None


def pkg_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def torch_info():
    try:
        import torch
        d = {
            "version": torch.__version__,
            "cuda_version": getattr(torch.version, 'cuda', None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, 'cudnn') else None,
        }
        if torch.cuda.is_available():
            d['devices'] = [{"index": i, "name": torch.cuda.get_device_name(i), "total_memory_gib": round(torch.cuda.get_device_properties(i).total_memory/1024**3, 2)} for i in range(torch.cuda.device_count())]
        return d
    except Exception as e:
        return {"error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--repo', default='.')
    ap.add_argument('--pip-freeze-output', default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    usage = shutil.disk_usage(repo)
    nvidia = run(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader']) if shutil.which('nvidia-smi') else {"ok": False, "error": "nvidia-smi not found"}
    nvcc = run(['nvcc','--version']) if shutil.which('nvcc') else {"ok": False, "error": "nvcc not found"}
    git_commit = run(['git','-C',str(repo),'rev-parse','HEAD'])
    git_branch = run(['git','-C',str(repo),'branch','--show-current'])
    git_status = run(['git','-C',str(repo),'status','--porcelain'])
    os_release = read_first_existing(['/etc/os-release'])
    selected_env = {k:v for k,v in os.environ.items() if k in {'CUDA_VISIBLE_DEVICES','NCCL_DEBUG','NCCL_SOCKET_IFNAME','OMP_NUM_THREADS','MKL_NUM_THREADS','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','CONDA_DEFAULT_ENV','VIRTUAL_ENV'} or k.startswith('NCCL_')}
    # MASTER_ADDR/PORT are not secrets; do not capture general env.
    data = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "git": {
            "commit": git_commit.get('stdout') if git_commit.get('ok') else None,
            "branch": git_branch.get('stdout') if git_branch.get('ok') else None,
            "dirty": bool(git_status.get('stdout')) if git_status.get('ok') else None,
            "status_porcelain": git_status.get('stdout') if git_status.get('ok') else None,
        },
        "os": {"platform": platform.platform(), "kernel": platform.release(), "machine": platform.machine(), "os_release": os_release},
        "cpu": {"model": cpu_model(), "logical_cpu_count": os.cpu_count()},
        "memory": {"total": mem_total()},
        "disk": {"path": str(repo), "total_gib": round(usage.total/1024**3,2), "used_gib": round(usage.used/1024**3,2), "free_gib": round(usage.free/1024**3,2)},
        "gpu": {"nvidia_smi_query": nvidia},
        "cuda_toolkit": {"nvcc": nvcc},
        "python": {"executable": sys.executable, "version": sys.version.replace('\n',' ')},
        "packages": {"torch": torch_info(), "deepspeed": pkg_version('deepspeed'), "accelerate": pkg_version('accelerate'), "transformers": pkg_version('transformers')},
        "selected_environment": selected_env,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True))
    freeze_path = Path(args.pip_freeze_output) if args.pip_freeze_output else out.with_suffix('.pip_freeze.txt')
    freeze = run([sys.executable,'-m','pip','freeze'])
    if freeze.get('ok'):
        freeze_path.write_text(freeze.get('stdout','') + '\n')
        data['pip_freeze_path'] = str(freeze_path)
        out.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(out)

if __name__ == '__main__':
    main()
