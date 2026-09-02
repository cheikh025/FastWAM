"""Standalone Randomized-phase-only RoboTwin scan.

`run_robotwin_manager.py` always launches the "clean" phase first for every
task and only starts "random" after "clean" completes (no config to skip
straight to random) -- see its `try_launch_pending`/scheduling loop. This
script reuses its exact subprocess-build and result-parsing logic
(`eval_robotwin_single.py`, `_parse_success_rate`, `_phase_result_filename`)
but schedules only the "random" phase, so a Randomized-only rescan doesn't
have to redundantly re-run Clean.

Usage: python experiments/robotwin/run_robotwin_random_only.py \
    --ckpt <path> --dataset-stats-path <path> --output-dir <dir> \
    [--num-gpus 4] [--max-tasks-per-gpu 2] [--num-episodes 5]
"""
import argparse
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_robotwin_manager import (  # noqa: E402
    PROJECT_ROOT,
    SINGLE_ENTRY,
    TERMINATE_TIMEOUT_SEC,
    POLL_INTERVAL_SEC,
    _load_all_tasks,
    _parse_success_rate,
    _phase_result_filename,
    _resolve_path,
    _resolve_ckpt_tag,
)


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    process: subprocess.Popen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset-stats-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--task", default="robotwin_uncond_3cam_384_multiembodiment_eval")
    ap.add_argument("--num-gpus", type=int, default=4)
    ap.add_argument("--max-tasks-per-gpu", type=int, default=2)
    ap.add_argument("--num-episodes", type=int, default=5)
    args = ap.parse_args()

    ckpt_path = _resolve_path(args.ckpt, base=PROJECT_ROOT)
    dataset_stats_path = _resolve_path(args.dataset_stats_path, base=PROJECT_ROOT)
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)
    output_dir = _resolve_path(args.output_dir, base=PROJECT_ROOT)
    run_ts = output_dir.name
    run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = run_output_dir / "manager.log"
    tasks = _load_all_tasks()
    gpu_ids = list(range(args.num_gpus))
    task_rates: dict[str, float | None] = {t: None for t in tasks}
    failed_records: list[dict] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def build_cmd(task_name: str, gpu_id: int) -> list[str]:
        return [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={ckpt_path}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            "EVALUATION.task_config=demo_randomized",
            f"EVALUATION.output_dir={run_output_dir}",
            f"task={args.task}",
            f"EVALUATION.dataset_stats_path={dataset_stats_path}",
            f"EVALUATION.eval_num_episodes={args.num_episodes}",
            "model.redirect_common_files=false",
        ]

    def launch(task_name: str, gpu_id: int) -> RunningState:
        cmd = build_cmd(task_name, gpu_id)
        log(f"launch task={task_name} phase=random gpu={gpu_id} cmd={' '.join(cmd)}")
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), text=True)
        return RunningState(task_name=task_name, gpu_id=gpu_id, process=proc)

    def gpu_running_count(gpu_id: int) -> int:
        return sum(1 for s in running_states if s.gpu_id == gpu_id and s.process.poll() is None)

    def try_launch_pending(gpu_id: int) -> None:
        while pending_tasks and gpu_running_count(gpu_id) < args.max_tasks_per_gpu:
            running_states.append(launch(pending_tasks.popleft(), gpu_id))

    def terminate_all() -> None:
        for s in running_states:
            if s.process.poll() is None:
                s.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for s in running_states:
            if s.process.poll() is None:
                try:
                    s.process.wait(timeout=max(0.0, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    s.process.kill()
                    s.process.wait()

    log(f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={args.max_tasks_per_gpu} output_dir={run_output_dir}")

    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)

    has_failure = False
    while running_states:
        progressed = False
        for state in list(running_states):
            rc = state.process.poll()
            if rc is None:
                continue
            progressed = True
            running_states.remove(state)
            if rc != 0:
                has_failure = True
                log(f"worker failed: task={state.task_name} gpu={state.gpu_id} rc={rc}")
                failed_records.append({"task_name": state.task_name, "gpu_id": state.gpu_id, "rc": rc})
                terminate_all()
                running_states.clear()
                break
            result_file = run_output_dir / state.task_name / _phase_result_filename("random")
            try:
                task_rates[state.task_name] = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                log(f"result parse failed: task={state.task_name} error={exc!r}")
                failed_records.append({"task_name": state.task_name, "gpu_id": state.gpu_id, "error": str(exc)})
                terminate_all()
                running_states.clear()
                break
            log(f"done task={state.task_name} phase=random gpu={state.gpu_id} success_rate={task_rates[state.task_name]:.4f}")
            try_launch_pending(state.gpu_id)
        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    import csv
    import json

    valid = [v for v in task_rates.values() if v is not None]
    mean = sum(valid) / len(valid) if valid else None

    with (run_output_dir / "summary_random.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_name", "random_success_rate"])
        for t in tasks:
            w.writerow([t, task_rates[t]])
        w.writerow(["__overall__", mean])

    (run_output_dir / "summary_random.json").write_text(json.dumps({
        "per_task": [{"task_name": t, "random_success_rate": task_rates[t]} for t in tasks],
        "overall": {"random_mean_success_rate": mean},
        "failed": failed_records,
    }, indent=2), encoding="utf-8")

    log(f"summary saved: {run_output_dir}/summary_random.json (mean={mean}, failures={len(failed_records)})")
    if has_failure:
        sys.exit(2)


if __name__ == "__main__":
    main()
