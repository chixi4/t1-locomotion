import argparse
import json
import math
import os
import re
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path("/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official")
TASK = "T1Shoulder4Pitch09Roll08_from7000LegFrozen_train1000"
OUT_DIR = ROOT / "artifacts" / "autoresearch" / "pitchroll_env_benchmark"


def poll_gpu(stop_event, samples):
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip().splitlines()
            if out:
                parts = [part.strip() for part in out[0].split(",")]
                samples.append(
                    {
                        "time": time.time(),
                        "gpu_util": float(parts[0]),
                        "memory_used_mb": float(parts[1]),
                        "memory_total_mb": float(parts[2]),
                        "power_w": float(parts[3]),
                    }
                )
        except Exception:
            pass
        stop_event.wait(1.0)


def summarize_gpu(samples):
    if not samples:
        return {}
    return {
        "gpu_util_avg": sum(s["gpu_util"] for s in samples) / len(samples),
        "gpu_util_max": max(s["gpu_util"] for s in samples),
        "memory_used_mb_max": max(s["memory_used_mb"] for s in samples),
        "power_w_avg": sum(s["power_w"] for s in samples) / len(samples),
        "power_w_max": max(s["power_w"] for s in samples),
    }


def run_case(num_envs, iterations, timeout_s):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"env{num_envs}_iters{iterations}.log"
    cmd = [
        "/opt/conda/bin/python",
        "-u",
        "train_shoulder4_frozen.py",
        "--task",
        TASK,
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(iterations),
        "--headless",
        "True",
    ]
    env = os.environ.copy()
    env["PATH"] = "/opt/conda/bin:" + env.get("PATH", "")
    epoch_times = []
    lines = []
    gpu_samples = []
    stop_event = threading.Event()
    monitor = threading.Thread(target=poll_gpu, args=(stop_event, gpu_samples), daemon=True)
    start = time.perf_counter()
    monitor.start()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    status = "ok"
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            now = time.perf_counter()
            lines.append(line)
            if re.search(r"epoch:\s+\d+/", line):
                epoch_times.append(now)
            if now - start > timeout_s:
                status = "timeout"
                proc.kill()
                break
        return_code = proc.wait(timeout=10)
        if return_code != 0 and status == "ok":
            status = "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
        proc.kill()
        return_code = proc.wait()
    finally:
        stop_event.set()
        monitor.join(timeout=2)
        elapsed = time.perf_counter() - start
        log_path.write_text("".join(lines), encoding="utf-8", errors="ignore")

    epoch_intervals = [b - a for a, b in zip(epoch_times, epoch_times[1:])]
    steady_intervals = epoch_intervals[2:] if len(epoch_intervals) > 4 else epoch_intervals
    mean_epoch_s = None
    env_steps_per_s = None
    ppo_iters_per_hour = None
    if steady_intervals:
        mean_epoch_s = sum(steady_intervals) / len(steady_intervals)
        env_steps_per_s = num_envs * 24 / mean_epoch_s
        ppo_iters_per_hour = 3600.0 / mean_epoch_s
    oom = any(("out of memory" in line.lower()) or ("cuda error" in line.lower()) for line in lines)
    if oom:
        status = "oom"
    result = {
        "num_envs": num_envs,
        "iterations_requested": iterations,
        "epochs_completed": len(epoch_times),
        "status": status,
        "return_code": return_code,
        "elapsed_s": elapsed,
        "mean_epoch_s": mean_epoch_s,
        "env_steps_per_s": env_steps_per_s,
        "ppo_iters_per_hour": ppo_iters_per_hour,
        "gpu": summarize_gpu(gpu_samples),
        "log": str(log_path.relative_to(ROOT)),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def planned_candidates(results):
    tested = {r["num_envs"] for r in results}
    ok = {r["num_envs"] for r in results if r["status"] == "ok" and r["epochs_completed"] >= 6}
    failed = {r["num_envs"] for r in results if r["status"] != "ok"}
    if 24576 not in tested:
        return 24576
    if 49152 not in tested:
        return 49152
    if 49152 in failed:
        for n in [32768, 36864, 40960, 45056, 47104]:
            if n not in tested:
                return n
        return None
    if 98304 not in tested:
        return 98304
    if 98304 in failed:
        for n in [61440, 73728, 81920, 86016, 90112]:
            if n not in tested:
                return n
        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--timeout_s", type=float, default=900.0)
    args = parser.parse_args()
    results = []
    while True:
        candidate = planned_candidates(results)
        if candidate is None:
            break
        results.append(run_case(candidate, args.iterations, args.timeout_s))
    ok_results = [r for r in results if r["env_steps_per_s"] is not None]
    best = max(ok_results, key=lambda r: r["env_steps_per_s"]) if ok_results else None
    payload = {
        "task": TASK,
        "iterations_per_case": args.iterations,
        "results": results,
        "best_by_env_steps_per_s": best,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Short-run benchmark only. The formal 1000-iteration training was not started.",
    }
    out_path = OUT_DIR / "pitchroll_env_benchmark_summary.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out_path.relative_to(ROOT)), "best": best}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
