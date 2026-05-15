import json
import time
from pathlib import Path

from benchmark_pitchroll_envs import ROOT, OUT_DIR, run_case


def main():
    candidates = [33792, 34816, 35840]
    results = []
    for num_envs in candidates:
        results.append(run_case(num_envs, 12, 900.0))
    ok_results = [r for r in results if r["env_steps_per_s"] is not None]
    best = max(ok_results, key=lambda r: r["env_steps_per_s"]) if ok_results else None
    payload = {
        "candidates": candidates,
        "results": results,
        "best_by_env_steps_per_s": best,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = OUT_DIR / "pitchroll_env_benchmark_fine_summary.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(out_path.relative_to(ROOT)), "best": best}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
