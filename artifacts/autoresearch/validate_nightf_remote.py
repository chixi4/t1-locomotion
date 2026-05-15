#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import yaml


def main() -> None:
    runs = json.loads(Path("artifacts/autoresearch/nightf_runs.json").read_text(encoding="utf-8"))
    for run in runs:
        cfg = yaml.safe_load(Path("envs", f"{run['task']}.yaml").read_text(encoding="utf-8"))
        assert cfg["basic"]["task"] == run["task"]
        assert cfg["basic"]["max_iterations"] == 400
        assert cfg["basic"]["leg_checkpoint"] == "logs/2026-05-05-11-09-07/nn/model_4000.pth"
        assert cfg["rewards"]["scales"]["anti_sway_vs_fixed_arm"] == 0.0
        asset = Path(cfg["asset"]["file"])
        assert asset.exists(), str(asset)
        print(
            run["code"],
            run["label"],
            cfg["control"]["target_clip"]["Left_Shoulder_Roll"],
            cfg["control"]["target_clip"]["Right_Shoulder_Roll"],
        )
    print("validation_ok")


if __name__ == "__main__":
    main()
