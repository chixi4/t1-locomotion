#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import yaml


def main() -> None:
    runs = json.loads(Path("artifacts/autoresearch/nightl_runs.json").read_text(encoding="utf-8"))
    for run in runs:
        cfg = yaml.safe_load(Path("envs", f"{run['task']}.yaml").read_text(encoding="utf-8"))
        assert cfg["basic"]["task"] == run["task"]
        assert cfg["basic"]["max_iterations"] == 400
        assert cfg["basic"]["leg_checkpoint"] == "logs/2026-05-05-11-09-07/nn/model_4000.pth"
        assert cfg["rewards"]["scales"]["anti_sway_vs_fixed_arm"] == 0.0
        scales = cfg["rewards"]["scales"]
        assert scales["shoulder_static_posture"] < 0.0
        assert scales["shoulder_dynamic_target"] < 0.0
        assert scales["shoulder_no_lazy_boundary"] < 0.0
        assert scales["shoulder_balance_smoothness"] < 0.0
        for old_name in [
            "shoulder_base_roll_target",
            "shoulder_roll",
            "shoulder_pitch_soft_limit",
            "shoulder_pair_symmetry",
            "shoulder_foot_phase_pitch",
            "shoulder_low_speed_same_down",
            "shoulder_lateral_roll_outward",
            "shoulder_lateral_roll_outward_margin",
            "shoulder_lateral_roll_wrong_sign_action",
            "shoulder_lateral_pitch_down",
        ]:
            assert scales.get(old_name, 0.0) == 0.0, old_name
        assert cfg["rewards"]["shoulder_dynamic_roll_base_amp"] >= 0.12
        assert cfg["rewards"]["shoulder_dynamic_roll_max"] <= 0.22
        assert cfg["rewards"]["shoulder_dynamic_roll_min_outward"] >= 0.05
        assert cfg["rewards"]["shoulder_boundary_roll_soft_limit"] >= 0.20
        assert cfg["control"]["target_clip"]["Left_Shoulder_Roll"][0] == 0.0
        assert cfg["control"]["target_clip"]["Right_Shoulder_Roll"][1] == 0.0
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
