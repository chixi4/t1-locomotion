from pathlib import Path

import yaml


TASK = "T1Shoulder4NightHFullBodyGrid15Scratch_train5000"
CFG = Path("envs") / f"{TASK}.yaml"


def main():
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    assert cfg["basic"]["task"] == TASK
    assert cfg["basic"]["checkpoint"] is None
    assert "leg_checkpoint" not in cfg["basic"]
    assert cfg["basic"]["max_iterations"] == 5000
    assert cfg["env"]["num_observations"] == 59
    assert cfg["env"]["num_actions"] == 16
    assert cfg["asset"]["file"].endswith("T1_locomotion_shoulder4_night_h_no_inward_limit.urdf")
    assert cfg["control"]["target_clip"]["Left_Shoulder_Roll"] == [0.0, 0.3]
    assert cfg["control"]["target_clip"]["Right_Shoulder_Roll"] == [-0.3, 0.0]
    assert cfg["commands"]["curriculum"] is True
    assert cfg["commands"]["sampling"] == "grid3d_circle"
    assert cfg["commands"]["grid3d_unlock_neighbors"] == "face6"
    assert cfg["commands"]["lin_vel_levels"] == 15
    assert cfg["commands"]["ang_vel_levels"] == 15
    assert "allowed_curriculum_checkpoint" not in cfg["commands"]
    assert cfg["commands"]["sway_curriculum"] is True
    scales = cfg["rewards"]["scales"]
    assert scales["tracking_lin_vel_x"] > 0
    assert scales["feet_swing"] > 0
    assert scales["shoulder_lateral_roll_outward"] < 0
    assert scales["shoulder_lateral_roll_outward_margin"] < 0
    assert cfg["rewards"]["shoulder_lateral_roll_left_sign"] == 1.0
    assert cfg["rewards"]["shoulder_lateral_roll_right_sign"] == -1.0
    print("validate_nighth_fullbody_remote: ok")


if __name__ == "__main__":
    main()
