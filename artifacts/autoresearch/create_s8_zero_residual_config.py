from pathlib import Path

import yaml


ROOT = Path("/mnt/c/Users/Administrator/Documents/dev/official_baselines/booster_gym_official")
SOURCE = ROOT / "envs/T1Upper9CameraStableOfficialOpenLeg18LegResidualSpeedUnlock_train300.yaml"
TARGET_NAME = "T1Upper9CameraStableOfficialOpenLeg18S8ZeroResidual_eval"
TARGET = ROOT / f"envs/{TARGET_NAME}.yaml"
INIT_FILE = ROOT / "envs/__init__.py"


def zero_mapping(mapping):
    return {key: 0.0 for key in mapping}


def main():
    cfg = yaml.load(SOURCE.read_text(), Loader=yaml.FullLoader)
    cfg["basic"]["description"] = (
        "Evaluation-only config for s8_omni_full as the frozen leg base, "
        "with residual effect disabled."
    )
    cfg["basic"]["leg_checkpoint"] = "checkpoints/s8_omni_full.pth"
    cfg["basic"]["checkpoint"] = "logs/2026-05-14-15-08-00/nn/model_500.pth"
    cfg["basic"]["max_iterations"] = 1
    cfg["algorithm"]["residual_effect_scale_by_dof"] = zero_mapping(
        cfg["algorithm"]["residual_effect_scale_by_dof"]
    )
    cfg["algorithm"]["residual_penalty"] = 0.0
    TARGET.write_text(yaml.safe_dump(cfg, sort_keys=False))

    init_text = INIT_FILE.read_text()
    alias = f"{TARGET_NAME} = T1\n"
    if alias not in init_text:
        INIT_FILE.write_text(init_text.rstrip() + "\n" + alias)
    print(TARGET)


if __name__ == "__main__":
    main()
