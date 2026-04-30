from dataclasses import dataclass
from typing import Dict, List, Optional


RUNNER_SAVE_INTERVAL = 50
EPISODE_SECONDS = 30.0
STAGE_TASK = "T1"
FORWARD_REWARD_SCALES = {
    "survival": 0.0,
    "tracking_lin_vel_x": 8.0,
    "tracking_lin_vel_y": 0.8,
    "tracking_ang_vel": 0.2,
    "action_rate": -0.2,
    "leg_action_magnitude_symmetry": -2.0,
    "torques": -1.0e-4,
    "dof_acc": -5.0e-8,
    "feet_slip": -0.3,
}
HIGH_SPEED_REWARD_SCALES = dict(FORWARD_REWARD_SCALES, survival=0.15, feet_slip=-0.5)
FORWARD_REWARD_PARAMS = {"tracking_sigma": 0.05}


@dataclass(frozen=True)
class CommandRange:
    still: float
    vx: List[float]
    vy: List[float]
    yaw: List[float]
    frequency: List[float]


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: CommandRange
    terrain: str
    randomization: str
    feet_swing_scale: Optional[float] = None
    reward_scales: Optional[Dict[str, float]] = None
    reward_params: Optional[Dict[str, float]] = None
    description: str = ""


STAGES: Dict[str, StageSpec] = {
    "s0_stand": StageSpec(
        "s0_stand",
        CommandRange(0.8, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.3]),
        "plane",
        "weak_noise",
        feet_swing_scale=1.0,
        description="stand and light stepping",
    ),
    "s1_forward_slow": StageSpec(
        "s1_forward_slow",
        CommandRange(0.05, [0.25, 0.45], [0.0, 0.0], [0.0, 0.0], [1.25, 1.8]),
        "plane",
        "flat",
        feet_swing_scale=2.0,
        reward_scales=FORWARD_REWARD_SCALES,
        reward_params=FORWARD_REWARD_PARAMS,
        description="slow forward walking",
    ),
    "s2_forward_05": StageSpec(
        "s2_forward_05",
        CommandRange(0.08, [0.3, 0.55], [0.0, 0.0], [0.0, 0.0], [1.3, 1.9]),
        "plane",
        "flat",
        feet_swing_scale=2.0,
        reward_scales=FORWARD_REWARD_SCALES,
        reward_params=FORWARD_REWARD_PARAMS,
        description="forward expansion to 0.5 m/s",
    ),
    "s2_forward_08": StageSpec(
        "s2_forward_08",
        CommandRange(0.05, [0.65, 0.9], [0.0, 0.0], [0.0, 0.0], [1.5, 2.0]),
        "plane",
        "flat",
        feet_swing_scale=2.0,
        reward_scales=HIGH_SPEED_REWARD_SCALES,
        reward_params=FORWARD_REWARD_PARAMS,
        description="forward expansion to 0.8 m/s",
    ),
    "s3_back_slow": StageSpec(
        "s3_back_slow",
        CommandRange(0.05, [-0.45, -0.15], [0.0, 0.0], [0.0, 0.0], [1.2, 1.7]),
        "plane",
        "flat",
        feet_swing_scale=2.0,
        reward_scales=HIGH_SPEED_REWARD_SCALES,
        reward_params=FORWARD_REWARD_PARAMS,
    ),
    "s3_forward_backward": StageSpec(
        "s3_forward_backward",
        CommandRange(0.05, [-0.45, 0.85], [0.0, 0.0], [0.0, 0.0], [1.3, 2.0]),
        "plane",
        "flat",
        feet_swing_scale=2.0,
        reward_scales=HIGH_SPEED_REWARD_SCALES,
        reward_params=FORWARD_REWARD_PARAMS,
    ),
    "s4_turn": StageSpec(
        "s4_turn",
        CommandRange(0.2, [0.0, 0.0], [0.0, 0.0], [-0.8, 0.8], [1.1, 1.6]),
        "plane",
        "flat",
        feet_swing_scale=2.0,
        reward_scales=dict(HIGH_SPEED_REWARD_SCALES, tracking_lin_vel_x=0.8, tracking_ang_vel=4.0),
        reward_params=FORWARD_REWARD_PARAMS,
    ),
    "s5_strafe": StageSpec(
        "s5_strafe",
        CommandRange(0.05, [0.0, 0.0], [-0.45, 0.45], [0.0, 0.0], [1.2, 1.8]),
        "plane",
        "flat",
        feet_swing_scale=2.0,
        reward_scales=dict(HIGH_SPEED_REWARD_SCALES, tracking_lin_vel_x=0.8, tracking_lin_vel_y=6.0, tracking_ang_vel=0.4),
        reward_params=FORWARD_REWARD_PARAMS,
    ),
    "s6_arc": StageSpec(
        "s6_arc",
        CommandRange(0.15, [-0.4, 0.8], [0.0, 0.0], [-0.8, 0.8], [1.1, 1.8]),
        "plane",
        "flat",
        feet_swing_scale=2.0,
        reward_scales=dict(HIGH_SPEED_REWARD_SCALES, tracking_lin_vel_x=5.0, tracking_ang_vel=3.0),
        reward_params=FORWARD_REWARD_PARAMS,
    ),
    "s7_diagonal": StageSpec(
        "s7_diagonal",
        CommandRange(0.15, [0.0, 0.8], [-0.5, 0.5], [0.0, 0.0], [1.1, 1.8]),
        "plane",
        "flat",
        feet_swing_scale=2.0,
        reward_scales=dict(HIGH_SPEED_REWARD_SCALES, tracking_lin_vel_x=5.0, tracking_lin_vel_y=4.0, tracking_ang_vel=0.4, feet_slip=-0.8, leg_action_magnitude_symmetry=-2.0),
        reward_params=FORWARD_REWARD_PARAMS,
    ),
    "s8_omni": StageSpec(
        "s8_omni",
        CommandRange(0.1, [-0.8, 1.0], [-0.8, 0.8], [-1.0, 1.0], [1.0, 2.0]),
        "plane",
        "flat",
        feet_swing_scale=2.0,
        reward_scales=dict(HIGH_SPEED_REWARD_SCALES, tracking_lin_vel_x=4.0, tracking_lin_vel_y=4.0, tracking_ang_vel=3.0, feet_slip=-0.9, leg_action_magnitude_symmetry=-3.0),
        reward_params=FORWARD_REWARD_PARAMS,
    ),
    "s9_noise": StageSpec(
        "s9_noise",
        CommandRange(0.1, [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, 2.0]),
        "plane",
        "weak_noise",
    ),
    "s9_actuator": StageSpec(
        "s9_actuator",
        CommandRange(0.1, [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, 2.0]),
        "plane",
        "actuator",
    ),
    "s9_friction": StageSpec(
        "s9_friction",
        CommandRange(0.1, [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, 2.0]),
        "plane",
        "friction",
        description="omnidirectional walking with friction randomization",
    ),
    "s9_push": StageSpec(
        "s9_push",
        CommandRange(0.1, [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, 2.0]),
        "plane",
        "push",
        description="omnidirectional walking with light pushes",
    ),
    "s9_terrain": StageSpec(
        "s9_terrain",
        CommandRange(0.1, [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, 2.0]),
        "trimesh",
        "push",
        description="omnidirectional walking on mixed terrain",
    ),
}


def apply_omni_stage(cfg: dict, stage_name: str) -> dict:
    if stage_name not in STAGES:
        raise ValueError(f"Unknown omni curriculum stage: {stage_name}")
    spec = STAGES[stage_name]
    _apply_command(cfg, spec.command)
    _apply_terrain(cfg, spec.terrain)
    _apply_randomization(cfg, spec.randomization)
    _apply_training_defaults(cfg, spec)
    return cfg


def _apply_command(cfg: dict, command: CommandRange) -> None:
    cfg["commands"]["curriculum"] = False
    cfg["commands"]["still_proportion"] = command.still
    cfg["commands"]["lin_vel_x"] = command.vx
    cfg["commands"]["lin_vel_y"] = command.vy
    cfg["commands"]["ang_vel_yaw"] = command.yaw
    cfg["commands"]["gait_frequency"] = command.frequency
    cfg["commands"]["resampling_time_s"] = [8.0, 12.0]


def _apply_terrain(cfg: dict, terrain_type: str) -> None:
    cfg["terrain"]["type"] = terrain_type
    if terrain_type == "plane":
        cfg["terrain"]["terrain_proportions"] = [1.0, 0.0, 0.0, 0.0]
        cfg["terrain"]["random_height"] = 0.0
        cfg["terrain"]["discrete_height"] = 0.0


def _apply_training_defaults(cfg: dict, spec: StageSpec) -> None:
    cfg["basic"]["task"] = STAGE_TASK
    cfg["basic"]["description"] = spec.description
    cfg["basic"]["run_name"] = spec.name
    cfg["runner"]["save_interval"] = RUNNER_SAVE_INTERVAL
    cfg["runner"]["use_wandb"] = False
    cfg["rewards"]["episode_length_s"] = EPISODE_SECONDS
    if spec.feet_swing_scale is not None:
        cfg["rewards"]["scales"]["feet_swing"] = spec.feet_swing_scale
    if spec.reward_scales is not None:
        cfg["rewards"]["scales"].update(spec.reward_scales)
    if spec.reward_params is not None:
        cfg["rewards"].update(spec.reward_params)


def _apply_randomization(cfg: dict, profile: str) -> None:
    _clear_randomization(cfg)
    if profile == "flat":
        return
    if profile == "weak_noise":
        _enable_observation_noise(cfg)
        return
    if profile == "actuator":
        _enable_observation_noise(cfg)
        _enable_actuator_randomization(cfg)
        return
    if profile == "friction":
        _enable_observation_noise(cfg)
        _enable_actuator_randomization(cfg)
        _enable_friction_randomization(cfg)
        return
    if profile == "push":
        _enable_observation_noise(cfg)
        _enable_actuator_randomization(cfg)
        _enable_friction_randomization(cfg)
        _enable_push_randomization(cfg)
        return
    raise ValueError(f"Unknown randomization profile: {profile}")


def _clear_randomization(cfg: dict) -> None:
    cfg["noise"] = {key: None for key in cfg["noise"]}
    for key in cfg["randomization"]:
        if not key.endswith("_interval_s") and not key.endswith("_duration_s"):
            cfg["randomization"][key] = None
    cfg["randomization"]["friction"] = _uniform_add(1.0, 1.0)
    cfg["randomization"]["compliance"] = _uniform_add(0.0, 0.0)
    cfg["randomization"]["restitution"] = _uniform_add(0.0, 0.0)
    cfg["randomization"]["base_com"] = _uniform_add(0.0, 0.0)
    cfg["randomization"]["base_mass"] = _uniform_scale(1.0, 1.0)


def _enable_observation_noise(cfg: dict) -> None:
    cfg["noise"]["gravity"] = _gaussian(0.0, 0.005)
    cfg["noise"]["lin_vel"] = _gaussian(0.0, 0.02)
    cfg["noise"]["ang_vel"] = _gaussian(0.0, 0.04)
    cfg["noise"]["dof_pos"] = _gaussian(0.0, 0.005)
    cfg["noise"]["dof_vel"] = _gaussian(0.0, 0.04)
    cfg["noise"]["height"] = _gaussian(0.0, 0.01)


def _enable_actuator_randomization(cfg: dict) -> None:
    cfg["randomization"]["dof_stiffness"] = _uniform_scale(0.95, 1.05)
    cfg["randomization"]["dof_damping"] = _uniform_scale(0.95, 1.05)
    cfg["randomization"]["dof_friction"] = _uniform_add(0.0, 1.0)


def _enable_friction_randomization(cfg: dict) -> None:
    cfg["randomization"]["friction"] = _uniform_add(0.5, 1.5)
    cfg["randomization"]["compliance"] = _uniform_add(0.8, 1.2)
    cfg["randomization"]["restitution"] = _uniform_add(0.0, 0.2)


def _enable_push_randomization(cfg: dict) -> None:
    cfg["randomization"]["kick_lin_vel"] = _gaussian(0.0, 0.05)
    cfg["randomization"]["kick_ang_vel"] = _gaussian(0.0, 0.01)
    cfg["randomization"]["push_force"] = _gaussian(0.0, 5.0)
    cfg["randomization"]["push_torque"] = _gaussian(0.0, 1.0)


def _gaussian(center: float, width: float) -> dict:
    return {"range": [center, width], "operation": "additive", "distribution": "gaussian"}


def _uniform_add(lower: float, upper: float) -> dict:
    return {"range": [lower, upper], "operation": "additive", "distribution": "uniform"}


def _uniform_scale(lower: float, upper: float) -> dict:
    return {"range": [lower, upper], "operation": "scaling", "distribution": "uniform"}
