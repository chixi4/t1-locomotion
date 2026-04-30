from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .mdp import arm_momentum, half_body_action
from .quiet_upper_arm_envelope_explore_env_cfg import (
    ELBOW_ACTION_SCALE,
    HEAD_ACTION_SCALE,
    LEG_ACTION_SCALE,
    SHOULDER_PITCH_ACTION_SCALE,
    SHOULDER_ROLL_ACTION_SCALE,
    WAIST_ACTION_SCALE,
    ZERO_COMMAND_RANGE,
    T1FlatHumanRefSymClockQuietUpperArmEnvelopeExploreV1EnvCfg,
    T1FlatHumanRefSymClockQuietUpperArmEnvelopeExploreV1EnvCfg_PLAY,
    T1QuietUpperArmEnvelopeExploreRewards,
)


DEFAULT_STANDING_RATIO = 0.15
FULL_SPEED_RANGE_MPS = (0.5, 1.5)
PLAY_SPEED_RANGE_MPS = (1.0, 1.0)
ARM_MOMENTUM_WEIGHT = 0.05
ARM_MOMENTUM_STD = 0.75
ARM_ENVELOPE_SAFETY_WEIGHT = -0.04
HALF_BODY_CYCLE_PERIOD_S = 0.70
HALF_BODY_RESIDUAL_SCALE = 0.08
FULL_SPEED_SHOULDER_PITCH_SCALE = 0.18
FULL_SPEED_SHOULDER_ROLL_SCALE = 0.05

FULL_SPEED_ACTION_SCALE = {
    "(Left|Right)_Hip_.*": LEG_ACTION_SCALE,
    "(Left|Right)_Knee_Pitch": LEG_ACTION_SCALE,
    "(Left|Right)_Ankle_.*": LEG_ACTION_SCALE,
    "Waist": WAIST_ACTION_SCALE,
    "(Left|Right)_Shoulder_Pitch": FULL_SPEED_SHOULDER_PITCH_SCALE,
    "(Left|Right)_Shoulder_Roll": FULL_SPEED_SHOULDER_ROLL_SCALE,
    "(Left|Right)_Elbow_.*": ELBOW_ACTION_SCALE,
    "AAHead_yaw": HEAD_ACTION_SCALE,
    "Head_pitch": HEAD_ACTION_SCALE,
}

HALF_BODY_ACTION_SCALE = {
    "(Left|Right)_Hip_.*": LEG_ACTION_SCALE,
    "(Left|Right)_Knee_Pitch": LEG_ACTION_SCALE,
    "(Left|Right)_Ankle_.*": LEG_ACTION_SCALE,
    "Waist": WAIST_ACTION_SCALE,
    "(Left|Right)_Shoulder_Pitch": FULL_SPEED_SHOULDER_PITCH_SCALE,
    "(Left|Right)_Shoulder_Roll": FULL_SPEED_SHOULDER_ROLL_SCALE,
    "(Left|Right)_Elbow_.*": ELBOW_ACTION_SCALE,
    "AAHead_yaw": HEAD_ACTION_SCALE,
    "Head_pitch": HEAD_ACTION_SCALE,
}


@configclass
class T1FullSpeedMomentumArmRewards(T1QuietUpperArmEnvelopeExploreRewards):
    arm_yaw_momentum_cancel = RewTerm(
        func=arm_momentum.arm_yaw_momentum_cancel_exp,
        weight=ARM_MOMENTUM_WEIGHT,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": ARM_MOMENTUM_STD,
        },
    )


@configclass
class T1FlatHumanRefSymClockFullSpeedMomentumArmEnvCfg(
    T1FlatHumanRefSymClockQuietUpperArmEnvelopeExploreV1EnvCfg
):
    rewards: T1FullSpeedMomentumArmRewards = T1FullSpeedMomentumArmRewards()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = FULL_SPEED_ACTION_SCALE
        self.commands.base_velocity.rel_standing_envs = DEFAULT_STANDING_RATIO
        self.commands.base_velocity.ranges.lin_vel_x = FULL_SPEED_RANGE_MPS
        self.commands.base_velocity.ranges.lin_vel_y = ZERO_COMMAND_RANGE
        self.commands.base_velocity.ranges.ang_vel_z = ZERO_COMMAND_RANGE
        self.rewards.hand_counter_swing = None
        self.rewards.arm_envelope_cylinder.weight = ARM_ENVELOPE_SAFETY_WEIGHT
        self.rewards.arm_yaw_momentum_cancel.weight = ARM_MOMENTUM_WEIGHT
        self.rewards.arm_yaw_momentum_cancel.params["std"] = ARM_MOMENTUM_STD


@configclass
class T1FlatHumanRefSymClockFullSpeedMomentumArmHalfBodyV1EnvCfg(
    T1FlatHumanRefSymClockFullSpeedMomentumArmEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = half_body_action.HalfBodyMirrorJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=HALF_BODY_ACTION_SCALE,
            use_default_offset=True,
            cycle_period_s=HALF_BODY_CYCLE_PERIOD_S,
            residual_scale=HALF_BODY_RESIDUAL_SCALE,
        )


@configclass
class T1FlatHumanRefSymClockFullSpeedMomentumArmEnvCfg_PLAY(
    T1FlatHumanRefSymClockQuietUpperArmEnvelopeExploreV1EnvCfg_PLAY
):
    rewards: T1FullSpeedMomentumArmRewards = T1FullSpeedMomentumArmRewards()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = PLAY_SPEED_RANGE_MPS
        self.commands.base_velocity.ranges.lin_vel_y = ZERO_COMMAND_RANGE
        self.commands.base_velocity.ranges.ang_vel_z = ZERO_COMMAND_RANGE
        self.rewards.hand_counter_swing = None
        self.rewards.arm_envelope_cylinder.weight = ARM_ENVELOPE_SAFETY_WEIGHT


@configclass
class T1FlatHumanRefSymClockFullSpeedMomentumArmHalfBodyV1EnvCfg_PLAY(
    T1FlatHumanRefSymClockFullSpeedMomentumArmEnvCfg_PLAY
):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = half_body_action.HalfBodyMirrorJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=HALF_BODY_ACTION_SCALE,
            use_default_offset=True,
            cycle_period_s=HALF_BODY_CYCLE_PERIOD_S,
            residual_scale=HALF_BODY_RESIDUAL_SCALE,
        )
