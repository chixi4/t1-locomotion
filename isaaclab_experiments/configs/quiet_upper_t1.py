from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .mdp import arm_posture
from .quiet_upper_fullsym_env_cfg import T1FlatHumanRefSymClockQuietUpperFullSymEnvCfg, T1QuietUpperFullSymRewards


LEG_ACTION_SCALE = 0.5
WAIST_ACTION_SCALE = 0.12
SHOULDER_PITCH_ACTION_SCALE = 0.22
SHOULDER_ROLL_ACTION_SCALE = 0.16
ELBOW_ACTION_SCALE = 0.12
HEAD_ACTION_SCALE = 0.06
DEFAULT_STANDING_RATIO = 0.15
TARGET_SPEED_RANGE_MPS = (1.40, 1.60)
PLAY_SPEED_RANGE_MPS = (1.5, 1.5)
ZERO_COMMAND_RANGE = (0.0, 0.0)
PLAY_NUM_ENVS = 32
PLAY_ENV_SPACING = 2.5
PLAY_EPISODE_LENGTH_S = 40.0
REFERENCE_LEG_POSE_WEIGHT = 0.0
OLD_ARM_POSTURE_WEIGHT = 0.0
HAND_COUNTER_SWING_WEIGHT = 0.06
HAND_COUNTER_SWING_AMPLITUDE_M = 0.04
HAND_COUNTER_SWING_STD_M = 0.08
LEG_HALF_CYCLE_MIRROR_WEIGHT = 0.12
FOOT_FORWARD_MIRROR_WEIGHT = 0.01
FOOT_LATERAL_MIRROR_WEIGHT = 0.05
CONTACT_BALANCE_WEIGHT = 0.07
JOINT_DEVIATION_ARMS_WEIGHT = -0.02
JOINT_DEVIATION_WAIST_WEIGHT = 0.0
PELVIS_MOTION_WEIGHT = -0.08
JOINT_VELOCITY_WAIST_WEIGHT = -0.02
JOINT_VELOCITY_ARMS_WEIGHT = -0.003
TORSO_YAW_RATE_WEIGHT = -0.01
FEET_SLIDE_WEIGHT = -0.04
DISABLED_REGULARIZER_WEIGHT = 0.0
PHASE_CONTACT_SCHEDULE_WEIGHT = 1.60
PHASE_FOOT_CLEARANCE_WEIGHT = 0.65
PHASE_STANCE_SLIP_WEIGHT = -0.25

ARM_ENVELOPE_ACTION_SCALE = {
    "(Left|Right)_Hip_.*": LEG_ACTION_SCALE,
    "(Left|Right)_Knee_Pitch": LEG_ACTION_SCALE,
    "(Left|Right)_Ankle_.*": LEG_ACTION_SCALE,
    "Waist": WAIST_ACTION_SCALE,
    "(Left|Right)_Shoulder_Pitch": SHOULDER_PITCH_ACTION_SCALE,
    "(Left|Right)_Shoulder_Roll": SHOULDER_ROLL_ACTION_SCALE,
    "(Left|Right)_Elbow_.*": ELBOW_ACTION_SCALE,
    "AAHead_yaw": HEAD_ACTION_SCALE,
    "Head_pitch": HEAD_ACTION_SCALE,
}


@configclass
class T1QuietUpperArmEnvelopeExploreRewards(T1QuietUpperFullSymRewards):
    arm_envelope_cylinder = RewTerm(
        func=arm_posture.arm_envelope_cylinder_penalty,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "radius": arm_posture.DEFAULT_ARM_ENVELOPE_RADIUS_M,
            "min_drop": arm_posture.DEFAULT_ARM_ENVELOPE_MIN_DROP_M,
        },
    )


@configclass
class T1FlatHumanRefSymClockQuietUpperArmEnvelopeExploreV1EnvCfg(
    T1FlatHumanRefSymClockQuietUpperFullSymEnvCfg
):
    rewards: T1QuietUpperArmEnvelopeExploreRewards = T1QuietUpperArmEnvelopeExploreRewards()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = ARM_ENVELOPE_ACTION_SCALE
        self.commands.base_velocity.rel_standing_envs = DEFAULT_STANDING_RATIO
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = TARGET_SPEED_RANGE_MPS
        self.commands.base_velocity.ranges.lin_vel_y = ZERO_COMMAND_RANGE
        self.commands.base_velocity.ranges.ang_vel_z = ZERO_COMMAND_RANGE
        self._apply_explore_reward_weights()
        self._apply_explore_curriculum()

    def _apply_explore_reward_weights(self) -> None:
        self.rewards.reference_leg_pose.weight = REFERENCE_LEG_POSE_WEIGHT
        self.rewards.hand_forward_reach.weight = OLD_ARM_POSTURE_WEIGHT
        self.rewards.hand_lateral_spread.weight = OLD_ARM_POSTURE_WEIGHT
        self.rewards.hand_lift_excess.weight = OLD_ARM_POSTURE_WEIGHT
        self.rewards.arm_outward_posture.weight = OLD_ARM_POSTURE_WEIGHT
        self.rewards.hand_counter_swing.weight = HAND_COUNTER_SWING_WEIGHT
        self.rewards.hand_counter_swing.params["amplitude"] = HAND_COUNTER_SWING_AMPLITUDE_M
        self.rewards.hand_counter_swing.params["std"] = HAND_COUNTER_SWING_STD_M
        self.rewards.leg_joint_half_cycle_mirror.weight = LEG_HALF_CYCLE_MIRROR_WEIGHT
        self.rewards.foot_forward_mirror.weight = FOOT_FORWARD_MIRROR_WEIGHT
        self.rewards.foot_lateral_mirror.weight = FOOT_LATERAL_MIRROR_WEIGHT
        self.rewards.episode_contact_balance.weight = CONTACT_BALANCE_WEIGHT
        self.rewards.joint_deviation_arms.weight = JOINT_DEVIATION_ARMS_WEIGHT
        self.rewards.joint_deviation_waist.weight = JOINT_DEVIATION_WAIST_WEIGHT
        self.rewards.pelvis_motion.weight = PELVIS_MOTION_WEIGHT
        self.rewards.joint_velocity_waist.weight = JOINT_VELOCITY_WAIST_WEIGHT
        self.rewards.joint_velocity_arms.weight = JOINT_VELOCITY_ARMS_WEIGHT
        self.rewards.torso_yaw_rate.weight = TORSO_YAW_RATE_WEIGHT
        self.rewards.feet_slide.weight = FEET_SLIDE_WEIGHT
        self.rewards.dof_torques_l2.weight = DISABLED_REGULARIZER_WEIGHT
        self.rewards.dof_acc_l2.weight = DISABLED_REGULARIZER_WEIGHT
        self.rewards.ang_vel_xy_l2.weight = DISABLED_REGULARIZER_WEIGHT

    def _apply_explore_curriculum(self) -> None:
        self.curriculum.phase_contact_schedule_weight.params["final_weight"] = PHASE_CONTACT_SCHEDULE_WEIGHT
        self.curriculum.phase_foot_clearance_weight.params["final_weight"] = PHASE_FOOT_CLEARANCE_WEIGHT
        self.curriculum.phase_stance_slip_weight.params["final_weight"] = PHASE_STANCE_SLIP_WEIGHT


@configclass
class T1FlatHumanRefSymClockQuietUpperArmEnvelopeExploreV1EnvCfg_PLAY(
    T1FlatHumanRefSymClockQuietUpperArmEnvelopeExploreV1EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = PLAY_NUM_ENVS
        self.scene.env_spacing = PLAY_ENV_SPACING
        self.episode_length_s = PLAY_EPISODE_LENGTH_S
        self.commands.base_velocity.ranges.lin_vel_x = PLAY_SPEED_RANGE_MPS
        self.commands.base_velocity.ranges.lin_vel_y = ZERO_COMMAND_RANGE
        self.commands.base_velocity.ranges.ang_vel_z = ZERO_COMMAND_RANGE
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
