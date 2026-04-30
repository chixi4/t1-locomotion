from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.t1.mdp import human_rewards

from .mdp import arm_posture, full_body_symmetry, symmetry_clock
from .symclock_env_cfg import QUIET_UPPER_BODY_ACTION_SCALE, T1FlatHumanRefSymClockEnvCfg, T1SymClockRewards


@configclass
class T1QuietUpperFullSymRewards(T1SymClockRewards):
    arm_swing_tracking = None
    air_time_balance = None
    paired_action_symmetry = None
    step_width_tracking = RewTerm(
        func=human_rewards.step_width_tracking_exp,
        weight=0.10,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "target_width": 0.18,
            "std": 0.05,
        },
    )
    pelvis_motion = RewTerm(func=human_rewards.pelvis_motion_penalty, weight=-0.12, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(Left|Right)_Shoulder_.*", "(Left|Right)_Elbow_.*"])},
    )
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["AAHead_yaw", "Head_pitch"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Waist"])},
    )
    joint_velocity_waist = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Waist"])},
    )
    joint_velocity_arms = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(Left|Right)_Shoulder_.*", "(Left|Right)_Elbow_.*"])},
    )
    joint_velocity_head = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["AAHead_yaw", "Head_pitch"])},
    )
    arm_outward_posture = RewTerm(
        func=arm_posture.arm_outward_posture_penalty,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "roll_margin": arm_posture.DEFAULT_SHOULDER_ROLL_MARGIN_RAD,
            "pitch_margin": arm_posture.DEFAULT_SHOULDER_PITCH_MARGIN_RAD,
        },
    )
    hand_lateral_spread = RewTerm(
        func=arm_posture.hand_lateral_spread_penalty,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_radius": arm_posture.DEFAULT_ARM_LATERAL_RADIUS_M,
            "margin": arm_posture.DEFAULT_ARM_LATERAL_MARGIN_M,
        },
    )
    hand_forward_reach = RewTerm(
        func=arm_posture.hand_forward_reach_penalty,
        weight=-0.14,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_radius": arm_posture.DEFAULT_ARM_FORWARD_RADIUS_M,
            "margin": arm_posture.DEFAULT_ARM_FORWARD_MARGIN_M,
        },
    )
    hand_lift_excess = RewTerm(
        func=full_body_symmetry.hand_lift_excess_penalty,
        weight=-0.10,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_drop": arm_posture.DEFAULT_HAND_MIN_DROP_M},
    )
    hand_pair_mirror = RewTerm(
        func=full_body_symmetry.hand_lateral_height_mirror_exp,
        weight=0.05,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot"), "std": 0.12},
    )
    hand_counter_swing = RewTerm(
        func=full_body_symmetry.hand_counter_swing_exp,
        weight=0.02,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "amplitude": full_body_symmetry.DEFAULT_HAND_COUNTER_SWING_AMPLITUDE_M,
            "std": full_body_symmetry.DEFAULT_HAND_COUNTER_SWING_STD_M,
        },
    )
    leg_joint_half_cycle_mirror = RewTerm(
        func=full_body_symmetry.leg_joint_half_cycle_mirror_exp,
        weight=0.04,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(full_body_symmetry.LEG_JOINT_PATTERNS)),
            "std": 0.45,
        },
    )
    arm_joint_mirror = RewTerm(
        func=full_body_symmetry.arm_joint_mirror_exp,
        weight=0.04,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(full_body_symmetry.ARM_JOINT_PATTERNS)),
            "std": 0.35,
        },
    )
    torso_yaw_rate = RewTerm(
        func=full_body_symmetry.torso_yaw_rate_l2,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    foot_lateral_mirror = RewTerm(
        func=symmetry_clock.foot_lateral_mirror_exp,
        weight=0.04,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "std": symmetry_clock.FOOT_LATERAL_MIRROR_STD_M,
        },
    )
    episode_contact_balance = RewTerm(
        func=symmetry_clock.episode_contact_balance_exp,
        weight=0.04,
        params={"command_name": "base_velocity", "std": symmetry_clock.CONTACT_BALANCE_STD},
    )


@configclass
class T1FlatHumanRefSymClockQuietUpperFullSymEnvCfg(T1FlatHumanRefSymClockEnvCfg):
    rewards: T1QuietUpperFullSymRewards = T1QuietUpperFullSymRewards()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = QUIET_UPPER_BODY_ACTION_SCALE
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.45
        self.rewards.joint_deviation_arms.weight = -0.08
        self.rewards.joint_deviation_head.weight = -0.04
        self.rewards.joint_deviation_waist.weight = -0.08
        self.rewards.feet_air_time.weight = 0.18
        self.rewards.feet_air_time.params["threshold"] = 0.28
        self.rewards.feet_slide.weight = -0.08
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.dof_acc_l2.weight = -3.0e-7


@configclass
class T1FlatHumanRefSymClockQuietUpperFullSymEnvCfg_PLAY(T1FlatHumanRefSymClockQuietUpperFullSymEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
