from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.t1.flat_env_cfg import (
    T1FlatHumanRefEnvCfg,
    T1HumanRefRewards,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.t1.mdp import human_rewards

from .mdp import arm_posture, symmetry_clock, symmetry_commands, symmetry_curriculums, symmetry_events, symmetry_rewards


ROLLOUT_STEPS_PER_ITERATION = 24
PHASE_REWARD_RAMP_START_ITERATION = 300
PHASE_REWARD_RAMP_END_ITERATION = 500
PHASE_REWARD_RAMP_START_STEP = ROLLOUT_STEPS_PER_ITERATION * PHASE_REWARD_RAMP_START_ITERATION
PHASE_REWARD_RAMP_END_STEP = ROLLOUT_STEPS_PER_ITERATION * PHASE_REWARD_RAMP_END_ITERATION
PHASE_CONTACT_SCHEDULE_FINAL_WEIGHT = 1.0
PHASE_FOOT_CLEARANCE_FINAL_WEIGHT = 0.4
PHASE_STANCE_SLIP_FINAL_WEIGHT = -0.2
PAIRED_ACTION_SYMMETRY_WEIGHT = 0.2
FOOT_FORWARD_MIRROR_WEIGHT = 0.0
FOOT_LATERAL_MIRROR_WEIGHT = 0.0
FOOT_FORWARD_PHASE_WEIGHT = 0.0
CONTACT_BALANCE_WEIGHT = 0.0
QUIET_UPPER_BODY_ACTION_SCALE = {
    "(Left|Right)_Hip_.*": 0.5,
    "(Left|Right)_Knee_Pitch": 0.5,
    "(Left|Right)_Ankle_.*": 0.5,
    "Waist": 0.12,
    "(Left|Right)_Shoulder_.*": 0.16,
    "(Left|Right)_Elbow_.*": 0.12,
    "AAHead_yaw": 0.06,
    "Head_pitch": 0.06,
}


@configclass
class T1SymClockMirrorRewards(T1HumanRefRewards):
    paired_action_symmetry = RewTerm(
        func=symmetry_rewards.paired_action_symmetry_exp,
        weight=PAIRED_ACTION_SYMMETRY_WEIGHT,
        params={"std": 1.5},
    )
    phase_contact_schedule = RewTerm(
        func=symmetry_clock.phase_contact_schedule_exp,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
            "force_scale": symmetry_clock.CONTACT_FORCE_NORMALIZER_N,
            "std": 0.35,
        },
    )
    phase_foot_clearance = RewTerm(
        func=symmetry_clock.phase_foot_clearance_exp,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "target_height": symmetry_clock.SWING_CLEARANCE_TARGET_M,
            "std": 0.04,
        },
    )
    phase_stance_slip = RewTerm(
        func=symmetry_clock.phase_stance_slip_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "contact_threshold": 1.0,
        },
    )
    foot_forward_mirror = RewTerm(
        func=symmetry_clock.foot_forward_mirror_exp,
        weight=FOOT_FORWARD_MIRROR_WEIGHT,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "std": symmetry_clock.FOOT_MIRROR_STD_M,
        },
    )
    foot_lateral_mirror = RewTerm(
        func=symmetry_clock.foot_lateral_mirror_exp,
        weight=FOOT_LATERAL_MIRROR_WEIGHT,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "std": symmetry_clock.FOOT_LATERAL_MIRROR_STD_M,
        },
    )
    phase_foot_forward_position = RewTerm(
        func=symmetry_clock.phase_foot_forward_position_exp,
        weight=FOOT_FORWARD_PHASE_WEIGHT,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "amplitude": symmetry_clock.FOOT_FORWARD_PHASE_AMPLITUDE_M,
            "std": symmetry_clock.FOOT_FORWARD_PHASE_STD_M,
        },
    )
    episode_contact_balance = RewTerm(
        func=symmetry_clock.episode_contact_balance_exp,
        weight=CONTACT_BALANCE_WEIGHT,
        params={"command_name": "base_velocity", "std": symmetry_clock.CONTACT_BALANCE_STD},
    )
    pelvis_motion = RewTerm(
        func=human_rewards.pelvis_motion_penalty,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_velocity_waist = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Waist"])},
    )
    joint_velocity_arms = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(Left|Right)_Shoulder_.*", "(Left|Right)_Elbow_.*"])},
    )
    arm_outward_posture = RewTerm(
        func=arm_posture.arm_outward_posture_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "roll_margin": arm_posture.DEFAULT_SHOULDER_ROLL_MARGIN_RAD,
            "pitch_margin": arm_posture.DEFAULT_SHOULDER_PITCH_MARGIN_RAD,
        },
    )
    hand_lateral_spread = RewTerm(
        func=arm_posture.hand_lateral_spread_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_radius": arm_posture.DEFAULT_ARM_LATERAL_RADIUS_M,
            "margin": arm_posture.DEFAULT_ARM_LATERAL_MARGIN_M,
        },
    )
    joint_velocity_head = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["AAHead_yaw", "Head_pitch"])},
    )
    air_time_balance = None
    step_width_tracking = RewTerm(
        func=human_rewards.step_width_tracking_exp,
        weight=0.08,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "target_width": 0.18,
            "std": 0.05,
        },
    )
    arm_swing_tracking = RewTerm(
        func=human_rewards.arm_swing_tracking_exp,
        weight=0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "shoulder_cfg": SceneEntityCfg("robot", joint_names=["Left_Shoulder_Pitch", "Right_Shoulder_Pitch"]),
            "hip_cfg": SceneEntityCfg("robot", joint_names=["Left_Hip_Pitch", "Right_Hip_Pitch"]),
            "scale": 0.55,
            "std": 0.3,
        },
    )
    reference_leg_pose = RewTerm(
        func=human_rewards.reference_leg_pose_tracking_exp,
        weight=0.14,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "hip_cfg": SceneEntityCfg("robot", joint_names=["Left_Hip_Pitch", "Right_Hip_Pitch"]),
            "knee_cfg": SceneEntityCfg("robot", joint_names=["Left_Knee_Pitch", "Right_Knee_Pitch"]),
            "ankle_cfg": SceneEntityCfg("robot", joint_names=["Left_Ankle_Pitch", "Right_Ankle_Pitch"]),
            "hip_scale": 0.4,
            "knee_scale": 0.7,
            "ankle_scale": 0.3,
            "std": 0.35,
        },
    )


@configclass
class T1SymClockRewards(T1SymClockMirrorRewards):
    paired_action_symmetry = None


@configclass
class T1SymClockMirrorCurriculum:
    terrain_levels = None
    phase_contact_schedule_weight = CurrTerm(
        func=symmetry_curriculums.linear_reward_weight_schedule,
        params={
            "term_name": "phase_contact_schedule",
            "initial_weight": 0.0,
            "final_weight": PHASE_CONTACT_SCHEDULE_FINAL_WEIGHT,
            "start_step": PHASE_REWARD_RAMP_START_STEP,
            "end_step": PHASE_REWARD_RAMP_END_STEP,
        },
    )
    phase_foot_clearance_weight = CurrTerm(
        func=symmetry_curriculums.linear_reward_weight_schedule,
        params={
            "term_name": "phase_foot_clearance",
            "initial_weight": 0.0,
            "final_weight": PHASE_FOOT_CLEARANCE_FINAL_WEIGHT,
            "start_step": PHASE_REWARD_RAMP_START_STEP,
            "end_step": PHASE_REWARD_RAMP_END_STEP,
        },
    )
    phase_stance_slip_weight = CurrTerm(
        func=symmetry_curriculums.linear_reward_weight_schedule,
        params={
            "term_name": "phase_stance_slip",
            "initial_weight": 0.0,
            "final_weight": PHASE_STANCE_SLIP_FINAL_WEIGHT,
            "start_step": PHASE_REWARD_RAMP_START_STEP,
            "end_step": PHASE_REWARD_RAMP_END_STEP,
        },
    )


@configclass
class T1FlatHumanRefSymClockMirrorEnvCfg(T1FlatHumanRefEnvCfg):
    rewards: T1SymClockMirrorRewards = T1SymClockMirrorRewards()
    curriculum: T1SymClockMirrorCurriculum = T1SymClockMirrorCurriculum()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = symmetry_commands.MirroredUniformVelocityCommandCfg(
            class_type=symmetry_commands.MirroredUniformVelocityCommand,
            asset_name="robot",
            resampling_time_range=(8.0, 12.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            ranges=symmetry_commands.MirroredUniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.2, 1.1),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-1.0, 1.0),
                heading=None,
            ),
        )
        self.observations.policy.phase_clock = ObsTerm(
            func=symmetry_clock.phase_clock_observation,
            params={"command_name": "base_velocity"},
        )
        self.events.reset_base.func = symmetry_events.paired_reset_root_state_uniform
        self.events.reset_robot_joints.func = symmetry_events.paired_reset_joints_by_scale
        self.events.push_robot.func = symmetry_events.paired_push_by_setting_velocity
        self.events.base_external_force_torque = None
        self.events.base_com.params["com_range"] = {"x": (-0.02, 0.02), "y": (0.0, 0.0), "z": (-0.01, 0.01)}
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_slide.weight = -0.05


@configclass
class T1FlatHumanRefSymClockEnvCfg(T1FlatHumanRefSymClockMirrorEnvCfg):
    rewards: T1SymClockRewards = T1SymClockRewards()


@configclass
class T1FlatHumanRefSymClockMirrorQuietUpperEnvCfg(T1FlatHumanRefSymClockMirrorEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = QUIET_UPPER_BODY_ACTION_SCALE


@configclass
class T1FlatHumanRefSymClockQuietUpperEnvCfg(T1FlatHumanRefSymClockEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale = QUIET_UPPER_BODY_ACTION_SCALE


@configclass
class T1FlatHumanRefSymClockMirrorEnvCfg_PLAY(T1FlatHumanRefSymClockMirrorEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
