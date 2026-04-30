# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.t1.mdp import human_rewards

from .t1_asset_cfg import T1_MINIMAL_CFG


@configclass
class T1Rewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(Left|Right)_Ankle_.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(Left|Right)_Shoulder_.*", "(Left|Right)_Elbow_.*"])},
    )
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["AAHead_yaw", "Head_pitch"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Waist"])},
    )


@configclass
class T1HumanLikeRewards(T1Rewards):
    step_width_tracking = RewTerm(
        func=human_rewards.step_width_tracking_exp,
        weight=0.15,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "target_width": 0.18,
            "std": 0.06,
        },
    )
    pelvis_motion = RewTerm(
        func=human_rewards.pelvis_motion_penalty,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    air_time_balance = RewTerm(
        func=human_rewards.air_time_balance_penalty,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
            "max_time": 0.6,
        },
    )
    arm_swing_tracking = RewTerm(
        func=human_rewards.arm_swing_tracking_exp,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "shoulder_cfg": SceneEntityCfg("robot", joint_names=["Left_Shoulder_Pitch", "Right_Shoulder_Pitch"]),
            "hip_cfg": SceneEntityCfg("robot", joint_names=["Left_Hip_Pitch", "Right_Hip_Pitch"]),
            "scale": 0.45,
            "std": 0.35,
        },
    )
    reference_leg_pose = RewTerm(
        func=human_rewards.reference_leg_pose_tracking_exp,
        weight=0.12,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "hip_cfg": SceneEntityCfg("robot", joint_names=["Left_Hip_Pitch", "Right_Hip_Pitch"]),
            "knee_cfg": SceneEntityCfg("robot", joint_names=["Left_Knee_Pitch", "Right_Knee_Pitch"]),
            "ankle_cfg": SceneEntityCfg("robot", joint_names=["Left_Ankle_Pitch", "Right_Ankle_Pitch"]),
            "hip_scale": 0.35,
            "knee_scale": 0.55,
            "ankle_scale": 0.25,
            "std": 0.45,
        },
    )


@configclass
class T1HumanRefRewards(T1HumanLikeRewards):
    step_width_tracking = RewTerm(
        func=human_rewards.step_width_tracking_exp,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
            "target_width": 0.18,
            "std": 0.05,
        },
    )
    arm_swing_tracking = RewTerm(
        func=human_rewards.arm_swing_tracking_exp,
        weight=0.08,
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
        weight=0.35,
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
class T1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: T1Rewards = T1Rewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = T1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self.scene.contact_forces.update_period = self.sim.dt

        self.events.push_robot.interval_range_s = (8.0, 12.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.35, 0.35), "y": (-0.25, 0.25)}
        self.events.add_base_mass.params["asset_cfg"].body_names = ["Trunk"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
        self.events.base_com.params["asset_cfg"].body_names = ["Trunk"]
        self.events.base_com.params["com_range"] = {"x": (-0.02, 0.02), "y": (-0.015, 0.015), "z": (-0.01, 0.01)}
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Trunk"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.2, 0.2),
            },
        }

        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.dof_torques_l2.weight = -5.0e-6
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -2.5e-7

        self.commands.base_velocity.resampling_time_range = (6.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (-0.25, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)

        self.terminations.base_contact.params["sensor_cfg"].body_names = ["Trunk"]


@configclass
class T1FlatEnvCfg_PLAY(T1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class T1FlatHumanLikeEnvCfg(T1FlatEnvCfg):
    rewards: T1HumanLikeRewards = T1HumanLikeRewards()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_deviation_arms.weight = -0.03
        self.rewards.joint_deviation_head.weight = -0.03
        self.rewards.joint_deviation_waist.weight = -0.03


@configclass
class T1FlatHumanLikeEnvCfg_PLAY(T1FlatHumanLikeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class T1FlatHumanRefEnvCfg(T1FlatHumanLikeEnvCfg):
    rewards: T1HumanRefRewards = T1HumanRefRewards()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 1.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class T1FlatHumanRefEnvCfg_PLAY(T1FlatHumanRefEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
