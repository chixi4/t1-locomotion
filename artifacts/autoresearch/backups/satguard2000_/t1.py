import os

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
)

assert gymtorch

import torch

import numpy as np
from .base_task import BaseTask

from utils.utils import apply_randomization


class T1(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()
        self.action_scale = self._make_dof_vector(self.cfg["control"].get("action_scale", 1.0), "action_scale").unsqueeze(0)
        self.dof_target_lower, self.dof_target_upper = self._make_dof_target_clip()

        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"].get("dof_friction"))

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])

        # prepare penalized and termination contact indices
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _make_dof_vector(self, spec, label):
        if isinstance(spec, (int, float)):
            return torch.full((self.num_dofs,), float(spec), dtype=torch.float, device=self.device)
        if not isinstance(spec, dict):
            raise TypeError(f"{label} must be a scalar or dict")
        values = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i, dof_name in enumerate(self.dof_names):
            found = False
            for key, value in spec.items():
                if key == "default":
                    continue
                if key == dof_name or key in dof_name:
                    values[i] = float(value)
                    found = True
                    break
            if not found:
                if "default" not in spec:
                    raise ValueError(f"{label} of joint {dof_name} was not defined")
                values[i] = float(spec["default"])
        return values

    def _make_dof_target_clip(self):
        lower = torch.full((1, self.num_dofs), -torch.inf, dtype=torch.float, device=self.device)
        upper = torch.full((1, self.num_dofs), torch.inf, dtype=torch.float, device=self.device)
        clip_cfg = self.cfg["control"].get("target_clip", {})
        for name, limits in clip_cfg.items():
            idx = self.dof_names.index(name)
            lower[:, idx] = float(limits[0])
            upper[:, idx] = float(limits[1])
        return lower, upper

    def _resolve_dof_indices(self, names):
        return torch.tensor([self.dof_names.index(name) for name in names], dtype=torch.long, device=self.device)

    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomization(
                    props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomization(
                    props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomization(
                    props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].mass, self.base_mass_scaled[i, 3] = apply_randomization(
                    props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True
                )
            else:
                props[j].com.x = apply_randomization(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = apply_randomization(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = apply_randomization(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = apply_randomization(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg["terrain"]["type"] == "plane":
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg["env"]["env_spacing"]
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
        else:
            num_cols = max(1.0, np.floor(np.sqrt(self.num_envs * self.terrain.env_length / self.terrain.env_width)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            self.env_origins[:, 0] = self.terrain.env_width / (num_rows + 1) * (xx.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 1] = self.terrain.env_length / (num_cols + 1) * (yy.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 2] = self.terrain.terrain_heights(self.env_origins)

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        if self.num_actions != self.num_dofs:
            raise ValueError(f"num_actions ({self.num_actions}) must match loaded asset DOFs ({self.num_dofs})")
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, self.num_bodies, 13)
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg["commands"]["num_commands"], dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.tilt_sq_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.tilt_bad_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.tilt_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if self.cfg["commands"].get("sampling", "box") == "grid3d_circle" and self.cfg["commands"].get("curriculum", False):
            self._init_grid3d_curriculum()
        elif self.cfg["commands"].get("sampling", "box") == "circle" and self.cfg["commands"].get("curriculum", False):
            self.curriculum_prob = torch.zeros(
                1 + self.cfg["commands"]["lin_vel_levels"],
                1 + 2 * self.cfg["commands"]["ang_vel_levels"],
                dtype=torch.float,
                device=self.device,
            )
            self.curriculum_prob[0, self.cfg["commands"]["ang_vel_levels"]] = 1.0
        else:
            self.curriculum_prob = torch.zeros(
                1 + 2 * self.cfg["commands"]["lin_vel_levels"],
                1 + 2 * self.cfg["commands"]["ang_vel_levels"],
                dtype=torch.float,
                device=self.device,
            )
            self.curriculum_prob[self.cfg["commands"]["lin_vel_levels"], self.cfg["commands"]["ang_vel_levels"]] = 1.0
        if not hasattr(self, "env_curriculum_level"):
            self.env_curriculum_level = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0
        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"]["default"]
        self.arm_indices = self._resolve_dof_indices(self.cfg["control"].get("arm_dof_names", []))
        self.leg_indices = self._resolve_dof_indices(self.cfg["control"].get("leg_dof_names", []))

    def _init_grid3d_curriculum(self):
        lin_levels = self.cfg["commands"]["lin_vel_levels"]
        yaw_levels = self.cfg["commands"]["ang_vel_levels"]
        lin_axis = torch.arange(-lin_levels, lin_levels + 1, dtype=torch.long, device=self.device)
        lx, ly = torch.meshgrid(lin_axis, lin_axis, indexing="ij")
        self.curriculum_mask = (lx.square() + ly.square()) <= lin_levels * lin_levels
        self.curriculum_mask = self.curriculum_mask.unsqueeze(-1).expand(-1, -1, 1 + 2 * yaw_levels)
        self.original_curriculum_mask = self.curriculum_mask.clone()
        allowed_checkpoint = self.cfg["commands"].get("allowed_curriculum_checkpoint")
        if allowed_checkpoint:
            checkpoint = torch.load(allowed_checkpoint, map_location=self.device, weights_only=True)
            allowed = checkpoint["curriculum"].to(device=self.device) > 0.5
            if tuple(allowed.shape) != (1 + 2 * lin_levels, 1 + 2 * lin_levels, 1 + 2 * yaw_levels):
                raise ValueError(f"Allowed curriculum shape {tuple(allowed.shape)} does not match this grid")
            self.curriculum_mask = self.curriculum_mask & allowed
        self.curriculum_prob = torch.zeros(
            1 + 2 * lin_levels,
            1 + 2 * lin_levels,
            1 + 2 * yaw_levels,
            dtype=torch.float,
            device=self.device,
        )
        self.curriculum_prob[lin_levels, lin_levels, yaw_levels] = 1.0
        self.curriculum_prob *= self.curriculum_mask.float()
        if self.curriculum_prob[lin_levels, lin_levels, yaw_levels] == 0:
            raise ValueError("Center curriculum cell is not allowed by the allowed mask")
        self.env_curriculum_level = torch.zeros(self.num_envs, 3, dtype=torch.long, device=self.device)

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

    def reset(self):
        """Reset all robots"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._resample_commands()
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._update_curriculum(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.tilt_sq_sum[env_ids] = 0.0
        self.tilt_bad_count[env_ids] = 0.0
        self.tilt_count[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        self.root_states[env_ids, :2] = apply_randomization(self.root_states[env_ids, :2], self.cfg["randomization"].get("init_base_pos_xy"))
        self.root_states[env_ids, 2] += self.terrain.terrain_heights(self.root_states[env_ids, :2])
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.rand(len(env_ids), device=self.device) * (2 * torch.pi),
        )
        self.root_states[env_ids, 7:9] = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_lin_vel_xy"),
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return
        out_x_min = self.root_states[:, 0] < -0.75 * self.terrain.border_size
        out_x_max = self.root_states[:, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min = self.root_states[:, 1] < -0.75 * self.terrain.border_size
        out_y_max = self.root_states[:, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size
        self.root_states[out_x_min, 0] += self.terrain.env_width + self.terrain.border_size
        self.root_states[out_x_max, 0] -= self.terrain.env_width + self.terrain.border_size
        self.root_states[out_y_min, 1] += self.terrain.env_length + self.terrain.border_size
        self.root_states[out_y_max, 1] -= self.terrain.env_length + self.terrain.border_size
        self.body_states[out_x_min, :, 0] += self.terrain.env_width + self.terrain.border_size
        self.body_states[out_x_max, :, 0] -= self.terrain.env_width + self.terrain.border_size
        self.body_states[out_y_min, :, 1] += self.terrain.env_length + self.terrain.border_size
        self.body_states[out_y_max, :, 1] -= self.terrain.env_length + self.terrain.border_size
        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self._refresh_feet_state()

    def _resample_commands(self):
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        if self.cfg["commands"]["curriculum"]:
            self._resample_curriculum_commands(env_ids)
        elif self.cfg["commands"].get("sampling", "box") == "circle":
            self._resample_circle_commands(env_ids)
        else:
            self.commands[env_ids, 0] = torch_rand_float(
                self.cfg["commands"]["lin_vel_x"][0], self.cfg["commands"]["lin_vel_x"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(
                self.cfg["commands"]["lin_vel_y"][0], self.cfg["commands"]["lin_vel_y"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 2] = torch_rand_float(
                self.cfg["commands"]["ang_vel_yaw"][0], self.cfg["commands"]["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        self.gait_frequency[env_ids] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0], self.cfg["commands"]["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        still_envs = env_ids[torch.randperm(len(env_ids))[: int(self.cfg["commands"]["still_proportion"] * len(env_ids))]]
        self.commands[still_envs, :] = 0.0
        self.gait_frequency[still_envs] = 0.0
        if self.cfg["commands"].get("sampling", "box") == "grid3d_circle" and self.cfg["commands"].get("curriculum", False):
            self.env_curriculum_level[still_envs, :] = 0
        resample_min = int(self.cfg["commands"]["resampling_time_s"][0] / self.dt)
        resample_max = int(self.cfg["commands"]["resampling_time_s"][1] / self.dt)
        if resample_max <= resample_min:
            resample_steps = torch.full((len(env_ids),), resample_min, dtype=torch.long, device=self.device)
        else:
            resample_steps = torch.randint(resample_min, resample_max, (len(env_ids),), device=self.device)
        self.cmd_resample_time[env_ids] += resample_steps

    def _resample_circle_commands(self, env_ids):
        speed_range = self.cfg["commands"].get(
            "linear_speed",
            [0.0, max(abs(self.cfg["commands"]["lin_vel_x"][0]), abs(self.cfg["commands"]["lin_vel_x"][1]), abs(self.cfg["commands"]["lin_vel_y"][0]), abs(self.cfg["commands"]["lin_vel_y"][1]))],
        )
        speed_min, speed_max = float(speed_range[0]), float(speed_range[1])
        if self.cfg["commands"].get("radial_distribution", "uniform_area") == "uniform_area":
            u = torch.rand(len(env_ids), dtype=torch.float, device=self.device)
            speed = torch.sqrt(speed_min * speed_min + (speed_max * speed_max - speed_min * speed_min) * u)
        else:
            speed = torch_rand_float(speed_min, speed_max, (len(env_ids), 1), device=self.device).squeeze(1)
        theta = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 0] = speed * torch.cos(theta)
        self.commands[env_ids, 1] = speed * torch.sin(theta)
        self.commands[env_ids, 2] = torch_rand_float(
            self.cfg["commands"]["ang_vel_yaw"][0], self.cfg["commands"]["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)

    def _update_curriculum(self, env_ids):
        if not self.cfg["commands"]["curriculum"]:
            return
        if self.cfg["commands"].get("sampling", "box") == "grid3d_circle":
            self._update_grid3d_curriculum(env_ids)
            return
        success = self.episode_length_buf[env_ids] > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt) * (
            1 - self.cfg["commands"]["episode_length_toler"]
        )
        success &= torch.abs(self.filtered_lin_vel[env_ids, 0] - self.commands[env_ids, 0]) < self.cfg["commands"]["lin_vel_x_toler"]
        success &= torch.abs(self.filtered_lin_vel[env_ids, 1] - self.commands[env_ids, 1]) < self.cfg["commands"]["lin_vel_y_toler"]
        success &= torch.abs(self.filtered_ang_vel[env_ids, 2] - self.commands[env_ids, 2]) < self.cfg["commands"]["ang_vel_yaw_toler"]
        for i in range(len(env_ids)):
            if success[i]:
                if self.cfg["commands"].get("sampling", "box") == "circle":
                    x = self.env_curriculum_level[env_ids[i], 0]
                    y = self.env_curriculum_level[env_ids[i], 1] + self.cfg["commands"]["ang_vel_levels"]
                else:
                    x = self.env_curriculum_level[env_ids[i], 0] + self.cfg["commands"]["lin_vel_levels"]
                    y = self.env_curriculum_level[env_ids[i], 1] + self.cfg["commands"]["ang_vel_levels"]
                self.curriculum_prob[x, y] += self.cfg["commands"]["update_rate"]
                if x > 0:
                    self.curriculum_prob[x - 1, y] += self.cfg["commands"]["update_rate"]
                if x < self.curriculum_prob.shape[0] - 1:
                    self.curriculum_prob[x + 1, y] += self.cfg["commands"]["update_rate"]
                if y > 0:
                    self.curriculum_prob[x, y - 1] += self.cfg["commands"]["update_rate"]
                if y < self.curriculum_prob.shape[1] - 1:
                    self.curriculum_prob[x, y + 1] += self.cfg["commands"]["update_rate"]
        self.curriculum_prob.clamp_(max=1.0)

    def _grid3d_radius_ratio(self, levels):
        lin_levels = float(self.cfg["commands"]["lin_vel_levels"])
        yaw_levels = float(self.cfg["commands"]["ang_vel_levels"])
        max_radius = np.sqrt(lin_levels * lin_levels + yaw_levels * yaw_levels)
        radius = torch.sqrt(torch.sum(levels[:, :2].float().square(), dim=1) + levels[:, 2].float().square())
        return torch.clamp(radius / max_radius, min=0.0, max=1.0)

    def _grid3d_tolerances(self, levels):
        radius_ratio = self._grid3d_radius_ratio(levels)
        lin_center = float(self.cfg["commands"].get("lin_vel_center_toler", self.cfg["commands"]["lin_vel_x_toler"]))
        lin_edge = float(self.cfg["commands"].get("lin_vel_edge_toler", lin_center))
        yaw_center = float(self.cfg["commands"].get("ang_vel_yaw_center_toler", self.cfg["commands"]["ang_vel_yaw_toler"]))
        yaw_edge = float(self.cfg["commands"].get("ang_vel_yaw_edge_toler", yaw_center))
        lin_toler = lin_center + radius_ratio * (lin_edge - lin_center)
        yaw_toler = yaw_center + radius_ratio * (yaw_edge - yaw_center)
        return lin_toler, yaw_toler

    def _grid3d_tracking_sigmas(self):
        if not (
            self.cfg["commands"].get("sampling", "box") == "grid3d_circle"
            and self.cfg["rewards"].get("tracking_sigma_curriculum", False)
        ):
            sigma = self.cfg["rewards"]["tracking_sigma"]
            return sigma, sigma
        lin_center = float(self.cfg["rewards"].get("tracking_lin_sigma_center", self.cfg["rewards"]["tracking_sigma"]))
        lin_edge = float(self.cfg["rewards"].get("tracking_lin_sigma_edge", lin_center))
        yaw_center = float(self.cfg["rewards"].get("tracking_ang_sigma_center", self.cfg["rewards"]["tracking_sigma"]))
        yaw_edge = float(self.cfg["rewards"].get("tracking_ang_sigma_edge", yaw_center))
        if lin_center == lin_edge and yaw_center == yaw_edge:
            return lin_center, yaw_center
        radius_ratio = self._grid3d_radius_ratio(self.env_curriculum_level)
        lin_sigma = lin_center + radius_ratio * (lin_edge - lin_center)
        yaw_sigma = yaw_center + radius_ratio * (yaw_edge - yaw_center)
        return lin_sigma, yaw_sigma

    def _grid3d_sway_rho(self, levels):
        lin_level = torch.linalg.norm(levels[:, :2].float(), dim=1) / float(self.cfg["commands"]["lin_vel_levels"])
        yaw_level = torch.abs(levels[:, 2].float()) / float(self.cfg["commands"]["ang_vel_levels"])
        return torch.clamp(torch.maximum(lin_level, yaw_level), min=0.0, max=1.0)

    def _grid3d_sway_limits(self, levels):
        rho = self._grid3d_sway_rho(levels)
        tilt_rms_limit = 0.042 + 0.023 * rho
        tilt_bad_limit = 0.085 + 0.040 * rho
        return rho, tilt_rms_limit, tilt_bad_limit

    def _update_sway_episode_stats(self):
        if not self.cfg["commands"].get("sway_curriculum", False):
            return
        valid = self.episode_length_buf > int(1.0 / self.dt)
        if not torch.any(valid):
            return
        tilt = torch.sqrt(self.projected_gravity[:, 0].square() + self.projected_gravity[:, 1].square())
        _, _, tilt_bad_limit = self._grid3d_sway_limits(self.env_curriculum_level)
        valid_float = valid.float()
        self.tilt_sq_sum += tilt.square() * valid_float
        self.tilt_bad_count += ((tilt > tilt_bad_limit) & valid).float()
        self.tilt_count += valid_float

    def _grid3d_sway_success(self, env_ids):
        levels = self.env_curriculum_level[env_ids]
        rho, tilt_rms_limit, _ = self._grid3d_sway_limits(levels)
        count = torch.clamp(self.tilt_count[env_ids], min=1.0)
        tilt_rms = torch.sqrt(self.tilt_sq_sum[env_ids] / count)
        tilt_bad_frac = self.tilt_bad_count[env_ids] / count
        lin_error = torch.linalg.norm(self.filtered_lin_vel[env_ids, :2] - self.commands[env_ids, :2], dim=1)
        yaw_error = torch.abs(self.filtered_ang_vel[env_ids, 2] - self.commands[env_ids, 2])
        success = self.episode_length_buf[env_ids] > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt) * (
            1 - self.cfg["commands"]["episode_length_toler"]
        )
        success &= self.tilt_count[env_ids] > 0
        success &= tilt_rms < tilt_rms_limit
        success &= tilt_bad_frac < float(self.cfg["commands"].get("tilt_bad_frac_limit", 0.03))
        success &= lin_error < (0.30 + 0.25 * rho)
        success &= yaw_error < (0.15 + 0.10 * rho)
        return success

    def _update_grid3d_curriculum(self, env_ids):
        if self.cfg["commands"].get("sway_curriculum", False):
            success = self._grid3d_sway_success(env_ids)
        else:
            success = self.episode_length_buf[env_ids] > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt) * (
                1 - self.cfg["commands"]["episode_length_toler"]
            )
            levels = self.env_curriculum_level[env_ids]
            lin_toler, yaw_toler = self._grid3d_tolerances(levels)
            lin_error = torch.linalg.norm(self.filtered_lin_vel[env_ids, :2] - self.commands[env_ids, :2], dim=1)
            yaw_error = torch.abs(self.filtered_ang_vel[env_ids, 2] - self.commands[env_ids, 2])
            success &= lin_error < lin_toler
            success &= yaw_error < yaw_toler

        lin_levels = self.cfg["commands"]["lin_vel_levels"]
        yaw_levels = self.cfg["commands"]["ang_vel_levels"]
        if not torch.any(success):
            return
        center = self.env_curriculum_level[env_ids[success]].clone()
        center[:, 0] += lin_levels
        center[:, 1] += lin_levels
        center[:, 2] += yaw_levels
        if self.cfg["commands"].get("grid3d_unlock_neighbors", "full26") == "face6":
            offsets = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
        else:
            offsets = tuple((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))
        for offset in offsets:
            neighbor = center + torch.tensor(offset, dtype=torch.long, device=self.device)
            valid = (neighbor[:, 0] >= 0) & (neighbor[:, 0] < self.curriculum_prob.shape[0])
            valid &= (neighbor[:, 1] >= 0) & (neighbor[:, 1] < self.curriculum_prob.shape[1])
            valid &= (neighbor[:, 2] >= 0) & (neighbor[:, 2] < self.curriculum_prob.shape[2])
            if not torch.any(valid):
                continue
            candidate = neighbor[valid]
            valid_mask = self.curriculum_mask[candidate[:, 0], candidate[:, 1], candidate[:, 2]]
            candidate = candidate[valid_mask]
            if len(candidate) > 0:
                self.curriculum_prob[candidate[:, 0], candidate[:, 1], candidate[:, 2]] = 1.0

    def _resample_curriculum_commands(self, env_ids):
        if self.cfg["commands"].get("sampling", "box") == "grid3d_circle":
            self._resample_grid3d_curriculum_commands(env_ids)
            return
        if self.cfg["commands"].get("sampling", "box") == "circle":
            self._resample_circle_curriculum_commands(env_ids)
            return
        grid_idx = torch.multinomial(self.curriculum_prob.flatten(), len(env_ids), replacement=True)
        lin_vel_level = grid_idx % self.curriculum_prob.shape[1] - self.cfg["commands"]["lin_vel_levels"]
        ang_vel_level = grid_idx // self.curriculum_prob.shape[1] - self.cfg["commands"]["ang_vel_levels"]
        self.env_curriculum_level[env_ids, 0] = lin_vel_level
        self.env_curriculum_level[env_ids, 1] = ang_vel_level
        self.mean_lin_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 0]).float())
        self.mean_ang_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 1]).float())
        self.max_lin_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 0]))
        self.max_ang_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 1]))
        self.commands[env_ids, 0] = (
            lin_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["lin_vel_x_resolution"]
        self.commands[env_ids, 1] = (
            torch.abs(lin_vel_level)
            * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
            * self.cfg["commands"]["lin_vel_y_resolution"]
        )
        self.commands[env_ids, 2] = (
            ang_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["ang_vel_resolution"]

    def _resample_grid3d_curriculum_commands(self, env_ids):
        unlocked = (self.curriculum_prob > 0.5) & self.curriculum_mask
        grid_idx = torch.multinomial(unlocked.flatten().float(), len(env_ids), replacement=True)

        lin_bins = 1 + 2 * self.cfg["commands"]["lin_vel_levels"]
        yaw_bins = 1 + 2 * self.cfg["commands"]["ang_vel_levels"]
        lx_idx = grid_idx // (lin_bins * yaw_bins)
        rem = grid_idx % (lin_bins * yaw_bins)
        ly_idx = rem // yaw_bins
        yaw_idx = rem % yaw_bins

        lin_vel_level_x = lx_idx - self.cfg["commands"]["lin_vel_levels"]
        lin_vel_level_y = ly_idx - self.cfg["commands"]["lin_vel_levels"]
        ang_vel_level = yaw_idx - self.cfg["commands"]["ang_vel_levels"]

        self.env_curriculum_level[env_ids, 0] = lin_vel_level_x
        self.env_curriculum_level[env_ids, 1] = lin_vel_level_y
        self.env_curriculum_level[env_ids, 2] = ang_vel_level

        lin_levels = self.cfg["commands"]["lin_vel_levels"]
        yaw_levels = self.cfg["commands"]["ang_vel_levels"]
        lin_radius = torch.linalg.norm(self.env_curriculum_level[:, :2].float(), dim=1)
        self.mean_lin_vel_level = torch.mean(lin_radius)
        self.mean_ang_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 2]).float())
        self.max_lin_vel_level = torch.max(lin_radius)
        self.max_ang_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 2]))

        speed_max = float(self.cfg["commands"].get("linear_speed", [0.0, 1.0])[1])
        lin_resolution = float(self.cfg["commands"].get("linear_speed_resolution", speed_max / max(1, lin_levels)))
        x = (lin_vel_level_x.float() + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)) * lin_resolution
        y = (lin_vel_level_y.float() + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)) * lin_resolution
        speed = torch.sqrt(x.square() + y.square())
        scale = torch.clamp(speed_max / torch.clamp(speed, min=1.0e-6), max=1.0)
        self.commands[env_ids, 0] = x * scale
        self.commands[env_ids, 1] = y * scale

        yaw_resolution = float(self.cfg["commands"].get("ang_vel_resolution", 1.0 / max(1, yaw_levels)))
        yaw = (ang_vel_level.float() + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)) * yaw_resolution
        self.commands[env_ids, 2] = torch.clamp(yaw, self.cfg["commands"]["ang_vel_yaw"][0], self.cfg["commands"]["ang_vel_yaw"][1])

    def _resample_circle_curriculum_commands(self, env_ids):
        yaw_bins = self.curriculum_prob.shape[1]
        grid_idx = torch.multinomial(self.curriculum_prob.flatten(), len(env_ids), replacement=True)
        speed_level = grid_idx // yaw_bins
        ang_vel_level = grid_idx % yaw_bins - self.cfg["commands"]["ang_vel_levels"]
        self.env_curriculum_level[env_ids, 0] = speed_level
        self.env_curriculum_level[env_ids, 1] = ang_vel_level
        self.mean_lin_vel_level = torch.mean(self.env_curriculum_level[:, 0].float())
        self.mean_ang_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 1]).float())
        self.max_lin_vel_level = torch.max(self.env_curriculum_level[:, 0])
        self.max_ang_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 1]))

        speed_max = float(self.cfg["commands"].get("linear_speed", [0.0, 1.0])[1])
        speed_resolution = float(
            self.cfg["commands"].get("linear_speed_resolution", speed_max / max(1, self.cfg["commands"]["lin_vel_levels"]))
        )
        speed = torch.clamp(
            (speed_level.float() + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)) * speed_resolution,
            min=0.0,
            max=speed_max,
        )
        theta = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 0] = speed * torch.cos(theta)
        self.commands[env_ids, 1] = speed * torch.sin(theta)
        self.commands[env_ids, 2] = (
            ang_vel_level.float() + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"].get("ang_vel_resolution", 0.1)

    def step(self, actions):
        # pre physics step
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        dof_targets = self.default_dof_pos + self.action_scale * self.actions
        dof_targets = torch.maximum(torch.minimum(dof_targets, self.dof_target_upper), self.dof_target_lower)

        # perform physics step
        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        # post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self._refresh_feet_state()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)
        self._update_sway_episode_stats()

        self._kick_robots()
        self._push_robots()
        self._check_termination()
        self._compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._reset_idx(env_ids)
        self._teleport_robot()
        self._resample_commands()

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            self.root_states[:, 7:10] = apply_randomization(self.root_states[:, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            self.root_states[:, 10:13] = apply_randomization(self.root_states[:, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]),
                self.cfg["randomization"].get("push_torque"),
            )
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(
            self.cfg["randomization"]["push_duration_s"] / self.dt
        ):
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.body_states[:, self.feet_indices, 3:7]
        roll, _, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        feet_edge_relative_pos = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
        feet_edge_pos = expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))
        self.feet_contact[:] = torch.any(
            (feet_edge_pos[:, 2] - self.terrain.terrain_heights(feet_edge_pos) < 0.01).reshape(
                self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2]
            ),
            dim=2,
        )

    def _check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.reset_buf |= self.root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.reset_buf |= self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        self.reset_buf |= self.time_out_buf
        self.time_out_buf |= self.episode_length_buf == self.cmd_resample_time

    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        self._tracking_lin_sigma, self._tracking_yaw_sigma = self._grid3d_tracking_sigmas()
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _compute_observations(self):
        """Computes observations"""
        commands_scale = torch.tensor(
            [self.cfg["normalization"]["lin_vel"], self.cfg["normalization"]["lin_vel"], self.cfg["normalization"]["ang_vel"]],
            device=self.device,
        )
        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                self.commands[:, :3] * commands_scale,
                (torch.cos(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                (torch.sin(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
                apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    # ------------ reward functions----------------
    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
        return torch.exp(torch.neg(torch.square(self.commands[:, 0] - self.filtered_lin_vel[:, 0]) / self._tracking_lin_sigma))

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        return torch.exp(torch.neg(torch.square(self.commands[:, 1] - self.filtered_lin_vel[:, 1]) / self._tracking_lin_sigma))

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        return torch.exp(torch.neg(torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2]) / self._tracking_yaw_sigma))

    def _reward_base_height(self):
        # Tracking of base height
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        return torch.square(base_height - self.cfg["rewards"]["base_height_target"])

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg["rewards"]["soft_dof_vel_limit"]).clip(min=0.0, max=1.0),
            dim=-1,
        )

    def _reward_torque_limits(self):
        # Penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg["rewards"]["soft_torque_limit"]).clip(min=0.0),
            dim=-1,
        )

    def _reward_torque_tiredness(self):
        # Penalize torque tiredness
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

    def _reward_power(self):
        # Penalize power
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_arm_action_rate(self):
        return torch.sum(torch.square(self.actions[:, self.arm_indices] - self.last_actions[:, self.arm_indices]), dim=-1)

    def _reward_arm_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel[:, self.arm_indices]), dim=-1)

    def _reward_arm_dof_acc(self):
        acc = (self.last_dof_vel[:, self.arm_indices] - self.dof_vel[:, self.arm_indices]) / self.dt
        return torch.sum(torch.square(acc), dim=-1)

    def _reward_arm_torques(self):
        return torch.sum(torch.square(self.torques[:, self.arm_indices]), dim=-1)

    def _reward_arm_power(self):
        power = self.torques[:, self.arm_indices] * self.dof_vel[:, self.arm_indices]
        return torch.sum(torch.abs(power), dim=-1)

    def _reward_shoulder_neutral_low_speed(self):
        q = self.dof_pos[:, self.arm_indices] - self.default_dof_pos[:, self.arm_indices]
        cmd_speed = torch.linalg.norm(self.commands[:, :2], dim=1)
        cmd_yaw = torch.abs(self.commands[:, 2])
        cmd_mag = cmd_speed + 0.5 * cmd_yaw
        low_speed_gate = torch.clamp(1.0 - cmd_mag / 0.60, min=0.0, max=1.0)
        return low_speed_gate * torch.sum(torch.square(q), dim=1)

    def _reward_shoulder_roll(self):
        q = self.dof_pos[:, self.arm_indices] - self.default_dof_pos[:, self.arm_indices]
        left_roll = q[:, 1]
        right_roll = q[:, 3]
        return left_roll.square() + right_roll.square()

    def _reward_shoulder_pitch_soft_limit(self):
        q = self.dof_pos[:, self.arm_indices] - self.default_dof_pos[:, self.arm_indices]
        left_pitch = q[:, 0]
        right_pitch = q[:, 2]
        left_excess = torch.relu(torch.abs(left_pitch) - 0.42)
        right_excess = torch.relu(torch.abs(right_pitch) - 0.42)
        return left_excess.square() + right_excess.square()

    def _reward_shoulder_pair_symmetry(self):
        q = self.dof_pos[:, self.arm_indices] - self.default_dof_pos[:, self.arm_indices]
        left_pitch = q[:, 0]
        left_roll = q[:, 1]
        right_pitch = q[:, 2]
        right_roll = q[:, 3]
        pitch_pair = left_pitch + right_pitch
        roll_pair = left_roll + right_roll
        return pitch_pair.square() + 2.0 * roll_pair.square()

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.feet_pos) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float()
        )

    def _reward_feet_vel_z(self):
        return torch.sum(torch.square((self.last_feet_pos - self.feet_pos) / self.dt)[:, :, 2], dim=-1)

    def _reward_feet_roll(self):
        return torch.sum(torch.square(self.feet_roll), dim=-1)

    def _reward_feet_yaw_diff(self):
        return torch.square((self.feet_yaw[:, 1] - self.feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_yaw_mean(self):
        feet_yaw_mean = self.feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(self.feet_yaw[:, 1] - self.feet_yaw[:, 0]) > torch.pi)
        return torch.square((get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_distance(self):
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_distance = torch.abs(
            torch.cos(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
            - torch.sin(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
        )
        return torch.clip(self.cfg["rewards"]["feet_distance_ref"] - feet_distance, min=0.0, max=0.1)

    def _reward_feet_swing(self):
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        return (left_swing & ~self.feet_contact[:, 0]).float() + (right_swing & ~self.feet_contact[:, 1]).float()
