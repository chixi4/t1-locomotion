import argparse
import glob
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from envs import *
from utils.buffer import ExperienceBuffer
from utils.model import ActorCritic
from utils.recorder import Recorder
from utils.utils import discount_values, surrogate_loss


class FullBodyResidualRunner:
    def __init__(self):
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()
        task_class = eval(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)
        self.device = self.cfg["basic"]["rl_device"]
        self.action_dim = self.env.num_actions
        self.leg_action_dim = len(self.env.leg_indices)
        self.arm_action_dim = len(self.env.arm_indices)

        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        self.residual_effect_scale = self._residual_effect_scale()
        self.residual_penalty = float(self.cfg["algorithm"].get("residual_penalty", 0.0))
        self.model = ActorCritic(
            self.action_dim,
            self.env.num_obs,
            self.env.num_privileged_obs,
            logstd_init=float(self.cfg["algorithm"].get("logstd_init", -3.2)),
            actor_mean_scale=self._residual_actor_mean_scale(),
            logstd_min=self.cfg["algorithm"].get("logstd_min"),
            logstd_max=self.cfg["algorithm"].get("logstd_max"),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.leg_model = ActorCritic(self.leg_action_dim, 47, self.env.num_privileged_obs).to(self.device)
        self.upper_model = ActorCritic(
            self.arm_action_dim,
            self.env.num_obs,
            self.env.num_privileged_obs,
            actor_mean_scale=self._upper_actor_mean_scale(),
            logstd_min=self.cfg["algorithm"].get("upper_logstd_min"),
            logstd_max=self.cfg["algorithm"].get("upper_logstd_max"),
        ).to(self.device)
        self._load_leg_model()
        self._load_upper_model()
        self.leg_model.eval()
        self.upper_model.eval()
        for param in self.leg_model.parameters():
            param.requires_grad = False
        for param in self.upper_model.parameters():
            param.requires_grad = False

        self.iteration_offset = 0
        if self.cfg["basic"].get("checkpoint"):
            self._load_residual_model()
        else:
            self._zero_residual_head()
            self._load_initial_curriculum()

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.action_dim,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)
        self.max_grad_norm = float(self.cfg["algorithm"].get("max_grad_norm", 0.5))
        self.log_ratio_clip = self.cfg["algorithm"].get("log_ratio_clip", 20.0)
        self.nonfinite_update_count = 0
        self.nonfinite_update_limit = int(self.cfg["algorithm"].get("nonfinite_update_limit", 16))

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str)
        parser.add_argument("--checkpoint", type=str)
        parser.add_argument("--leg_checkpoint", type=str)
        parser.add_argument("--upper_checkpoint", type=str)
        parser.add_argument("--num_envs", type=int)
        parser.add_argument("--headless", type=bool)
        parser.add_argument("--sim_device", type=str)
        parser.add_argument("--rl_device", type=str)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--max_iterations", type=int)
        self.args = parser.parse_args()

    def _update_cfg_from_args(self):
        import yaml

        cfg_file = os.path.join("envs", f"{self.args.task}.yaml")
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        for arg in vars(self.args):
            value = getattr(self.args, arg)
            if value is None:
                continue
            if arg == "num_envs":
                self.cfg["env"][arg] = value
            else:
                self.cfg["basic"][arg] = value
        self.cfg["viewer"]["record_video"] = False

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print(f"Setting seed: {self.cfg['basic']['seed']}")
        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _residual_actor_mean_scale(self):
        by_dof = self.cfg["algorithm"].get("residual_actor_mean_scale_by_dof")
        if not by_dof:
            return self.cfg["algorithm"].get("actor_mean_scale")
        scales = []
        for name in self.env.dof_names:
            if name not in by_dof:
                raise KeyError(f"residual_actor_mean_scale_by_dof missing {name}")
            scales.append(float(by_dof[name]))
        print(f"Using residual per-DOF mean scale: {dict(zip(self.env.dof_names, scales))}")
        return scales

    def _residual_effect_scale(self):
        by_dof = self.cfg["algorithm"].get("residual_effect_scale_by_dof")
        if not by_dof:
            return torch.ones(1, self.env.num_actions, dtype=torch.float, device=self.device)
        values = []
        for name in self.env.dof_names:
            if name not in by_dof:
                raise KeyError(f"residual_effect_scale_by_dof missing {name}")
            values.append(float(by_dof[name]))
        print(f"Using residual effect scale: {dict(zip(self.env.dof_names, values))}")
        return torch.tensor(values, dtype=torch.float, device=self.device).view(1, -1)

    def _upper_actor_mean_scale(self):
        checkpoint = self.cfg["basic"]["upper_checkpoint"]
        data = torch.load(checkpoint, map_location=self.device, weights_only=True)
        return data.get("actor_mean_scale")

    def _load_leg_model(self):
        checkpoint = self.cfg["basic"]["leg_checkpoint"]
        print(f"Loading baseline leg model from {checkpoint}")
        model_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.leg_model.load_state_dict(model_dict["model"], strict=True)

    def _load_upper_model(self):
        checkpoint = self.cfg["basic"]["upper_checkpoint"]
        print(f"Loading baseline upper model from {checkpoint}")
        model_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.upper_model.load_state_dict(model_dict["model"], strict=False)

    def _zero_residual_head(self):
        with torch.no_grad():
            self.model.actor[-1].weight.zero_()
            self.model.actor[-1].bias.zero_()
        print("Initialized residual actor head to zero.")

    def _load_residual_model(self):
        checkpoint = self.cfg["basic"]["checkpoint"]
        if checkpoint == "-1" or checkpoint == -1:
            checkpoint = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print(f"Loading residual model from {checkpoint}")
        model_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_dict["model"], strict=False)
        try:
            self._load_curriculum(model_dict["curriculum"])
        except Exception as exc:
            print(f"Failed to load residual checkpoint curriculum: {exc}")
        try:
            if self.cfg["basic"].get("reset_optimizer_on_resume", False):
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate
                print(f"Reset optimizer on resume; using configured learning rate {self.learning_rate}")
            else:
                self.optimizer.load_state_dict(model_dict["optimizer"])
                self.learning_rate = self.optimizer.param_groups[0]["lr"]
                print(f"Loaded optimizer learning rate {self.learning_rate}")
        except Exception as exc:
            print(f"Failed to load optimizer: {exc}")
        self.iteration_offset = int(model_dict.get("global_iteration", model_dict.get("iteration_offset", 0)))

    def _load_initial_curriculum(self):
        checkpoint = self.cfg["basic"].get("curriculum_checkpoint")
        if not checkpoint:
            self._seed_initial_curriculum()
            return
        print(f"Loading initial curriculum from {checkpoint}")
        model_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self._load_curriculum(model_dict["curriculum"])
        self._seed_initial_curriculum()

    def _seed_initial_curriculum(self):
        radius = self.cfg["basic"].get("curriculum_seed_radius")
        if radius is None or not hasattr(self.env, "curriculum_prob") or self.env.curriculum_prob.dim() != 3:
            return
        lin_levels = int(self.env.cfg["commands"]["lin_vel_levels"])
        yaw_levels = int(self.env.cfg["commands"]["ang_vel_levels"])
        yaw_radius = int(self.cfg["basic"].get("curriculum_seed_yaw_radius", yaw_levels))
        radius = float(radius)
        lx = torch.arange(-lin_levels, lin_levels + 1, dtype=torch.float, device=self.device).view(-1, 1, 1)
        ly = torch.arange(-lin_levels, lin_levels + 1, dtype=torch.float, device=self.device).view(1, -1, 1)
        yaw = torch.arange(-yaw_levels, yaw_levels + 1, dtype=torch.float, device=self.device).view(1, 1, -1)
        seeded = (lx.square() + ly.square()).sqrt() <= radius
        seeded = seeded & (yaw.abs() <= float(yaw_radius))
        if hasattr(self.env, "curriculum_mask"):
            seeded = seeded & self.env.curriculum_mask
        self.env.curriculum_prob.zero_()
        self.env.curriculum_prob[seeded] = 1.0
        center = (lin_levels, lin_levels, yaw_levels)
        self.env.curriculum_prob[center] = 1.0
        print(f"Seeded curriculum radius={radius} yaw_radius={yaw_radius}; unlocked {int(torch.count_nonzero(self.env.curriculum_prob > 0.5))}.")

    def _load_curriculum(self, curriculum):
        target = self.env.curriculum_prob
        curriculum = curriculum.to(device=target.device, dtype=target.dtype)
        if tuple(curriculum.shape) == tuple(target.shape):
            self.env.curriculum_prob = curriculum
            print(f"Loaded curriculum shape {tuple(curriculum.shape)}")
            return
        expanded = torch.zeros_like(target)
        src_slices = []
        dst_slices = []
        for src_size, dst_size in zip(curriculum.shape, target.shape):
            copy_size = min(src_size, dst_size)
            src_start = (src_size - copy_size) // 2
            dst_start = (dst_size - copy_size) // 2
            src_slices.append(slice(src_start, src_start + copy_size))
            dst_slices.append(slice(dst_start, dst_start + copy_size))
        expanded[tuple(dst_slices)] = curriculum[tuple(src_slices)]
        if hasattr(self.env, "curriculum_mask") and tuple(self.env.curriculum_mask.shape) == tuple(expanded.shape):
            expanded *= self.env.curriculum_mask.to(device=expanded.device, dtype=expanded.dtype)
        center = tuple(size // 2 for size in expanded.shape)
        if torch.count_nonzero(expanded > 0.5) == 0:
            expanded[center] = 1.0
        self.env.curriculum_prob = expanded
        print(f"Loaded curriculum shape {tuple(curriculum.shape)} into {tuple(target.shape)}; unlocked {int(torch.count_nonzero(expanded > 0.5))}.")

    def _build_old_leg_obs(self, last_leg_action):
        commands_scale = torch.tensor(
            [
                self.env.cfg["normalization"]["lin_vel"],
                self.env.cfg["normalization"]["lin_vel"],
                self.env.cfg["normalization"]["ang_vel"],
            ],
            device=self.device,
        )
        gait_active = (self.env.gait_frequency > 1.0e-8).float()
        return torch.cat(
            (
                self.env.projected_gravity * self.env.cfg["normalization"]["gravity"],
                self.env.base_ang_vel * self.env.cfg["normalization"]["ang_vel"],
                self.env.commands[:, :3] * commands_scale,
                (torch.cos(2 * torch.pi * self.env.gait_process) * gait_active).unsqueeze(-1),
                (torch.sin(2 * torch.pi * self.env.gait_process) * gait_active).unsqueeze(-1),
                (self.env.dof_pos[:, self.env.leg_indices] - self.env.default_dof_pos[:, self.env.leg_indices])
                * self.env.cfg["normalization"]["dof_pos"],
                self.env.dof_vel[:, self.env.leg_indices] * self.env.cfg["normalization"]["dof_vel"],
                last_leg_action,
            ),
            dim=-1,
        )

    def _baseline_action(self, obs, last_leg_action):
        old_leg_obs = self._build_old_leg_obs(last_leg_action)
        leg_action = torch.clamp(self.leg_model.act(old_leg_obs).loc, -1.0, 1.0)
        upper_action = torch.clamp(self.upper_model.act(obs).loc, -1.0, 1.0)
        base_action = torch.zeros(self.env.num_envs, self.env.num_actions, dtype=torch.float, device=self.device)
        base_action[:, self.env.leg_indices] = leg_action
        base_action[:, self.env.arm_indices] = upper_action
        return base_action

    def train(self):
        self.recorder = Recorder(self.cfg)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        last_leg_action = torch.zeros(self.env.num_envs, self.leg_action_dim, dtype=torch.float, device=self.device)

        max_iterations = self.cfg["basic"]["max_iterations"]
        total_iterations = self.iteration_offset + max_iterations
        train_start_time = time.time()
        for it in range(max_iterations):
            global_it = self.iteration_offset + it
            save_it = global_it + 1
            for n in range(self.cfg["runner"]["horizon_length"]):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                with torch.no_grad():
                    base_action = self._baseline_action(obs, last_leg_action)
                    residual_dist = self.model.act(obs)
                    residual_raw_action = torch.clamp(residual_dist.sample(), -1.0, 1.0)
                    residual_action = residual_raw_action * self.residual_effect_scale
                    full_action = torch.clamp(base_action + residual_action, -1.0, 1.0)

                obs, rew, done, infos = self.env.step(full_action)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                residual_penalty = self.residual_penalty * torch.sum(torch.square(residual_action), dim=-1)
                if self.residual_penalty > 0.0:
                    rew = rew - residual_penalty
                last_leg_action[:] = full_action[:, self.env.leg_indices]
                last_leg_action[done] = 0.0

                self.buffer.update_data("actions", n, residual_raw_action)
                self.buffer.update_data("rewards", n, rew)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                ep_info = {"reward": rew}
                ep_info.update(infos["rew_terms"])
                if self.residual_penalty > 0.0:
                    ep_info["residual_penalty"] = -residual_penalty
                self.recorder.record_episode_statistics(done, ep_info, global_it, n == (self.cfg["runner"]["horizon_length"] - 1))

            with torch.no_grad():
                old_dist = self.model.act(self.buffer["obses"])
                old_actions_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)

            mean_value_loss = 0.0
            mean_actor_loss = 0.0
            mean_bound_loss = 0.0
            mean_entropy = 0.0
            kl_mean = torch.tensor(0.0, device=self.device)
            successful_updates = 0
            for _ in range(self.cfg["runner"]["mini_epochs"]):
                values = self.model.est_value(self.buffer["obses"], self.buffer["privileged_obses"])
                last_values = self.model.est_value(obs, privileged_obs)
                with torch.no_grad():
                    self.buffer["rewards"][self.buffer["time_outs"]] = values[self.buffer["time_outs"]]
                    advantages = discount_values(
                        self.buffer["rewards"],
                        self.buffer["dones"] | self.buffer["time_outs"],
                        values,
                        last_values,
                        self.cfg["algorithm"]["gamma"],
                        self.cfg["algorithm"]["lam"],
                    )
                    returns = values + advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                value_loss = F.mse_loss(values, returns)
                dist = self.model.act(self.buffer["obses"])
                actions_log_prob = dist.log_prob(self.buffer["actions"]).sum(dim=-1)
                actor_loss = surrogate_loss(
                    old_actions_log_prob,
                    actions_log_prob,
                    advantages,
                    log_ratio_clip=self.log_ratio_clip,
                )
                bound_loss = torch.clip(dist.loc - 1.0, min=0.0).square().mean() + torch.clip(dist.loc + 1.0, max=0.0).square().mean()
                entropy = dist.entropy().sum(dim=-1)
                loss = (
                    value_loss
                    + actor_loss
                    + self.cfg["algorithm"]["bound_coef"] * bound_loss
                    + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                )
                if not torch.isfinite(loss):
                    self.nonfinite_update_count += 1
                    print(f"Non-finite loss at {save_it}; skipping update", flush=True)
                    if self.nonfinite_update_count >= self.nonfinite_update_limit:
                        raise FloatingPointError("Too many non-finite PPO updates")
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    kl = torch.sum(
                        torch.log(dist.scale / old_dist.scale)
                        + 0.5 * (torch.square(old_dist.scale) + torch.square(dist.loc - old_dist.loc)) / torch.square(dist.scale)
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(torch.nan_to_num(kl, nan=0.0, posinf=1.0e6, neginf=0.0))
                    if self.cfg["algorithm"].get("adaptive_lr", True):
                        if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                            self.learning_rate = max(float(self.cfg["algorithm"].get("learning_rate_min", 1e-5)), self.learning_rate / 1.5)
                        elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                            self.learning_rate = min(float(self.cfg["algorithm"].get("learning_rate_max", 1e-4)), self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate
                mean_value_loss += value_loss.item()
                mean_actor_loss += actor_loss.item()
                mean_bound_loss += bound_loss.item()
                mean_entropy += entropy.mean()
                successful_updates += 1

            denom = max(1, successful_updates)
            active_count = torch.clamp(getattr(self.env, "tilt_count", torch.ones(self.env.num_envs, device=self.device)), min=1.0)
            tilt_rms = torch.sqrt(getattr(self.env, "tilt_sq_sum", torch.zeros(self.env.num_envs, device=self.device)) / active_count)
            arm_sat_frac = getattr(self.env, "arm_saturation_sum", torch.zeros(self.env.num_envs, device=self.device)) / active_count
            residual_loc = self.model.act(self.buffer["obses"]).loc * self.residual_effect_scale
            residual_mean = residual_loc.abs().mean()
            residual_max = residual_loc.abs().max()
            self.recorder.record_statistics(
                {
                    "value_loss": mean_value_loss / denom,
                    "actor_loss": mean_actor_loss / denom,
                    "bound_loss": mean_bound_loss / denom,
                    "entropy": mean_entropy / denom,
                    "kl_mean": kl_mean,
                    "lr": self.learning_rate,
                    "ppo/successful_updates": successful_updates,
                    "ppo/nonfinite_updates": self.nonfinite_update_count,
                    "residual/mean_abs": residual_mean,
                    "residual/max_abs": residual_max,
                    "curriculum/active_tilt_rms_mean": tilt_rms.mean(),
                    "curriculum/active_arm_saturation_frac_mean": arm_sat_frac.mean(),
                    "curriculum/unlocked_cells": torch.count_nonzero((self.env.curriculum_prob > 0.5) & self.env.curriculum_mask),
                    "curriculum/allowed_cells": torch.count_nonzero(self.env.curriculum_mask),
                    "curriculum/mean_lin_vel_level": self.env.mean_lin_vel_level,
                    "curriculum/mean_ang_vel_level": self.env.mean_ang_vel_level,
                    "curriculum/max_lin_vel_level": self.env.max_lin_vel_level,
                    "curriculum/max_ang_vel_level": self.env.max_ang_vel_level,
                },
                global_it,
            )
            if save_it % self.cfg["runner"]["save_interval"] == 0 or it == (max_iterations - 1):
                self.recorder.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "curriculum": self.env.curriculum_prob,
                        "global_iteration": save_it,
                        "is_fullbody_residual": True,
                        "leg_checkpoint": self.cfg["basic"]["leg_checkpoint"],
                        "upper_checkpoint": self.cfg["basic"]["upper_checkpoint"],
                        "dof_names": self.env.dof_names,
                        "arm_dof_names": [self.env.dof_names[i] for i in self.env.arm_indices.tolist()],
                        "leg_dof_names": [self.env.dof_names[i] for i in self.env.leg_indices.tolist()],
                    },
                    save_it,
                )
            elapsed_s = time.time() - train_start_time
            avg_s = elapsed_s / max(1, it + 1)
            eta_s = avg_s * max(0, max_iterations - it - 1)
            print(
                "epoch: {}/{} (local {}/{}) elapsed={:.1f}m eta={:.1f}m".format(
                    save_it,
                    total_iterations,
                    it + 1,
                    max_iterations,
                    elapsed_s / 60.0,
                    eta_s / 60.0,
                ),
                flush=True,
            )
