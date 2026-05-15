import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from envs import *
from utils.buffer import ExperienceBuffer
from utils.model import ActorCritic
from utils.recorder import Recorder
from utils.utils import discount_values, surrogate_loss


class FrozenLegShoulderRunner:

    def __init__(self):
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()
        task_class = eval(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)

        self.device = self.cfg["basic"]["rl_device"]
        self.arm_action_dim = len(self.env.arm_indices)
        self.leg_action_dim = len(self.env.leg_indices)
        if self.arm_action_dim != 4 or self.leg_action_dim != 12:
            raise ValueError(f"Expected 4 arm and 12 leg DOFs, got {self.arm_action_dim} and {self.leg_action_dim}")

        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        self.model = ActorCritic(
            self.arm_action_dim,
            self.env.num_obs,
            self.env.num_privileged_obs,
            logstd_init=float(self.cfg["algorithm"].get("logstd_init", -2.5)),
            actor_mean_scale=self.cfg["algorithm"].get("actor_mean_scale"),
            logstd_min=self.cfg["algorithm"].get("logstd_min"),
            logstd_max=self.cfg["algorithm"].get("logstd_max"),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.leg_model = ActorCritic(self.leg_action_dim, 47, self.env.num_privileged_obs).to(self.device)
        self._load_leg_model()
        self.leg_model.eval()
        for param in self.leg_model.parameters():
            param.requires_grad = False

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.arm_action_dim,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)
        self.iteration_offset = int(self.cfg["basic"].get("iteration_offset", 0))
        self.max_grad_norm = float(self.cfg["algorithm"].get("max_grad_norm", 1.0))
        self.log_ratio_clip = self.cfg["algorithm"].get("log_ratio_clip", 20.0)
        self.nonfinite_update_count = 0
        self.nonfinite_update_limit = int(self.cfg["algorithm"].get("nonfinite_update_limit", 16))

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Optional shoulder model checkpoint to load.")
        parser.add_argument("--leg_checkpoint", type=str, help="Frozen 12D leg model checkpoint.")
        parser.add_argument("--num_envs", type=int, help="Number of environments to create.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window.")
        parser.add_argument("--sim_device", type=str, help="Physics simulation device.")
        parser.add_argument("--rl_device", type=str, help="RL device.")
        parser.add_argument("--seed", type=int, help="Random seed.")
        parser.add_argument("--max_iterations", type=int, help="Maximum training iterations.")
        self.args = parser.parse_args()

    def _update_cfg_from_args(self):
        import yaml

        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
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
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))
        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _load_leg_model(self):
        checkpoint = self.cfg["basic"]["leg_checkpoint"]
        if checkpoint == "-1" or checkpoint == -1:
            checkpoint = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print("Loading frozen leg model from {}".format(checkpoint))
        model_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.leg_model.load_state_dict(model_dict["model"], strict=True)

    def _load_arm_model(self):
        checkpoint = self.cfg["basic"].get("checkpoint")
        if not checkpoint:
            return
        print("Loading shoulder model from {}".format(checkpoint))
        model_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_dict["model"], strict=False)
        if self.cfg["basic"].get("reset_optimizer_on_resume", False):
            print(f"Resetting optimizer on resume; using configured learning rate {self.learning_rate}")
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
            return
        try:
            self.optimizer.load_state_dict(model_dict["optimizer"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
            print(f"Loaded optimizer learning rate {self.learning_rate}")
        except Exception as e:
            print(f"Failed to load optimizer: {e}")

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

    def train(self):
        self._load_arm_model()
        self.recorder = Recorder(self.cfg)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        last_leg_action = torch.zeros(self.env.num_envs, self.leg_action_dim, dtype=torch.float, device=self.device)

        max_iterations = self.cfg["basic"]["max_iterations"]
        total_iterations = self.iteration_offset + max_iterations
        for it in range(max_iterations):
            global_it = self.iteration_offset + it
            save_it = global_it + 1
            for n in range(self.cfg["runner"]["horizon_length"]):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                with torch.no_grad():
                    old_leg_obs = self._build_old_leg_obs(last_leg_action)
                    leg_action = torch.clamp(self.leg_model.act(old_leg_obs).loc, -1.0, 1.0)
                    arm_dist = self.model.act(obs)
                    arm_action = torch.clamp(arm_dist.sample(), -1.0, 1.0)
                    full_action = torch.zeros(self.env.num_envs, self.env.num_actions, dtype=torch.float, device=self.device)
                    full_action[:, self.env.arm_indices] = arm_action
                    full_action[:, self.env.leg_indices] = leg_action

                obs, rew, done, infos = self.env.step(full_action)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                last_leg_action[:] = leg_action
                last_leg_action[done] = 0.0

                self.buffer.update_data("actions", n, arm_action)
                self.buffer.update_data("rewards", n, rew)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                ep_info = {"reward": rew}
                ep_info.update(infos["rew_terms"])
                self.recorder.record_episode_statistics(done, ep_info, global_it, n == (self.cfg["runner"]["horizon_length"] - 1))

            with torch.no_grad():
                old_dist = self.model.act(self.buffer["obses"])
                old_actions_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)

            mean_value_loss = 0
            mean_actor_loss = 0
            mean_bound_loss = 0
            mean_entropy = 0
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
                    lr_min = float(self.cfg["algorithm"].get("learning_rate_min", 1e-5))
                    self.learning_rate = max(lr_min, self.learning_rate / 2.0)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                    print(
                        "Non-finite PPO loss at global iteration "
                        f"{save_it}: value={value_loss.item()} actor={actor_loss.item()} "
                        f"bound={bound_loss.item()} entropy={entropy.mean().item()} "
                        f"-> skip update, lr={self.learning_rate}",
                        flush=True,
                    )
                    if self.nonfinite_update_count >= self.nonfinite_update_limit:
                        raise FloatingPointError(
                            f"Too many non-finite PPO updates ({self.nonfinite_update_count}) at global iteration {save_it}"
                        )
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
                    kl = torch.nan_to_num(kl, nan=0.0, posinf=1.0e6, neginf=0.0)
                    kl_mean = torch.mean(kl)
                    if self.cfg["algorithm"].get("adaptive_lr", True):
                        if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                            lr_min = float(self.cfg["algorithm"].get("learning_rate_min", 1e-5))
                            self.learning_rate = max(lr_min, self.learning_rate / 1.5)
                        elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                            lr_max = float(self.cfg["algorithm"].get("learning_rate_max", 1e-2))
                            self.learning_rate = min(lr_max, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                mean_value_loss += value_loss.item()
                mean_actor_loss += actor_loss.item()
                mean_bound_loss += bound_loss.item()
                mean_entropy += entropy.mean()
                successful_updates += 1

            denom = max(1, successful_updates)
            mean_value_loss /= denom
            mean_actor_loss /= denom
            mean_bound_loss /= denom
            mean_entropy /= denom
            unlocked = torch.count_nonzero((self.env.curriculum_prob > 0.5) & self.env.curriculum_mask)
            active_count = torch.clamp(getattr(self.env, "tilt_count", torch.ones(self.env.num_envs, device=self.device)), min=1.0)
            tilt_rms = torch.sqrt(getattr(self.env, "tilt_sq_sum", torch.zeros(self.env.num_envs, device=self.device)) / active_count)
            arm_sat_frac = getattr(self.env, "arm_saturation_sum", torch.zeros(self.env.num_envs, device=self.device)) / active_count
            with torch.no_grad():
                policy_mean_abs_max = self.model.act(self.buffer["obses"]).loc.abs().max()
            self.recorder.record_statistics(
                {
                    "value_loss": mean_value_loss,
                    "actor_loss": mean_actor_loss,
                    "bound_loss": mean_bound_loss,
                    "entropy": mean_entropy,
                    "kl_mean": kl_mean,
                    "lr": self.learning_rate,
                    "ppo/successful_updates": successful_updates,
                    "ppo/nonfinite_updates": self.nonfinite_update_count,
                    "policy/mean_abs_max": policy_mean_abs_max,
                    "curriculum/active_tilt_rms_mean": tilt_rms.mean(),
                    "curriculum/active_arm_saturation_frac_mean": arm_sat_frac.mean(),
                    "curriculum/unlocked_cells": unlocked,
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
                        "iteration_offset": self.iteration_offset,
                        "global_iteration": save_it,
                        "actor_mean_scale": self.cfg["algorithm"].get("actor_mean_scale"),
                        "logstd_min": self.cfg["algorithm"].get("logstd_min"),
                        "logstd_max": self.cfg["algorithm"].get("logstd_max"),
                        "leg_checkpoint": self.cfg["basic"]["leg_checkpoint"],
                        "arm_dof_names": [self.env.dof_names[i] for i in self.env.arm_indices.tolist()],
                        "leg_dof_names": [self.env.dof_names[i] for i in self.env.leg_indices.tolist()],
                    },
                    save_it,
                )
            print("epoch: {}/{} (local {}/{})".format(save_it, total_iterations, it + 1, max_iterations))
