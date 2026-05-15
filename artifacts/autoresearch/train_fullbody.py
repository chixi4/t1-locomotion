import argparse
import glob
import os
import random
import sys
import time

import isaacgym  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.getcwd())

from envs import *  # noqa: F401,F403
from utils.buffer import ExperienceBuffer
from utils.model import ActorCritic
from utils.recorder import Recorder
from utils.utils import discount_values


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def clipped_surrogate_loss(old_actions_log_prob, actions_log_prob, advantages, e_clip=0.2, log_ratio_clip=None):
    log_ratio = actions_log_prob - old_actions_log_prob
    if log_ratio_clip is not None:
        log_ratio = torch.clamp(log_ratio, -float(log_ratio_clip), float(log_ratio_clip))
    ratio = torch.exp(log_ratio)
    surrogate = -advantages * ratio
    surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - e_clip, 1.0 + e_clip)
    return torch.max(surrogate, surrogate_clipped).mean()


class FullBodyRunner:
    def __init__(self):
        self._get_args()
        self._load_cfg()
        self._set_seed()
        task_class = eval(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)
        self.device = self.cfg["basic"]["rl_device"]
        self.learning_rate = float(self.cfg["algorithm"]["learning_rate"])
        self.max_grad_norm = float(self.cfg["algorithm"].get("max_grad_norm", 1.0))
        self.log_ratio_clip = self.cfg["algorithm"].get("log_ratio_clip", 20.0)
        self.nonfinite_update_count = 0
        self.nonfinite_update_limit = int(self.cfg["algorithm"].get("nonfinite_update_limit", 16))
        self.model = ActorCritic(
            self.env.num_actions,
            self.env.num_obs,
            self.env.num_privileged_obs,
            logstd_init=float(self.cfg["algorithm"].get("logstd_init", -2.0)),
            actor_mean_scale=self.cfg["algorithm"].get("actor_mean_scale"),
            logstd_min=self.cfg["algorithm"].get("logstd_min"),
            logstd_max=self.cfg["algorithm"].get("logstd_max"),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._load_checkpoint()
        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True)
        parser.add_argument("--checkpoint")
        parser.add_argument("--num_envs", type=int)
        parser.add_argument("--max_iterations", type=int)
        parser.add_argument("--run_name")
        parser.add_argument("--headless", type=str_to_bool)
        parser.add_argument("--sim_device")
        parser.add_argument("--rl_device")
        parser.add_argument("--seed", type=int)
        parser.add_argument("--use_wandb", type=str_to_bool)
        parser.add_argument("--load_optimizer", type=str_to_bool)
        parser.add_argument("--reset_logstd", type=float)
        self.args = parser.parse_args()

    def _load_cfg(self):
        cfg_file = os.path.join("envs", f"{self.args.task}.yaml")
        with open(cfg_file, "r", encoding="utf-8") as file:
            self.cfg = yaml.load(file.read(), Loader=yaml.FullLoader)
        for arg, value in vars(self.args).items():
            if value is None:
                continue
            if arg == "num_envs":
                self.cfg["env"]["num_envs"] = value
            elif arg == "use_wandb":
                self.cfg["runner"]["use_wandb"] = value
            elif arg in {"checkpoint", "max_iterations", "run_name", "headless", "sim_device", "rl_device", "seed", "load_optimizer", "reset_logstd"}:
                self.cfg["basic"][arg] = value
        self.cfg["basic"]["task"] = self.args.task
        self.cfg["viewer"]["record_video"] = False

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        seed = int(self.cfg["basic"]["seed"])
        print(f"Setting seed: {seed}", flush=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _load_checkpoint(self):
        checkpoint = self.cfg["basic"].get("checkpoint")
        if not checkpoint:
            return
        if checkpoint == "-1" or checkpoint == -1:
            checkpoint = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print(f"Loading model from {checkpoint}", flush=True)
        model_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if self.cfg["basic"].get("reset_logstd") is not None and "logstd" in model_dict["model"]:
            model_dict["model"]["logstd"][:] = float(self.cfg["basic"]["reset_logstd"])
        self.model.load_state_dict(model_dict["model"], strict=False)
        try:
            self.env.curriculum_prob = model_dict["curriculum"].to(self.device)
        except Exception as exc:
            print(f"Failed to load curriculum: {exc}", flush=True)
        if self.cfg["basic"].get("load_optimizer", True):
            try:
                self.optimizer.load_state_dict(model_dict["optimizer"])
                self.learning_rate = self.optimizer.param_groups[0]["lr"]
            except Exception as exc:
                print(f"Failed to load optimizer: {exc}", flush=True)

    def train(self):
        self.recorder = Recorder(self.cfg)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        max_iterations = int(self.cfg["basic"]["max_iterations"])
        train_started = time.time()
        for it in range(max_iterations):
            for n in range(self.cfg["runner"]["horizon_length"]):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                with torch.no_grad():
                    dist = self.model.act(obs)
                    action = torch.clamp(dist.sample(), -1.0, 1.0)
                obs, rew, done, infos = self.env.step(action)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                self.buffer.update_data("actions", n, action)
                self.buffer.update_data("rewards", n, rew)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                ep_info = {"reward": rew}
                ep_info.update(infos["rew_terms"])
                self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))

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
                actor_loss = clipped_surrogate_loss(
                    old_actions_log_prob,
                    actions_log_prob,
                    advantages,
                    log_ratio_clip=self.log_ratio_clip,
                )
                bound_loss = torch.clip(dist.loc - 1.0, min=0.0).square().mean() + torch.clip(dist.loc + 1.0, max=0.0).square().mean()
                entropy = dist.entropy().sum(dim=-1)
                loss = value_loss + actor_loss + self.cfg["algorithm"]["bound_coef"] * bound_loss + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                if not torch.isfinite(loss):
                    self.nonfinite_update_count += 1
                    self._lower_lr()
                    print(
                        f"Non-finite update at iteration {it + 1}: value={value_loss.item()} actor={actor_loss.item()} "
                        f"bound={bound_loss.item()} entropy={entropy.mean().item()} lr={self.learning_rate}",
                        flush=True,
                    )
                    if self.nonfinite_update_count >= self.nonfinite_update_limit:
                        raise FloatingPointError(f"Too many non-finite PPO updates: {self.nonfinite_update_count}")
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
                    self._adapt_lr(kl_mean)
                mean_value_loss += value_loss.item()
                mean_actor_loss += actor_loss.item()
                mean_bound_loss += bound_loss.item()
                mean_entropy += entropy.mean().item()
                successful_updates += 1

            denom = max(1, successful_updates)
            stats = {
                "value_loss": mean_value_loss / denom,
                "actor_loss": mean_actor_loss / denom,
                "bound_loss": mean_bound_loss / denom,
                "entropy": mean_entropy / denom,
                "kl_mean": kl_mean,
                "lr": self.learning_rate,
                "ppo/successful_updates": successful_updates,
                "ppo/nonfinite_updates": self.nonfinite_update_count,
                "curriculum/mean_lin_vel_level": self.env.mean_lin_vel_level,
                "curriculum/mean_ang_vel_level": self.env.mean_ang_vel_level,
                "curriculum/max_lin_vel_level": self.env.max_lin_vel_level,
                "curriculum/max_ang_vel_level": self.env.max_ang_vel_level,
            }
            if hasattr(self.env, "curriculum_mask"):
                stats["curriculum/unlocked_cells"] = torch.count_nonzero((self.env.curriculum_prob > 0.5) & self.env.curriculum_mask)
                stats["curriculum/allowed_cells"] = torch.count_nonzero(self.env.curriculum_mask)
            if hasattr(self.env, "tilt_count"):
                active_count = torch.clamp(self.env.tilt_count, min=1.0)
                stats["curriculum/active_tilt_rms_mean"] = torch.sqrt(self.env.tilt_sq_sum / active_count).mean()
                stats["curriculum/active_arm_saturation_frac_mean"] = (self.env.arm_saturation_sum / active_count).mean()
            self.recorder.record_statistics(stats, it)
            save_it = it + 1
            if save_it % self.cfg["runner"]["save_interval"] == 0 or (self.cfg["runner"].get("save_final", True) and save_it == max_iterations):
                self._save(save_it)
            elapsed = time.time() - train_started
            print(f"epoch: {save_it}/{max_iterations} elapsed_s={elapsed:.1f}", flush=True)

    def _adapt_lr(self, kl_mean):
        if not self.cfg["algorithm"].get("adaptive_lr", True):
            return
        desired_kl = float(self.cfg["algorithm"]["desired_kl"])
        if kl_mean > desired_kl * 2:
            self._lower_lr()
        elif kl_mean < desired_kl / 2:
            lr_max = float(self.cfg["algorithm"].get("learning_rate_max", 1.0e-2))
            self.learning_rate = min(lr_max, self.learning_rate * 1.5)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _lower_lr(self):
        lr_min = float(self.cfg["algorithm"].get("learning_rate_min", 1.0e-5))
        self.learning_rate = max(lr_min, self.learning_rate / 1.5)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def _save(self, iteration):
        self.recorder.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "curriculum": self.env.curriculum_prob,
                "global_iteration": iteration,
                "dof_names": self.env.dof_names,
                "arm_dof_names": [self.env.dof_names[i] for i in self.env.arm_indices.tolist()],
                "leg_dof_names": [self.env.dof_names[i] for i in self.env.leg_indices.tolist()],
            },
            iteration,
        )


if __name__ == "__main__":
    FullBodyRunner().train()
