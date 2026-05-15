import argparse
import glob
import os
import random
import time

import isaacgym
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from envs import *
from utils.model import ActorCritic


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(task, num_envs=None):
    with open(os.path.join("envs", f"{task}.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg["viewer"]["record_video"] = False
    cfg["basic"]["headless"] = True
    if num_envs is not None:
        cfg["env"]["num_envs"] = int(num_envs)
    return cfg


def load_checkpoint(path):
    if path == "-1" or path == -1:
        path = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    return path, torch.load(path, map_location="cuda:0", weights_only=True)


def build_old_leg_obs(env, last_leg_action):
    commands_scale = torch.tensor(
        [
            env.cfg["normalization"]["lin_vel"],
            env.cfg["normalization"]["lin_vel"],
            env.cfg["normalization"]["ang_vel"],
        ],
        device=env.device,
    )
    gait_active = (env.gait_frequency > 1.0e-8).float()
    return torch.cat(
        (
            env.projected_gravity * env.cfg["normalization"]["gravity"],
            env.base_ang_vel * env.cfg["normalization"]["ang_vel"],
            env.commands[:, :3] * commands_scale,
            (torch.cos(2 * torch.pi * env.gait_process) * gait_active).unsqueeze(-1),
            (torch.sin(2 * torch.pi * env.gait_process) * gait_active).unsqueeze(-1),
            (env.dof_pos[:, env.leg_indices] - env.default_dof_pos[:, env.leg_indices]) * env.cfg["normalization"]["dof_pos"],
            env.dof_vel[:, env.leg_indices] * env.cfg["normalization"]["dof_vel"],
            last_leg_action,
        ),
        dim=-1,
    )


def copy_matching_prefix(full_model, upper_model):
    full_state = full_model.state_dict()
    upper_state = upper_model.state_dict()
    with torch.no_grad():
        for key, value in upper_state.items():
            if key in full_state and tuple(full_state[key].shape) == tuple(value.shape):
                full_state[key].copy_(value)
    full_model.load_state_dict(full_state, strict=True)


def seed_output_heads(full_model, upper_model, leg_model, env):
    full_last = full_model.actor[-1]
    upper_last = upper_model.actor[-1]
    leg_last = leg_model.actor[-1]
    arm_names = env.cfg["control"]["arm_dof_names"]
    leg_names = env.cfg["control"]["leg_dof_names"]
    with torch.no_grad():
        for src, name in enumerate(arm_names):
            dst = env.dof_names.index(name)
            full_last.weight[dst].copy_(upper_last.weight[src])
            full_last.bias[dst].copy_(upper_last.bias[src])
        for src, name in enumerate(leg_names):
            dst = env.dof_names.index(name)
            full_last.weight[dst].copy_(leg_last.weight[src])
            full_last.bias[dst].copy_(leg_last.bias[src])


def seed_logstd(full_model, upper_ckpt, leg_ckpt, env):
    upper_logstd = upper_ckpt["model"].get("logstd")
    leg_logstd = leg_ckpt["model"].get("logstd")
    arm_names = env.cfg["control"]["arm_dof_names"]
    leg_names = env.cfg["control"]["leg_dof_names"]
    with torch.no_grad():
        full_model.logstd.fill_(-3.0)
        if upper_logstd is not None:
            upper_logstd = upper_logstd.to(device=full_model.logstd.device, dtype=full_model.logstd.dtype)
            for src, name in enumerate(arm_names):
                dst = env.dof_names.index(name)
                full_model.logstd[0, dst] = torch.minimum(upper_logstd[0, src], torch.tensor(-2.0, device=full_model.logstd.device))
        if leg_logstd is not None:
            leg_logstd = leg_logstd.to(device=full_model.logstd.device, dtype=full_model.logstd.dtype)
            for src, name in enumerate(leg_names):
                dst = env.dof_names.index(name)
                full_model.logstd[0, dst] = leg_logstd[0, src]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--leg-checkpoint", required=True)
    parser.add_argument("--upper-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--dagger-start-frac", type=float, default=0.35)
    parser.add_argument("--dagger-final-blend", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_cfg(args.task, args.num_envs)
    cfg["basic"]["checkpoint"] = None
    task_class = eval(cfg["basic"]["task"])
    env = task_class(cfg)
    device = cfg["basic"]["rl_device"]

    leg_path, leg_ckpt = load_checkpoint(args.leg_checkpoint)
    upper_path, upper_ckpt = load_checkpoint(args.upper_checkpoint)
    leg_model = ActorCritic(len(env.leg_indices), 47, env.num_privileged_obs).to(device)
    leg_model.load_state_dict(leg_ckpt["model"], strict=True)
    leg_model.eval()

    actor_mean_scale = upper_ckpt.get("actor_mean_scale")
    upper_model = ActorCritic(
        len(env.arm_indices),
        env.num_obs,
        env.num_privileged_obs,
        actor_mean_scale=actor_mean_scale,
        logstd_min=upper_ckpt.get("logstd_min"),
        logstd_max=upper_ckpt.get("logstd_max"),
    ).to(device)
    upper_model.load_state_dict(upper_ckpt["model"], strict=False)
    upper_model.eval()

    full_model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs, logstd_init=-2.1).to(device)
    copy_matching_prefix(full_model, upper_model)
    seed_output_heads(full_model, upper_model, leg_model, env)
    seed_logstd(full_model, upper_ckpt, leg_ckpt, env)
    optimizer = torch.optim.Adam(full_model.parameters(), lr=args.lr)

    obs, infos = env.reset()
    obs = obs.to(device)
    last_leg_action = torch.zeros(env.num_envs, len(env.leg_indices), dtype=torch.float, device=device)
    action_weight = torch.ones(env.num_actions, dtype=torch.float, device=device)
    action_weight[env.leg_indices] = 2.5
    action_weight[env.arm_indices] = 1.5
    if len(env.elbow_indices) > 0:
        action_weight[env.elbow_indices] = 1.2
    if len(env.waist_indices) > 0:
        action_weight[env.waist_indices] = 1.2

    start = time.time()
    ema_loss = None
    for step in range(args.steps):
        with torch.no_grad():
            leg_obs = build_old_leg_obs(env, last_leg_action)
            leg_action = torch.clamp(leg_model.act(leg_obs).loc, -1.0, 1.0)
            upper_action = torch.clamp(upper_model.act(obs).loc, -1.0, 1.0)
            target = torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=device)
            target[:, env.leg_indices] = leg_action
            target[:, env.arm_indices] = upper_action

        pred = full_model.act(obs).loc
        loss = torch.mean(((pred - target) * action_weight.unsqueeze(0)).square())
        loss = loss + 0.05 * torch.relu(pred.abs() - 1.0).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
        optimizer.step()
        ema_loss = loss.item() if ema_loss is None else 0.98 * ema_loss + 0.02 * loss.item()

        dagger_start = int(args.steps * args.dagger_start_frac)
        if step < dagger_start:
            blend = 0.0
        else:
            blend = args.dagger_final_blend * (step - dagger_start + 1) / max(1, args.steps - dagger_start)
        with torch.no_grad():
            pred_action = torch.clamp(pred.detach(), -1.0, 1.0)
            exec_action = torch.clamp((1.0 - blend) * target + blend * pred_action, -1.0, 1.0)
            obs, rew, done, infos = env.step(exec_action)
            obs = obs.to(device)
            last_leg_action[:] = exec_action[:, env.leg_indices]
            last_leg_action[done] = 0.0

        if (step + 1) % 50 == 0 or step == 0:
            elapsed = time.time() - start
            print(
                f"bc_step: {step + 1}/{args.steps} loss={loss.item():.6f} ema={ema_loss:.6f} "
                f"blend={blend:.2f} elapsed={elapsed / 60.0:.1f}m",
                flush=True,
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "model": full_model.state_dict(),
        "curriculum": upper_ckpt.get("curriculum", leg_ckpt.get("curriculum")),
        "bc": {
            "task": args.task,
            "leg_checkpoint": leg_path,
            "upper_checkpoint": upper_path,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "lr": args.lr,
            "dagger_start_frac": args.dagger_start_frac,
            "dagger_final_blend": args.dagger_final_blend,
            "final_loss": float(loss.item()),
            "ema_loss": float(ema_loss),
            "arm_dof_names": [env.dof_names[i] for i in env.arm_indices.tolist()],
            "leg_dof_names": [env.dof_names[i] for i in env.leg_indices.tolist()],
        },
    }
    torch.save(payload, args.out)
    print(f"saved_fullbody_bc: {args.out}", flush=True)


if __name__ == "__main__":
    main()
