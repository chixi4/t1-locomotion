from dataclasses import dataclass
from typing import Dict, List

import torch
from isaacgym.torch_utils import quat_rotate


EPS = 1.0e-6
STILL_COMMAND_NORM = 0.05
FORWARD_COMMAND_MIN = 0.1
ACTION_JOINT_SUFFIXES = (
    "Hip_Pitch",
    "Hip_Roll",
    "Hip_Yaw",
    "Knee_Pitch",
    "Ankle_Pitch",
    "Ankle_Roll",
)


@dataclass(frozen=True)
class JointPair:
    left: int
    right: int


class T1OmniMetrics:
    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        pairs = _find_joint_pairs(env.dof_names)
        self.left_action_ids = _to_index_tensor([pair.left for pair in pairs], self.device)
        self.right_action_ids = _to_index_tensor([pair.right for pair in pairs], self.device)
        self._init_buffers()

    def reset(self, env_ids: torch.Tensor) -> None:
        for name in self._scalar_names:
            getattr(self, name)[env_ids] = 0.0
        for name in self._foot_names:
            getattr(self, name)[env_ids, :] = 0.0
        self.last_contact[env_ids, :] = False
        self.touchdown_ready[env_ids, :] = False
        self.last_touchdown_proj[env_ids, :] = 0.0

    def update(self) -> None:
        self.steps += 1.0
        self._update_body_metrics()
        self._update_command_metrics()
        self._update_foot_metrics()
        self._update_action_metrics()
        self.last_contact[:] = self.env.feet_contact

    def pop_episode_metrics(self, env_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        metrics = self.episode_metrics()
        selected = {}
        for key, value in metrics.items():
            selected[key] = torch.zeros(self.num_envs, device=self.device)
            selected[key][env_ids] = value[env_ids]
        self.reset(env_ids)
        return selected

    def step_metrics(self) -> Dict[str, torch.Tensor]:
        return {key: torch.mean(value) for key, value in self.episode_metrics().items()}

    def episode_metrics(self) -> Dict[str, torch.Tensor]:
        stance = _pair_asym(self.stance_time[:, 0], self.stance_time[:, 1])
        swing = _pair_asym(self.swing_time[:, 0], self.swing_time[:, 1])
        clearance_mean = _safe_div(self.clearance_sum, self.clearance_steps)
        step_length_mean = _safe_div(self.step_length_sum, self.step_count)
        slip_mean = _safe_div(self.slip_sum, self.slip_steps)
        clearance = _pair_asym(clearance_mean[:, 0], clearance_mean[:, 1])
        step_length = _pair_asym(step_length_mean[:, 0], step_length_mean[:, 1])
        slip_asym = _pair_asym(slip_mean[:, 0], slip_mean[:, 1])
        return {
            "Metrics/base_height_error": _safe_div(self.base_height_error, self.steps),
            "Metrics/base_roll_pitch": _safe_div(self.base_roll_pitch, self.steps),
            "Metrics/policy_forward_speed": _safe_div(self.policy_forward_speed, self.steps),
            "Metrics/fast_policy_forward_speed": _safe_div(self.fast_forward_speed, self.fast_forward_steps),
            "Metrics/policy_lateral_speed": _safe_div(self.policy_lateral_speed, self.steps),
            "Metrics/policy_yaw_rate": _safe_div(self.policy_yaw_rate, self.steps),
            "Metrics/standing_policy_speed": _safe_div(self.standing_speed, self.standing_steps),
            "Metrics/command_error_x": _safe_div(self.command_error_x, self.steps),
            "Metrics/command_error_y": _safe_div(self.command_error_y, self.steps),
            "Metrics/command_error_yaw": _safe_div(self.command_error_yaw, self.steps),
            "Metrics/moving_command_error_xy": _safe_div(self.moving_error_xy, self.moving_steps),
            "Metrics/moving_command_error_yaw": _safe_div(self.moving_error_yaw, self.moving_steps),
            "Metrics/stance_time_asymmetry": stance,
            "Metrics/swing_time_asymmetry": swing,
            "Metrics/step_length_asymmetry": step_length,
            "Metrics/foot_clearance_asymmetry": clearance,
            "Metrics/foot_slip_asymmetry": slip_asym,
            "Metrics/left_stance_time": self.stance_time[:, 0],
            "Metrics/right_stance_time": self.stance_time[:, 1],
            "Metrics/left_step_count": self.step_count[:, 0],
            "Metrics/right_step_count": self.step_count[:, 1],
            "Metrics/foot_slip_energy": _safe_div(self.slip_sum.sum(dim=1), self.steps),
            "Metrics/leg_action_magnitude_asymmetry": _safe_div(self.action_asymmetry, self.steps),
            "Metrics/action_jerk": _safe_div(self.action_jerk, self.steps),
            "Metrics/time_out_rate": self.env.time_out_buf.float(),
            "Metrics/fall_rate": self.env.reset_buf.float() * (~self.env.time_out_buf).float(),
        }

    def _init_buffers(self) -> None:
        self._scalar_names = _scalar_buffer_names()
        self._foot_names = _foot_buffer_names()
        for name in self._scalar_names:
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))
        for name in self._foot_names:
            setattr(self, name, torch.zeros(self.num_envs, 2, device=self.device))
        self.last_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.touchdown_ready = torch.zeros_like(self.last_contact)
        self.last_touchdown_proj = torch.zeros(self.num_envs, 2, device=self.device)

    def _update_body_metrics(self) -> None:
        terrain_height = self.env.terrain.terrain_heights(self.env.base_pos)
        height = self.env.base_pos[:, 2] - terrain_height
        target = self.env.cfg["rewards"]["base_height_target"]
        self.base_height_error += torch.abs(height - target)
        self.base_roll_pitch += torch.norm(self.env.projected_gravity[:, :2], dim=1)

    def _update_command_metrics(self) -> None:
        lin_error = torch.abs(self.env.filtered_lin_vel[:, :2] - self.env.commands[:, :2])
        yaw_error = torch.abs(self.env.filtered_ang_vel[:, 2] - self.env.commands[:, 2])
        cmd_norm = torch.norm(self.env.commands[:, :2], dim=1) + torch.abs(self.env.commands[:, 2])
        still_mask = cmd_norm < STILL_COMMAND_NORM
        moving_mask = ~still_mask
        forward_mask = self.env.commands[:, 0] > FORWARD_COMMAND_MIN
        self.command_error_x += lin_error[:, 0]
        self.command_error_y += lin_error[:, 1]
        self.command_error_yaw += yaw_error
        self.policy_forward_speed += self.env.filtered_lin_vel[:, 0]
        self.policy_lateral_speed += torch.abs(self.env.filtered_lin_vel[:, 1])
        self.policy_yaw_rate += torch.abs(self.env.filtered_ang_vel[:, 2])
        self.standing_speed += torch.norm(self.env.filtered_lin_vel[:, :2], dim=1) * still_mask.float()
        self.standing_steps += still_mask.float()
        self.moving_error_xy += torch.norm(lin_error, dim=1) * moving_mask.float()
        self.moving_error_yaw += yaw_error * moving_mask.float()
        self.moving_steps += moving_mask.float()
        self.fast_forward_speed += self.env.filtered_lin_vel[:, 0] * forward_mask.float()
        self.fast_forward_steps += forward_mask.float()

    def _update_foot_metrics(self) -> None:
        contact = self.env.feet_contact.float()
        no_contact = 1.0 - contact
        foot_velocity = (self.env.feet_pos - self.env.last_feet_pos) / self.env.dt
        foot_height = self._relative_foot_height()
        self.stance_time += contact * self.env.dt
        self.swing_time += no_contact * self.env.dt
        self.clearance_sum += foot_height * no_contact
        self.clearance_steps += no_contact
        self.slip_sum += torch.norm(foot_velocity[:, :, :2], dim=2) * contact
        self.slip_steps += contact
        self._update_touchdowns()

    def _update_touchdowns(self) -> None:
        touchdown = (~self.last_contact) & self.env.feet_contact
        projection = self._project_feet_on_command()
        ready_touchdown = touchdown & self.touchdown_ready
        length = torch.abs(projection - self.last_touchdown_proj) * ready_touchdown.float()
        self.step_length_sum += length
        self.step_count += ready_touchdown.float()
        self.last_touchdown_proj = torch.where(touchdown, projection, self.last_touchdown_proj)
        self.touchdown_ready |= touchdown

    def _update_action_metrics(self) -> None:
        left_mag = torch.mean(torch.abs(self.env.actions[:, self.left_action_ids]), dim=1)
        right_mag = torch.mean(torch.abs(self.env.actions[:, self.right_action_ids]), dim=1)
        self.action_asymmetry += _pair_asym(left_mag, right_mag)
        self.action_jerk += torch.mean(torch.abs(self.env.actions - self.env.last_actions), dim=1)

    def _relative_foot_height(self) -> torch.Tensor:
        flat_feet = self.env.feet_pos.reshape(-1, 3)
        terrain_height = self.env.terrain.terrain_heights(flat_feet).reshape(self.num_envs, 2)
        return torch.clip(self.env.feet_pos[:, :, 2] - terrain_height, min=0.0)

    def _project_feet_on_command(self) -> torch.Tensor:
        direction = self._world_command_direction()
        return torch.sum(self.env.feet_pos[:, :, :2] * direction.unsqueeze(1), dim=2)

    def _world_command_direction(self) -> torch.Tensor:
        local_direction = torch.zeros(self.num_envs, 3, device=self.device)
        local_direction[:, :2] = self.env.commands[:, :2]
        local_norm = torch.norm(local_direction[:, :2], dim=1)
        fallback = torch.zeros_like(local_direction)
        fallback[:, 0] = 1.0
        local_direction = torch.where((local_norm > STILL_COMMAND_NORM).unsqueeze(1), local_direction, fallback)
        world_direction = quat_rotate(self.env.base_quat, local_direction)[:, :2]
        return world_direction / torch.clamp(torch.norm(world_direction, dim=1, keepdim=True), min=EPS)

def _find_joint_pairs(dof_names: List[str]) -> List[JointPair]:
    pairs = []
    for suffix in ACTION_JOINT_SUFFIXES:
        pairs.append(JointPair(_find_index(dof_names, f"Left_{suffix}"), _find_index(dof_names, f"Right_{suffix}")))
    return pairs


def _find_index(dof_names: List[str], target: str) -> int:
    matches = [index for index, name in enumerate(dof_names) if name == target]
    if len(matches) != 1:
        raise ValueError(f"Expected one joint named {target}, found {len(matches)}")
    return matches[0]


def _to_index_tensor(indices: List[int], device: str) -> torch.Tensor:
    return torch.tensor(indices, dtype=torch.long, device=device)


def _pair_asym(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.abs(left - right) / torch.clamp(torch.abs(left) + torch.abs(right), min=EPS)


def _safe_div(value: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    return value / torch.clamp(count, min=EPS)


def _scalar_buffer_names() -> List[str]:
    return [
        "steps",
        "base_height_error",
        "base_roll_pitch",
        "policy_forward_speed",
        "fast_forward_speed",
        "fast_forward_steps",
        "policy_lateral_speed",
        "policy_yaw_rate",
        "standing_speed",
        "standing_steps",
        "command_error_x",
        "command_error_y",
        "command_error_yaw",
        "moving_error_xy",
        "moving_error_yaw",
        "moving_steps",
        "action_asymmetry",
        "action_jerk",
    ]


def _foot_buffer_names() -> List[str]:
    return [
        "stance_time",
        "swing_time",
        "clearance_sum",
        "clearance_steps",
        "slip_sum",
        "slip_steps",
        "step_length_sum",
        "step_count",
    ]
