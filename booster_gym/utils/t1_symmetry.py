import torch


OBS_DIM = 47
ACTION_DIM = 12
DOF_START = 11
DOF_COUNT = 12
ACTION_START = 35
MIRROR_INDEX = (6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5)
MIRROR_SIGN = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0)


def mirror_t1_action(action: torch.Tensor) -> torch.Tensor:
    _check_last_dim(action, ACTION_DIM, "action")
    index = _index_tensor(action)
    sign = _sign_tensor(action)
    return action[..., index] * sign


def mirror_t1_observation(obs: torch.Tensor) -> torch.Tensor:
    _check_last_dim(obs, OBS_DIM, "observation")
    mirrored = obs.clone()
    mirrored[..., 0:3] = obs[..., 0:3] * _vector_sign(obs, (1.0, -1.0, 1.0))
    mirrored[..., 3:6] = obs[..., 3:6] * _vector_sign(obs, (-1.0, 1.0, -1.0))
    mirrored[..., 6:9] = obs[..., 6:9] * _vector_sign(obs, (1.0, -1.0, -1.0))
    mirrored[..., 9:11] = -obs[..., 9:11]
    mirrored[..., DOF_START : DOF_START + DOF_COUNT] = mirror_t1_action(obs[..., DOF_START : DOF_START + DOF_COUNT])
    mirrored[..., 23:35] = mirror_t1_action(obs[..., 23:35])
    mirrored[..., ACTION_START : ACTION_START + ACTION_DIM] = mirror_t1_action(obs[..., ACTION_START : ACTION_START + ACTION_DIM])
    return mirrored


def _index_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.tensor(MIRROR_INDEX, dtype=torch.long, device=value.device)


def _sign_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.tensor(MIRROR_SIGN, dtype=value.dtype, device=value.device)


def _vector_sign(value: torch.Tensor, signs: tuple) -> torch.Tensor:
    return torch.tensor(signs, dtype=value.dtype, device=value.device)


def _check_last_dim(value: torch.Tensor, expected: int, name: str) -> None:
    if value.shape[-1] != expected:
        raise ValueError(f"Expected T1 {name} dimension {expected}, got {value.shape[-1]}")
