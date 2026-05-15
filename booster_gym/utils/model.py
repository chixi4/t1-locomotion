import torch
import torch.nn.functional as F


class ActorCritic(torch.nn.Module):

    def __init__(
        self,
        num_act,
        num_obs,
        num_privileged_obs,
        logstd_init=-2.0,
        actor_mean_scale=None,
        logstd_min=None,
        logstd_max=None,
    ):
        super().__init__()
        self.actor_mean_scale = actor_mean_scale
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_obs + num_privileged_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_act),
        )
        self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_act), fill_value=logstd_init), requires_grad=True)

    def act(self, obs):
        action_mean = self.actor(obs)
        if self.actor_mean_scale is not None:
            if isinstance(self.actor_mean_scale, (list, tuple)):
                mean_scale = torch.tensor(self.actor_mean_scale, dtype=action_mean.dtype, device=action_mean.device).view(1, -1)
            else:
                mean_scale = float(self.actor_mean_scale)
            action_mean = torch.tanh(action_mean) * mean_scale
        logstd = self.logstd
        if self.logstd_min is not None or self.logstd_max is not None:
            min_val = -float("inf") if self.logstd_min is None else float(self.logstd_min)
            max_val = float("inf") if self.logstd_max is None else float(self.logstd_max)
            logstd = torch.clamp(logstd, min=min_val, max=max_val)
        action_std = torch.exp(logstd).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

    def est_value(self, obs, privileged_obs):
        critic_input = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)
