import torch
from torch.utils.tensorboard import SummaryWriter
import os
import re
import time
import wandb
import yaml


MAX_RUN_NAME_CHARS = 96
RAW_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
PASSTHROUGH_PREFIXES = ("Metrics/", "Episode_", "Command/", "Symmetry/")


class Recorder:

    def __init__(self, cfg):
        self.cfg = cfg
        name = self._make_log_name()
        self.dir = os.path.join("logs", name)
        os.makedirs(self.dir)
        self.model_dir = os.path.join(self.dir, "nn")
        os.mkdir(self.model_dir)
        self.writer = SummaryWriter(os.path.join(self.dir, "summaries"))
        if self.cfg["runner"]["use_wandb"]:
            wandb.init(
                project=self.cfg["basic"]["task"],
                dir=self.dir,
                name=name,
                notes=self.cfg["basic"]["description"],
                config=self.cfg,
            )

        self.episode_statistics = {}
        self.last_episode = {}
        self.last_episode["steps"] = []
        self.episode_steps = None

        with open(os.path.join(self.dir, "config.yaml"), "w") as file:
            yaml.dump(self.cfg, file)

    def record_episode_statistics(self, done, ep_info, it, write_record=False):
        if self.episode_steps is None:
            self.episode_steps = torch.zeros_like(done, dtype=int)
        else:
            self.episode_steps += 1
        for val in self.episode_steps[done]:
            self.last_episode["steps"].append(val.item())
        self.episode_steps[done] = 0

        for key, value in ep_info.items():
            if self.episode_statistics.get(key) is None:
                self.episode_statistics[key] = torch.zeros_like(value)
            self.episode_statistics[key] += value
            if self.last_episode.get(key) is None:
                self.last_episode[key] = []
            for done_value in self.episode_statistics[key][done]:
                self.last_episode[key].append(done_value.item())
            self.episode_statistics[key][done] = 0

        if write_record:
            for key in self.last_episode.keys():
                path = self._tensorboard_path(key)
                value = self._mean(self.last_episode[key])
                self.writer.add_scalar(path, value, it)
                if self.cfg["runner"]["use_wandb"]:
                    wandb.log({path: value}, step=it)
                self.last_episode[key].clear()

    def record_statistics(self, statistics, it):
        for key, value in statistics.items():
            self.writer.add_scalar(key, float(value), it)
            if self.cfg["runner"]["use_wandb"]:
                wandb.log({key: float(value)}, step=it)

    def save(self, model_dict, it):
        path = os.path.join(self.model_dir, "model_{}.pth".format(it))
        print("Saving model to {}".format(path))
        torch.save(model_dict, path)

    def _mean(self, data):
        if len(data) == 0:
            return 0.0
        else:
            return sum(data) / len(data)

    def _make_log_name(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        raw_name = self.cfg["basic"].get("run_name")
        if not raw_name:
            return timestamp
        clean_name = RAW_NAME_PATTERN.sub("_", raw_name).strip("._-")
        if not clean_name:
            raise ValueError("run_name must contain at least one filesystem-safe character")
        return f"{timestamp}_{clean_name[:MAX_RUN_NAME_CHARS]}"

    def _tensorboard_path(self, key):
        if key in {"steps", "reward"} or key.startswith(PASSTHROUGH_PREFIXES):
            return key
        return "episode/" + key
