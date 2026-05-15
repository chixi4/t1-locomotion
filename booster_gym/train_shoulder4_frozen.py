import isaacgym
from utils.shoulder_runner import FrozenLegShoulderRunner


if __name__ == "__main__":
    runner = FrozenLegShoulderRunner()
    runner.train()
