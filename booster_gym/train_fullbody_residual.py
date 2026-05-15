import isaacgym
from utils.fullbody_residual_runner import FullBodyResidualRunner


if __name__ == "__main__":
    runner = FullBodyResidualRunner()
    runner.train()
