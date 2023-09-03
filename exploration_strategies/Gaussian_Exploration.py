"""
Gaussian Exploration
"""
import mindspore as ms
from mindspore import ops
from mindspore.nn.probability.distribution import Normal
from exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy


class Gaussian_Exploration(Base_Exploration_Strategy):
    """Gaussian noise exploration strategy"""

    def __init__(self, config):
        super().__init__(config)
        self.action_noise_std = self.config.hyperparameters["action_noise_std"]
        self.action_noise_distribution = Normal(ms.Tensor([0.0]), ms.Tensor([self.action_noise_std]))
        self.action_noise_clipping_range = self.config.hyperparameters["action_noise_clipping_range"]

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action = action_info["action"]
        action_noise = self.action_noise_distribution.sample(shape=action.shape)
        action_noise = action_noise.squeeze(-1)
        clipped_action_noise = ops.clamp(
            action_noise, min=-self.action_noise_clipping_range, max=self.action_noise_clipping_range
        )
        action += clipped_action_noise
        return action

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")
