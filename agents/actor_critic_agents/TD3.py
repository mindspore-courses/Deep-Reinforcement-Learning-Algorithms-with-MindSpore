"""
TD3 Agent
"""
import mindspore as ms
from mindspore import ops, nn
from agents.Base_Agent import Base_Agent
from exploration_strategies.Gaussian_Exploration import Gaussian_Exploration
from .DDPG import DDPG



class TD3(DDPG):
    """A TD3 Agent from the paper Addressing Function Approximation Error in Actor-Critic Methods (Fujimoto et al. 2018)
    https://arxiv.org/abs/1802.09477"""
    agent_name = "TD3"

    def __init__(self, config):
        DDPG.__init__(self, config)
        self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                             key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                              key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.critic_optimizer_2 = nn.Adam(
            self.critic_local_2.trainable_params(),
            learning_rate=self.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4
        )
        self.exploration_strategy_critic = Gaussian_Exploration(self.config)

        # grads
        self.critic_grad_fn = ms.value_and_grad(
            self.compute_critic_loss,
            grad_position=None,
            weights=self.critic_local.trainable_params(),
            has_aux=False
        )
        self.critic_grad_fn_2 = ms.value_and_grad(
            self.compute_critic_loss_2,
            grad_position=None,
            weights=self.critic_local_2.trainable_params(),
            has_aux=False
        )

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        actions_next = self.actor_target(next_states)
        actions_next_with_noise = self.exploration_strategy_critic.perturb_action_for_exploration_purposes(
            {"action": actions_next}
        )
        critic_targets_next_1 = self.critic_target(
            ops.cat((next_states, actions_next_with_noise), 1)
        )
        critic_targets_next_2 = self.critic_target_2(
            ops.cat((next_states, actions_next_with_noise), 1)
        )
        critic_targets_next = ops.min(
            ops.cat((critic_targets_next_1, critic_targets_next_2), axis=1), axis=1
        )[0].unsqueeze(-1)

        return critic_targets_next

    def critic_learn(self, states, actions, rewards, next_states, dones):
        """Runs a learning iteration for both the critics"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)

        self.compute_critic_loss(critic_targets, states, actions)
        _, critic_grads_1 = self.critic_grad_fn(critic_targets, states, actions)
        _, critic_grads_2 = self.critic_grad_fn(critic_targets, states, actions)

        self.take_optimisation_step(self.critic_optimizer, critic_grads_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, critic_grads_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])

        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def compute_critic_loss(self, critic_targets, states, actions):
        """Compute critic local 1 loss"""
        critic_expected = self.critic_local(ops.cat((states, actions), 1))
        critic_loss = ops.mse_loss(critic_expected, critic_targets)

        return critic_loss

    def compute_critic_loss_2(self, critic_targets, states, actions):
        """Compute critic local 2 loss"""
        critic_expected = self.critic_local_2(ops.cat((states, actions), 1))
        critic_loss = ops.mse_loss(critic_expected, critic_targets)

        return critic_loss
