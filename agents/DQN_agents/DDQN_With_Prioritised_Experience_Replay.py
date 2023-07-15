"""
DDQN with Prioritised experience replay
"""
# pylint: disable=C0103
import mindspore as ms
from mindspore import ops
from agents.DQN_agents.DDQN import DDQN
from utilities.data_structures.Prioritised_Replay_Buffer import Prioritised_Replay_Buffer


class DDQN_With_Prioritised_Experience_Replay(DDQN):
    """A DQN agent with prioritised experience replay"""
    agent_name = "DDQN with Prioritised Replay"

    def __init__(self, config):
        DDQN.__init__(self, config)
        self.memory = Prioritised_Replay_Buffer(self.hyperparameters, config.seed)
        # 所以重新定义梯度计算公式
        self.grad_fn = ms.value_and_grad(
            self.compute_loss_and_td_errors, grad_position=None, weights=self.q_network_local.trainable_params(),
            has_aux=True
        )

    def learn(self, experiences=None):
        # experiences用于保持一致，并不使用
        """Runs a learning iteration for the Q network after sampling from the replay buffer in a prioritised way"""
        sampled_experiences, importance_sampling_weights = self.memory.sample()
        states, actions, rewards, next_states, dones = sampled_experiences

        # loss, td_errors = self.compute_loss_and_td_errors(
        #     states, next_states, rewards, actions, dones, importance_sampling_weights
        # )
        (_, td_errors), grads = self.grad_fn(
                states, next_states, rewards, actions, dones, importance_sampling_weights
            )
        self.take_optimisation_step(
            self.q_network_optimizer, grads, self.hyperparameters["gradient_clipping_norm"]
        )
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target, self.hyperparameters["tau"])
        self.memory.update_td_errors(td_errors.squeeze(1))

    def save_experience(self, memory=None, experience=None):
        # 变量仅用于保持一致，并不使用
        """Saves the latest experience including the td_error"""
        max_td_error_in_experiences = self.memory.give_max_td_error() + 1e-9
        self.memory.add_experience(
            max_td_error_in_experiences, self.state, self.action, self.reward, self.next_state, self.done
        )

    def compute_loss_and_td_errors(self, states, next_states, rewards, actions, dones, importance_sampling_weights):
        """Calculates the loss for the local Q network. It weighs each observations loss according to the importance
        sampling weights which come from the prioritised replay buffer"""
        Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = self.loss_fn(Q_expected, Q_targets)
        loss = loss * importance_sampling_weights
        loss = ops.mean(loss)
        td_errors = Q_targets.numpy() - Q_expected.numpy()
        return loss, td_errors
