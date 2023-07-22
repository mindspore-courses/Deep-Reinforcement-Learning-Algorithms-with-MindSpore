"""
PPO
"""
# pylint: disable=C0103

import sys
import mindspore as ms
from mindspore import nn, ops
import numpy as np
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.Parallel_Experience_Generator import Parallel_Experience_Generator
from utilities.Utility_Functions import normalise_rewards, create_actor_distribution


class PPO(Base_Agent):
    """
    Proximal Policy Optimization agent
    """
    agent_name = "PPO"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.policy_output_size = self.calculate_policy_output_size()
        self.policy_new = self.create_NN(input_dim=self.state_size, output_dim=self.policy_output_size)
        self.policy_old = self.create_NN(input_dim=self.state_size, output_dim=self.policy_output_size)
        self.policy_old.set_train(mode=False)
        # self.policy_old.load_state_dict(copy.deepcopy(self.policy_new.state_dict()))
        self.copy_model_over(self.policy_new, self.policy_old)
        self.policy_new_optimizer = nn.Adam(
            self.policy_new.trainable_params(), learning_rate=self.hyperparameters["learning_rate"], eps=1e-4
        )
        self.episode_number = 0
        self.many_episode_states = []
        self.many_episode_actions = []
        self.many_episode_rewards = []
        self.experience_generator = Parallel_Experience_Generator(self.environment, self.policy_new, self.config.seed,
                                                                  self.hyperparameters, self.action_size)
        self.exploration_strategy = Epsilon_Greedy_Exploration(self.config)

        self.calculate_loss_grad_fn = ms.value_and_grad(
            self.calculate_loss, grad_position=None, weights=self.policy_new.trainable_params(), has_aux=False
        )

    def calculate_policy_output_size(self):
        """Initialises the policies"""
        result = 0
        if self.action_types == "DISCRETE":
            result = self.action_size
        elif self.action_types == "CONTINUOUS":
            result = self.action_size * 2  # Because we need 1 parameter for mean and 1 for std of distribution
        return result

    def step(self):
        """Runs a step for the PPO agent"""
        exploration_epsilon = self.exploration_strategy.get_updated_epsilon_exploration(
            {"episode_number": self.episode_number}
        )
        self.many_episode_states, self.many_episode_actions, self.many_episode_rewards = \
            self.experience_generator.play_n_episodes(
                self.hyperparameters["episodes_per_learning_round"], exploration_epsilon
            )
        self.episode_number += self.hyperparameters["episodes_per_learning_round"]
        self.policy_learn()
        self.update_learning_rate(self.hyperparameters["learning_rate"], self.policy_new_optimizer)
        self.equalise_policies()

    def policy_learn(self):
        """A learning iteration for the policy"""
        all_discounted_returns = self.calculate_all_discounted_returns()
        if self.hyperparameters["normalise_rewards"]:
            all_discounted_returns = normalise_rewards(all_discounted_returns)
        all_discounted_returns = ms.Tensor(all_discounted_returns, dtype=ms.float32)
        for _ in range(self.hyperparameters["learning_iterations_per_round"]):
            all_states = [state for states in self.many_episode_states for state in states]
            all_states = ops.stack([ms.Tensor(states, dtype=ms.float32) for states in all_states])

            all_actions = [
                [action] if self.action_types == "DISCRETE" else action
                for actions in self.many_episode_actions for action in actions
            ]
            all_actions = ops.stack([ms.Tensor(actions, dtype=ms.float32) for actions in all_actions])
            all_actions = all_actions.view(-1, len(all_states))

            old_policy_distribution_log_prob = self.calculate_log_probability_of_actions(
                self.policy_old, all_states, all_actions
            )
            # all_ratio_of_policy_probabilities = self.calculate_all_ratio_of_policy_probabilities()
            self.calculate_loss(all_discounted_returns, all_states, all_actions, old_policy_distribution_log_prob)
            _, grads = self.calculate_loss_grad_fn(
                all_discounted_returns, all_states, all_actions, old_policy_distribution_log_prob
            )
            self.take_policy_new_optimisation_step(grads)

    def calculate_all_discounted_returns(self):
        """Calculates the cumulative discounted return for each episode which we will then use in a learning
        iteration"""
        all_discounted_returns = []
        for episode, many_episode_states_element in enumerate(self.many_episode_states):
            discounted_returns = [0]
            for ix in range(len(many_episode_states_element)):
                return_value = self.many_episode_rewards[episode][-(ix + 1)] + \
                               self.hyperparameters["discount_rate"] * discounted_returns[-1]
                discounted_returns.append(return_value)
            discounted_returns = discounted_returns[1:]
            all_discounted_returns.extend(discounted_returns[::-1])
        return all_discounted_returns

    def calculate_all_ratio_of_policy_probabilities(self, all_states, all_actions, old_policy_distribution_log_prob):
        """For each action calculates the ratio of the probability that the new policy would have picked the action vs.
         the probability the old policy would have picked it. This will then be used to inform the loss"""
        new_policy_distribution_log_prob = self.calculate_log_probability_of_actions(
            self.policy_new, all_states, all_actions
        )
        # old_policy_distribution_log_prob = self.calculate_log_probability_of_actions(
        #     self.policy_old, all_states, all_actions
        # )
        ratio_of_policy_probabilities = \
            ops.exp(new_policy_distribution_log_prob) / (ops.exp(old_policy_distribution_log_prob) + 1e-6)
        return ratio_of_policy_probabilities

    def calculate_log_probability_of_actions(self, policy, states, actions):
        """Calculates the log probability of an action occuring given a policy and starting state"""
        policy_output = policy(states)
        policy_distribution = create_actor_distribution(self.action_types, policy_output, self.action_size)
        policy_distribution_log_prob = policy_distribution.log_prob(actions)
        return policy_distribution_log_prob

    def calculate_loss(self, all_discounted_returns, all_states, all_actions, old_policy_distribution_log_prob):
        """Calculates the PPO loss"""
        policy_output = self.policy_new(all_states)
        policy_distribution = create_actor_distribution(self.action_types, policy_output, self.action_size)
        policy_distribution_log_prob = policy_distribution.log_prob(all_actions)

        new_policy_distribution_log_prob = policy_distribution_log_prob
        ratio_of_policy_probabilities = ops.exp(new_policy_distribution_log_prob - old_policy_distribution_log_prob)
        # 下式精度损失太大导致不收敛, 好像不是？
        # ratio_of_policy_probabilities = \
        #     ops.exp(new_policy_distribution_log_prob) / (ops.exp(old_policy_distribution_log_prob) + 1e-6)

        all_ratio_of_policy_probabilities = [
            ratio_of_policy_probabilities
        ]
        all_ratio_of_policy_probabilities = ops.squeeze(ops.stack(all_ratio_of_policy_probabilities))
        all_ratio_of_policy_probabilities = ops.clamp(input=all_ratio_of_policy_probabilities,
                                                      min=-sys.maxsize,
                                                      max=sys.maxsize)
        # mindspore无需特别指定来保证在同一设备
        # all_discounted_returns = ms.Tensor(all_discounted_returns).to(all_ratio_of_policy_probabilities)
        all_discounted_returns = ms.Tensor(all_discounted_returns, dtype=ms.float32)
        potential_loss_value_1 = all_discounted_returns * all_ratio_of_policy_probabilities
        potential_loss_value_2 = all_discounted_returns * self.clamp_probability_ratio(
            all_ratio_of_policy_probabilities)
        loss = ops.min(ops.vstack((potential_loss_value_1, potential_loss_value_2)), axis=0)[0]
        # loss = ops.min(potential_loss_value_1, potential_loss_value_2)
        loss = -ops.mean(loss)
        return loss

    def clamp_probability_ratio(self, value):
        """Clamps a value between a certain range determined by hyperparameter clip epsilon"""
        return ops.clamp(input=value, min=1.0 - self.hyperparameters["clip_epsilon"],
                         max=1.0 + self.hyperparameters["clip_epsilon"])

    def take_policy_new_optimisation_step(self, grads):
        """Takes an optimisation step for the new policy"""
        grads = ops.clip_by_global_norm(grads, self.hyperparameters["gradient_clipping_norm"])
        self.policy_new_optimizer(grads)

    def equalise_policies(self):
        """Sets the old policy's parameters equal to the new policy's parameters"""
        self.copy_model_over(self.policy_new, self.policy_old)
        # for old_param, new_param in zip(self.policy_old.parameters(), self.policy_new.parameters()):
        #     old_param.data.copy_(new_param.data)

    def save_result(self):
        """Save the results seen by the agent in the most recent experiences"""
        for episode_reward in self.many_episode_rewards:
            total_reward = np.sum(episode_reward)
            self.game_full_episode_scores.append(total_reward)
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()

    # def play_n_episodes(self, n, exploration_epsilon=None):
    #     """Plays n episodes in parallel using the fixed policy and returns the data"""
    #     self.exploration_epsilon = exploration_epsilon
    #     x = self(0)
    #     with closing(Pool(processes=n)) as pool:
    #         args = [_ for _ in range(n)]
    #         results = pool.map(self, args)
    #         pool.terminate()
    #
    #     # results = [x, self(1)]
    #     states_for_all_episodes = [episode[0] for episode in results]
    #     actions_for_all_episodes = [episode[1] for episode in results]
    #     rewards_for_all_episodes = [episode[2] for episode in results]
    #     return states_for_all_episodes, actions_for_all_episodes, rewards_for_all_episodes
