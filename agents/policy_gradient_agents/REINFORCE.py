"""
REINFORCE
"""
import mindspore as ms
from mindspore import nn, ops
from mindspore.nn.probability.distribution import Categorical
from agents.Base_Agent import Base_Agent


EPS = 1e-6


class REINFORCE(Base_Agent):
    """
    REINFORCE
    """
    agent_name = "REINFORCE"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        config.num_episodes_to_run = 1000
        config.hyperparameters['final_layer_activation'] = "SOFTMAX"

        self.policy = self.create_NN(
            input_dim=self.state_size, output_dim=self.action_size, hyperparameters=config.hyperparameters
        )
        self.optimizer = nn.Adam(
            self.policy.trainable_params(), learning_rate=self.hyperparameters["learning_rate"]
        )
        self.episode_rewards = []
        self.episode_log_states = []
        self.episode_log_actions = []
        self.episode_step_number = 0

        # grads
        self.grad_fn = ms.value_and_grad(
            self.calculate_policy_loss_on_episode,
            grad_position=None,
            weights=self.policy.trainable_params(),
            has_aux=False
        )

    def reset_game(self):
        """Resets the game information, so we are ready to play a new episode"""
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_rewards = []
        self.episode_log_states = []
        self.episode_log_actions = []
        self.episode_step_number = 0

    def step(self):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.pick_and_conduct_action_and_save_log_probabilities()
            # self.update_next_state_reward_done_and_score()
            self.store_reward()
            if self.time_to_learn():
                self.actor_learn()
            self.state = self.next_state  # this is to set the state for the next iteration
            self.episode_step_number += 1
        self.episode_number += 1
        self.update_learning_rate(self.hyperparameters["learning_rate"], self.optimizer)

    def pick_and_conduct_action_and_save_log_probabilities(self):
        """Picks and then conducts actions. Then saves the log probabilities of the actions it conducted to be used for
        learning later"""
        action, _, state = self.pick_action_and_get_log_probabilities()
        action = action.numpy()
        # self.store_log_probabilities(log_probabilities)
        self.store_log_states(state)
        self.store_action(action)
        self.conduct_action(action)

    def pick_action_and_get_log_probabilities(self):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        # PyTorch only accepts mini-batches and not individual observations so we have to add
        # a "fake" dimension to our observation using un-squeeze.
        state = ms.Tensor(self.state, ms.float32).unsqueeze(dim=0)
        action_probabilities = self.policy(state).clip(min=EPS, max=1 - EPS)
        action_probabilities /= action_probabilities.sum()
        # this creates a distribution to sample from
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample()
        return action.item(), action_distribution.log_prob(action), state

    def store_log_states(self, log_states):
        """Stores the log probabilities of picked actions to be used for learning later"""
        self.episode_log_states.append(log_states)

    def store_action(self, action):
        """Stores the action picked"""
        self.action = action
        self.episode_log_actions.append(action.item(0))

    def store_reward(self):
        """Stores the reward picked"""
        self.episode_rewards.append(self.reward)

    def actor_learn(self):
        """Runs a learning iteration for the policy"""
        total_discounted_reward = self.calculate_episode_discounted_reward()
        # policy_loss = self.calculate_policy_loss_on_episode(total_discounted_reward)
        _, grads = self.grad_fn(total_discounted_reward)
        self.take_optimisation_step(optimizer=self.optimizer, grads=grads, )

    def calculate_episode_discounted_reward(self):
        """Calculates the cumulative discounted return for the episode"""
        discounted_rewards = []
        rewards = self.episode_rewards
        gamma = self.hyperparameters["discount_rate"]
        # for t in range(len(rewards)):
        #     Gt = 0
        #     pw = 0
        #     for r in rewards[t:]:
        #         Gt += gamma ** pw * r
        #         pw += 1
        #     discounted_rewards.append(Gt)
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            discounted_rewards.append(R)
        discounted_rewards.reverse()

        # total_discounted_reward = np.dot(discounts, self.episode_rewards)
        total_discounted_reward = ms.Tensor(discounted_rewards, dtype=ms.float32).reshape(-1, 1)
        total_discounted_reward = (total_discounted_reward - total_discounted_reward.mean()) / \
                                  (total_discounted_reward.std() + 1e-9)
        return total_discounted_reward

    def calculate_policy_loss_on_episode(self, total_discounted_reward):
        """Calculates the loss from an episode"""
        policy_loss = 0
        if len(self.episode_log_states) != 0:
            states = ops.vstack(self.episode_log_states)
            actions = ms.Tensor(self.episode_log_actions).reshape(-1, 1).long()
            prob = self.policy(states).gather(actions, axis=1, batch_dims=1)

            log_prob = prob.log()
            policy_loss = (-log_prob * total_discounted_reward).sum()
        else:
            print("not learn")
        # for log_prob in self.episode_log_probabilities:
        #     policy_loss.append(-log_prob * total_discounted_reward)
        # We need to add up the losses across the mini-batch to get 1 overall loss
        #     policy_loss = ops.cat(
        #         policy_loss
        #     ).sum()
        return policy_loss

    def time_to_learn(self):
        """Tells us whether it is time for the algorithm to learn. With REINFORCE we only learn at the end of every
        episode so this just returns whether the episode is over"""
        return self.done
