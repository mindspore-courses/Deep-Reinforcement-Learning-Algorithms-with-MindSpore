""""
SAC
"""
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.nn.probability.distribution import Normal
from utilities.OU_Noise import OU_Noise
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from agents.Base_Agent import Base_Agent


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class SAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this GitHub
    implementation https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the
    agent is also trained to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"

    def __init__(self, config, _assert=True):
        Base_Agent.__init__(self, config)
        if _assert:
            assert self.action_types == "CONTINUOUS", \
                "Action types must be continuous. Use SAC Discrete instead for discrete actions"
            assert self.config.hyperparameters["Actor"]["final_layer_activation"] != "Softmax", \
                "Final actor layer must not be softmax"
        self.hyperparameters = config.hyperparameters
        self.critic_local_1 = self.create_NN(
            input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic"
        )
        self.critic_local_2 = self.create_NN(
            input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic",
            override_seed=self.config.seed + 1
        )
        self.critic_optimizer_1 = nn.Adam(
            self.critic_local_1.trainable_params(), learning_rate=self.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4
        )
        self.critic_optimizer_2 = nn.Adam(
            self.critic_local_2.trainable_params(), learning_rate=self.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4
        )
        self.critic_target_1 = self.create_NN(
            input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic"
        )
        self.critic_target_2 = self.create_NN(
            input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic"
        )
        Base_Agent.copy_model_over(self.critic_local_1, self.critic_target_1)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(
            self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"], self.config.seed
        )
        self.actor_local = self.create_NN(
            input_dim=self.state_size, output_dim=self.action_size * 2, key_to_use="Actor"
        )
        self.actor_optimizer = nn.Adam(
            self.actor_local.trainable_params(), learning_rate=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4
        )
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            # heuristic value from the paper
            self.target_entropy = -ops.prod(ms.Tensor(self.environment.action_space.shape)).numpy().item()
            self.log_alpha = ms.Parameter(ops.zeros(1), name="log_alpha")
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = nn.Adam(
                [self.log_alpha], learning_rate=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4
            )
            # grads
            self.alpha_grad_fn = ms.value_and_grad(
                self.calculate_entropy_tuning_loss, grad_position=None, weights=[self.log_alpha], has_aux=False
            )
        else:
            if self.hyperparameters["entropy_term_weight"] is not None:
                self.alpha = self.hyperparameters["entropy_term_weight"]
            else:
                self.alpha = 1

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

        # grads
        self.mse_loss = ops.mse_loss
        self.actor_grad_fn = ms.value_and_grad(
            self.calculate_actor_loss, grad_position=None, weights=self.actor_local.trainable_params(),
            has_aux=True
        )
        self.q1_grad_fn = ms.value_and_grad(
            self.calculate_qf1_grads, grad_position=None, weights=self.critic_local_1.trainable_params(),
            has_aux=False
        )
        self.q2_grad_fn = ms.value_and_grad(
            self.calculate_qf2_grads, grad_position=None, weights=self.critic_local_2.trainable_params(),
            has_aux=False
        )
        # self.critic_loss = ms.Tensor([0])

        # standard normal
        self.normal = Normal(mean=0, sd=1)
        self.sample = self.normal.sample

        self.episode_step_number_val = None
        self.action = None
        self.state = None

    def save_result(self):
        """Saves the result of an episode of the game. Overriding the method in Base Agent that does this because we
        only want to keep track of the results during the evaluation episodes"""
        if self.episode_number == 1 or not self.do_evaluation_iterations:
            total_episode_score_so_far = self.total_episode_score_so_far
            if isinstance(self.total_episode_score_so_far, ms.Tensor):
                total_episode_score_so_far = total_episode_score_so_far.numpy()
            self.game_full_episode_scores.extend([total_episode_score_so_far])
            rolling_results = self.game_full_episode_scores[-1 * self.rolling_score_window:]
            self.rolling_results.append(np.mean(rolling_results))
            self.save_max_result_seen()

        elif (self.episode_number - 1) % TRAINING_EPISODES_PER_EVAL_EPISODE == 0:
            self.game_full_episode_scores.extend(
                [self.total_episode_score_so_far for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.rolling_results.extend(
                [np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]) for _ in
                 range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.save_max_result_seen()

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        if self.add_extra_noise:
            self.noise.reset()

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            if not eval_ep:
                self.save_experience(experience=(self.state, self.action, self.reward, self.next_state, mask))
            self.state = self.next_state
            self.global_step_number += 1
        # print(self.total_episode_score_so_far)
        if eval_ep:
            self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
        2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is
        False. The difference between evaluation and training mode is that training mode does more exploration"""
        if state is None:
            state = self.state
        if eval_ep:
            action = self.actor_pick_action(state=state, _eval=True)
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            action = self.environment.action_space.sample()
            # print("Picking random action ", action)
        else:
            action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        return action

    def actor_pick_action(self, state=None, _eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None:
            state = self.state
        # state = torch.FloatTensor([state])
        state = ms.Tensor([state])
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if not _eval:
            # action, _, _ = self.produce_action_and_action_info(state)
            action = self.not_eval_produce_action_and_action_info(state)
        else:
            # _, z, action = self.produce_action_and_action_info(state)
            action = self.eval_produce_action_and_action_info(state)
        action = action.numpy()
        return action[0]

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        normal = Normal(mean=mean, sd=std)
        x_t = mean + self.sample(shape=mean.shape) * std
        action = ops.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= ops.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdims=True)
        return action, log_prob, ops.tanh(mean)

    def action_log_prob_produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        # normal = Normal(mean=mean, sd=std)
        x_t = mean + self.sample(shape=mean.shape) * std
        action = ops.tanh(x_t)
        # log_prob = normal.log_prob(x_t)
        # 尝试不使用normal, 分母防止0出现
        log_prob = self.normal.log_prob((x_t-mean)/(std+EPSILON))
        log_prob -= ops.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdims=True)
        return action, log_prob

    def not_eval_produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        state = state.float()
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        x_t = mean + self.sample(shape=mean.shape) * std
        action = ops.tanh(x_t)
        return action

    def eval_produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_local(state)
        mean, _ = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        return ops.tanh(mean)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
            self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
                "update_every_n_steps"] == 0

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        next_q_value = self.pre_calculate_next_q_value(reward_batch, next_state_batch, mask_batch)
        # qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch,
        #                                                   mask_batch)
        # self.calculate_qf1_grads(state_batch, action_batch, next_q_value)
        _, qf1_grads = self.q1_grad_fn(state_batch, action_batch, next_q_value)
        _, qf2_grads = self.q2_grad_fn(state_batch, action_batch, next_q_value)
        # self.update_critic_parameters(
        #     critic_loss_1=qf1_loss, critic_loss_2=qf2_loss, critic_grads_1=qf1_grads, critic_grads_2=qf2_grads
        # )
        self.update_critic_parameters(
            # critic_loss_1=qf1_loss, critic_loss_2=qf2_loss,
            critic_grads_1=qf1_grads, critic_grads_2=qf2_grads
        )

        (_, log_pi), policy_grads = self.actor_grad_fn(state_batch)
        if self.automatic_entropy_tuning:
            # alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
            _, alpha_grads = self.alpha_grad_fn(log_pi)
        else:
            alpha_grads = None
        self.update_actor_parameters(
            # actor_loss=policy_loss, alpha_loss=alpha_loss,
            actor_grads=policy_grads, alpha_grads=alpha_grads
        )

    def sample_experiences(self):
        """Sample"""
        return self.memory.sample()

    def pre_calculate_next_q_value(self, reward_batch, next_state_batch, mask_batch):
        """
        pre caculate next q value
        """
        # with torch.no_grad():
        # next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch)
        next_state_action, next_state_log_pi = self.action_log_prob_produce_action_and_action_info(next_state_batch)
        qf1_next_target = self.critic_target_1(ops.cat((next_state_batch, next_state_action), axis=1))
        qf2_next_target = self.critic_target_2(ops.cat((next_state_batch, next_state_action), axis=1))
        min_qf_next_target = ops.stack((qf1_next_target, qf2_next_target)).min(axis=0) - self.alpha * next_state_log_pi
        next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * min_qf_next_target
        return next_q_value

    def calculate_qf1_grads(self, state_batch, action_batch, next_q_value):
        """Calculates qf1 gradient"""
        qf1 = self.critic_local_1(ops.cat((state_batch, action_batch), axis=1))
        qf1_loss = self.mse_loss(qf1, next_q_value)
        return qf1_loss

    def calculate_qf2_grads(self, state_batch, action_batch, next_q_value):
        """Calculates qf2 gradient"""
        qf2 = self.critic_local_2(ops.cat((state_batch, action_batch), axis=1))
        qf2_loss = self.mse_loss(qf2, next_q_value)
        return qf2_loss

    # def calculate_critic_losses(
    #         self, state_batch, action_batch, next_q_value
    # ):
    #     """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
    #      term is taken into account"""
    #     qf1 = self.critic_local(ops.cat((state_batch, action_batch), axis=1))
    #     qf2 = self.critic_local_2(ops.cat((state_batch, action_batch), axis=1))
    #     qf1_loss = ops.mse_loss(qf1, next_q_value)
    #     qf2_loss = ops.mse_loss(qf2, next_q_value)
    #     return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        # action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        action, log_pi = self.action_log_prob_produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local_1(ops.cat((state_batch, action), axis=1))
        qf2_pi = self.critic_local_2(ops.cat((state_batch, action), axis=1))
        # min_qf_pi = torch.min(qf1_pi, qf2_pi)
        min_qf_pi = ops.stack((qf1_pi, qf2_pi)).min(axis=0)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if
        self.automatic_entropy_tuning is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy)).mean()
        return alpha_loss

    def update_critic_parameters(self, critic_grads_1, critic_grads_2):
        """Updates the parameters for both critics"""
        self.take_optimisation_step(self.critic_optimizer_1, critic_grads_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, critic_grads_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local_1, self.critic_target_1,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_grads, alpha_grads):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, actor_grads,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_grads is not None:
            self.take_optimisation_step(self.alpha_optim, alpha_grads, None)
            self.alpha = self.log_alpha.exp()

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print(f"Episode score {self.total_episode_score_so_far} ")
        print("----------------------------")
