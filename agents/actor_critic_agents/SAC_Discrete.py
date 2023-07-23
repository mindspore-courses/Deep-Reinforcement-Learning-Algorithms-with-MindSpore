""""
SAC Discrete
"""
# pylint: disable=W0233
# pylint: disable=W0231
import mindspore as ms
from mindspore import ops, nn
import numpy as np
from agents.Base_Agent import Base_Agent
from agents.actor_critic_agents.SAC import SAC
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from utilities.Utility_Functions import create_actor_distribution


class SAC_Discrete(SAC):
    """The Soft Actor Critic for discrete actions. It inherits from SAC for continuous actions and only changes a few
    methods."""
    agent_name = "SAC"

    def __init__(self, config):
        # super().__init__(config)
        Base_Agent.__init__(self, config)
        assert self.action_types == "DISCRETE", "Action types must be discrete. Use SAC instead for continuous actions"
        assert self.config.hyperparameters["Actor"][
                   "final_layer_activation"] == "Softmax", "Final actor layer must be softmax"
        self.hyperparameters = config.hyperparameters
        self.critic_local_1 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                             key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                             key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer_1 = nn.Adam(
            self.critic_local_1.trainable_params(), learning_rate=self.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4
        )
        self.critic_optimizer_2 = nn.Adam(
            self.critic_local_2.trainable_params(), learning_rate=self.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4
        )
        self.critic_target_1 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                              key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                              key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local_1, self.critic_target_1)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(
            self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"], self.config.seed
        )

        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_optimizer = nn.Adam(
            self.actor_local.trainable_params(), learning_rate=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4
        )
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = ms.Parameter(ops.zeros(1), name='log_alpha')
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = nn.Adam(
                [self.log_alpha], learning_rate=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4
            )
            # grads
            self.alpha_grad_fn = ms.value_and_grad(
                self.calculate_entropy_tuning_loss, grad_position=None, weights=[self.log_alpha], has_aux=False
            )
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]
        assert not self.hyperparameters[
            "add_extra_noise"], "There is no add extra noise option for the discrete version of SAC at moment"
        self.add_extra_noise = False
        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

        self.q1_grad_fn = ms.value_and_grad(
            self.calculate_qf1_grads, grad_position=None, weights=self.critic_local_1.trainable_params(),
            has_aux=False
        )
        self.q2_grad_fn = ms.value_and_grad(
            self.calculate_qf2_grads, grad_position=None, weights=self.critic_local_2.trainable_params(),
            has_aux=False
        )
        self.actor_grad_fn = ms.value_and_grad(
            self.calculate_actor_loss, grad_position=None, weights=self.actor_local.trainable_params(),
            has_aux=True
        )

        self.critic_loss = ms.Tensor([0])

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        state = state.float()
        action_probabilities = self.actor_local(state)
        max_probability_action = ops.argmax(action_probabilities, dim=-1)
        action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = ops.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

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
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            # _, z, action = self.produce_action_and_action_info(state)
            _, _, action = self.produce_action_and_action_info(state)
        action = action.numpy()
        return action[0]

    def learn(self):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        _, (action_probabilities, log_action_probabilities), _ = \
            self.produce_action_and_action_info(next_state_batch)
        qf1_next_target = self.critic_target_1(next_state_batch)
        qf2_next_target = self.critic_target_2(next_state_batch)
        min_q_nex_target = ops.stack((qf1_next_target, qf2_next_target)).min(axis=0)
        min_qf_next_target = action_probabilities * (
                min_q_nex_target - self.alpha * log_action_probabilities
        )
        min_qf_next_target = min_qf_next_target.sum(axis=1).unsqueeze(-1)
        next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * min_qf_next_target

        _, qf1_grads = self.q1_grad_fn(state_batch, action_batch, next_q_value)
        _, qf2_grads = self.q2_grad_fn(state_batch, action_batch, next_q_value)
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

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        # with torch.no_grad():
        _, (
            action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
        qf1_next_target = self.critic_target_1(next_state_batch)
        qf2_next_target = self.critic_target_2(next_state_batch)
        min_qf_next_target = action_probabilities * (
                self.self_min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
        min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
        next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
            min_qf_next_target)

        qf1 = self.critic_local_1(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = ops.mse_loss(qf1, next_q_value)
        qf2_loss = ops.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        _, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local_1(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = ops.stack((qf1_pi, qf2_pi)).min(axis=0)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(axis=1).mean()
        log_action_probabilities = ops.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities

    def calculate_qf1_grads(self, state_batch, action_batch, next_q_value):
        qf1 = self.critic_local_1(state_batch).gather(action_batch.long(), axis=1, batch_dims=1)
        qf1_loss = ops.mse_loss(qf1, next_q_value)
        return qf1_loss

    def calculate_qf2_grads(self, state_batch, action_batch, next_q_value):
        qf2 = self.critic_local_2(state_batch).gather(action_batch.long(), axis=1, batch_dims=1)
        qf2_loss = ops.mse_loss(qf2, next_q_value)
        return qf2_loss
