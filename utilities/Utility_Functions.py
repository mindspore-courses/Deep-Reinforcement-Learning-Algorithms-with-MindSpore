"""Function"""
from abc import ABCMeta
import numpy as np
import mindspore as ms
import mindspore.nn.probability.distribution as msd
from mindspore import ops


# import torch
# from torch.distributions import Categorical, normal, MultivariateNormal

def abstract(cls):
    """Abstract class"""
    return ABCMeta(cls.__name__, cls.__bases__, dict(cls.__dict__))


def save_score_results(file_path, results):
    """Saves results as a numpy file at given path"""
    np.save(file_path, results)


def normalise_rewards(rewards):
    """Normalises rewards to mean 0 and standard deviation 1"""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return (rewards - mean_reward) / (std_reward + 1e-8)  # 1e-8 added for stability


def create_actor_distribution(action_types, actor_output, action_size):
    """Creates a distribution that the actor can then use to randomly draw actions"""
    eps = 1e-7
    if action_types == "DISCRETE":
        assert actor_output.shape[1] == action_size, "Actor output the wrong size"
        # action_distribution = Categorical(actor_output)  # this creates a distribution to sample from
        actor_output = ms.numpy.clip(actor_output, xmin=eps, xmax=1-eps)
        actor_output /= actor_output.sum(axis=1).unsqueeze(-1)
        action_distribution = msd.Categorical(actor_output)  # mindspore: this creates a distribution to sample from
    else:
        assert actor_output.shape[1] == action_size * 2, "Actor output the wrong size"
        means = actor_output[:, :action_size]
        stds = actor_output[:, action_size:]
        if len(means.shape) == 2:
            means = means.squeeze(-1)
        if len(stds.shape) == 2:
            stds = stds.squeeze(-1)
        if len(stds.shape) > 1 or len(means.shape) > 1:
            raise ValueError(f"Wrong mean and std shapes - {stds.shape} -- {means.shape}")
        action_distribution = msd.Normal(means, ops.abs(stds)+eps)
    return action_distribution


# class SharedAdam(ms.nn.Adam):
#     """Creates an adam optimizer object that is shareable between processes. Useful for algorithms like A3C. Code
#     taken from https://github.com/ikostrikov/pytorch-a3c/blob/master/my_optim.py"""
#     # 由于没有multiprocessing，暂不实现该算法
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
#         super(SharedAdam, self).__init__(params, learning_rate=lr, beta1=betas[0], beta2=betas[1], eps=eps,
#                                          weight_decay=weight_decay, use_amsgrad=amsgrad)
#         self.params_memory = list()
#         for i, p in enumerate(params):
#             self.parameters.append(
#                 {
#                     'step': ops.zeros(1),
#                     'exp_avg': ops.zeros_like(p.data),
#                     'exp_avg_sq': ops.zeros_like(p.data),
#                 }
#             )
#
#     def share_memory(self):
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'].share_memory_()
#                 state['exp_avg'].share_memory_()
#                 state['exp_avg_sq'].share_memory_()
#
#     @jit
#     def construct(self, gradients):
#         """Performs a single optimization step.
#         """
#         params = self._parameters
#         for i, p in enumerate(params):
#             if not p.requires_grad:
#                 continue
#             grad = gradients[i]
#             amsgrad = self.use_amsgrad
#             state = self.params_memory[i]
#             exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#
#             beta1, beta2 = self.beta1, self.beta2
#             state['step'] += 1
#
#             weight_decay = self.get_weight_decay()
#             if weight_decay != 0:
#                 grad += weight_decay * p.data
#             # Decay the first and second moment running average coefficient
#             exp_avg = exp_avg * beta1 + (1 - beta1) * grad
#             exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * grad * grad
#
#             if amsgrad:
#                 # 事实上，这里从未定义过
#                 max_exp_avg_sq = state['max_exp_avg_sq']
#                 # Maintains the maximum of all 2nd moment running avg. till now
#                 max_exp_avg_sq = ops.max(ops.vstack((max_exp_avg_sq.unsqueeze(0), exp_avg_sq.unsqueeze(0))), axis=0)[0]
#                 # Use the max. for normalizing running avg. of gradient
#                 denom = max_exp_avg_sq.sqrt() + self.eps
#             else:
#                 denom = exp_avg_sq.sqrt() + self.eps
#
#             bias_correction1 = 1 - beta1 ** state['step'].item()
#             bias_correction2 = 1 - beta2 ** state['step'].item()
#             step_size = self.get_lr() * ms.numpy.sqrt(bias_correction2) / bias_correction1
#
#             p.data = p.data.addcdiv(value=-step_size, tensor1=exp_avg, tensor2=denom)


def flatten_action_id_to_actions(action_id_to_actions, global_action_id_to_primitive_action, num_primitive_actions):
    """Converts the values in an action_id_to_actions dictionary back to the primitive actions they represent"""
    flattened_action_id_to_actions = {}
    for key in action_id_to_actions.keys():
        actions = action_id_to_actions[key]
        raw_actions = backtrack_action_to_primitive_actions(actions, global_action_id_to_primitive_action,
                                                            num_primitive_actions)
        flattened_action_id_to_actions[key] = raw_actions
    return flattened_action_id_to_actions


def backtrack_action_to_primitive_actions(action_tuple, global_action_id_to_primitive_action, num_primitive_actions):
    """Converts an action tuple back to the primitive actions it represents in a recursive way."""
    print("Recursing to backtrack on ", action_tuple)
    primitive_actions = range(num_primitive_actions)
    if all(action in primitive_actions for action in action_tuple):
        return action_tuple  # base case
    new_action_tuple = []
    for action in action_tuple:
        if action in primitive_actions:
            new_action_tuple.append(action)
        else:
            converted_action = global_action_id_to_primitive_action[action]
            print(new_action_tuple)
            new_action_tuple.extend(converted_action)
            print("Should have changed: ", new_action_tuple)
    new_action_tuple = tuple(new_action_tuple)
    return new_action_tuple
