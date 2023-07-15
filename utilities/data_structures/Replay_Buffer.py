"""
Replay Buffer
"""
from collections import namedtuple, deque
import random
import mindspore.numpy as np


class Replay_Buffer:
    """Replay buffer to store experiences that the agent can then use for training data"""

    def __init__(self, buffer_size, batch_size, seed):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # 因为random.seed不返回任何值
        # self.seed = random.seed(seed)
        random.seed(seed)

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        # if type(dones) == list:
        if isinstance(dones, list):
            # assert type(dones[0]) != list, "A done shouldn't be a list"
            assert not isinstance(dones[0], list), "A done shouldn't be a list"
            experiences = [
                self.experience(state, action, reward, next_state, done)
                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones)
            ]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        result = None

        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            result = states, actions, rewards, next_states, dones
        else:
            result = experiences

        return result

    def separate_out_data_types(self, experiences):
        """separate_out_data_types"""
        states, actions, rewards, next_states, dones = [], [], [], [], []
        states_append, actions_append, rewards_append, next_states_append, dones_append \
            = states.append, actions.append, rewards.append, next_states.append, dones.append
        for e in experiences:
            if e is not None:
                states_append(e.state)
                actions_append(e.action)
                rewards_append(e.reward)
                next_states_append(e.next_state)
                dones_append(int(e.done))

        states, actions, rewards = np.array(states), np.array(actions).unsqueeze(-1), np.array(rewards).unsqueeze(-1)
        next_states, dones = np.array(next_states), np.array(dones).unsqueeze(-1)

        actions = actions.squeeze(-1) if len(actions.shape) == 3 else actions

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def pick_experiences(self, num_experiences=None):
        """
        pick experience
        """
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
