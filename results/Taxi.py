"""
taxi
"""
import gym
# from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.DQN import DQN
# from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config

height = 15
width = 15
random_goal_place = False
num_possible_states = (height * width) ** (1 + 1 * random_goal_place)
embedding_dimensions = [[num_possible_states, 20]]

config = Config()
config.seed = 1
config.environment = gym.make("CartPole-v0")
config.env_parameters = {}
config.num_episodes_to_run = 500
config.file_to_save_data_results = "results/data_and_graphs/texi/Cart_Pole_data.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/taxi/Cart_Poke.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False

# Loss is not drawing a random sample! otherwise wouldnt jump around that much!!

linear_hidden_units = [32, 32]
learning_rate = 0.005  # 0.001 taxi
buffer_size = 1000000
batch_size = 256
batch_norm = False
embedding_dimensionality = 10
gradient_clipping_norm = 0.5  # needs to be optimised
update_every_n_steps = 1
learning_iterations = 1
epsilon_decay_rate_denominator = 2  # 150
episodes_per_round = 50  # 80
discount_rate = 0.99
tau = 0.004
sequitur_k = 2
pre_training_learning_iterations_multiplier = 0
episodes_to_run_with_no_exploration = 0
action_balanced_replay_buffer = True
copy_over_hidden_layers = True

num_top_results_to_use = 10
action_frequency_required_in_top_results = 0.8

random_episodes_to_run = 0

action_length_reward_bonus = 0.0

only_train_new_actions = True
only_train_final_layer = True
reduce_macro_action_appearance_cutoff_throughout_training = False
add_1_macro_action_at_a_time = True

calculate_q_values_as_increments = True
abandon_ship = True
clip_rewards = True
use_relative_counts = True

config.debug_mode = False

config.hyperparameters = {
    "SNN_HRL": {
        "SKILL_AGENT": {
            "num_skills": 20,
            "regularisation_weight": 1.5,
            "visitations_decay": 0.9999,
            "episodes_for_pretraining": 300,
            "batch_size": 256,
            # "learning_rate": 0.001,
            "learning_rate": 0.001,
            # "buffer_size": 40000,
            "buffer_size": 4000000,
            # "linear_hidden_units": [20, 10],
            "linear_hidden_units": [128, 128],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1],
            "embedding_dimensions": [embedding_dimensions[0],
                                     [20, 6]],
            "batch_norm": False,
            "gradient_clipping_norm": 2,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 500,
            "discount_rate": 0.999,
            "learning_iterations": 1,
            "tau": 0.01,
            "clip_rewards": False
        },
    },
    "h_DQN": {
        "CONTROLLER": {
            "batch_size": 256,
            # "learning_rate": 0.01,
            "learning_rate": 0.01,
            # "buffer_size": 40000,
            "buffer_size": 400000,
            # "linear_hidden_units": [20, 10],
            "linear_hidden_units": [128, 128],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1],
            "embedding_dimensions": [[config.environment.observation_space.shape[0],
                                      max(4, int(config.environment.observation_space.shape[0] / 10.0))],
                                     [config.environment.observation_space.shape[0],
                                      max(4, int(config.environment.observation_space.shape[0] / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 1500,
            "discount_rate": 0.999,
            "learning_iterations": 1,
            "tau": 0.01
        },
        "META_CONTROLLER": {
            "batch_size": 256,
            # "learning_rate": 0.001,
            # "buffer_size": 40000,
            # "linear_hidden_units": [20, 10],
            "learning_rate": 0.001,
            "buffer_size": 400000,
            "linear_hidden_units": [128, 128],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": [[config.environment.observation_space.shape[0],
                                      max(4, int(config.environment.observation_space.shape[0] / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 2500,
            "discount_rate": 0.999,
            "learning_iterations": 1,
            "clip_rewards": False,
            "tau": 0.01
        },
        "clip_rewards": False,
    },

    "HRL": {
        "linear_hidden_units": linear_hidden_units,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "final_layer_activation": "None",
        # "columns_of_data_to_be_embedded": [0],
        # "embedding_dimensions": [[config.environment.observation_space.n, embedding_dimensionality]],
        "batch_norm": batch_norm,
        "gradient_clipping_norm": gradient_clipping_norm,
        "update_every_n_steps": update_every_n_steps,
        "epsilon_decay_rate_denominator": epsilon_decay_rate_denominator,
        "discount_rate": discount_rate,
        "learning_iterations": learning_iterations,
        "tau": tau,
        "sequitur_k": sequitur_k,
        "use_relative_counts": use_relative_counts,
        "action_length_reward_bonus": action_length_reward_bonus,
        "pre_training_learning_iterations_multiplier": pre_training_learning_iterations_multiplier,
        "episodes_to_run_with_no_exploration": episodes_to_run_with_no_exploration,
        "action_balanced_replay_buffer": action_balanced_replay_buffer,
        "copy_over_hidden_layers": copy_over_hidden_layers,
        "random_episodes_to_run": random_episodes_to_run,
        "only_train_new_actions": only_train_new_actions,
        "only_train_final_layer": only_train_final_layer,
        "num_top_results_to_use": num_top_results_to_use,
        "action_frequency_required_in_top_results": action_frequency_required_in_top_results,
        "reduce_macro_action_appearance_cutoff_throughout_training": reduce_macro_action_appearance_cutoff_throughout_training,
        "add_1_macro_action_at_a_time": add_1_macro_action_at_a_time,
        "calculate_q_values_as_increments": calculate_q_values_as_increments,
        "episodes_per_round": episodes_per_round,
        "abandon_ship": abandon_ship,
        "clip_rewards": clip_rewards
    },

    "DQN_Agents": {
        "linear_hidden_units": linear_hidden_units,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "final_layer_activation": "None",
        # "columns_of_data_to_be_embedded": [0],
        # "embedding_dimensions": [[config.environment.observation_space.n, embedding_dimensionality]],
        "batch_norm": batch_norm,
        "gradient_clipping_norm": gradient_clipping_norm,
        "update_every_n_steps": update_every_n_steps,
        "epsilon_decay_rate_denominator": epsilon_decay_rate_denominator,
        "discount_rate": discount_rate,
        "learning_iterations": learning_iterations,
        "tau": tau,
        "clip_rewards": clip_rewards
    },

    "Actor_Critic_Agents": {
        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            # "columns_of_data_to_be_embedded": [0],
            # "embedding_dimensions": [[config.environment.observation_space.n, embedding_dimensionality]],
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            # "columns_of_data_to_be_embedded": [0],
            # "embedding_dimensions": [[config.environment.observation_space.n, embedding_dimensionality]],
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 10000,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True,
        "clip_rewards": clip_rewards
    },
}

if __name__ == "__main__":
    # AGENTS = [DDQN] #] #DDQN, ,  ] #] ##  ] #, SAC_Discrete] #, DQN]
    AGENTS = [DQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
