"""
Four Rooms
"""
# from agents.DQN_agents.DDQN import DDQN
# from agents.hierarchical_agents.SNN_HRL import SNN_HRL
from agents.hierarchical_agents.DIAYN import DIAYN
from agents.Trainer import Trainer
from environments.Four_Rooms_Environment import Four_Rooms_Environment
from utilities.data_structures.Config import Config

config = Config()
config.seed = 1

height = 15
width = 15
random_goal_place = False
num_possible_states = (height * width) ** (1 + 1 * random_goal_place)
embedding_dimensions = [[num_possible_states, 20]]
print("Num possible states ", num_possible_states)

config.environment = Four_Rooms_Environment(
    height, width, stochastic_actions_probability=0.0, random_start_user_place=True,
    random_goal_place=random_goal_place
)

config.num_episodes_to_run = 1000
config.file_to_save_data_results = "results/data_and_graphs/Four_Rooms.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/Four_Rooms.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False
clip_rewards = False

actor_critic_agent_hyperparameters = {
        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [128, 128],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [128, 128],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True,
        "clip_rewards": clip_rewards
    }

dqn_agent_hyperparameters = {
    "learning_rate": 0.005,
    "batch_size": 128,
    "buffer_size": 40000,
    "epsilon": 1.0,
    "epsilon_decay_rate_denominator": 3,
    "discount_rate": 0.99,
    "tau": 0.01,
    "alpha_prioritised_replay": 0.6,
    "beta_prioritised_replay": 0.1,
    "incremental_td_error": 1e-8,
    "update_every_n_steps": 3,
    "linear_hidden_units": [30, 15],
    "final_layer_activation": "None",
    "batch_norm": False,
    "gradient_clipping_norm": 5,
    "clip_rewards": False,
    "learning_iterations": 2
}

manager_hyperparameters = dqn_agent_hyperparameters
manager_hyperparameters.update({"timesteps_to_give_up_control_for": 5})

config.hyperparameters = {
    "DQN_Agents": {
        "linear_hidden_units": [30, 10],
        "learning_rate": 0.01,
        "buffer_size": 40000,
        "batch_size": 256,
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": embedding_dimensions,
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "update_every_n_steps": 1,
        "epsilon_decay_rate_denominator": 10,
        "discount_rate": 0.99,
        "tau": 0.01,
        "exploration_cycle_episodes_length": None,
        "learning_iterations": 1,
        "clip_rewards": False
    },

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

        "MANAGER": {
            "timesteps_before_changing_skill": 6,
            # "linear_hidden_units": [10, 5],
            "linear_hidden_units": [128, 128],
            # "learning_rate": 0.01,
            "learning_rate": 0.01,
            # "buffer_size": 40000,
            "buffer_size": 4000000,
            "batch_size": 256,
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": embedding_dimensions,
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 50,
            "discount_rate": 0.99,
            "learning_iterations": 1,
            "tau": 0.01,
            "clip_rewards": False

        }

    },

    "Actor_Critic_Agents": {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],

        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": embedding_dimensions,
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 50.0,
        "normalise_rewards": True,
        "clip_rewards": False

    },

    "DIAYN": {
        "num_unsupservised_episodes": 100,
        "MANAGER": manager_hyperparameters,
        "num_skills": 5,
        "DISCRIMINATOR": {
            # "learning_rate": 0.01,
            # "linear_hidden_units": [20, 10],
            "learning_rate": 0.003,
            "linear_hidden_units": [128, 128],
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": embedding_dimensions,
            "final_layer_activation": None,
            "gradient_clipping_norm": 5,
        },

        "AGENT": actor_critic_agent_hyperparameters,
    },

    "HRL": {
        "linear_hidden_units": [10, 5],
        "learning_rate": 0.01,
        "buffer_size": 40000,
        "batch_size": 256,
        "final_layer_activation": "None",
        "columns_of_data_to_be_embedded": [0],
        "embedding_dimensions": embedding_dimensions,
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "update_every_n_steps": 1,
        "epsilon_decay_rate_denominator": 400,
        "discount_rate": 0.99,
        "learning_iterations": 1,
        "tau": 0.01

    }

}

if __name__ == '__main__':
    # AGENTS = [DDQN] #DIAYN] # A3C] #SNN_HRL] #, DDQN]
    AGENTS = [DIAYN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
