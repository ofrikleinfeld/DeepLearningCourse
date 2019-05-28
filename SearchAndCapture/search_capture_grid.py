import numpy as np


class BaseGrid(object):
    def __init__(self, states_dict, init_states, goal_states):
        self.states_dict = states_dict
        self.init_states = init_states
        self.goal_states = goal_states
        self.states_rewards = {}

    def get_states(self):
        return list(self.states_dict.keys())

    def get_init_states(self):
        return self.init_states

    def get_goal_states(self):
        return self.goal_states

    def get_all_states_rewards(self):
        return self.states_rewards

    def get_state_reward(self, state):
        return self.states_rewards.get(state, None)

    def get_state_neighbors(self, state):
        return self.states_dict[state]["neighbors"]

    def is_goal_state(self, state):
        return state in self.goal_states

    def get_state_weight(self, state):
        return int(self.states_dict[state]["weight"])

    def _set_reward_for_goal_states(self, goal_reward):
        self.states_rewards = {state: goal_reward for state in self.goal_states}

    def normalize_rewards(self):
        if len(self.states_rewards) > 0:
            reward_values = list(self.states_rewards.values())
            values_mean = np.mean(reward_values)
            values_std = np.std(reward_values)

            self.states_rewards = {state: (self.states_rewards[state] - values_mean) / values_std
                                   for state in self.states_rewards}


class WeightRankRewardGrid(BaseGrid):
    def __init__(self, states_dict, init_states, goal_states):
        super(WeightRankRewardGrid, self).__init__(states_dict, init_states, goal_states)

    def compute_states_rewards(self, goal_reward=10):
        self._set_reward_for_goal_states(goal_reward)

        for state in self.states_dict:

            if state not in self.states_rewards:
                normalize_weight = int(self.states_dict[state]["weight"]) / 100
                rank = int(self.states_dict[state]["rank"])
                reward = -1 * normalize_weight / (normalize_weight + rank)

                self.states_rewards[state] = reward

        return self.states_rewards


class WeightRankGrid(BaseGrid):

    def __init__(self, states_dict, init_states, goal_states):
        super(WeightRankGrid, self).__init__(states_dict, init_states, goal_states)

    def compute_states_rewards(self, goal_reward=10):
        self._set_reward_for_goal_states(goal_reward)

        for state in self.states_dict:

            if state not in self.states_rewards:
                normalize_weight = int(self.states_dict[state]["weight"]) / 100
                reward = -1 * normalize_weight

                self.states_rewards[state] = reward

        return self.states_rewards
