import random
import math
import numpy as np


class BaseValueIterationThief(object):

    @staticmethod
    def _compute_rms(v, new_v):
        num_states = len(v)
        sum_squares = 0

        for state in v:
            previous_value = v[state]
            new_value = new_v[state]
            values_square_diff = (new_value - previous_value) ** 2
            sum_squares += values_square_diff

        return math.sqrt(sum_squares) / num_states

    def __init__(self, grid_object):
        self.grid = grid_object
        self.path = []
        self.total_weight = 0
        self.v = {}

    def get_path(self):
        return self.path

    def get_path_weight(self):
        return self.total_weight

    def compute_path(self, init_state):
        current_state = init_state
        path = [current_state]
        path_weight = 0

        while True:
            neighbors = self.grid.get_state_neighbors(current_state)
            path_weight += self.grid.get_state_weight(current_state)
            found_goal_neighbor = False
            neighbors_values = []

            for n_state in neighbors:

                if n_state not in path:

                    if self.grid.is_goal_state(n_state):
                        path.append(n_state)
                        found_goal_neighbor = True
                        break

                    else:
                        neighbors_values.append((n_state, self.v[n_state]))

            if found_goal_neighbor:
                # found goal state - algorithm is done
                self.path = path
                self.total_weight = path_weight
                return True

            elif len(neighbors_values) > 0:
                # choose next state according to class policy
                next_state = self.choose_next_state(neighbors_values)
                path.append(next_state)
                current_state = next_state

            else:
                # if there is no valid neighbor - algorithm failed
                print(path)
                return False

    def choose_next_state(self, neighbors_values):
        raise NotImplementedError("A Sub class must implement the choose next state method")

    def value_iteration(self, gamma=0.99, convergence_rate=0.0001, debug=False):
        converged = False
        states = self.grid.get_states()
        states_rewards = self.grid.get_all_states_rewards()

        v = states_rewards
        num_iterations = 0

        while not converged:
            new_v = {}

            for s in states:
                neighbors = self.grid.get_state_neighbors(s)
                neighbors_values = [v[n] for n in neighbors]

                # Bellman Equation - where T(S,a,S') = 1 if S' is reachable from S
                new_v[s] = states_rewards[s] + gamma * max(neighbors_values)

            rms = self._compute_rms(v, new_v)
            v = new_v

            if debug:
                num_iterations += 1
                if num_iterations % 100 == 0:
                    print(f'current iteration for value iteration algorithm: {num_iterations}')

            if rms < convergence_rate:
                converged = True
                if debug:
                    print(f'finished value iteration after {num_iterations} iterations')

        self.v = v
        return v


class GreedyPathThief(BaseValueIterationThief):

    def __init__(self, grid_object):
        super(GreedyPathThief, self).__init__(grid_object)

    def choose_next_state(self, neighbors_values):
        # choose best neighbor according to states values
        best_neighbor, _ = max(neighbors_values, key=lambda x: x[1])
        return best_neighbor


class ProbabilisticGreedyPathThief(BaseValueIterationThief):

    def __init__(self, grid_object):
        super(ProbabilisticGreedyPathThief, self).__init__(grid_object)

    def normalize_states_values(self):
        state_values = list(self.v.values())
        values_mean = np.mean(state_values)
        value_std = np.std(state_values)

        self.v = {s: (self.v[s] - values_mean) / value_std for s in self.v}

    def compute_path(self, init_state):
        self.normalize_states_values()
        super(ProbabilisticGreedyPathThief, self).compute_path(init_state)

    def choose_next_state(self, neighbors_values):
        values_sum = sum([np.exp(value) for n, value in neighbors_values])
        values_distribution = [(n, np.exp(value) / values_sum) for n, value in neighbors_values]
        # values_sum = sum([np.exp(value) if value >= 1e-10 else 0 for n, value in neighbors_values])
        # values_distribution = [(n, np.exp(value) / values_sum) if value >= 1e-10 else (n, 0)
        #                        for n, value in neighbors_values]
        sorted_probabilities = sorted(values_distribution, key=lambda x: x[1], reverse=True)

        probability = random.random()
        next_state, next_state_prob = sorted_probabilities[0]
        cumulative_probability = next_state_prob
        for i in range(1, len(sorted_probabilities)):

            if probability <= cumulative_probability:
                return next_state

            else:
                next_state, next_state_prob = sorted_probabilities[i]
                cumulative_probability += next_state_prob

        return next_state
