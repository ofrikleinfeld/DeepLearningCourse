import math


class ValueIterationThief(object):

    def __init__(self, grid_object):
        self.grid = grid_object
        self.v = {}

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
                    print(f'current iteration {num_iterations}')

            if rms < convergence_rate:
                converged = True
                if debug:
                    print(f'finished after iteration {num_iterations}')

        self.v = v
        return v


class GreedyPathThief(ValueIterationThief):

    def __init__(self, grid_object):
        super(GreedyPathThief, self).__init__(grid_object)
        self.path = []
        self.total_weight = 0

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
                # choose best valued neighbor and move to it
                best_neighbors, _ = max(neighbors_values, key=lambda x: x[1])
                path.append(best_neighbors)
                current_state = best_neighbors

            else:
                # if there is no valid neighbor - algorithm failed
                return False
