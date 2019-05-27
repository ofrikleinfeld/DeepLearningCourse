import math
import numpy as np

DEFAULT_CONVERGENCE_RATE = 0.0001
GAMMA = 0.99


def root_mean_square(v, new_v):
    num_states = len(v)
    sum_squares = 0

    for cell_id in v:
        previous_cell_value = v[cell_id]
        new_cell_value = new_v[cell_id]
        values_square_diff = (new_cell_value - previous_cell_value) ** 2
        sum_squares += values_square_diff

    return math.sqrt(sum_squares) / num_states


def normalize_values(values_dict):
    values = list(values_dict.values())
    values_mean = np.mean(values)
    values_std = np.std(values)

    norm_values = {cell_id: (values_dict[cell_id] - values_mean) / values_std for cell_id in values_dict}
    return norm_values


def value_iteration(cells_dict, cells_rewards, convergence_rate=DEFAULT_CONVERGENCE_RATE):
    converged = False
    v = {cell_id: cells_rewards[cell_id] for cell_id in cells_dict}

    num_iterations = 0

    while not converged:
        new_v = {}

        for cell_id in cells_dict:
            neighbors = cells_dict[cell_id]["neighbors"]
            neighbors_values = [v[n] for n in neighbors]

            # Bellman Equation - where T(S,a,S') = 1 if S' is reachable from S
            new_v[cell_id] = cells_rewards[cell_id] + GAMMA * max(neighbors_values)

        rms = root_mean_square(v, new_v)
        v = new_v
        if rms < convergence_rate:
            print(f'finished after iteration {num_iterations}')
            converged = True

        num_iterations += 1
        if num_iterations % 100 == 0:
            print(f'current iteration {num_iterations}')

    return v


def get_best_route_path(cells_dict, cells_values, init_state, goal_states):
    path = [init_state]
    path_total_weight = 0
    reached_goal = False
    current_state = init_state

    while not reached_goal:
        path_total_weight += int(cells_dict[current_state]["weight"])
        neighbors = cells_dict[current_state]["neighbors"]
        found_goal_neighbor = False
        neighbors_values = []

        for neighbor_id in neighbors:
            if neighbor_id in goal_states and neighbor_id not in path:
                path.append(neighbor_id)
                found_goal_neighbor = True
                break
            else:
                neighbors_values.append((neighbor_id, cells_values[neighbor_id]))

        if found_goal_neighbor:
            reached_goal = True

        else:
            sorted_neighbors_values = sorted(neighbors_values, key=lambda x: x[1], reverse=True)
            has_possible_state = False
            for cell_id, _ in sorted_neighbors_values:
                if cell_id not in path:
                    path.append(cell_id)
                    current_state = cell_id
                    has_possible_state = True
                    break

            if not has_possible_state:
                print(f'got stuck!!!!!!')
                print(path)
                exit(-1)

    print(path)
    print(f'path length is: {len(path)}')
    print(f'path total weight is: {path_total_weight}')
    return path


