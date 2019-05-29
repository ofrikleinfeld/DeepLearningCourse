import random

import data_processing
from search_capture_grid import WeightRankRewardGrid, WeightRankGrid
from thief_algorithms import GreedyPathThief, ProbabilisticGreedyPathThief

states = data_processing.load_grid("grid_file.csv")
init_states = data_processing.load_states_info("init_cells.txt")
goal_states = data_processing.load_states_info("goal_cells.txt")

# grid = WeightRankRewardGrid(states, init_states, goal_states)
grid = WeightRankGrid(states, init_states, goal_states )
grid.compute_states_rewards()
grid.normalize_rewards()

init_state = random.sample(init_states, 1)[0]
# greedy_thief = GreedyPathThief(grid)
# greedy_thief.value_iteration(debug=True)
probabilistic_thief = ProbabilisticGreedyPathThief(grid)
probabilistic_thief.value_iteration(debug=False)


valid_path = False
max_tries = 1000
current_try = 1
while not valid_path:
    valid_path = probabilistic_thief.compute_path(init_state)
    current_try += 1
    if current_try > max_tries:
        print(f"failed to find a valid path without loops. tried for {max_tries} iterations")
        break

current_try = 1
if not valid_path:
    print("trying greedy thief approach")
    greedy_thief = GreedyPathThief(grid)
    greedy_thief.value_iteration(debug=False)
    while not valid_path:
        valid_path = greedy_thief.compute_path(init_state)
        current_try += 1
        if current_try > max_tries:
            print(f"failed to find a valid path without loops also for greedy algorithm. tried for {max_tries} iterations")
            print(f"existing with error")
            break

if valid_path:
    path = probabilistic_thief.get_path()
    path_weight = probabilistic_thief.get_path_weight()
    print(f"the path from init state {init_state} to some goal state is:\n{path}")
    print(f"the path total weight is {path_weight}")
    print(f"the path total length is {len(path)}")
    data_processing.output_path_to_file(path)
