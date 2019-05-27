import random
import data_processing
import paths_logic

cells = data_processing.load_grid("grid_file.csv")
init_states = data_processing.load_states_info("init_cells.txt")
goal_states = data_processing.load_states_info("goal_cells.txt")
cells_rewards = data_processing.compute_states_rewards(cells, goal_states)

values = paths_logic.value_iteration(cells, cells_rewards)
norm_values = paths_logic.normalize_values(values)
init_state = random.sample(init_states, 1)[0]
paths_logic.get_best_route_path(cells, norm_values, init_state, goal_states)
