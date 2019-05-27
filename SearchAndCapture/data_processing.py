DESTINATION_REWARD = 10


def load_grid(file_name):
    cells = {}
    with open(file_name, "r") as f:
        # read headline
        f.readline()

        # read data lines
        for line in f:
            id_, weight, rank, neighbors = line.split(",")
            neighbors = neighbors[1:-2].split(";")
            cells[id_] = {"weight": weight, "rank": rank, "neighbors": neighbors}

    return cells


def load_states_info(file_name):
    with open(file_name, "r") as f:
        states = set(f.readline().split(","))

    return states


def compute_states_rewards(cells_dict, goal_cells):
    rewards = {}
    for cell_id in cells_dict:
        if cell_id in goal_cells:
            reward = DESTINATION_REWARD
        else:
            normalize_weight = int(cells_dict[cell_id]["weight"]) / 100
            rank = int(cells_dict[cell_id]["rank"])
            reward = -1 * normalize_weight / (normalize_weight + rank)

        rewards[cell_id] = reward

    return rewards

