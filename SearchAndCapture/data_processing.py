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
        states = set(f.readline()[:-1].split(","))
        states.remove("")

    return states
