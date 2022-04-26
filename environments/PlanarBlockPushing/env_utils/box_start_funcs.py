import numpy as np


def box_start_func_0(env):
    return (env.box_start_pos, env.box_start_or)


def box_start_func_1(env):
    box_pos = env.box_start_pos
    new_x = np.random.choice(np.array([box_pos[0], -1 * box_pos[0]]), size=1)[
        0
    ]  # todo: this should use env random number generator; same for all other random ops

    return ([new_x, box_pos[1], box_pos[2]], env.box_start_or)


def box_start_func_2(env):
    max_dist = 1
    min_dist = 0.5

    def in_bounds(x, y):
        dist = np.sqrt((x ** 2) + (y ** 2))
        return min_dist <= dist <= max_dist

    while True:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(0, 1)
        if in_bounds(x, y) and y > 0.25:
            return ([x, y, env.box_start_pos[2]], env.box_start_or)


box_start_funcs = {
    "box_start_func_0": box_start_func_0,
    "box_start_func_1": box_start_func_1,
    "box_start_func_2": box_start_func_2,
}
