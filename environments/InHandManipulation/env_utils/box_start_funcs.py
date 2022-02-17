import numpy as np
import pybullet as p


def box_start_func_0(env):
    d_max = 0.25

    while True:
        x = np.random.uniform(-d_max, d_max)
        y = np.random.uniform(-d_max, d_max)

        if np.linalg.norm(np.array([x, y])) <= d_max:
            box_start_or = [0, 0, np.random.uniform(0, np.pi / 2)]
            return (np.array([x, y, env.box_start_pos[2]]), np.array(box_start_or))


box_start_funcs = {
    "box_start_func_0": box_start_func_0,
}
