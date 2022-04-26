import numpy as np
import pybullet as p


# todo: give a better name (one that more clearly describes what this function does / when it should be used)
def target_start_func_0(env):
    # y = 4
    # z = -2
    # return [0.4, y, z], env.target_start_or

    rz_min = 6
    rz_max = 10
    ry_min = 3
    ry_max = 7

    target_x_orientation = None

    def in_bounds(y, z):
        inside_outer = ((y ** 2) / (ry_max ** 2)) + ((z ** 2) / (rz_max ** 2)) <= 1
        outside_inner = ((y ** 2) / (ry_min ** 2)) + ((z ** 2) / (rz_min ** 2)) > 1
        return inside_outer and outside_inner and z > 1

    while True:
        y = np.random.uniform(-ry_max, ry_max)
        z = np.random.uniform(0, rz_max)
        if in_bounds(y, z):
            return (
                [env.target_start_pos[0], y, z],
                env.target_start_or,
                target_x_orientation,
            )


# todo: give a better name (one that more clearly describes what this function does / when it should be used)
def target_start_func_1(env):
    """
    todo: description
    """
    z_min = 5
    z_max = 8
    y_min = 2.75
    y_max = 4

    # while True:
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)
    # obstacle_dist = np.linalg.norm(np.array([y, z]) - np.array(env.obstacle_pos[1:]))
    # if obstacle_dist > 1:

    target_x_orientation = None
    return ([env.target_start_pos[0], y, z], env.target_start_or, target_x_orientation)


def target_start_random_pos_and_orX(env):
    """
    todo: description
    """
    z_min = 5
    z_max = 8
    y_min = 2.75
    y_max = 4

    # while True:
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)
    # obstacle_dist = np.linalg.norm(np.array([y, z]) - np.array(env.obstacle_pos[1:]))
    # if obstacle_dist > 1:

    target_x_orientation = np.random.uniform(-np.pi, np.pi)
    orig_target_start_or = env.run_config["target_start_or"]
    target_start_or = [
        target_x_orientation,
        orig_target_start_or[1],
        orig_target_start_or[2],
    ]
    target_or_quat = p.getQuaternionFromEuler(target_start_or)
    return ([env.target_start_pos[0], y, z], target_or_quat, target_x_orientation)


target_start_funcs = {
    "target_start_func_0": target_start_func_0,
    "target_start_func_1": target_start_func_1,
    "target_start_random_pos_and_orX": target_start_random_pos_and_orX,
}


# from matplotlib import pyplot as plt

# num_pts = 1000
# ys = np.zeros(num_pts)
# zs = np.zeros(num_pts)
# for i in range(1000):
#     [ys[i], zs[i]] = target_start_func_0(None)

# fig, ax = plt.subplots()
# ax.scatter(ys, zs)
# ax.scatter([1.5], [6])
# ax.set_ylim(0, 12)
# ax.set_xlim(-8, 8)
# ax.set_aspect(1)
# plt.show()
