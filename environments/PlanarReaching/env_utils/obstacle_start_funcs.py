import numpy as np


def obstacle_start_func_0(env):
    z_min = 5
    z_max = 8
    y_min = 1.3
    y_max = 1.5

    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)

    return ([env.obstacle_pos[0], y, z], env.obstacle_or)


obstacle_start_funcs = {
    "obstacle_start_func_0": obstacle_start_func_0,
}


# from matplotlib import pyplot as plt

# num_pts = 1000
# ys = np.zeros(num_pts)
# zs = np.zeros(num_pts)
# for i in range(1000):
#     [ys[i], zs[i]] = obstacle_start_func_0(None)

# fig, ax = plt.subplots()
# ax.scatter(ys, zs)
# ax.scatter([1.5], [6])
# ax.set_ylim(0, 12)
# ax.set_xlim(-8, 8)
# ax.set_aspect(1)
# plt.show()
