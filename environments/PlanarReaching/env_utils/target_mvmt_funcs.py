import numpy as np
import pybullet as p


def target_mvmt_func_0(env):
    curr_target_pos = np.array(
        p.getBasePositionAndOrientation(bodyUniqueId=env.target_id)[0]
    ).flatten()
    curr_target_v = np.array(p.getBaseVelocity(bodyUniqueId=env.target_id)[0]).flatten()
    target_mass = 1
    delta_t = env.run_config["action_time"]

    max_dist = 1.2
    min_dist = 0.2

    def in_bounds(x, y):
        dist = np.sqrt((x ** 2) + (y ** 2))
        return min_dist <= dist <= max_dist

    if True:  # env.step_count % 2 == 0:
        F = np.array(list(np.random.uniform(-1, 1, 2)) + [0]) * 0.6
    else:
        F = np.array([0, 0, 0])

    for i in range(1000):
        new_a = F / target_mass
        new_v = curr_target_v + delta_t * new_a

        next_pos = curr_target_pos + delta_t * new_v

        x, y = next_pos[0:2]
        if in_bounds(x, y) and y > 0.25:
            print("in_bounds")
            p.resetBaseVelocity(env.target_id, linearVelocity=new_v)
            return

        F = np.array(list(np.random.uniform(-1, 1, 2)) + [0])

    print("ERROR: COULD NOT FIND SUITABLE TARGET MOVE")
    p.resetBaseVelocity(env.target_id, linearVelocity=-1 * curr_target_v)
    return


target_mvmt_funcs = {
    "target_mvmt_func_0": target_mvmt_func_0,
}
