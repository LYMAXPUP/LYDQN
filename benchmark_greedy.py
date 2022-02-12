import numpy as np
from env import NetEnv, Flow


def try_step(env, action):
    capacity = env.capacitys_tar[action]
    feature = env.features_tar[action]
    r = env.BETA * (capacity / env.speed_cur[action]) + (1-env.BETA) * feature
    return r


def greedy(seed):
    env.reset(seed)
    print(env.reqs)
    done = 0
    incomes = 0.
    all_invests = 0.
    sequence = []
    for i, invest in enumerate(env.invests):
        actions = []
        while invest > 0 and not done:
            action = -1
            r_max = float("-inf")
            mask = env.mask * np.array(env.costs <= invest)
            for c_id, cluster in enumerate(env.Cs_cur):
                if mask[c_id] == 0:
                    continue
                r = try_step(env, c_id)
                if r_max < r:
                    action = c_id
                    r_max = r
            if action == -1:
                break
            invest -= env.costs[action]
            all_invests += env.costs[action]
            done = env.step(action)
            actions.append(action)
        incomes += Flow(env).get_incomes()
        env.reqs = Flow(env).one_episode_growth()
        sequence.append(actions)
        if done:
            ROI = incomes / all_invests + 1000 / (i+1)   # steps = i+1
            print("sequence=", sequence)
            print("incomes=", incomes)
            print("all_invest=", all_invests)
            print("ROI=", ROI)
            break


if __name__ == "__main__":
    _NET = 1
    invest = [50, 20, 30, 20, 30, 30]
    speed = [1.15, 1.25, 1.3, 1.2, 1.25]

    # _NET = 2
    # invest = [110, 110, 110, 110, 110, 110, 110]
    # speed = [1.3, 1.2, 1.2, 1.25, 1.3, 1.15, 1.25]

    # _NET = 3
    # # invest = [500, 550, 550, 300, 200]
    # invest = [500, 550, 550, 500]
    # speed = [1.6, 1.5, 1.4, 1.25, 1.4, 1.8, 1.4, 1.2, 1.15, 1.15]

    for k in range(1):
        env = NetEnv(_NET, invest, speed)
        greedy(seed=k)
















