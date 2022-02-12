import numpy as np
from env import NetEnv, Flow


def permute(nums):
    """生成长度为num的所有排列组合
    """
    def dfs(nums, size, depth, path, used, res):
        if depth == size:
            res.append(path[:])
            return
        for i in range(size):
            if not used[i]:
                used[i] = True
                path.append(nums[i])
                dfs(nums, size, depth + 1, path, used, res)
                used[i] = False
                path.pop()

    size = len(nums)
    used = [False for _ in range(size)]
    res = []
    dfs(nums, size, 0, [], used, res)
    return res


def best(seed):
    max_incomes = float("-inf")
    max_sequence = []
    max_all_invests = 0.
    max_ROI = 0.
    res = permute([i for i in range(env.num_clusters)])

    for seq in res:
        env.reset(seed)
        done = 0
        incomes = 0.
        all_invests = 0
        index = 0
        sequence = []
        steps = 1
        for i, invest in enumerate(env.invests):
            actions = []
            while invest > 0 and not done:
                mask = env.mask * np.array(env.costs <= invest)
                if index >= len(seq) or mask[seq[index]] == 0:
                    break
                invest -= env.costs[seq[index]]
                all_invests += env.costs[seq[index]]
                done = env.step(seq[index])
                actions.append(seq[index])
                index += 1
            incomes += Flow(env).get_incomes()
            env.reqs = Flow(env).one_episode_growth()
            sequence.append(actions)
            if done:
                steps = i+1
                break
        if done:
            ROI = incomes / all_invests + 1000 / steps   # 只要加上一个很大的权重，就代表优先目标是尽早结束
            if ROI > max_ROI:
                max_sequence = sequence
                max_incomes = incomes
                max_all_invests = all_invests
                max_ROI = ROI
    print("sequence=", max_sequence)
    print("incomes=", max_incomes)
    print("all_invest=", max_all_invests)
    print("ROI=", max_ROI)


if __name__== "__main__":
    _NET = 1
    invest = [50, 20, 30, 20, 30, 30]
    speed = [1.15, 1.25, 1.3, 1.2, 1.25]

    # _NET = 2
    # invest = [110, 110, 110, 110, 110, 110, 110]
    # speed = [1.3, 1.2, 1.2, 1.25, 1.3, 1.15, 1.25]

    # sequence= [[5, 6], [1, 3], [0], [4], [2]]
    # incomes= 3020.1909121599974
    # all_invest= 427.0
    # ROI= 207.0730466326932

    # _NET = 3
    # invest = [550, 550]
    # speed = [1.5, 1.3, 1.3, 1.05]

    env = NetEnv(_NET, invest, speed)
    for k in range(10):
        best(k)