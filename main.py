from env import NetEnv, Flow
from d3qn import D3QN
import numpy as np


def get_best_actions(env, invest, done):
    actions = [0 for _ in range(num_clusters)]    # =1表示选了这个动作
    env_mask = env.mask.copy()
    while invest > 0 and not done:
        mask = env_mask * np.array(env.costs <= invest)
        s = env.get_state(mask)
        a = rl.choose_action(s, mask, epsilon=1)
        if a == -1:
            break
        actions[a] = 1
        env_mask[a] = 0
        done = 1 if sum(env_mask) == 0 else 0
        invest -= env.costs[a]
    return actions


def train():
    for k in range(1000):     # 尝试不同环境：改变reqs/ invests/ speed
        env.reset()
        done = 0
        all_incomes = 0.
        all_invests = 0.
        all_best = []
        all = []
        for i, invest in enumerate(env.invests):
            best_actions = get_best_actions(env, invest, done)
            all_best.append(best_actions)

            actions = [0 for _ in range(num_clusters)]   # 选了哪个哪个为1
            stage_incomes_before = Flow(env).get_incomes()
            s_before = env.get_state(env.mask.copy())
            while invest > 0 and not done:
                mask = env.mask * np.array(env.costs <= invest)
                s = env.get_state(mask)
                a = rl.choose_action(s, mask, epsilon=0.9)
                if a == -1:   # 满足while的条件但是没有可选的a
                    break
                done = env.step(a)   # s_step拓扑变、capacity变、簇变、mask变；单步奖励为0
                invest -= env.costs[a]
                all_invests += env.costs[a]
                actions[a] = 1
            stage_incomes_after = Flow(env).get_incomes()
            r = stage_incomes_after - stage_incomes_before
            if done:
                r += (len(env.invests) - (i+1)) * 1000
            all_incomes += stage_incomes_after
            env.reqs = Flow(env).one_episode_growth()
            s_ = env.get_state(env.mask.copy())
            rl.replay_buffer.push(s_before, actions, r, s_, done, best_actions)
            all.append(actions)

            if rl.replay_buffer.memory_full:
                rl.learn()

            if done:
                steps = i + 1
                ROI = all_incomes / all_invests + 1000 / steps
                print('Episode: %i | ROI: %f | T: %i' % (k, ROI, steps))
                break
        if not done:
            print('Episode: %i | --- ' %(k))
    rl.save()


def eval():
    rl.load()
    for k in range(10):
        env.reset()
        done = 0
        all_incomes = 0.
        all_invests = 0.
        sequence = []
        for i, invest in enumerate(env.invests):
            actions = [0 for _ in range(num_clusters)]
            actions_list = []
            while invest > 0 and not done:
                mask = env.mask * np.array(env.costs <= invest)
                s = env.get_state(mask)
                a = rl.choose_action(s, mask, epsilon=1)
                if a == -1:
                    break
                done = env.step(a)
                all_invests += env.costs[a]
                invest -= env.costs[a]
                actions[a] = 1
                actions_list.append(a)
            all_incomes += Flow(env).get_incomes()
            env.reqs = Flow(env).one_episode_growth()
            sequence.append(actions_list)
            if done:
                steps = i + 1
                ROI = all_incomes / all_invests + 1000 / steps
                print('Episode: %i | ROI: %f | T: %i' % (k, ROI, steps))
                print("sequence=", sequence)
                break

        if not done:
            print('Episode: %i | --- ' %(k))


if __name__ == '__main__':
    _NET = 1
    invest = [50, 20, 30, 20, 30, 30]
    speed = [1.15, 1.25, 1.3, 1.2, 1.25]

    # _NET = 2
    # invest = [110, 110, 110, 110, 110, 110, 110]
    # speed = [1.3, 1.2, 1.2, 1.25, 1.3, 1.15, 1.25]

    # _NET = 3
    # invest = [500, 550, 550, 300, 200]
    # speed = [1.6, 1.5, 1.4, 1.25, 1.4, 1.8, 1.4, 1.2, 1.15, 1.15]

    env = NetEnv(_NET, invest, speed)
    num_clusters = env.num_clusters
    state_dim = 6 * num_clusters
    action_dim = num_clusters
    rl = D3QN(state_dim, action_dim)

    train()

    print("==============")
    print("start eval...")
    print("==============")
    eval()

