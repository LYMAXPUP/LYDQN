import numpy as np
import networkx as nx
from utils.load_data import build_network, load_clusters


class NetEnv(object):
    ALPHA = 0.1  # 初始reqs = ALPHA * 节点度数和
    BETA = 0.7  # 收入=BETA * 流量和 + (1-BETA) * 节点度数和

    def __init__(self, net, invest, speed):
        self.G_old = build_network(file_network="datasets/Net" + str(net) + "_old.csv")
        self.G_tar = build_network(file_network="datasets/Net" + str(net) + "_target.csv")
        self.G_cur = nx.Graph()

        self.Cs_old = load_clusters(file_clusters="datasets/Cluster" + str(net) + "_old.csv")
        self.Cs_tar = load_clusters(file_clusters="datasets/Cluster" + str(net) + "_target.csv")
        self.num_clusters = len(self.Cs_old)
        self.Cs_cur = np.array([])

        self.capacitys_cur = np.array([])
        self.capacitys_tar = NetCompute().get_capacitys(self.G_tar, self.Cs_tar)

        self.reqs = np.array([])

        self.features_cur = np.array([])
        self.features_tar = NetCompute().get_features(self.G_tar, self.Cs_tar)

        self.speed_old = speed
        self.speed_cur = np.array([])

        self.mask = np.array([])  # 1可选，0不可选

        self.costs = NetCompute().init_costs(self.G_old, self.G_tar, self.Cs_old, self.Cs_tar)
        self.invests = invest

    def reset(self, seed=None):
        self.G_cur = self.G_old.copy()
        self.Cs_cur = self.Cs_old.copy()
        self.capacitys_cur = NetCompute().get_capacitys(self.G_cur, self.Cs_cur)
        self.features_cur = NetCompute().get_features(self.G_cur, self.Cs_cur)
        if seed is not None:
            self.reqs = Flow(self).init_reqs(seed)
        else:
            self.reqs = Flow(self).init_reqs()
        self.speed_cur = self.speed_old.copy()
        self.mask = np.ones(self.num_clusters)

    def get_state(self, mask):
        s = np.concatenate((self.reqs, self.speed_cur, self.capacitys_cur,
                            self.capacitys_tar, self.features_cur, self.features_tar))
        return s

    # 单步状态改变: G变、C变、capacity变、feature变、speed变、mask变
    def step(self, action):
        C_cur = self.Cs_cur[action]
        C_tar = self.Cs_tar[action]
        for i in C_cur:
            edges = self.G_cur[i]
            for j, wei in dict(edges).items():
                if j in C_cur:
                    self.G_cur.remove_edge(i, j)
        for node, neighbors in dict(self.G_cur.adj).items():    # 移除孤立节点
            if len(neighbors) == 0:
                self.G_cur.remove_node(node)
        for i in C_tar:
            edges = self.G_tar[i]
            for j, wei in dict(edges).items():
                if j in C_tar:
                    self.G_cur.add_edge(i, j, weight=wei['weight'])

        self.Cs_cur[action] = C_tar

        for c_id in range(self.num_clusters):
            # 1:1000, 2:1000, 3:5000
            self.speed_cur[c_id] += (self.capacitys_tar[action] - self.capacitys_cur[action]) / 1000

        self.capacitys_cur[action] = self.capacitys_tar[action]

        self.features_cur[action] = self.features_tar[action]

        self.mask[action] = 0
        done = 1 if sum(self.mask) == 0 else 0        # 1=True, 0=False

        return done


class Flow(object):
    def __init__(self, env):
        self.env = env

    def init_reqs(self, *args):
        # 初始流量分布=alpha * 度数和 + 随机数
        if args:
            np.random.seed(args)
        # 1: ALPHA = 0.1 / 1, 2: ALPHA = 0.1 / 5, 3: ALPHA = 0.02 / 5
        reqs = self.env.ALPHA * self.env.features_cur + 1 * np.random.rand(self.env.num_clusters)
        return reqs

    def one_episode_growth(self):
        # 所有簇的流量需求按速率v增长
        return np.multiply(self.env.reqs, self.env.speed_cur)

    def get_incomes(self):
        flow = 0.
        for c_id, cluster in enumerate(self.env.Cs_cur):
            if self.env.reqs[c_id] <= self.env.capacitys_cur[c_id]:
                flow += self.env.reqs[c_id]
            else:
                flow += self.env.capacitys_cur[c_id]
        return self.env.BETA * flow + (1 - self.env.BETA) * np.sum(self.env.features_cur)


class NetCompute(object):
    def init_costs(self, G_old, G_tar, Cs_old, Cs_tar):
        # 改造成本与新增链路数成正比
        costs = []
        for c1, c2 in zip(Cs_old, Cs_tar):
            edge_set1 = set()
            for n1 in c1:
                for n2 in G_old.neighbors(n1):
                    if n2 in c1:
                        weight = G_old[n1][n2]['weight']
                        edge_set1.add((n1, n2, weight))
                        edge_set1.add((n2, n1, weight))
            edge_set2 = set()
            for n1 in c2:
                for n2 in G_tar.neighbors(n1):
                    if n2 in c2:
                        weight = G_tar[n1][n2]['weight']
                        edge_set2.add((n1, n2, weight))
                        edge_set2.add((n2, n1, weight))
            new_edges = edge_set2 - edge_set1 & edge_set2
            cost = 0.
            for i, j, wei in new_edges:
                cost += wei
            costs.append(cost / 2.)
        return np.array(costs)

    def get_capacitys(self, G, Cs):
        capacitys = []
        for C in Cs:
            capacity = 0.
            for i in C:
                edges = G[i]
                for j, wei in dict(edges).items():
                    if j in C:
                        capacity += wei['weight']
            capacity = capacity / 2.
            capacitys.append(capacity)
        return np.array(capacitys)

    def get_features(self, G, Cs):
        features = []
        for C in Cs:
            feature = 0.
            for node in C:
                feature += self.weighted_degree(G, node)
            features.append(feature)
        return np.array(features)

    def weighted_degree(self, G, i):
        degree = 0.
        for j in nx.neighbors(G, i):
            degree += G[i][j]['weight']
        return degree






