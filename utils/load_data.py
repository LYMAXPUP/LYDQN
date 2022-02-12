import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice


def build_network(file_network):
    G = nx.Graph()
    with open(file_network, 'r') as fr:
        lines = fr.readlines()
    for line in islice(lines, 1, None):
        n = line.strip().split(',')
        e = (n[0], n[1])
        if e in G.edges():
            G.add_edge(e[0], e[1], weight=max(G[e[0]][e[1]]['weight'], float(n[2])))
        else:
            G.add_edge(e[0], e[1], weight=float(n[2]))
    return G


def load_clusters(file_clusters):
    Cs = []
    with open(file_clusters, 'r') as fr:
        lines = fr.readlines()
    for line in lines:
        nodes = line.strip().split(',')
        C = [n for n in nodes]
        Cs.append(C)
    return np.array(Cs, dtype=list)


def draw_network(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=10)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    plt.show()
    # nx.write_gexf(G, path="network.gexf")


if __name__ == '__main__':
    G = build_network(file_network="../datasets/Net3_target.csv")
    Cs = load_clusters(file_clusters="../datasets/Cluster3_old.csv")
    # draw_network(G)