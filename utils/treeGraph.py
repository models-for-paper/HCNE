import networkx as nx
import numpy as np

def load_graph(file_path):
    G = nx.Graph()
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if len(items) != 2:
                continue
            G.add_edge(int(items[0]), int(items[1]))
    return G
def load_tree(file_path):
    # 一个有向图
    G = nx.DiGraph()
    n, m = None, None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if len(items) != 2:
                continue
            if n is None:
                # 第一个点记录下来，不放入图
                n, m = int(items[0]), int(items[1])
            else:
                G.add_edge(int(items[0]), int(items[1]))
    return G, n, m

def transfer_to_matrix(graph):
    n = graph.number_of_nodes()
    mat = np.zeros([n, n])
    for e in graph.edges():
        mat[e[0]][e[1]] = 1
        mat[e[1]][e[0]] = 1
    return mat