import networkx as nx
import numpy as np
from collections import defaultdict

def load_tree(file_path='./data/tree2_hamilton'):
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

def load_graph(file_path='./data/edges_hamilton.txt'):
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
def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("../cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = [float(x) for x in info[1:-1]]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("../cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def load_tree_sr():
    import networkx as nx
    Cg = load_graph()
    G, n, m = load_tree()
    num_nodes = len(G.node)
    leaf_num = len(Cg.node)
    feat_data = np.ones((num_nodes, 1), dtype=np.float)

    flags_dict = dict()
    with open("./data/flag_hamilton.txt" , "r") as file:
        for f in file:
            f = f.replace("\n", "")
            a = f.split('\t')
            if (len(a) == 2):
                flags_dict[int(a[0])] = int(a[1])
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    for i in range(leaf_num):
        labels[i] = flags_dict[i]

    for j in range(leaf_num, num_nodes):
        labels[j] = -1

    adj_lists = defaultdict(set)
    for e in G.edges:
        adj_lists[e[0]].add(e[1])
        adj_lists[e[1]].add(e[0])
    for e in Cg.edges:
        adj_lists[e[0]].add(e[1])
        adj_lists[e[1]].add(e[0])
     return feat_data, labels, adj_lists