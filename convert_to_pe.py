import pandas as pd
import networkx as nx



def load_tree(file_path='../../CHE/data/tree2_hamilton'):
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

def load_graph(file_path='../../CHE/data/edges_hamilton.txt'):
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


def load_label(file_path='../../CHE/data/flag_hamilton.txt'):
    df = pd.read_csv(file_path, header=None, sep='\t')
    ids = []
    labels = []
    for i in range(len(df)):
        ids.append(df.iloc[i,0])
        labels.append(df.iloc[i,1])
    return ids, labels


name = "amherst"
Tg, n, m = load_tree(f"../../CHE/data/tree2_{name}")
Cg = load_graph(f"../../CHE/data/edges_{name}.txt")

#### 构建闭包
id1 = []
id2 = []
weight = []
for edge in Tg.edges:
    id1.append(edge[1])
    id2.append(edge[0])
    weight.append(1)
for edge in Cg.edges:
    id1.append(edge[1])
    id2.append(edge[0])
    weight.append(1)
D = pd.DataFrame()
D['id1'] = id1
D['id2'] = id2
D['weight'] = weight
D.to_csv(f"{name}_closure.csv", index=False, sep=',')
#### label文件

ids, labels = load_label(f"../../CHE/data/flag_{name}.txt")
D = pd.DataFrame()
D['id'] = ids
D['label'] = labels
D.to_csv(f"{name}_label.csv", index=False, sep=',')
