import pandas as pd
import numpy as np
import networkx as nx
# files = ['../poincare-embeddings/wordnet/noun_closure.csv', '../poincare-embeddings/wordnet/noun_label.csv']
files = ['../poincare-embeddings/wordnet/mammal_closure.csv', '../poincare-embeddings/wordnet/mammal_label.csv']
# root_name = "entity.n.01"
root_name = "mammal.n.01"
# name = 'mammal'
# name = 'wordnet'
G = nx.DiGraph()
f = pd.read_csv(files[0], header=0, sep=',')
d = pd.DataFrame()
d['parent'] = f['id2']
d['child'] = f['id1']
d['w'] = f['weight']
d = d[~(d['parent'] == d['child'])]
G.add_weighted_edges_from(d.values)
leafs = []
no_leafs = []
for node in G.node:
    outdegree = G.out_degree(node)
    if outdegree == 0:
        leafs.append(node)
    elif node != root_name:
        no_leafs.append(node)

i = 0
node_dict = dict()
for node in leafs:
    node_dict[node] = i
    i = i+1
for node in no_leafs:
    node_dict[node] = i
    i=i+1
node_dict[root_name] = i
no_leafs.append(root_name)

f = pd.read_csv(files[1], header=0, sep=',')
name_ids = []
labels = []
for i in range(len(f)):
    if node_dict[f.iloc[i,0]] >= len(leafs):
        continue
    name_ids.append(node_dict[f.iloc[i,0]])
    labels.append(f.iloc[i,1])

d = pd.DataFrame()
d['id'] = name_ids
d['label'] = labels
# d.to_csv('./data/flag_wordnet.txt', header=False, index=False, sep='\t')
d.to_csv('./data/flag_mammal.txt', header=False, index=False, sep='\t')


e1 = [len(leafs)+len(no_leafs)]
e2 = [len(leafs)]
for edge in G.edges():
    e1.append(node_dict[edge[0]])
    e2.append(node_dict[edge[1]])

d = pd.DataFrame()
d['p'] = e1
d['c'] = e2
# d.to_csv('./data/tree2_wordnet', header=False, index=False, sep='\t')
d.to_csv('./data/tree2_mammal', header=False, index=False, sep='\t')


tl=[]
for l in leafs:
    tl.append(node_dict[l])
tl.sort()

d = pd.DataFrame()
d['p'] = tl
d['c'] = tl
# d.to_csv('./data/edges_wordnet.txt', header=False, index=False, sep='\t')
d.to_csv('./data/edges_mammal.txt', header=False, index=False, sep='\t')












