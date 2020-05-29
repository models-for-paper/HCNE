import pandas as pd
import numpy as np
import networkx as nx
files = ['../poincare-embeddings/wordnet/noun_closure.csv', '../poincare-embeddings/wordnet/noun_label.csv']

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
    elif node != 'entity.n.01':
        no_leafs.append(node)

i = 0
node_dict = dict()
for node in leafs:
    node_dict[node] = i
    i = i+1
for node in no_leafs:
    node_dict[node] = i
    i=i+1
node_dict['entity.n.01'] = i
no_leafs.append('entity.n.01')

f = pd.read_csv(files[1], header=0, sep=',')
name_ids = []
labels = []
for i in range(len(f)):
    name_ids.append(node_dict[f.iloc[i,0]])
    labels.append(f.iloc[i,1])

d = pd.DataFrame()
d['id'] = name_ids
d['label'] = labels
d.to_csv('./data/flag_wordnet.txt', header=False, index=False, sep='\t')

e1 = [len(leafs)+len(no_leafs)]
e2 = [len(leafs)]
for edge in G.edges():
    e1.append(node_dict[edge[0]])
    e2.append(node_dict[edge[1]])

d = pd.DataFrame()
d['c'] = e1
d['p'] = e2
d.to_csv('./data/tree2_wordnet', header=False, index=False, sep='\t')


tl=[]
for l in leafs:
    tl.append(node_dict[l])
tl.sort()

d = pd.DataFrame()
d['c'] = tl
d['p'] = tl
d.to_csv('./data/edges_wordnet.txt', header=False, index=False, sep='\t')











