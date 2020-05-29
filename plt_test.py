import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from utils.Data import testSet
from torch.utils.data import DataLoader
from metric.metric import reconstruction_worker
name = 'mammal'
net = torch.load(f'./data/{name}_1_best_s98.pkl')
device = torch.device('cuda:0')
net.to(device)
net.eval()
flag_file = f"./data/flag_{name}.txt"
edge_file = f"./data/edges_{name}.txt"
tree_file = f"./data/tree2_{name}"
ts = testSet()
ts.trav(flag_file)

split_rate = 0.9

import utils.treeGraph as tg
import os
cg = tg.load_graph(edge_file)
g, n, m = tg.load_tree(tree_file)
adj = cg.adj
objects = np.array(list(adj.keys()))
# r = reconstruction_worker(adj, net.model, objects)
# print(r[0]/r[1], ' ', r[2]/r[3])

# real = net.model.complex_coordinates_real(x)
# img = net.model.complex_coordinates_img(x)
# feature = torch.cat((real, img), dim=-1)

node_ids = torch.tensor(list(g.node)).to(device)
real = net.model.complex_coordinates_real(node_ids)
img = net.model.complex_coordinates_img(node_ids)

import matplotlib.pyplot as plt
p = []
c = []

def get_coor(net, edge):
    node_ids = torch.tensor([edge[0], edge[1]]).to(device)
    x = net.model.complex_coordinates_real(node_ids)
    y = net.model.complex_coordinates_img(node_ids)
    # x = real*torch.cos(img)
    # y = real*torch.sin(img)
    return x.detach().cpu().numpy(), y.detach().cpu().numpy()

plt.figure(figsize=(10,10))
for e in g.edges:
    x, y = get_coor(net, e)
    plt.scatter(x, y, s=50, c='b')
    plt.plot(x,y, 'r', linewidth=0.2)
plt.show()
plt.savefig('./plt/mammal-best-98-bigpoint.pdf', bbox_inches='tight', dpi=1000)


