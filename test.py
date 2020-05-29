import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from utils.Data import testSet
from torch.utils.data import DataLoader
from metric.metric import reconstruction_worker, link_prediction_worker
name = 'amherst'
# net = torch.load(f'./data/{name}_16_test.pkl')
net = torch.load(f'./data/lp_{name}_16_s9.pkl')

device = torch.device('cuda:0')
net.to(device)
net.eval()
flag_file = f"./data/flag_{name}.txt"
edge_file = f"./data/edges_{name}.txt"
tree_file = f"./data/tree2_{name}"
ts = testSet()
ts.trav(flag_file)

split_rate = 0.9

class Classficator(nn.Module):
    def __init__(self, dim, m_dim):
        super(Classficator, self).__init__()
        self.l1 = nn.Linear(dim, m_dim)
        self.ac1 = nn.ReLU()
        self.l3 = nn.Linear(m_dim, 40)
        self.ac3 = nn.ReLU()
        self.l2 = nn.Linear(40, 6)
        # self.ac2 = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.l2(self.ac3(self.l3(self.ac1(self.l1(x)))))

LRmodel = Classficator(32, 64)
LRmodel.to(device)
LRmodel.train()
optim = Adam(LRmodel.parameters(), lr=0.0001)

cri = nn.CrossEntropyLoss()
test_len = int(len(ts)*split_rate)
train_set, test_set = torch.utils.data.random_split(ts, [len(ts)-test_len, test_len])
train_data = DataLoader(train_set, batch_size=256, drop_last=False)

test_data = DataLoader(test_set, batch_size=16, drop_last=False)
acc_rates = []

import utils.treeGraph as tg
import os
cg = tg.load_graph(edge_file)
g, n, m = tg.load_tree(tree_file)
adj = cg.adj
objects = np.array(list(adj.keys()))
print(objects)
# r = reconstruction_worker(adj, net.model, objects)
# print(r[0]/r[1], ' ', r[2]/r[3])

r = link_prediction_worker(adj, net.model, objects, len(cg.nodes))
print(r[0]/r[1], ' ', r[2]/r[3])

'''

for e in range(1):
    LRmodel.train()
    for b, data in enumerate(train_data):
        x, y = data[0], data[1]
        x, y = x.to(device), y.to(device)
        real = net.model.complex_coordinates_real(x)
        img = net.model.complex_coordinates_img(x)
        feature = torch.cat((real, img), dim=-1)
        
        logit = LRmodel(feature)
        loss = cri(logit, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        if (b % 1000 == 1):
            print(e, " ", b, " ", loss.item())


        # break
    LRmodel.eval()
    acc = 0
    for b, data in enumerate(test_data):
        x, y = data[0], data[1]
        x, y = x.to(device), y.to(device)
        real = net.model.complex_coordinates_real(x)
        img = net.model.complex_coordinates_img(x)
        feature = torch.cat((real, img), dim=-1)
        
        logit = LRmodel(feature)
        acc += (torch.max(logit, dim=-1)[1] == y).sum().item()

    acc_rate = acc / float(len(test_set))
    acc_rates.append(acc_rate)
    print(acc_rate)
    # break

print(max(acc_rates))
print(acc_rates)


'''