import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import utils.treeGraph as tg
from model.CHE import CHE
import numpy as np
from model.CLS import CLS
from metric.metric import reconstruction_worker
from utils.Data import TreeDataset
from tqdm import trange, tqdm
from time import time

def testData(data_path, file_name, cfile):
    # 加载社区图
    print("loag net graphy from path:", os.path.join(data_path, cfile))
    cg = tg.load_graph(os.path.join(data_path, cfile))
    # 构建好图
    print("loag net graphy from path:", os.path.join(data_path, file_name))
    g, n, m = tg.load_tree(os.path.join(data_path, file_name))
    n = g.number_of_nodes()

    root_id = n - 1

    # 设置pad node
    pad_node_id = n
    n = n + 1

    # treeGraph, graph, node_num, leaf_num, root_id, negative_num, pad_id
    td = TreeDataset(g, cg, n, m, root_id, 20, pad_node_id)
    td.trav_tree()
    return td, n, pad_node_id

def mask_graph(g, rate):
    remove_e = []
    for e in g.edges:
        if np.random.rand() < rate:
            remove_e.append(e)
    mask_graph = g.copy()
    mask_graph.remove_edges_from(remove_e)
    return mask_graph
def testData_link_prediction(data_path, file_name, cfile):
    # 加载社区图
    print("loag net graphy from path:", os.path.join(data_path, cfile))
    cg = tg.load_graph(os.path.join(data_path, cfile))
    # 构建好图
    print("loag net graphy from path:", os.path.join(data_path, file_name))
    g, n, m = tg.load_tree(os.path.join(data_path, file_name))
    n = g.number_of_nodes()

    root_id = n - 1

    # 设置pad node
    pad_node_id = n
    n = n + 1
    # leaf node link mask
    mask_rate = 0.2
    cg_mask = mask_graph(cg, mask_rate)



    # treeGraph, graph, node_num, leaf_num, root_id, negative_num, pad_id
    td = TreeDataset(g, cg_mask, n, m, root_id, 20, pad_node_id)
    td.trav_tree()
    return td, n, pad_node_id


def extract_tree(data_path, file_name, cfile):
    # 加载社区图
    print("loag net graphy from path:", os.path.join(data_path, cfile))
    cg = tg.load_graph(os.path.join(data_path, cfile))
    # 构建好图
    print("loag net graphy from path:", os.path.join(data_path, file_name))
    g, n, m = tg.load_tree(os.path.join(data_path, file_name))
    n = g.number_of_nodes()
    
    # 设置pad node
    pad_node_id = n
    n = n + 1

    # 得到链接矩阵
    g_mat = tg.transfer_to_matrix(g)


    return g_mat, tree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='./data/input/',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--hidden_dim",
                        default=32,
                        type=int,
                        required=True)
    parser.add_argument("--radius_scale_factor",
                        default=0.9,
                        type=float,
                        required=True)
    parser.add_argument("--class_num",
                        default=2,
                        type=int,
                        required=True)
    parser.add_argument("--coor_dim",
                        default=2,
                        type=int,
                        required=True)
    parser.add_argument("--batch_num",
                        default=32,
                        type=int,
                        required=True)
    parser.add_argument("--epoch_num",
                        default=1,
                        type=int,
                        required=True)
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        required=True)
    parser.add_argument("--radius_0",
                        default=128,
                        type=float,
                        required=True)
    parser.add_argument("--graph_file",
                        default='edges_hamilton.txt',
                        type=str,
                        required=True)
    parser.add_argument("--tree_file",
                        default='tree2_hamilton',
                        type=str,
                        required=True)



    args = parser.parse_args()

    device = torch.device('cuda:2')

    #* read tree
    # extract_tree(args.data_dir, args.tree_file, args.graph_file)
    #* build dataset
    '''data form
    a line of input: <child id, parent id>
    a line of output: <predict label of child>
    '''
    dataset, node_num, pad_node_id = testData(args.data_dir, args.tree_file, args.graph_file)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_num, drop_last=False)


    #* build model
    # node_num, hidden_dim, radius_scale_factor, radius_0, pad_id
    model = CHE(node_num, args.hidden_dim, args.radius_scale_factor, args.radius_0, pad_node_id)
    # model, hidden_dim, class_num, coor_dim=2
    net = CLS(model, args.hidden_dim, args.class_num, args.coor_dim)
    # criticize = nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    #* opti
    optimizer = Adam(net.parameters(), lr=args.learning_rate)
    cg = tg.load_graph(os.path.join(args.data_dir, args.graph_file))
    g, n, m = tg.load_tree(os.path.join(args.data_dir, args.tree_file))
    adj = g.adj
    objects = np.array(list(adj.keys()))
    #* train model
    begin_time = time()
    for epoch in range(args.epoch_num):
        for i, data in enumerate(tqdm(train_loader)):
            # if (i < 34000):
            #     continue
            data = data.to(device)
            # children, brothers, parents, brothers_parents, unbrothers, need_loss=True
            logits, loss, feature = net(data[:,0], data[:,1], data[:,2], data[:,3], data[:,4:], True)
            if (torch.isnan(loss)):
                print(i, data)
                print(dataset[i-1])
                for name, parms in net.named_parameters():	
                    print('-->name:', name, '-->grad_requirs:',parms.requires_grad,\
                        ' -->grad_value:',parms.grad)
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i % 300 == 0):
                # r = reconstruction_worker(adj, net.model, objects)
                # print(r[0]/r[1], ' ', r[2]/r[3])
                print(epoch, " ", i, " ", loss.item())
    end_time = time()
    run_time = end_time-begin_time
    # print('该循环程序运行时间：',run_time)
            

    #* save model
    torch.save(net, f"./data/mammal_{args.hidden_dim}_best_s98.pkl")


    #* test model



def train_link_prediction():
    print("Linl prediction start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='./data/input/',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--hidden_dim",
                        default=32,
                        type=int,
                        required=True)
    parser.add_argument("--radius_scale_factor",
                        default=0.9,
                        type=float,
                        required=True)
    parser.add_argument("--class_num",
                        default=2,
                        type=int,
                        required=True)
    parser.add_argument("--coor_dim",
                        default=2,
                        type=int,
                        required=True)
    parser.add_argument("--batch_num",
                        default=32,
                        type=int,
                        required=True)
    parser.add_argument("--epoch_num",
                        default=1,
                        type=int,
                        required=True)
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        required=True)
    parser.add_argument("--radius_0",
                        default=128,
                        type=float,
                        required=True)
    parser.add_argument("--graph_file",
                        default='edges_hamilton.txt',
                        type=str,
                        required=True)
    parser.add_argument("--tree_file",
                        default='tree2_hamilton',
                        type=str,
                        required=True)



    args = parser.parse_args()

    device = torch.device('cuda:2')

    #* read tree
    # extract_tree(args.data_dir, args.tree_file, args.graph_file)
    #* build dataset
    '''data form
    a line of input: <child id, parent id>
    a line of output: <predict label of child>
    '''
    dataset, node_num, pad_node_id = testData_link_prediction(args.data_dir, args.tree_file, args.graph_file)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_num, drop_last=False)


    #* build model
    # node_num, hidden_dim, radius_scale_factor, radius_0, pad_id
    model = CHE(node_num, args.hidden_dim, args.radius_scale_factor, args.radius_0, pad_node_id)
    # model, hidden_dim, class_num, coor_dim=2
    net = CLS(model, args.hidden_dim, args.class_num, args.coor_dim)
    # criticize = nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    #* opti
    optimizer = Adam(net.parameters(), lr=args.learning_rate)
    cg = tg.load_graph(os.path.join(args.data_dir, args.graph_file))
    g, n, m = tg.load_tree(os.path.join(args.data_dir, args.tree_file))
    adj = g.adj
    objects = np.array(list(adj.keys()))
    #* train model
    begin_time = time()
    for epoch in range(args.epoch_num):
        for i, data in enumerate(tqdm(train_loader)):
            # if (i < 34000):
            #     continue
            data = data.to(device)
            # children, brothers, parents, brothers_parents, unbrothers, need_loss=True
            logits, loss, feature = net(data[:,0], data[:,1], data[:,2], data[:,3], data[:,4:], True)
            if (torch.isnan(loss)):
                print(i, data)
                print(dataset[i-1])
                for name, parms in net.named_parameters():	
                    print('-->name:', name, '-->grad_requirs:',parms.requires_grad,\
                        ' -->grad_value:',parms.grad)
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i % 300 == 0):
                # r = reconstruction_worker(adj, net.model, objects)
                # print(r[0]/r[1], ' ', r[2]/r[3])
                print(epoch, " ", i, " ", loss.item())
    end_time = time()
    run_time = end_time-begin_time
    # print('该循环程序运行时间：',run_time)
            

    #* save model
    torch.save(net, f"./data/lp_amherst_{args.hidden_dim}_s9.pkl")


    #* test model


if __name__ == "__main__":
    # main()
    train_link_prediction()
