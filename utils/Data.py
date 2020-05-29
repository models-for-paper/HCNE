from torch.utils.data import Dataset
import torch
import numpy as np
class GraphDataset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index):
        pass
    def __len__(self):
        pass

    
class TreeDataset(Dataset):
    def __init__(self, treeGraph, graph, node_num, leaf_num, root_id, negative_num, pad_id):
        self.treeGraph = treeGraph
        self.node_num = node_num
        self.negative_num = negative_num
        self.pad_id = pad_id
        self.root_id = root_id
        self.leaf_num = leaf_num
        self.len = 0
        self.data = []
        self.graph = graph
    def negativeSampling(self):
        return np.random.randint(0,self.pad_id, size=self.negative_num).tolist()
    # def getBrothers(self, p, c):
    #     b = []
    #     for p_b in list(self.treeGraph.out_edges(p)):
    #         if p_b[1] != c:
    #             b.append(p_b[1])
    #     if len(b) == 0:
    #         b.append(self.pad_id)
    #     return b
    def getBrothers(self, c):
        if c < self.leaf_num:
            bb =  list(self.graph.adj[c])
        else:
            bb =  list(self.treeGraph.adj[c])
        if len(bb) == 0:
            bb = [self.pad_id]
        return bb
    def getCoLevelBrothersParents(self, gp, p):
        cbp = []
        ps = list(self.treeGraph.out_edges(gp))
        for p_ in ps:
            if (p_[1] != p):
                cbp.append(p_[1])
        if len(cbp) == 0:
            cbp = [self.pad_id]
        return cbp
    #  children, brothers, parents, colevel_brothers_parents, unbrothers
    def trav_tree(self):
        # self.data.append([self.root_id, self.pad_id, self.pad_id, self.pad_id])
        for c in range(self.node_num):
            child = c
            if (c == self.root_id or c == self.pad_id):
                continue
            parent = list(self.treeGraph.in_edges(c))[0][0]
            t_p = list(self.treeGraph.in_edges(parent))
            if len(t_p) == 0:
                grand_parent = self.pad_id
                colevel_brothers_parents = [self.pad_id]
            else:
                grand_parent = t_p[0][0]
                colevel_brothers_parents = self.getCoLevelBrothersParents(grand_parent, parent)
            brothers = self.getBrothers(c)

            for b in brothers:
                for cb in colevel_brothers_parents:
                    self.data.append([child, b, parent, cb])
            


    def __getitem__(self, index):
        c = self.data[index][0]
        if c < self.leaf_num:
            na = self.negativeSampling()
        else:
            na = [self.pad_id]*self.negative_num
        line = self.data[index] + na
        line = torch.tensor(line)
        return line

    def __len__(self):
        return len(self.data)


class testSet(Dataset):
    def __init__(self):
        self.x = []
        self.y = []

    def trav(self, path):
        with open(path, "r") as file:
            for f in file:
                f = f.replace("\n", "")
                a = f.split('\t')
                if (len(a) == 2):
                    # if (int(a[1]) == 0 or int(a[1] == 1)):
                    self.x.append(int(a[0]))
                    self.y.append(int(a[1]))
                    # else:
                    #     print(a)

    def __getitem__(self, index):
        return torch.tensor(self.x[index]), torch.tensor(self.y[index])

    def __len__(self):
        return len(self.x)