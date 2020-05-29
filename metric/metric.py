import numpy as np
from sklearn.metrics import average_precision_score
import torch
def Distence(u, v):
    return (u - v).pow(2).sum(dim=-1)


def reconstruction_worker(adj, model, objects, progress=False):
    ranksum = nranks = ap_scores = iters = 0
    labels = np.empty(model.complex_coordinates_real.weight.size(0))
    for object in tqdm(objects) if progress else objects:
        labels.fill(0)
        neighbors = np.array(list(adj[object]))
        if (len(neighbors) == 0):
            continue
        real, img = model.complex_coordinates_real.weight[None, object], model.complex_coordinates_img.weight[None, object]
        obj_features = torch.cat((real, img), dim = -1)
        all_features = torch.cat((model.complex_coordinates_real.weight, model.complex_coordinates_img.weight), dim = -1)
        dists = Distence(obj_features, all_features)
        dists[object] = 1e12
        sorted_dists, sorted_idx = dists.sort()
        ranks, = np.where(np.in1d(sorted_idx.detach().cpu().numpy(), neighbors))
        # The above gives us the position of the neighbors in sorted order.  We
        # want to count the number of non-neighbors that occur before each neighbor
        ranks += 1
        N = ranks.shape[0]

        # To account for other positive nearer neighbors, we subtract (N*(N+1)/2)
        # As an example, assume the ranks of the neighbors are:
        # 0, 1, 4, 5, 6, 8
        # For each neighbor, we'd like to return the number of non-neighbors
        # that ranked higher than it.  In this case, we'd return 0+0+2+2+2+3=14
        # Another way of thinking about it is to return
        # 0 + 1 + 4 + 5 + 6 + 8 - (0 + 1 + 2 + 3 + 4 + 5)
        # (0 + 1 + 2 + ... + N) == (N * (N + 1) / 2)
        # Note that we include `N` to account for the source embedding itself
        # always being the nearest neighbor
        ranksum += ranks.sum() - (N * (N - 1) / 2)
        nranks += ranks.shape[0]
        labels[neighbors] = 1
        ap_scores += average_precision_score(labels, -dists.detach().cpu().numpy())
        iters += 1
    return float(ranksum), nranks, ap_scores, iters


def link_prediction_worker(adj, model, objects, leaf_num, progress=False):
    ranksum = nranks = ap_scores = iters = 0
    # labels = np.empty(model.complex_coordinates_real.weight.size(0))
    labels = np.empty(leaf_num)
    for object in tqdm(objects) if progress else objects:
        if object >= leaf_num:
            continue
        labels.fill(0)
        neighbors = np.array(list(adj[object]))
        if (len(neighbors) == 0):
            continue
        real, img = model.complex_coordinates_real.weight[None, object], model.complex_coordinates_img.weight[None, object]
        obj_features = torch.cat((real, img), dim = -1)
        all_features = torch.cat((model.complex_coordinates_real.weight[:leaf_num], model.complex_coordinates_img.weight[:leaf_num]), dim = -1)
        dists = Distence(obj_features, all_features)
        dists[object] = 1e12
        sorted_dists, sorted_idx = dists.sort()
        ranks, = np.where(np.in1d(sorted_idx.detach().cpu().numpy(), neighbors))
        # The above gives us the position of the neighbors in sorted order.  We
        # want to count the number of non-neighbors that occur before each neighbor
        ranks += 1
        N = ranks.shape[0]

        # To account for other positive nearer neighbors, we subtract (N*(N+1)/2)
        # As an example, assume the ranks of the neighbors are:
        # 0, 1, 4, 5, 6, 8
        # For each neighbor, we'd like to return the number of non-neighbors
        # that ranked higher than it.  In this case, we'd return 0+0+2+2+2+3=14
        # Another way of thinking about it is to return
        # 0 + 1 + 4 + 5 + 6 + 8 - (0 + 1 + 2 + 3 + 4 + 5)
        # (0 + 1 + 2 + ... + N) == (N * (N + 1) / 2)
        # Note that we include `N` to account for the source embedding itself
        # always being the nearest neighbor
        ranksum += ranks.sum() - (N * (N - 1) / 2)
        nranks += ranks.shape[0]
        labels[neighbors] = 1
        ap_scores += average_precision_score(labels, -dists.detach().cpu().numpy())
        iters += 1
    return float(ranksum), nranks, ap_scores, iters