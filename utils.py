import torch
import numpy as np
from params import args
import torch.nn.functional as F


def sp_mat_to_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()


def inner_product(x1, x2):
    return torch.sum(torch.mul(x1, x2), dim=-1)


def compute_bpr_loss(x1, x2):
    #return -torch.sum(torch.log((x1.view(-1) - x2.view(-1)).sigmoid() + 1e-8))
    return -torch.sum(F.logsigmoid(x1-x2))

def compute_infoNCE_loss(x1, x2, temp):
    return torch.logsumexp((x2 - x1[:, None]) / temp, dim=1)


def compute_reg_loss(w1, w2, w3):
    return 0.5 * torch.sum(torch.pow(w1, 2) + torch.pow(w2, 2) + torch.pow(w3, 2))


def compute_metric(ratings, test_item):
    hit = 0
    DCG = 0.
    iDCG = 0.

    _, shoot_index = torch.topk(ratings, args.k)
    shoot_index = shoot_index.cpu().tolist()

    for i in range(len(shoot_index)):
        if shoot_index[i] in test_item:
            hit += 1
            DCG += 1 / np.log2(i + 2)
        if i < test_item.size()[0]:
            iDCG += 1 / np.log2(i + 2)

    recall = hit / test_item.size()[0]
    NDCG = DCG / iDCG

    return recall, NDCG
