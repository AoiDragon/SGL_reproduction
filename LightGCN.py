import torch
import torch.nn as nn
from params import args


class LightGCN(nn.Module):
    def __init__(self, user_num, item_num, embed_dim, layer_num):
        super(LightGCN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.dropout = nn.Dropout(p=0.1)

        self.user_embedding = nn.Embedding(self.user_num, self.embed_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.embed_dim)

        self.reset_params()

    def reset_params(self):
        init = torch.nn.init.xavier_uniform_
        init(self.user_embedding.weight)
        init(self.item_embedding.weight)

    def forward(self, norm_adj):
        ego_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embedding = [ego_embedding]

        for i in range(self.layer_num):
            ego_embedding = torch.sparse.mm(norm_adj, ego_embedding)
            all_embedding += [ego_embedding]

        all_embedding = torch.stack(all_embedding, dim=1).mean(dim=1)
        user_embedding, item_embedding = torch.split(all_embedding, [self.user_num, self.item_num], dim=0)

        return user_embedding, item_embedding





