import datetime
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from params import args
import scipy.sparse as sp
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from LightGCN import LightGCN
from dataset import RecDataset_train, RecDataset_test
from utils import sp_mat_to_tensor, inner_product, compute_infoNCE_loss, compute_bpr_loss, compute_reg_loss, \
    compute_metric


class Model:

    def __init__(self):
        self.train_data_path = args.train_data_path
        self.test_data_path = args.test_data_path
        self.behavior_mats = {}
        self.behavior_mats_T = {}

        now_time = datetime.datetime.now()
        self.time = datetime.datetime.strftime(now_time, '%Y_%m_%d__%H_%M_%S')

        self.epoch = 0
        self.cnt = 0
        self.train_loss = []
        self.bpr_loss = []
        self.infoNCE_loss = []
        self.reg_loss = []
        self.recall_history = []
        self.NDCG_history = []
        self.best_recall = 0
        self.best_NDCG = 0
        self.best_epoch = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load data
        train_data = pd.read_csv(self.train_data_path, sep=',', header=None, names=['user', 'item'])
        test_data = pd.read_csv(self.test_data_path, sep=',', header=None, names=['user', 'item'])
        all_data = pd.concat([train_data, test_data])
        self.user_num = max(all_data['user']) + 1
        self.item_num = max(all_data['item']) + 1

        self.train_dataset = RecDataset_train(train_data, self.user_num, self.item_num)
        self.train_loader = dataloader.DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, pin_memory=True)

        self.test_dataset = RecDataset_test(test_data)
        self.test_loader = dataloader.DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=10, pin_memory=True)

        # Model Config
        self.embed_dim = args.embed_dim
        self.layer_num = args.layer_num
        self.lr = args.lr
        self.model = LightGCN(self.user_num, self.item_num, self.embed_dim, self.layer_num).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.graph = self.create_adj_mat(is_subgraph=False)
        self.graph = sp_mat_to_tensor(self.graph).to(self.device)

    def run(self):
        for epoch in range(1, args.epoch_num + 1):
            self.epoch += 1

            epoch_loss, bpr_loss, infoNCE_loss, reg_loss= self.train_epoch()
            self.train_loss.append(epoch_loss)
            self.bpr_loss.append(bpr_loss)
            self.infoNCE_loss.append(infoNCE_loss)
            self.reg_loss.append(reg_loss)
            print(f"Epoch {self.epoch}:  loss:{epoch_loss/self.train_dataset.interact_num} \
                    bpr_loss:{bpr_loss/self.train_dataset.interact_num} \
                    info_NCE_loss:{infoNCE_loss/self.train_dataset.interact_num} \
                    reg_loss:{reg_loss/self.train_dataset.interact_num}")

            epoch_recall, epoch_NDCG = self.test_epoch()
            self.recall_history.append(epoch_recall)
            self.NDCG_history.append(epoch_NDCG)
            print(f"Epoch {self.epoch}:  recall:{epoch_recall}, NDCG:{epoch_NDCG}")

            if epoch_recall > self.best_recall:
                self.cnt = 0
                self.best_recall = epoch_recall
                self.best_epoch = self.epoch

            if epoch_NDCG > self.best_NDCG:
                self.cnt = 0
                self.best_NDCG = epoch_NDCG
                self.best_epoch = self.epoch

            if epoch_recall < self.best_recall and epoch_NDCG < self.best_NDCG:
                self.cnt += 1

            self.save_metrics()

            if self.cnt == args.stop_cnt:
                print(f"Early stop at {self.best_epoch}: best Recall: {self.best_recall}, best_NDCG: {self.best_NDCG}\n")
                self.save_metrics()
                break

    def train_epoch(self):
        epoch_loss = 0
        epoch_bpr_loss = 0
        epoch_infoNCE_loss = 0
        epoch_reg_loss = 0
        sub_graph1 = self.create_adj_mat(is_subgraph=True)
        sub_graph1 = sp_mat_to_tensor(sub_graph1).to(self.device)
        sub_graph2 = self.create_adj_mat(is_subgraph=True)
        sub_graph2 = sp_mat_to_tensor(sub_graph2).to(self.device)

        for batch_user, batch_pos_item, batch_neg_item in tqdm(self.train_loader):
            batch_user = batch_user.long().to(self.device)
            batch_pos_item = batch_pos_item.long().to(self.device)
            batch_neg_item = batch_neg_item.long().to(self.device)

            all_user_embedding, all_item_embedding = self.model(self.graph)
            SSL_user_embedding1, SSL_item_embedding1 = self.model(sub_graph1)
            SSL_user_embedding2, SSL_item_embedding2 = self.model(sub_graph2)

            # 归一化，消除嵌入的模对相似度衡量的影响+
            SSL_user_embedding1 = F.normalize(SSL_user_embedding1)
            SSL_user_embedding2 = F.normalize(SSL_user_embedding2)
            SSL_item_embedding1 = F.normalize(SSL_item_embedding1)
            SSL_item_embedding2 = F.normalize(SSL_item_embedding2)

            batch_user_embedding = all_user_embedding[batch_user]
            batch_pos_item_embedding = all_item_embedding[batch_pos_item]
            batch_neg_item_embedding = all_item_embedding[batch_neg_item]
            batch_SSL_user_embedding1 = SSL_user_embedding1[batch_user]
            batch_SSL_user_embedding2 = SSL_user_embedding2[batch_user]
            batch_SSL_item_embedding1 = SSL_item_embedding1[batch_pos_item]
            batch_SSL_item_embedding2 = SSL_item_embedding2[batch_pos_item]

            # [batch_size]
            pos_score = inner_product(batch_user_embedding, batch_pos_item_embedding)  # [2048]
            neg_score = inner_product(batch_user_embedding, batch_neg_item_embedding)

            # [batch_size]
            SSL_user_pos_score = inner_product(batch_SSL_user_embedding1, batch_SSL_user_embedding2)  # 全1
            SSL_user_neg_score = torch.matmul(batch_SSL_user_embedding1, torch.transpose(SSL_user_embedding2, 0, 1))

            SSL_item_pos_score = inner_product(batch_SSL_item_embedding1, batch_SSL_item_embedding2)
            SSL_item_neg_score = torch.matmul(batch_SSL_item_embedding1, torch.transpose(SSL_item_embedding2, 0, 1))

            bpr_loss = compute_bpr_loss(pos_score, neg_score)  # 1419

            infoNCE_user_loss = compute_infoNCE_loss(SSL_user_pos_score, SSL_user_neg_score, args.SSL_temp)
            infoNCE_item_loss = compute_infoNCE_loss(SSL_item_pos_score, SSL_item_neg_score, args.SSL_temp)
            infoNCE_loss = torch.sum(infoNCE_user_loss + infoNCE_item_loss, dim=-1)  # 22375

            reg_loss = compute_reg_loss(  # 11
                self.model.user_embedding(batch_user),
                self.model.item_embedding(batch_pos_item),
                self.model.item_embedding(batch_neg_item)
            )

            loss = bpr_loss + infoNCE_loss * args.SSL_reg + reg_loss * args.reg  # 3657
            epoch_loss += loss
            epoch_bpr_loss += bpr_loss
            epoch_infoNCE_loss += infoNCE_loss * args.SSL_reg
            epoch_reg_loss += reg_loss * args.reg
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return epoch_loss, epoch_bpr_loss, epoch_infoNCE_loss, epoch_reg_loss

    def test_epoch(self):
        test_user_pos_dict = self.test_dataset.user_pos_dict
        train_user_pos_dict = self.train_dataset.user_pos_dict

        epoch_recall = 0
        epoch_NDCG = 0
        tot = 0

        for test_user in self.test_loader:
            user_num = test_user.size()[0]
            test_user = test_user.long().to(self.device)
            test_item = [torch.from_numpy(test_user_pos_dict[int(u)]).long().to(self.device) for u in test_user]

            all_user_embedding, all_item_embedding = self.model(self.graph)
            test_user_embedding = all_user_embedding[test_user]
            ratings = torch.matmul(test_user_embedding, all_item_embedding.T)

            # 消除训练集数据的影响
            for idx, user in enumerate(test_user):
                #if user in train_user_pos_dict and len(train_user_pos_dict[user]) > 0:
                train_items = train_user_pos_dict[int(user.cpu())]
                ratings[idx][train_items] = -np.inf

            for i in range(user_num):
                recall, NDCG = compute_metric(ratings[i], test_item[i])
                epoch_recall += recall
                epoch_NDCG += NDCG

            tot += user_num

        epoch_recall /= tot
        epoch_NDCG /= tot

        return epoch_recall, epoch_NDCG

    def create_adj_mat(self, is_subgraph):
        node_num = self.user_num + self.item_num
        user_np, item_np = self.train_dataset.user_index, self.train_dataset.item_index

        if is_subgraph:
            sample_size = int(user_np.shape[0]*(1-args.SSL_dropout_ratio))
            keep_index = np.arange(user_np.shape[0])
            np.random.shuffle(keep_index)
            keep_index = keep_index[:sample_size]
            # keep_index = np.random.randint(user_np.shape[0], size=3*sample_size)
            # keep_index = np.unique(keep_index)
            # keep_index = keep_index[:sample_size]
            # keep_idx = np.random.randint(user_np.shape[0], size=int(user_np.shape[0]*(1-args.SSL_dropout_ratio)))
            # keep_idx = np.random.choice(user_np, size=int(user_np.shape[0]*(1-args.SSL_dropout_ratio)), replace=False)
            user_np = np.array(user_np)[keep_index]
            item_np = np.array(item_np)[keep_index]
            ratings = np.ones_like(user_np)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(node_num, node_num))


            # keep_idx = np.random.choice(user, size=int(len(user) * (1 - args.SSL_dropout_ratio)), replace=True)
            # keep_idx.tolist()
            # sub_user = np.array(user)[keep_idx]
            # sub_item = np.array(item)[keep_idx]
            # # rating = np.ones_like(sub_user, dtype=np.float32)
            # c = np.ones_like(sub_user, dtype=np.float32)
            # c = torch.ones(sub_user.shape[0])
            # # tmp_adj = sp.csr_matrix((rating, (sub_user, sub_item + self.user_num)), shape=(node_num, node_num))
            # a = sp.csr_matrix( (c, (sub_user, sub_item + self.user_num)), shape=(node_num, node_num))
            # b = sp.csr_matrix((c, (sub_user, sub_item)), shape=(node_num, node_num))
            # tmp_adj = sp.csr_matrix((c, (sub_user, sub_item + self.user_num)), shape=(node_num, node_num))
        else:
            rating = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((rating, (user_np, item_np + self.user_num)), shape=(node_num, node_num))
        adj = tmp_adj + tmp_adj.T

        row_sum = np.array(adj.sum(1))
        d = np.power(row_sum, -0.5).flatten()
        d[np.isinf(d)] = 0.
        d_mat = sp.diags(d)
        norm_adj = d_mat.dot(adj)
        norm_adj = norm_adj.dot(d_mat)

        return norm_adj

    def save_metrics(self):
        path = './runs/' + self.time + '/' + str(self.epoch) + '/'
        writer = SummaryWriter(path)
        for i in range(self.epoch):
            writer.add_scalar('Loss', self.train_loss[i], i)
            writer.add_scalar('bpr_loss', self.bpr_loss[i], i)
            writer.add_scalar('infoNCE_loss', self.infoNCE_loss[i], i)
            writer.add_scalar('reg_loss', self.reg_loss[i], i)
            writer.add_scalar('Recall', self.recall_history[i], i)
            writer.add_scalar('NDCG', self.NDCG_history[i], i)


if __name__ == '__main__':
    model = Model()
    model.run()
