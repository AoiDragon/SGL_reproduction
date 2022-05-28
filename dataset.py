import torch.utils.data as data
from scipy.sparse import csr_matrix
import numpy as np
from collections import OrderedDict, defaultdict, Iterable


class RecDataset_train(data.Dataset):
    def __init__(self, data, user_num, item_num):
        self.data = data
        self.user_num = user_num
        self.item_num = item_num

        self.user_item_pair = self.data.values
        self.user_index = self.user_item_pair[:, 0].flatten()
        self.item_index = self.user_item_pair[:, 1].flatten()
        self.interact_num = len(self.user_item_pair)

        self.user_pos_dict = OrderedDict()
        grouped_user = self.data.groupby('user')
        for user, user_data in grouped_user:
            self.user_pos_dict[user] = user_data['item'].to_numpy(dtype=np.int32)

        self.user_list, self.pos_item_list, self.neg_item_list = self.sample()

    def sample(self):
        """
        Sample user, pos_item, neg_item
        """
        user_arr = np.array(list(self.user_pos_dict.keys()), dtype=np.int32)
        user_list = np.random.choice(user_arr, size=self.interact_num, replace=True)

        user_pos_len = defaultdict(int)
        for u in user_list:
            user_pos_len[u] += 1

        user_pos_sample = dict()
        user_neg_sample = dict()
        for user, pos_len in user_pos_len.items():
            pos_item = self.user_pos_dict[user]
            pos_idx = np.random.choice(pos_item, size=pos_len, replace=True)
            user_pos_sample[user] = list(pos_idx)

            neg_item = np.random.randint(low=0, high=self.item_num, size=pos_len)
            for i in range(len(neg_item)):
                idx = neg_item[i]
                while idx in pos_item:
                    idx = np.random.randint(low=0, high=self.item_num)
                neg_item[i] = idx
            user_neg_sample[user] = list(neg_item)

        pos_item_list = [user_pos_sample[user].pop() for user in user_list]
        neg_item_list = [user_neg_sample[user].pop() for user in user_list]
        return user_list, pos_item_list, neg_item_list

    def __len__(self):
        return self.interact_num

    def __getitem__(self, idx):
        return self.user_list[idx], self.pos_item_list[idx], self.neg_item_list[idx]


class RecDataset_test(data.Dataset):
    def __init__(self, data):
        self.data = data

        self.user_item_pair = self.data.values

        self.user_pos_dict = OrderedDict()
        grouped_user = self.data.groupby('user')
        for user, user_data in grouped_user:
            self.user_pos_dict[user] = user_data['item'].to_numpy(dtype=np.int32)

        self.user_list = np.array(list(self.user_pos_dict.keys()))  # 用户不重复

    def __len__(self):
        return self.user_list.shape[0]

    def __getitem__(self, idx):
        return self.user_list[idx]
