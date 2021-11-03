import argparse
import copy
import logging
import os
import re

import pickle as pkl
from collections import defaultdict
from multiprocessing import Queue, Process
from tqdm import tqdm
import gc
import networkx as nx

import pandas as pd

from scipy.io import loadmat
from time import time
import torch.multiprocessing
import scipy.sparse as sp
import joblib
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')


np.set_printoptions(threshold=1000000)



# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def get_nbr(u2u, user, nbr_maxlen):
    nbr = np.zeros([nbr_maxlen, ], dtype=np.int64)
    nbr_len = len(u2u[user])
    if nbr_len == 0:
        pass
    elif nbr_len > nbr_maxlen:
        np.random.shuffle(u2u[user])
        nbr[:] = u2u[user][:nbr_maxlen]
    else:
        nbr[:nbr_len] = u2u[user]

    return nbr


def get_nbr_iids(user_train, user, nbrs, time_splits):
    start_idx = np.nonzero(time_splits)[0][0]
    user_first_ts = time_splits[start_idx]
    user_last_ts = time_splits[-1]
    nbr_maxlen = len(nbrs)
    seq_maxlen = len(time_splits)
    nbrs_iids = np.zeros((nbr_maxlen, seq_maxlen))

    for i, nbr in enumerate(nbrs):
        if nbr == 0 or nbr == user:
            continue

        nbr_hist = user_train[nbr]

        if len(nbr_hist) == 0:
            continue

        nbr_first_ts = nbr_hist[0][1]
        nbr_last_ts = nbr_hist[-1][1]

        if nbr_first_ts > user_last_ts or nbr_last_ts <= user_first_ts:
            continue

        sample_list = list()
        for j in range(start_idx + 1, seq_maxlen):
            start_time = time_splits[j - 1]
            end_time = time_splits[j]

            if start_time != end_time:
                sample_list = list(filter(None, map(
                    lambda x: x[0] if x[1] > start_time and x[1] <= end_time else None, nbr_hist
                )))

            if len(sample_list):
                # print('st={} et={} sl={}'.format(start_time, end_time, sample_list))
                nbrs_iids[i, j] = np.random.choice(sample_list)

    return nbrs_iids


def sample_function(seq_list, pos_list, meta_list, user_train, u2u, user_num, item_num, batch_size, seq_maxlen, nbr_maxlen, meta_maxlen, time_limit, result_queue, seed):
    # print('seed=', seed)

    def sample(seq_list, pos_list, meta_list):
        user = np.random.randint(1, user_num)
        while len(user_train[user]) <= 1: user = np.random.randint(1, user_num)

        seq = seq_list[user]
        pos = pos_list[user]
        ts = np.zeros(seq_maxlen, dtype=np.int64)
        neg = np.zeros(seq_maxlen, dtype=np.int64)
        nxt = user_train[user][-1, 0]
        idx = seq_maxlen - 1
        meta = meta_list[user]

        exclude_items = set(user_train[user][:, 0].tolist())
        for (item, time_stamp) in reversed(user_train[user][:-1]):
            ts[idx] = time_stamp
            if nxt != 0:
                neg[idx] = random_neq(1, item_num, exclude_items)
            nxt = item
            idx -= 1
            if idx == -1: break

        nbr = get_nbr(u2u, user, nbr_maxlen)
        nbr_iids = get_nbr_iids(user_train, user, nbr, ts)

        return user, seq, pos, neg, nbr, nbr_iids, meta


    np.random.seed(seed)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(seq_list, pos_list, meta_list))

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, que_len, seq_list, pos_list, meta_list, user_train, u2u, user_num, item_num, batch_size, seq_maxlen, nbr_maxlen, meta_maxlen, time_limit, n_workers=1, seed=0):
        self.result_queue = Queue(maxsize=n_workers * que_len)
        self.processors = []
        np.random.seed(seed)
        for i in range(n_workers):
            self.processors.append(Process(
                target=sample_function,
                args=(seq_list, pos_list, meta_list, user_train, u2u, user_num, item_num,
                      batch_size, seq_maxlen, nbr_maxlen, meta_maxlen,time_limit,
                      self.result_queue, np.random.randint(low=1, high=1e8))))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def preprocess_uir(df, prepro='origin', binary=False, pos_threshold=None, level='ui'):
    # set rating >= threshold as positive samples
    if pos_threshold is not None:
        df = df.query(f'rate >= {pos_threshold}').reset_index(drop=True)

    # reset rating to interaction, here just treat all rating as 1
    if binary:
        df['rate'] = 1.0

    # which type of pre-dataset will use
    if prepro == 'origin':
        pass

    elif prepro.endswith('filter'):
        pattern = re.compile(r'\d+')
        filter_num = int(pattern.findall(prepro)[0])

        tmp1 = df.groupby(['user'], as_index=False)['item'].count()
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
        if level == 'ui':
            df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'u':
            df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'i':
            df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()
        else:
            raise ValueError(f'Invalid level value: {level}')

        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()

    elif prepro.endswith('core'):
        pattern = re.compile(r'\d+')
        core_num = int(pattern.findall(prepro)[0])

        def filter_user(df):
            tmp = df.groupby(['user'], as_index=False)['item'].count()
            tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
            df = df.merge(tmp, on=['user'])
            df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)

            return df

        def filter_item(df):
            tmp = df.groupby(['item'], as_index=False)['user'].count()
            tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp, on=['item'])
            df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)

            return df

        if level == 'ui':
            while 1:
                df = filter_user(df)
                df = filter_item(df)
                chk_u = df.groupby('user')['item'].count()
                chk_i = df.groupby('item')['user'].count()
                if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
                    break
        elif level == 'u':
            df = filter_user(df)
        elif level == 'i':
            df = filter_item(df)
        else:
            raise ValueError(f'Invalid level value: {level}')

        gc.collect()

    else:
        raise ValueError('Invalid dataset preprocess type, origin/Ncore/Nfilter (N is int number) expected')

    return df


# def load_ds(dataset='Ciao'):
#     # Ciao Raw #u2i=284086, #u2u=57544
#     # Epin Raw #u2i=922267, #u2u=355813
#
#     rating = pd.DataFrame()
#
#     rating_mat = loadmat(f'./datasets/{dataset}/rating_with_timestamp.mat')
#     if dataset == 'Ciao':
#         rating = rating_mat['rating']
#     elif dataset == 'Epinions':
#         rating = rating_mat['rating_with_timestamp']
#
#     df = pd.DataFrame(data=rating, columns=['user', 'item', 'cate', 'rate', 'help', 'ts'])
#     df.drop(columns=['cate', 'help'], inplace=True)
#     df.drop_duplicates(subset=['user', 'item', 'ts'], keep='first', inplace=True)
#     df = preprocess_uir(df, prepro='origin', binary=True, pos_threshold=3)
#     df.drop(columns=['rate'], inplace=True)
#     df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
#
#     # data = df.values
#     # users = []
#     # items = []
#     # for i in data:
#     #     users.append(i[0])
#     #     items.append(i[1])
#     # users.sort()
#     # items.sort()
#     # user_id = 1
#     # user_map = {}
#     # for i in range(len(users)):
#     #     if i == 0:
#     #         user_map[users[i]] = user_id
#     #         user_id += 1
#     #     else:
#     #         if users[i] != users[i - 1]:
#     #             user_map[users[i]] = user_id
#     #             user_id += 1
#     # user_num = user_id - 1
#     #
#     # item_id = 1
#     # item_map = {}
#     # for i in range(len(items)):
#     #     if i == 0:
#     #         item_map[items[i]] = item_id
#     #         item_id += 1
#     #     else:
#     #         if items[i] != items[i - 1]:
#     #             item_map[items[i]] = item_id
#     #             item_id += 1
#     # item_num = item_id - 1
#     # for i in range(len(data)):
#     #     data[i, 0] = user_map[data[i, 0]]
#     #     data[i, 1] = item_map[data[i, 1]]
#     #
#     # df = pd.DataFrame(data, columns=['user', 'item', 'ts'])
#
#     uu_elist = loadmat(f'./datasets/{dataset}/trust.mat')['trust']
#     g = nx.Graph()
#     g.add_nodes_from(list(range(user_num)))
#     g.add_edges_from(uu_elist)
#     g.add_edges_from([[u, u] for u in g.nodes])  # add self-loop to avoid NaN attention scores
#     u2u = nx.to_dict_of_lists(g)
#
#     print(f'Loaded {dataset} dataset with {user_num} users, {item_num} items, '
#           f'{len(df.values)} u2i, {len(uu_elist)} u2u. ')
#
#     print('Average neighbors: {:.4f}'.format(np.mean([len(v) for k, v in u2u.items()])))
#
#     return df, u2u, user_num, item_num

def load_ds(dataset='Ciao'):
    # Ciao Raw #u2i=284086, #u2u=57544
    # Epin Raw #u2i=922267, #u2u=355813

    rating = pd.DataFrame()

    rating_mat = loadmat(f'datasets/{dataset}/rating_with_timestamp.mat')
    if dataset == 'Ciao':
        rating = rating_mat['rating']
    elif dataset == 'Epinions':
        rating = rating_mat['rating_with_timestamp']

    df = pd.DataFrame(data=rating, columns=['user', 'item', 'cate', 'rate', 'help', 'ts'])
    df.drop(columns=['cate', 'help'], inplace=True)
    df.drop_duplicates(subset=['user', 'item', 'ts'], keep='first', inplace=True)
    df = preprocess_uir(df, prepro='origin', binary=True, pos_threshold=3)
    df.drop(columns=['rate'], inplace=True)
    df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)

    user_num = df['user'].max() + 1
    item_num = df['item'].max() + 1

    uu_elist = loadmat(f'datasets/{dataset}/trust.mat')['trust']
    g = nx.Graph()
    g.add_nodes_from(list(range(user_num)))
    g.add_edges_from(uu_elist)
    g.add_edges_from([[u, u] for u in g.nodes])  # add self-loop to avoid NaN attention scores
    u2u = nx.to_dict_of_lists(g)

    print(f'Loaded {dataset} dataset with {user_num} users, {item_num} items, '
          f'{len(df.values)} u2i, {len(uu_elist)} u2u. ')

    print('Average neighbors: {:.4f}'.format(np.mean([len(v) for k, v in u2u.items()])))

    return df, u2u, user_num, item_num


def data_partition(df):
    print('Splitting train/val/test set...')
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)

    item_train = defaultdict(list)

    user_items_dict = defaultdict(list)
    item_train = defaultdict(list)

    def apply_fn1(grp):
        key_id = grp['user'].values[0]
        user_items_dict[key_id] = grp[['item', 'ts']].values



    def apply_fn2(grp):
        key_id = grp['item'].values[0]
        item_train[key_id] = grp[['user','ts']].values


    df.groupby('user').apply(apply_fn1)
    df.groupby('item').apply(apply_fn2)

    for user in user_items_dict:
        nfeedback = len(user_items_dict[user])
        if nfeedback < 5:
            user_train[user] = user_items_dict[user]
            user_valid[user] = []
            user_test[user] = []

        else:
            user_train[user] = user_items_dict[user][:-2]
            user_valid[user] = []
            user_valid[user].append(user_items_dict[user][-2])
            user_test[user] = []
            user_test[user].append(user_items_dict[user][-1])
            for i,j in enumerate(item_train[user_items_dict[user][-2,0]]):
                if j[0] == user:
                    tmp = item_train[user_items_dict[user][-2, 0]].tolist()
                    tmp.pop(i)
                    item_train[user_items_dict[user][-2, 0]] = np.array(tmp)
                    break
            for i,j in enumerate(item_train[user_items_dict[user][-1,0]]):
                if j[0] == user:
                    tmp = item_train[user_items_dict[user][-1, 0]].tolist()
                    tmp.pop(i)
                    item_train[user_items_dict[user][-1, 0]] = np.array(tmp)
                    break

    return user_train, user_valid, user_test, item_train


def parse_batch(uid, seq, pos, neg, nbr, nbr_iid, meta):
    uid = np.array(uid, dtype=np.int32)
    seq = np.array(seq, dtype=np.int32)
    pos = np.array(pos, dtype=np.int32)
    neg = np.array(neg, dtype=np.int32)
    nbr = np.array(nbr, dtype=np.int32)
    nbr_iid = np.array(nbr_iid, dtype=np.int32)
    meta = np.array(meta, dtype=np.int32)
    tmp_list = []
    for tmp in nbr_iid:
        tmp_list.append(sp.csr_matrix(tmp, dtype=np.int32))
    nbr_iid = tmp_list
    batch = [uid, seq, pos, neg, nbr, nbr_iid, meta]
    return batch


def generate_train_samples(sampler, num_batch):
    print('Reseting train samples...', end='')
    train_set = list()
    st = time()
    check_meta_len = []
    for step in tqdm(range(num_batch)):
        uid, seq, pos, neg, nbr, nbr_iid, meta = sampler.next_batch()
        # tmp_check_meta_len = list(tmp_check_meta_len)
        # for i in tmp_check_meta_len:
        #     check_meta_len = check_meta_len + i
        # print(tmp_check_meta_len)

        batch = parse_batch(uid, seq, pos, neg, nbr, nbr_iid, meta)
        train_set.append([batch])
    print('ava meta num',np.array(check_meta_len).mean())

    print(' Time: {:.2f}s'.format(time() - st))

    return train_set

def save_pkl(file, obj):
    with open(file, 'wb') as f:
        pkl.dump(obj, f)


def load_pkl(file):
    with open(file, 'rb') as f:
        data = pkl.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description='Get Train Samples')
    parser.add_argument('--dataset', default='Epinions')
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--seq_maxlen', type=int, default=50)
    parser.add_argument('--nbr_maxlen', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--meta_maxlen', type=int, default=10)
    parser.add_argument('--time_limit', type=int, default=2500000)
    parser.add_argument('--que_len', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=100)
    args = parser.parse_args()

    if os.path.exists(f'./datasets/{args.dataset}/u2idf.csv'):
        df = pd.read_csv(f'./datasets/{args.dataset}/u2idf.csv')
        tmp = np.load(f'./datasets/{args.dataset}/usernum_itmenum.npy')
        user_num, item_num = tmp[0], tmp[1]
        print('user_num',user_num,'item_num',item_num)
        all_d = np.load(f'./datasets/{args.dataset}/{args.time_limit}process_data.npz',allow_pickle=True)
        user_train = all_d['user_train'][()]
        user_valid = all_d['user_valid'][()]
        user_test = all_d['user_test'][()]
        item_train= all_d['item_train'][()]
        u2u = all_d['u2u_dict'][()]
        u2u_dict = u2u
    else:
        df, u2u, user_num, item_num = load_ds(args.dataset)
        u2u_dict = u2u
        df.to_csv(f'./datasets/{args.dataset}/u2idf.csv')
        np.save(f'./datasets/{args.dataset}/usernum_itmenum.npy', np.array([user_num, item_num]))


        user_train, user_valid, user_test, item_train = data_partition(df)
        # save_pkl(f'./datasets/{args.dataset}/{args.time_limit}process_data.pkl',[u2u,user_train,user_valid,user_test,item_train])
        np.savez(f'./datasets/{args.dataset}/{args.time_limit}process_data.npz',u2u_dict=u2u,user_train=user_train,user_valid=user_valid,user_test=user_test,item_train=item_train)
    print('user_train_len',len(user_train),'user_num',user_num)
    print('user_num',user_num,'item_num',item_num)
    print('Average sequence length: %.4f' % (sum(len(user_train[u]) for u in user_train) / len(user_train)))
    num_batch = len(user_train) // args.batch_size
    seq_maxlen = args.seq_maxlen
    meta_maxlen = args.meta_maxlen
    time_limit = args.time_limit
    nbr_maxlen = args.nbr_maxlen
    que_len = args.que_len
    def sample_one_user(user):
        seq = np.zeros(seq_maxlen, dtype=np.int32)
        pos = np.zeros(seq_maxlen, dtype=np.int32)
        ts = np.zeros(seq_maxlen, dtype=np.int32)
        nxt = user_train[user][-1, 0]
        idx = seq_maxlen - 1
        meta = np.zeros((seq_maxlen, meta_maxlen), dtype=np.int64)
        check_meta_len = []
        for (item, time_stamp) in reversed(user_train[user][:-1]):
            seq[idx] = item
            ts[idx] = time_stamp
            pos[idx] = nxt
            tmp_list = []
            for i in item_train[item]:
                if abs(i[1] - time_stamp) < time_limit:
                    tmp_list.append(i[0])
            if len(tmp_list) >= meta_maxlen:
                meta[idx] = np.array(np.random.choice(tmp_list, size=meta_maxlen), dtype=np.int64)
            else:
                for i, tmp_user in enumerate(tmp_list):
                    meta[idx, i] = tmp_user
            check_meta_len.append(len(tmp_list))
            nxt = item
            idx -= 1
            if idx == -1: break
        if len(np.nonzero(ts)[0]) == 0:
            ts[0] = 1
        nbr = get_nbr(u2u_dict, user, nbr_maxlen)
        nbr_iid = get_nbr_iids(user_train, user, nbr, ts)
        nbr_iid = sp.csr_matrix(nbr_iid, dtype=np.int32)

        return user, seq, pos, nbr, nbr_iid, meta, np.array(check_meta_len).mean()


    if os.path.exists(f'datasets/{args.dataset}/user_seq_pos_nbr_nbriid_meta_avameta.npz'):
        all_data = np.load(f'datasets/{args.dataset}/user_seq_pos_nbr_nbriid_meta_avameta.npz',allow_pickle=True)
        seq_list = all_data['seq_list']
        pos_list = all_data['pos_list']
        nbr_list = all_data['nbr_list']
        nbr_iid_list = all_data['nbr_iid_list']
        meta_list = all_data['meta_list']
        ava_meta_list = all_data['ava_meta_list']
        print('seq',len(seq_list),'nbr',len(nbr_list),'nbr_iid',len(nbr_iid_list))
    else:
        uid_list = [[]]
        seq_list = [[]]
        pos_list = [[]]
        nbr_list = [[]]
        nbr_iid_list = [[]]
        meta_list = [[]]
        ava_meta_list = [[]]
        print("pre uid seq pos meta")
        for user in tqdm(range(1, user_num)):
            if len(user_train[user])<=1:
                uid_list.append([])
                seq_list.append([])
                pos_list.append([])
                nbr_list.append([])
                nbr_iid_list.append([])
                meta_list.append([])
                ava_meta_list.append([])
            else:
                user, seq, pos, nbr, nbr_iid, meta, ava_meta = sample_one_user(user)
                uid_list.append(user)
                seq_list.append(seq)
                pos_list.append(pos)
                nbr_list.append(nbr)
                nbr_iid_list.append(nbr_iid)
                meta_list.append(meta)
                ava_meta_list.append(ava_meta)

        np.savez(f'datasets/{args.dataset}/user_seq_pos_nbr_nbriid_meta_avameta.npz',
                 seq_list=seq_list,
                 pos_list=pos_list,
                 nbr_list=nbr_list,
                 nbr_iid_list=nbr_iid_list,
                 meta_list=meta_list,
                 ava_meta_list=ava_meta_list)

    seed = args.seed
    np.random.seed(seed)

    sampler_train = WarpSampler(
        que_len=que_len,
        seq_list=seq_list,
        pos_list=pos_list,
        meta_list=meta_list,
        user_train=user_train,
        u2u=u2u,
        user_num=user_num,
        item_num=item_num,
        batch_size=args.batch_size,
        seq_maxlen=args.seq_maxlen,
        nbr_maxlen=args.nbr_maxlen,
        meta_maxlen=args.meta_maxlen,
        time_limit=args.time_limit,
        n_workers=args.num_workers,
        seed=seed)

    num_epoch = args.epoch
    num_batch = num_epoch * num_batch
    train_set = generate_train_samples(sampler_train, num_batch)


    file_name = f'datasets/{args.dataset}/processed_train_set_n{args.epoch}_bs{args.batch_size}_sl{args.seq_maxlen}_nl{args.nbr_maxlen}meta{args.meta_maxlen}timelimit{args.time_limit}.pkl'
    save_pkl(file_name, train_set)
    print('saved at', file_name)
    pkl_file = load_pkl(file_name)



if __name__ == '__main__':
    main()
