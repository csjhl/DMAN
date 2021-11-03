import argparse
import copy
import logging
import os
import re
import sys
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from datetime import datetime
from multiprocessing import Queue, Process
from tqdm import tqdm
import gc
import networkx as nx
import numpy as np
import pandas as pd
import torch
from typing import Optional
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from time import time
import torch.multiprocessing
import joblib

class Short_term(nn.Module):
    def __init__(self, seq_num, seq_len,  dim, device):
        super(Short_term, self).__init__()
        self.device = device
        self.T = seq_len
        self.S = seq_num
        self.D = dim
        self.ATT = nn.MultiheadAttention(embed_dim=dim, kdim=dim+dim, vdim=dim+dim, num_heads=1)


    def forward(self, Ht, H):
        #Ht T*(B*S)*D
        #H T*(B*S)*(D+D)
        batch_size = int(Ht.shape[1]/self.S)
        attention_mask = ~torch.tril(torch.ones((self.T, self.T), dtype=torch.bool)).to(self.device)
        # print("SHORT Ht",Ht.shape,"H",H.shape)
        new_Ht,_ = self.ATT(Ht, H, H, attn_mask=attention_mask)#T*(B*S)*D
        tmp_H = torch.cat((torch.zeros(self.T,1,self.D).to(self.device), Ht.detach()), dim=1)[:,:-1,:]#T*(B*S)*D
        # tmp_H = torch.cat((torch.zeros(self.T, 1, self.D).to(self.device), Ht), dim=1)[:, :-1, :]  # T*(B*S)*D
        tmp_H[:,list(range(0, Ht.shape[1], batch_size)),:] = 0
        new_H = torch.cat((new_Ht, tmp_H),dim=-1)#T*(B*S)*(D+D)

        return new_Ht, new_H

class Long_term(nn.Module):
    def __init__(self, seq_num, seq_len,  dim, device):
        super(Long_term, self).__init__()
        self.device = device
        self.T = seq_len
        self.S = seq_num
        self.D = dim
        self.ATT = nn.MultiheadAttention(embed_dim=dim, num_heads=1)

    def forward(self, Hh, M_emb):
        #Hh T*(B*S)*D
        #M_emb M*(B)*D
        new_Hh, _ = self.ATT(Hh, M_emb, M_emb)#T*(B*S)*D
        return new_Hh



# class Long_term(nn.Module):
#     def __init__(self, seq_num, seq_len,  dim):
#         super(Long_term, self).__init__()
#         self.T = seq_len
#         self.S = seq_num
#         self.D = dim
#         self.ATT = nn.MultiheadAttention(embed_dim=dim, num_heads=1)
#
#     def forward(self, Hh, M_emb):
#         #Hh T*(B)*D
#         #M_emb M*(B)*D
#
#         new_Hh, _ = self.ATT(Hh, M_emb, M_emb)#T*(B)*D
#         return new_Hh

class  Gating(nn.Module):
    def __init__(self, seq_num, seq_len,  dim, device):
        super( Gating, self).__init__()
        self.device = device
        self.T = seq_len
        self.S = seq_num
        self.D = dim
        self.short_linear = nn.Linear(dim, dim)
        self.long_linear = nn.Linear(dim, dim)

    def forward(self, Ht, Hh):
        #Ht B*S*T*D
        #Hh B*S*T*D
        G = torch.sigmoid(self.short_linear(Ht) + self.long_linear(Hh)) #B*S*T*D
        V = G*Ht + (1-G)*Hh
        return V


class DynamicMemory(nn.Module):
    def __init__(self, seq_num, seq_len, dim, M, device):
        super(DynamicMemory, self).__init__()
        self.device = device
        self.T = seq_len
        self.S = seq_num
        self.D = dim
        self.M = M
        self.W = nn.Parameter(torch.Tensor(self.M, self.M+self.T,dim, dim))
        nn.init.uniform_(self.W, a=-0.0000001, b=0.0000001)

    def forward(self, M_emb, Ht_n):
        # Ht B*T*D n-1时刻的Ht
        # M_emb B*M*D
        # print("M_emb",M_emb[0,0],"Htn",Ht_n[0,0])
        eps = 0.0001
        M = M_emb.shape[1]
        batch_size = M_emb.shape[0]
        new_M_emb = torch.rand((batch_size, M, self.D)).to(self.device)#B*M*D
        cat_emb = torch.cat((M_emb, Ht_n), dim=1)#B*(T+M)*D
        for i in range(3):
            b = torch.einsum('bid, itdk, btk -> bit', new_M_emb, self.W, cat_emb)#B*M*(T+M)
            # print("b",b.shape)
            alph_u = torch.exp(b)

            # alph_u = torch.sigmoid(b)+eps
            alph_d = alph_u.sum(-1, keepdim=True) #不可能等于0
            alph = alph_u/ alph_d  # B*M*(T+M)
            # print('alph_u',alph_u)
            # print('alph_d',alph_d)
            # print('alph',alph)
            s = torch.einsum('bit, itdk, btk -> bid', alph, self.W, cat_emb)#B*M*D
            # print("s",s.shape)
            # print("s",s)
            s = torch.tanh(s)
            tmp = torch.norm(s, dim=-1, keepdim=True)
            # # tmp_d = tmp.clone()  # todo!!!
            # # tmp_d[tmp_d == 0] = 1.0
            tmp = tmp + eps
            # print('tmp',tmp)
            new_M_emb = ((tmp * tmp) / (1 + tmp * tmp)) * (s / tmp)  # B*M*D #todo

        return new_M_emb#todo


class DMAN_layer(nn.Module):
    def __init__(self, seq_num, seq_len, dim, M, device):
        super(DMAN_layer, self).__init__()
        self.device = device
        self.T = seq_len
        self.S = seq_num
        self.D = dim
        self.M = M
        self.short_net = Short_term(seq_num, seq_len,  dim, device)
        self.long_net = Long_term(seq_num, seq_len,  dim, device)
        self.DM = DynamicMemory(seq_num, seq_len,  dim, M, device)

    def forward(self, Ht, H, Hh, M_emb):
        # Ht B*S*T*D
        # H B*S*T*D
        # Hh B*S*T*D
        # M_emb B*M*D
        # print('Layer H',Ht.shape,"M_emb",M_emb.shape)
        M = M_emb.shape[0]
        short_Ht = Ht.reshape(-1,self.T, self.D).transpose(0, 1)
        short_H = H.reshape(-1, self.T, 2*self.D).transpose(0, 1)
        # print('short Ht',short_Ht.shape,'short_H',short_H.shape)
        new_Ht, new_H = self.short_net(short_Ht, short_H)#T*(B*S)*D  T*(B*S)*(D+D)
        long_Hh = Hh.reshape(-1, self.T, self.D).transpose(0, 1)#T*(B*S)*D
        tmp_M_emb = M_emb.unsqueeze(1).expand(-1, self.S, -1, -1)#B*S*M*D
        long_M_emb = tmp_M_emb.reshape(-1, M, self.D).transpose(0, 1)#M*(B*S)*D
        new_Hh = self.long_net(long_Hh, long_M_emb)#T*(B*S)*D

        new_M_emb = M_emb
        for i in range(self.S):
            new_M_emb = self.DM(new_M_emb, Ht[:, i, :, :])
        # new_M_emb = self.DM(new_M_emb, Ht[:, 0, :, :])#todo
        new_Ht = new_Ht.transpose(0, 1).reshape(-1, self.S, self.T, self.D)
        new_H = new_H.transpose(0, 1).reshape(-1, self.S, self.T, self.D)
        new_Hh = new_Hh.transpose(0, 1).reshape(-1, self.S, self.T, self.D)

        return new_Ht, new_H, new_Hh, new_M_emb


class DMAN(nn.Module):
    def __init__(self, seq_num, seq_len, dim, layer_num, item_num, M, device):
        super(DMAN, self).__init__()
        self.T = seq_len
        self.S = seq_num
        self.D = dim
        self.L = layer_num
        self.M = M
        self.device = device
        self.Layers = nn.ModuleList([DMAN_layer(seq_num, seq_len, dim, M, device) for _ in range(layer_num)])
        self.Gate = Gating(seq_num, seq_len,  dim, device)
        self.item_emb = nn.Embedding(item_num, dim, padding_idx=0)
        nn.init.uniform_(self.item_emb.weight[1:], a=-0.5 / item_num, b=0.5 / item_num)

    def forward(self, batch):
        seq, pos, neg = batch
        seq = seq.to(self.device)
        pos = pos.to(self.device)
        neg = neg.to(self.device)
        #seq B*S*T
        #pos B*S*T
        #neg B*S*T*neg_size
        neg_size = neg.shape[-1]
        Ht = self.item_emb(seq)
        H = torch.cat((Ht.clone(), Ht.clone()),dim=-1)#B*S*T*(D+D)
        Hh = Ht.clone()
        M_emb = torch.rand(seq.shape[0], self.M, self.D).to(self.device)
        # print("ST H",H.shape,"Ht",Ht.shape,"M_EMB",M_emb.shape)
        for layer in self.Layers:
            Ht, H, Hh, M_emb = layer(Ht, H, Hh, M_emb)
        V = self.Gate(Ht, Hh)#B*S*T*D

        pos_emb = self.item_emb(pos) #B*S*T*D
        pos_logits = (V * pos_emb).sum(-1,keepdim=True) #B*S*T*1

        neg_emb = self.item_emb(neg) #B*S*T*neg*D
        V = V.unsqueeze(-2)
        neg_logits = (V * neg_emb).sum(-1) #B*S*T*neg

        all_items = torch.cat([pos.unsqueeze(-1), neg], dim=-1)  # B*S*T (1 + ns)
        all_indices = torch.where(all_items != 0)
        logits = torch.cat([pos_logits, neg_logits], dim=-1)  # B*S*T (1 + ns)
        logits = logits[all_indices].view(-1, 1 + neg_size)
        labels = torch.zeros((logits.shape[0])).long().to(self.device)

        loss = F.cross_entropy(logits, labels)
        item_norm = pos_emb.norm(2, dim=-1).pow(2).mean() + neg_emb.norm(2, dim=-1).pow(2).mean()
        loss += item_norm
        return loss

    def get_parameters(self):
        param_list = []
        for i in range(self.L):
            param_list.append({'params': self.Layers[i].short_net.ATT.parameters()})
            param_list.append({'params': self.Layers[i].long_net.ATT.parameters()})
            param_list.append({'params': self.Layers[i].DM.W})
        param_list.append({'params': self.Gate.short_linear.parameters()})
        param_list.append({'params': self.Gate.long_linear.parameters()})
        param_list.append({'params': self.item_emb.parameters(), 'weight_decay': 0})

        return param_list


    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for i, eval_batch in enumerate(eval_loader):
                seq, eval_iid = eval_batch
                seq = seq.long().to(self.device)
                seq = seq.reshape(-1, self.S, self.T)
                eval_iid = eval_iid.long().to(self.device)
                Ht = self.item_emb(seq)
                H = torch.cat((Ht.clone(), Ht.clone()),dim=-1)
                Hh = Ht.clone()
                # print("eval Ht",Ht.shape,'H',H.shape)
                M_emb = torch.rand(seq.shape[0], self.M, self.D).to(self.device)
                for layer in self.Layers:
                    Ht, H, Hh, M_emb = layer(Ht, H, Hh, M_emb)

                eval_item_emb = self.item_emb(eval_iid)  # B*item_len*D
                V = self.Gate(Ht, Hh)  # B*S*T*D
                V = V[:,-1,-1,:].unsqueeze(-2).expand_as(eval_item_emb)#B*item_len*D
                batch_score = (V * eval_item_emb).sum(-1)
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores

    def epin_eval_all_users(self, eval_batch):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            seq, eval_iid = eval_batch
            seq = seq.long().to(self.device)
            seq = seq.reshape(-1, self.S, self.T)
            eval_iid = eval_iid.long().to(self.device)
            Ht = self.item_emb(seq)
            H = torch.cat((Ht.clone(), Ht.clone()),dim=-1)
            Hh = Ht.clone()
            # print("eval Ht",Ht.shape,'H',H.shape)
            M_emb = torch.rand(seq.shape[0], self.M, self.D).to(self.device)
            for layer in self.Layers:
                Ht, H, Hh, M_emb = layer(Ht, H, Hh, M_emb)

            eval_item_emb = self.item_emb(eval_iid)  # B*item_len*D
            V = self.Gate(Ht, Hh)  # B*S*T*D
            V = V[:,-1,-1,:].unsqueeze(-2).expand_as(eval_item_emb)#B*item_len*D
            batch_score = (V * eval_item_emb).sum(-1).squeeze()

        return batch_score



