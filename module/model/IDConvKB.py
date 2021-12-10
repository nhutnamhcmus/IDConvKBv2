import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

import numpy as np

class IDConvKB(Model):
    def __init__(self, ent_tot, rel_tot, dim=100, perm=1, k_h=10, k_w=5, input_drop=0.2, hidden_drop=0.5, feature_drop=0.5, num_of_filters=32, kernel_size=3):
        super().__init__(ent_tot, rel_tot)
        self.perm = perm 
        self.k_h = k_h
        self.k_w = k_w
        self.num_of_filters = num_of_filters

        self.input_drop_rate = input_drop
        self.hidden_drop_rate = hidden_drop
        self.feature_drop_rate = feature_drop
        self.kernel_size = kernel_size

        self.dim_e = dim
        self.dim_r = dim
        # Khởi tạo embedding cho thực thể và quan hệ
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)

        self.ent_transfer = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r)

        self.inp_drop = torch.nn.Dropout(self.input_drop_rate)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
        self.feature_map_drop = torch.nn.Dropout2d(self.feature_drop_rate)

        flat_sz_h = self.k_h
        flat_sz_w = 2 * self.k_w

        self.flat_sz = flat_sz_h * flat_sz_w * self.num_of_filters*self.perm
        self.padding = 0

        self.bn0 = torch.nn.BatchNorm2d(self.perm)
        self.bn1 = torch.nn.BatchNorm2d(self.num_of_filters*self.perm)
        self.bn2 = torch.nn.BatchNorm1d(1)

        self.fc = torch.nn.Linear(self.flat_sz, 1)

        self.register_parameter('bias', nn.Parameter(torch.zeros(self.ent_tot)))
        self.register_parameter('conv_filt', nn.Parameter(torch.zeros(self.num_of_filters, 1, self.kernel_size,  self.kernel_size)))

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_transfer.weight.data)
        nn.init.xavier_uniform_(self.rel_transfer.weight.data)
        nn.init.xavier_normal_(self.conv_filt)
        self.chequer_perm = self.get_chequer_perm()

    def _resize(self, tensor, axis, size):
        shape = tensor.size()
        osize = shape[axis]
        if osize == size:
            return tensor
        if (osize > size):
            return torch.narrow(tensor, axis, 0, size)
        paddings = []
        for i in range(len(shape)):
            if i == axis:
                paddings = [0, size - osize] + paddings
            else:
                paddings = [0, 0] + paddings
        return F.pad(tensor, paddings = paddings, mode = "constant", value = 0)

    def _transfer(self, e, e_transfer, r_transfer):
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], e.shape[-1])
            e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
            r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
            e = F.normalize(
				self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
				p = 2, 
				dim = -1
			)	
            return e.view(-1, e.shape[-1])
        else:
            return F.normalize(
				self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
				p = 2, 
				dim = -1
			)

    def circular_padding_chw(self, batch, padding):
        upper_pad	= batch[..., -padding:, :]
        lower_pad	= batch[..., :padding, :]
        temp		= torch.cat([upper_pad, batch, lower_pad], dim=2)
        
        left_pad	= temp[..., -padding:]
        right_pad	= temp[..., :padding]
        padded		= torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def get_chequer_perm(self):
        ent_perm    = np.int32([np.random.permutation(self.dim_e) for _ in range(self.perm)])
        rel_perm    = np.int32([np.random.permutation(self.dim_r) for _ in range(self.perm)])
        comb_idx    = []
        for k in range(self.perm):
            temp = []
            ent_idx, rel_idx = 0, 0
            
            for i in range(self.k_h):
                for j in range(self.k_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx]+self.dim_r)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx]+self.dim_r)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx]); ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx]+self.dim_e)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx]+self.dim_e)
                            rel_idx += 1
            comb_idx.append(temp)
        
        chequer_perm = torch.LongTensor(np.int32(comb_idx)).cuda()
        return chequer_perm

    def _calc(self, h, r, t, mode):
        if mode == 'head_batch':
            comb_emb = torch.cat([t, r], dim=1)
            chequer_perm	= comb_emb[:, self.chequer_perm]
            stack_inp	= chequer_perm.reshape((-1, self.perm, 2*self.k_w, self.k_h))
            stack_inp	= self.bn0(stack_inp)
            x		= self.inp_drop(stack_inp)
            x		= self.circular_padding_chw(x, self.kernel_size//2)
            x		= F.conv2d(x, self.conv_filt.repeat(self.perm, 1, 1, 1), padding=self.padding, groups=self.perm)
            x		= self.bn1(x)
            x		= F.relu(x)
            x		= self.feature_map_drop(x)
            x		= x.view(-1, self.flat_sz)
            x		= self.fc(x)
            x		= self.hidden_drop(x)
            x		= self.bn2(x)
            x       = torch.sigmoid(x)
        else:
            comb_emb = torch.cat([h, r], dim=1)
            chequer_perm	= comb_emb[:, self.chequer_perm]
            stack_inp	= chequer_perm.reshape((-1, self.perm, 2*self.k_w, self.k_h))
            stack_inp	= self.bn0(stack_inp)
            x		= self.inp_drop(stack_inp)
            x		= self.circular_padding_chw(x, self.kernel_size//2)
            x		= F.conv2d(x, self.conv_filt.repeat(self.perm, 1, 1, 1), padding=self.padding, groups=self.perm)
            x		= self.bn1(x)
            x		= F.relu(x)
            x		= self.feature_map_drop(x)
            x		= x.view(-1, self.flat_sz)
            x		= self.fc(x)
            x		= self.hidden_drop(x)
            x		= self.bn2(x)
            x       = torch.sigmoid(x)
        return x

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t) 
        r = self.rel_embeddings(batch_r)

        h_transfer = self.ent_transfer(batch_h)
        r_transfer = self.rel_transfer(batch_r)
        t_transfer = self.ent_transfer(batch_t)

        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)

        score = self._calc(h, r, t, mode)
        return score

    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t) 
        r = self.rel_embeddings(batch_r)

        h_transfer = self.ent_transfer(batch_h)
        r_transfer = self.rel_transfer(batch_r)
        t_transfer = self.ent_transfer(batch_t)

        regul = (torch.mean(h ** 2) +
                torch.mean(t ** 2) +
                torch.mean(r ** 2) + 
                torch.mean(h_transfer ** 2) +
                torch.mean(t_transfer ** 2) +
                torch.mean(r_transfer ** 2)
        ) / 6

        return regul




