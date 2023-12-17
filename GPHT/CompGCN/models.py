import torch
from torch.nn import Parameter
from torch.nn import functional as F

from .compgcn_conv import CompGCNConv
from .compgcn_conv_basis import CompGCNConvBasis
from .helper import get_param


class CompGCNBase(torch.nn.Module):
    def __init__(self, num_rel, params=None):
        super(CompGCNBase, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()

        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))

        if self.p.num_bases > 0:
            self.init_rel = get_param((self.p.num_bases, self.p.init_dim))
        else:
            if self.p.score_func == 'transe':
                self.init_rel = get_param((num_rel, self.p.init_dim))
            else:
                self.init_rel = get_param((num_rel*2, self.p.init_dim))

        if self.p.num_bases > 0:
            self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
        else:
            self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)

        if self.p.gcn_layer == 2:
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act, params=self.p)

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.NoneType = type(None)

    def forward_base(self, sub, rel, drop1, drop2, edge_index, edge_type):
        r = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
        if torch.any(torch.isnan(r)):
            print(1)
        x, r = self.conv1(self.init_embed, edge_index, edge_type, rel_embed=r)
        if torch.any(torch.isnan(x)):
            print(1)
        if torch.any(torch.isnan(r)):
            print(1)
        x = drop1(x)
        if torch.any(torch.isnan(x)):
            print(1)
        if self.p.gcn_layer == 2:
            x, r = self.conv2(x, edge_index, edge_type, rel_embed=r)
            x = drop2(x)

        if type(sub) == self.NoneType or type(rel) == self.NoneType:
            return x, r

        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, x

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class CompGCN_TransE(CompGCNBase):
    def __init__(self, params=None):
        super(self.__class__, self).__init__(params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel, edge_index, edge_type):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop, edge_index, edge_type)
        obj_emb = sub_emb + rel_emb

        x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)

        return score


class CompGCN_DistMult(CompGCNBase):
    def __init__(self, params=None):
        super(self.__class__, self).__init__(params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel, edge_index, edge_type):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop, edge_index, edge_type)
        obj_emb = sub_emb * rel_emb

        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class CompGCN_ConvE(CompGCNBase):
    def __init__(self, params=None):
        super(self.__class__, self).__init__(params.num_rel, params)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(
            self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2*self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h*flat_sz_w*self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed. view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel, edge_index, edge_type):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, edge_index, edge_type)
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score
