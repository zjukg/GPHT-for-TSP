import logging
from math import sqrt, ceil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from CompGCN.models import CompGCNBase
from data import DataBase
from tqdm import tqdm

class GNNComplete(nn.Module):
    def __init__(self, database: DataBase, para) -> None:
        super().__init__()
        self.database = database
        self.para = para
        self.pad_idx = para.padIdx
        self.topk = para.topk
        self.gamma = para.gamma
        self.device = para.device
        self.score_func = para.score_func.lower() #'hake'

        self.embed_dim = para.embed_dim

        self.drop = torch.nn.Dropout(para.hid_drop)
        self.gnn = CompGCNBase(para.num_rel, para)

        self.mlp = nn.Sequential(nn.Linear(para.embed_dim * 2 + 1 + para.num_rel * 2, 2048), nn.LeakyReLU(negative_slope=1e-4), nn.Dropout(p=0.7),
                                 nn.Linear(2048, 512), nn.LeakyReLU(negative_slope=1e-3), nn.Dropout(p=0.4),
                                 nn.Linear(512, 1), nn.Sigmoid())

        if self.score_func == 'pairre':
            self.trans = nn.Linear(para.embed_dim, para.embed_dim//2)
        elif self.score_func == 'hake':
            self.trans = nn.Linear(para.embed_dim, para.embed_dim//3*2)
        else:
            self.trans = None
        if self.score_func == 'hake':
            self.q = nn.Linear(para.embed_dim // 3 * 2, 128)
            self.k = nn.Linear(para.embed_dim // 3 * 2, 128)
        elif self.score_func == 'pairre':
            self.q = nn.Linear(para.embed_dim // 2, 128)
            self.k = nn.Linear(para.embed_dim // 2, 128)
        self.kgeloss = {'transe': KGELoss.transe, 'rotate': KGELoss.rotate, 'hake': KGELoss.hake, 'pairre': KGELoss.pairre}[self.score_func]

    def decoder(self, graphEmb: torch.Tensor):
        pred = F.relu(self.tmp_decode(graphEmb))
        return pred.chunk(3, dim=-1)

    def forward(self, edge_index, edge_type, toTest, mode, negEnts=None, triples=None):
        all_ent_emb, all_rel_emb = self.gnn.forward_base(None, None, self.drop, self.drop, edge_index, edge_type) #[2378， 1024] [24, 1024]  #在117 step时候， all_ent_emb 和all_rel_emb都为nan
        fetch = {}
        if torch.any(torch.isnan(all_ent_emb)):
            print(1)
        if mode == "train":
            fetch["ent_emb"] = all_ent_emb
            fetch["rel_emb"] = all_rel_emb
            if self.para.pretrain:
                h, r, t = triples[:, 0].unsqueeze(-1), triples[:, 1].unsqueeze(-1), negEnts.unsqueeze(0)
                fetch["kge"] = self.kgeloss(h, r, t, all_ent_emb, all_rel_emb, self.trans)
            else:
                fetch["kge"] = []
                cnt = ceil(edge_type.shape[0] / 4096)
                for ed_in, ed_ty in zip(torch.chunk(edge_index, cnt, dim=-1), torch.chunk(edge_type, cnt, dim=-1)):
                    fetch["kge"].append(self.kgeloss(ed_in[0], ed_ty, ed_in[1], all_ent_emb, all_rel_emb, self.trans))
                fetch["kge"] = torch.cat(fetch["kge"], dim=-1)

        if (not self.para.pretrain) and (type(toTest) != type(None)):
            if self.score_func == 'hake':
                cnt = ceil(toTest.shape[0] / 4096)
                fetch["testSco"] = []
                phase_relation, mod_relation, bias_relation = torch.chunk(all_rel_emb, 3, dim=-1)
                t_bias_relation = (1-bias_relation).clamp(min=1e-6)
                mod_relation = (mod_relation + bias_relation) / t_bias_relation
                
                for tes in torch.chunk(toTest, cnt, dim=0):
                    testEmb = all_ent_emb[tes]
                    entsEmb = self.trans(testEmb)
                    atta = self.q(entsEmb[:, 0]) * self.k(entsEmb[:, 1])
                    atta = (atta.sum(-1) / sqrt(self.embed_dim)).reshape(-1, 1)
                    h_p, h_m = torch.chunk(entsEmb[:, 0], 2, dim=-1)
                    h_m = h_m.clamp(min=1e-6)
                    t_p, t_m = torch.chunk(entsEmb[:, 1], 2, dim=-1)
                    relSim = (t_p - h_p) @ phase_relation.t() + (t_m / h_m) @ mod_relation.t()
                    
                    cat = torch.cat([testEmb[:, 0], testEmb[:, 1], atta, relSim], dim=1)
                    ht_w = self.mlp(cat)
                    fetch["testSco"].append(ht_w)
                fetch["testSco"] = torch.cat(fetch["testSco"], dim=0)
                if torch.any(torch.isnan(fetch["testSco"])):
                        print(1)
            elif self.score_func == 'pairre':
                cnt = ceil(toTest.shape[0] / 4096)
                fetch["testSco"] = []
                head_relation, tail_relation = torch.chunk(all_rel_emb, 2, dim=-1)  #[24, 512]
                
                for tes in torch.chunk(toTest, cnt, dim=0):
                    testEmb = all_ent_emb[tes]  #tes:[1521, 2]   #[1521, 2, 1024]
                    entsEmb = self.trans(testEmb)  #[1521, 2, 512]
                    atta = self.q(entsEmb[:, 0]) * self.k(entsEmb[:, 1])
                    atta = (atta.sum(-1) / sqrt(self.embed_dim)).reshape(-1, 1)
                    
                    
                    relSim = entsEmb[:,0]@head_relation.t() - entsEmb[:,1]@tail_relation.t()
                    fetch["testSco"].append(self.mlp(torch.cat([testEmb[:, 0], testEmb[:, 1], atta, relSim], dim=1)))
                fetch["testSco"] = torch.cat(fetch["testSco"], dim=0)
                


        return fetch

    def loss_pre(self, pred, true_label):
        pred = pred / pred.sum(dim=-1).unsqueeze(-1)
        return pred[true_label].sum()

    def loss(self, pred, label, toAdd) -> torch.Tensor:
        if torch.any(torch.isnan(pred[~label])):
            print(1)
        loss = pred[~label].mean()
        if torch.any(torch.isnan(loss)):
            print(1)
        if toAdd.any():
            loss = loss + (1 - pred[toAdd]).mean()
        return loss


class KGELoss:
    @staticmethod
    def transe(h, r, t, embE, embR, trans, scos=None) -> torch.Tensor:
        head = torch.index_select(embE, 0, h)
        rel = torch.index_select(embR, 0, r)
        tail = torch.index_select(embE, 0, t)
        if type(scos) == type(None):
            score = (head + rel - tail).norm(-1)
        else:
            score = ((head - tail + rel).norm(-1) * scos.unsqueeze(1)).view(-1)
        return score

    @staticmethod
    def rotate(h, r, t, embE, embR, trans, scos=None, embedding_range=10) -> torch.Tensor:
        head = torch.index_select(embE, 0, h)
        relation = trans(torch.index_select(embR, 0, r))
        tail = torch.index_select(embE, 0, t)

        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        phase_relation = relation / (embedding_range/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).norm(dim=-1)

        if type(scos) != type(None):
            score = (score * scos.unsqueeze(1)).view(-1)
        return score

    @staticmethod
    def hake(h, r, t, embE, embR, trans, scos=None, embedding_range=10, phase_weight=0.5, modulus_weight=1.0) -> torch.Tensor:
        head = trans(embE[h]) #[1,1,682]
        relation = embR[r]  #[1, 1, 1023]
        tail = trans(embE[t])

        phase_head, mod_head = torch.chunk(head, 2, dim=-1)
        phase_relation, mod_relation, bias_relation = torch.chunk(relation, 3, dim=-1)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=-1)

        pi = 3.14159265358979323846
        phase_head = phase_head / (embedding_range / pi)
        phase_relation = phase_relation / (embedding_range / pi)
        phase_tail = phase_tail / (embedding_range / pi)

        if type(scos) == type(None):
            phase_score = (phase_head + phase_relation) - phase_tail
        else:
            phase_score = (phase_head - phase_tail) + phase_relation

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=-1) * phase_weight
        r_score = torch.norm(r_score, dim=-1) * modulus_weight

        score = phase_score + r_score
        if type(scos) != type(None):
            score = (score * scos.unsqueeze(1)).view(-1)

        return score
    @staticmethod
    def pairre(h, r, t, embE, embR, trans, scos=None) ->torch.Tensor:
        head = F.normalize(trans(embE[h]), 2, -1)
        tail = F.normalize(trans(embE[t]), 2, -1)
        relation = embR[r]
        re_head, re_tail = torch.chunk(relation, 2, dim=-1)
        score = torch.norm((head*re_head - tail*re_tail), p=1, dim=-1)
        if type(scos) != type(None):
            score = (score * scos.unsqueeze(1)).view(-1)
        
        return score

