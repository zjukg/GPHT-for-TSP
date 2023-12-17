import logging
from math import sqrt
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict as ddict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import BatchType, TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class KGEModel(nn.Module, ABC):
    @abstractmethod
    def func(self, head, rel, tail, batch_type):
        ...

    def forward(self, sample, batch_type=BatchType.SINGLE):
        if batch_type == BatchType.SINGLE:
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.HEAD_BATCH:
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.TAIL_BATCH:
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)   #[100, 1, 1000]

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)   #[100, 1, 1500]

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)  #[100, 64, 1000]

        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))

        return self.func(head, relation, tail, batch_type)  #[1,1, 1000] ,[1,1,1500],

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        positive_sample = positive_sample.cuda() #[100, 3]
        negative_sample = negative_sample.cuda()  #[100, 64]
        subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), batch_type=batch_type)

        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args, steps, r_dict):
        logging.info('test_step')
        model.eval()

        bigData = True
        scoreSum = 0

        nents = len(data_reader.entity_dict)
        nrels = len(data_reader.relation_dict)
        ents = torch.LongTensor(range(nents))
        scores = {} if bigData else []

        thrs = [args.thr]
        length = nents * nents * nrels
        if args.thr < 0:
            #thrs = [500, 250, 200, 150, 100, 50, 20]
            thrs = [20,10,5, 3, 1, 0.1, 0.5]
            #if args.model == 'HAKE':
                #thrs = [1500,1000,800,650,500,150,100,60,50,30,15,10,5,1,0.05,0.01]
                #thrs = [20,15,12.5,10,7.5,5,3,1]
            if args.model == 'PairRE':
                thrs = [5000,3000, 2800, 2500, 2400, 2350, 2300, 2250, 2200, 2150,2100, 2050, 2000]
                #thrs = [2000,1700,1500,1450,1400,1350,1300,1250,1200,1150,1100,1050,1000,800,750,700,650,600,550,510,500,200,150,100,75,50,20]
        minThr = min(thrs) # 0.01

        def idx2hrt(ind):
            return (ind//nents//nrels, ind//nents % nrels, ind % nents)

        with torch.no_grad():
            for head in tqdm(range(nents), desc="-Head", ncols=60):
                for rel in tqdm(range(nrels), desc="    Relation", ncols=60, leave=False):
                    positive_sample = torch.LongTensor([[head, rel, -1]]).cuda()  #[1, 3]
                    negative_sample = ents.cuda().clone().unsqueeze(0) #[1, 2378]

                    score = model((positive_sample, negative_sample), BatchType.TAIL_BATCH)  #[1, 2378]
                    if bigData:
                        score.exp_()
                        scoreSum += score.sum().item()
                    else:
                        scores.extend(score.clone().detach().cpu())

        if bigData:
            scores = {}
            with torch.no_grad():
                for head in tqdm(range(nents), desc="+Head", ncols=60):
                    for rel in tqdm(range(nrels), desc="    Relation", ncols=60, leave=False):
                        positive_sample = torch.LongTensor([[head, rel, -1]]).cuda()
                        negative_sample = ents.cuda().clone().unsqueeze(0)
                        score = model((positive_sample, negative_sample), BatchType.TAIL_BATCH)[0]
                        score = score.exp() * length / scoreSum
                        idx = torch.nonzero(score>minThr, as_tuple=False).squeeze()
                        num_idx = torch.numel(idx)
                        if num_idx == 0:
                            continue
                        hrt = head*nrels*nents + rel*nents + idx
                        sc = score[idx]
                        if num_idx == 1:
                            hrt = hrt.unsqueeze(0).tolist()
                            sc = sc.unsqueeze(0).tolist()
                        else:
                            hrt = hrt.tolist()
                            sc = sc.tolist()
                        for k, v in zip(hrt, sc):
                            scores[k] = max(scores.setdefault(k, -float('inf')), v)

        

            scores = sorted(scores.items(), key=lambda x: -x[1])
        else:
            scores = torch.cat(scores, dim=0).softmax(dim=0).unsqueeze(0) * length
            raise "torch.cat may change indices +-1"
            values, indices = [x.cpu().tolist() for x in scores.sort(dim=1, descending=True)]
            scores = torch.cat(scores.sort(dim=1, descending=True), dim=0).T.tolist()

        bestF1 = -1
        bestMRR = -100
        #for owl in [True, False]:  #这里是直接close world 和open world 都考虑了, true为close world  false为open world
        for owl in [False]:    
            for test in ['VALID', 'TEST']:
                truthTriples = set(data_reader.train_data + (data_reader.valid_data if test == 'TEST' else []))
                testTriples = set(data_reader.test_data if test == 'TEST' else data_reader.valid_data)
                metrics = {}
                rank = 0
                f_len = 0
                trueCnt, addedTrueCnt, idx = 0, 0, 0
                truthRank, truthRevRank, Len = 0, 0, len(scores)
                for thr in sorted(thrs, reverse=True):
                    while idx < Len and scores[idx][1] >= thr: #67858608
                        hrt = idx2hrt(scores[idx][0])                        
                        if hrt in truthTriples:
                            idx +=1
                            continue
                        r_other = r_dict[hrt[1]]
                        trueCnt +=1
                        if (hrt[0], hrt[1], hrt[2]) in testTriples:
                            addedTrueCnt +=1
                            rank +=1/trueCnt
                            idx +=1
                            continue
                        #判断是否是错误的
                        for r_o in r_other:                                                       
                            if (hrt[0], r_o, hrt[2]) in truthTriples:
                                rank -=1/trueCnt
                                f_len +=1
                                break
                        idx += 1
                    metrics['add'] = trueCnt   #74076， 72569， 7  预测的总的三元组
                    metrics['true'] = addedTrueCnt 
                    metrics['op'] = addedTrueCnt + f_len
                    metrics['pre'] = (addedTrueCnt/trueCnt + addedTrueCnt/metrics['op'])/2 if trueCnt and metrics['op'] else -1
                    metrics['recall'] = sqrt(metrics['true'] / len(testTriples))
                
                    metrics['MRR'] = rank
                    metrics['F1'] = 2*metrics['pre']*metrics['recall'] / (metrics['pre']+metrics['recall']) if (metrics['pre']+metrics['recall']) else 0
                    if test == 'VALID' and args.best_evaluate == 'F1':
                        if metrics['F1'] > bestF1: bestF1 = metrics['F1']
                    elif test == 'VALID' and args.best_evaluate == 'MRR':
                        if metrics['MRR'] > bestMRR: bestMRR = metrics['MRR']
                    
                    with open(f"{args.save_path}/../metric-False-SP{args.perfix}-data{args.data}-model{args.model}-eval{args.best_evaluate}_1.txt", 'a') as me:
                        me.write("save_path:{}".format(args.save_path))
                        me.write("\n")
                        me.write('"split":"{}", "owl":"{}", "label":"{}-{}-{}",  "len_test":{}, "F1":{}, "add":{}, "true":{},"pre":{},"recall":{},"op":{},"MMR":{},'.format(
                            test, owl, args.model, steps, thr, len(testTriples), *[metrics[m] for m in ['F1','add','true','pre','recall','op','MRR']]))
                        me.write("}\n")
        logging.info('test_step done')
        if args.best_evaluate =='F1':
            metrics['F1'] = bestF1
        elif args.best_evaluate == 'MRR':
            metrics['MRR'] = bestMRR
        return metrics

    def GNNtest(self, model, data_reader, ht_scos, args, steps, r_dict):
        logging.info('GNNtest')
        model.eval()
        nent = len(data_reader.entity_dict)
        nrel = len(data_reader.relation_dict)

        cnt = ht_scos[0].shape[0] // 4096
        batches = [torch.chunk(x, cnt, dim=0) for x in ht_scos]
        batches = [x for x in zip(*batches)]

        addscores = {}
        scoreSum = 0
        with torch.no_grad():
            for ht, sco in tqdm(batches, desc="1-ht", ncols=60):
                head = self.entity_embedding[ht[:, 0]].unsqueeze(1)
                tail = self.entity_embedding[ht[:, 1]].unsqueeze(1)
                relation = self.relation_embedding.unsqueeze(0)

                score = self.func(head, relation, tail, BatchType.RELA_BATCH)
                scoreSum += score.view(-1).exp().sum().item()

        def idx2hrt(ind):
            return (ind//nent//nrel, ind//nent % nrel, ind % nent)

        thrs = [args.thr]
        if args.data == 'wiki_select_new'  and  args.model=='HAKE':    
            #thrs = [1500,1200,1000,800,650,500, 150, 100, 60, 50, 30, 15, 10, 5]
            thrs = [ 100, 60, 50, 30, 15, 10, 5, 3, 1, 0.5]
        elif args.data == 'wiki_select_new'  and  args.model=='PairRE':
            thrs = [500, 150, 100, 60, 50, 30, 15, 10, 5, 3, 1, 0.5]
        elif args.data == 'wiki_select_new_3'  and  args.model== 'HAKE':    
            #thrs = [500, 150, 100, 60, 50, 40, 30, 10]
            #thrs = [600,500,350,300,250,200,150,100,75,50,30,20,10,5,3,1,0.5,0.25,0.1,0.05,0.01]
            #thrs = [5000,3000,2000]
            thrs = [10,5,3,1,0.5,0.1,0.05,0.01]
        elif args.data =="wiki_select_new_3" and args.model == 'PairRE':
            thrs = [1,5,10,20,30,50,100]
        
        minThr = min(thrs)
        length = ht_scos[1].shape[0] * nrel
        addscores = {}
        with torch.no_grad():
            for ht, sco in tqdm(batches, desc="2-ht", ncols=60):
                head = self.entity_embedding[ht[:, 0]].unsqueeze(1)
                tail = self.entity_embedding[ht[:, 1]].unsqueeze(1)
                relation = self.relation_embedding.unsqueeze(0)

                score = self.func(head, relation, tail, BatchType.RELA_BATCH)
                score = score.view(-1).exp() * length / scoreSum

                addscores.update({hrt: sc for idx in torch.where(score >= minThr)[0] if addscores.get(
                    (hrt := (ht[(n := idx.item()//nrel),0]*nrel*nent + (idx%nrel)*nent + ht[n,1]).item()), -1e9) < (sc := score[idx].item())})

        addscores = sorted(addscores.items(), key=lambda x: -x[1])

        bestF1 = -1
        bestMRR = -1000
        for owl in [False]:    
            for test in ['VALID', 'TEST']:
                
                truthTriples = set(data_reader.train_data + (data_reader.valid_data if test == 'TEST' else []))
                testTriples = set(data_reader.test_data if test == 'TEST' else data_reader.valid_data)
                metrics = {}
                rank = 0
                f_len = 0
                trueCnt, addedTrueCnt, idx , truth_idx= 0, 0, 0, 0
                truthRank, truthRevRank, Len = 0, 0, len(addscores)
                for thr in sorted(thrs, reverse=True):
                    while idx < Len and addscores[idx][1] >= thr: #67858608
                        hrt = idx2hrt(addscores[idx][0])                        
                        if hrt in truthTriples:
                            idx +=1
                            continue
                        
                        r_other = r_dict[hrt[1]]
                        trueCnt +=1
                        if trueCnt >100000:
                            break
                        if (hrt[0], hrt[1], hrt[2]) in testTriples:
                            addedTrueCnt +=1
                            rank +=1/trueCnt
                            idx +=1
                            continue
                        #判断是否是错误的
                        
                        for r_o in r_other:                                                       
                            if (hrt[0], r_o, hrt[2]) in truthTriples:
                                rank -=1/trueCnt
                                f_len +=1
                                break
                       
                        idx += 1
                    metrics['add'] = trueCnt   #74076， 72569， 7  预测的总的三元组
                    metrics['true'] = addedTrueCnt
                    metrics['op'] = addedTrueCnt + f_len
                    metrics['pre'] = (addedTrueCnt/trueCnt + addedTrueCnt/metrics['op'])/2 if trueCnt and metrics['op'] else -1
                    metrics['recall'] = sqrt(metrics['true'] / len(testTriples))
                    metrics['MRR'] = rank
                    metrics['F1'] = 2*metrics['pre']*metrics['recall'] / (metrics['pre']+metrics['recall']) if (metrics['pre']+metrics['recall']) else 0
                    
                    if test == 'VALID' and args.best_evaluate == 'F1':
                        if metrics['F1'] > bestF1: bestF1 = metrics['F1']
                    elif test == 'VALID' and args.best_evaluate == 'MRR':
                        if metrics['MRR'] > bestMRR: bestMRR = metrics['MRR']
                    with open(f"{args.save_path}/../{args.bs}-GNN-metric-False-SP{args.perfix}-data{args.data}-model{args.model}-eval{args.best_evaluate}.txt", 'a') as me:
                        me.write("save_path:{}".format(args.save_path))
                        me.write("}\n")
                        me.write('"split":"{}", "owl":"{}", "label":"{}-{}-{}",  "len_test":{}, "F1":{}, "add":{}, "true":{},"pre":{},"recall":{},"op":{},"MRR":{},'.format(
                            test, owl, args.model, steps, thr, len(testTriples), *[metrics[m] for m in ['F1','add','true','pre','recall',"op",'MRR']]))
                        me.write("}\n")
        logging.info('GNNtest done')
        if args.best_evaluate =='F1':
            metrics['F1'] = bestF1
        elif args.best_evaluate == 'MRR':
            metrics['MRR'] = bestMRR
        return metrics
        


class HAKE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, modulus_weight=1.0, phase_weight=0.5):
        super(HAKE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 3))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.ones_(
            tensor=self.relation_embedding[:, hidden_dim:2 * hidden_dim]
        )

        nn.init.zeros_(
            tensor=self.relation_embedding[:, 2 * hidden_dim:3 * hidden_dim]
        )

        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))

        self.pi = 3.14159262358979323846

    def func(self, head, rel, tail, batch_type):
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        if batch_type == BatchType.HEAD_BATCH:
            phase_score = phase_head + (phase_relation - phase_tail)
        elif batch_type == BatchType.RELA_BATCH:
            phase_score = (phase_head - phase_tail) + phase_relation
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)

class PairRE(KGEModel):
    
    def __init__(self, num_entity, num_relation, hidden_dim, gamma):
        super(PairRE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False,
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False,
        )
        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a = -self.embedding_range.item(),
            b = self.embedding_range.item(),
        )
        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a = -self.embedding_range.item(),
            b = self.embedding_range.item(), 
        )

    def func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.
        
        The formula for calculating the score is :math:`\gamma - || h \circ r^H - t  \circ r^T ||`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        re_head, re_tail = torch.chunk(relation_emb, 2, dim=2)

        head = F.normalize(head_emb, 2, -1)
        tail = F.normalize(tail_emb, 2, -1)

        score = head * re_head - tail  * re_tail
        return self.gamma.item() - torch.norm(score, p=1, dim=2)
    

