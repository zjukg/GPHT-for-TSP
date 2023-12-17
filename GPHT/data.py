import logging
import os
import math
import pickle
import random
import sys
from collections import defaultdict as ddict

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class __LINE__(object):
    def __repr__(self):
        try:
            raise Exception
        except:
            return str(sys.exc_info()[2].tb_frame.f_back.f_lineno)

__LINE__ = __LINE__()

class DataBase:
    def __init__(self, para: object) -> None:
        with open(f'{para.data_dir}/{para.dataset}/data/entities.dict', 'r') as e2i:
            lines = [line.strip().split() for line in e2i]
            self.ent2id = {e: int(i) for i, e in lines}
        with open(f'{para.data_dir}/{para.dataset}/data/relations.dict', 'r') as r2i:
            lines = [line.strip().split() for line in r2i]
            self.rel2id = {r: int(i) for i, r in lines}
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(self.rel2id)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.num_ent = para.num_ent = len(self.ent2id)
        self.num_rel = para.num_rel = len(self.rel2id) // 2
        self.para = para

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'valid', 'test']:
            for line in open(f'{para.data_dir}/{para.dataset}/data/{para.perfix}{split}.txt'):
                sub, rel, obj = line.strip().split('\t')
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]

                self.data[split].append((sub, rel, obj))
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.num_rel)].add(sub)
            if split == 'train':
                self.sr2o = {k: list(v) for k, v in sr2o.items()}

        self.data = dict(self.data)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()} #(h,r):[t1, t2, t3]

        neighbors = ddict(lambda: ddict(set))
        for h, r, t in self.data['train']:
            neighbors[h][r].add(t)
            neighbors[t][r + self.num_rel].add(h)

        self.neighbors = {h: {r: list(t) for r, t in rt.items()} for h, rt in neighbors.items()}  #h:{r1:[t1, t2,,,], r2:[t1,t2,,,]}
        dumpedFile = f'{para.data_dir}/{para.dataset}/data/split-subgraph-{para.perfix}J{para.subgraph}.pkl'
        if os.path.exists(dumpedFile):
            with open(dumpedFile, 'rb') as f:
                self.subgraph = pickle.load(f)
        else:
            logging.info('extract subgraph...')
            self.cntParts()  #抽取得到一个一个的子图
            bestMetric, bestSub = [1e9, 1e9, 1e9], None
            for i in range(50):
                desc = self.extractSubgraph(para.subgraph, para.dataset) # 3
                print(i, desc)
                if bestMetric[0] > desc[0]:  #desc[0]理解为最大子图中包含的三元组个数
                    bestMetric = desc
                    bestSub = self.subgraph
            desc = "{}-{}-{}".format(*bestMetric)
            with open(dumpedFile, 'wb') as f:
                pickle.dump(bestSub, f)
            exit()
        self.nheads = len(self.subgraph)

        self.htPair = {split: {(h,t) for h,_,t in self.data[split]} for split in ['valid', 'test']}

        if para.pretrain or para.testKGE != 0:
            edge_index, edge_type = [], []
            for sub, rel, obj in self.data['train']:
                edge_index.append((sub, obj))
                edge_type.append(rel)
            for sub, rel, obj in self.data['train']:
                edge_index.append((obj, sub))
                edge_type.append(rel + self.num_rel)
            self.edge_index = torch.LongTensor(edge_index).t()
            self.edge_type = torch.LongTensor(edge_type)
    
    def cntParts(self):
        self.neighborsHT = {h:{t for ts in rt.values() for t in ts} for h,rt in self.neighbors.items()}  #训练集中所有的ht 对
        unseen = {e for e in range(self.num_ent)}
        self.parts = []
        while len(unseen) != 0:   #广度优先搜索遍历得到独立的实体组
            self.parts.append(set())
            queue = [random.choice(list(unseen))]
            while len(queue) != 0:
                root = queue.pop()
                unseen -= {root}
                self.parts[-1].add(root)
                queue.extend([t for t in self.neighborsHT[root] if t in unseen])  #根据广度优先搜索得到一个个子图
        self.parts.sort(key=lambda p:len(p)) #这句话不是很懂什么意思
        print(f"{len(self.parts)=}   {[len(x) for x in self.parts]=}") #按照集合中每个元素中个数的长度进行排序
        # degreeAve = math.ceil(len(self.data['train']) / self.num_ent)  
        # maxEntsInSub = degreeAve * math.ceil(degreeAve / 2) * math.ceil(degreeAve / 3)  #这句话啥意思
        # minEntsInSub = maxEntsInSub // 3
        # print(f"{maxEntsInSub=}  {minEntsInSub=}  {degreeAve=}")

    def extractSubgraph(self, hops, dataset):
        dataset = dataset.lower()
        if dataset in ('family', 'wn18rr', 'fb15k237'):
            degreeAve = {'family':9, 'wn18rr':3, 'fb15k237':20}[dataset]
            maxEntsInSub = {'family':135, 'wn18rr':100, 'fb15k237':400}[dataset]
            minEntsInSub = {'family':45, 'wn18rr':9, 'fb15k237':100}[dataset]
            minThrs = {'family':3, 'wn18rr':3, 'fb15k237':8}[dataset]
            maxThrs = {'family':9, 'wn18rr':6, 'fb15k237':20}[dataset]
        else:
            maxThrs = degreeAve = max(math.ceil(len(self.data['train']) / self.num_ent), 2)
            minThrs = maxThrs / 3
            maxEntsInSub = degreeAve * math.ceil(degreeAve / 2) * math.ceil(degreeAve / 3)
            minEntsInSub = maxEntsInSub // 3
        degree = {e:len(self.neighborsHT[e]) for e in range(self.num_ent)}

        def selNeis(ent):
            nodes = [ent]
            length = [0]
            for jump in range(hops):
                length.append(len(nodes))
                for idx in range(length[jump], length[jump+1]):
                    parent = nodes[idx]
                    nodes += list({t for t in self.neighborsHT[parent] if t not in nodes})
                    if len(nodes) >= minEntsInSub:
                        return True
            return False

        def shift(hop, cnt):
            return (degreeAve / cnt / 2) ** 0.5

        unExpandEnts = set(range(self.num_ent)) #2378
        subgraphEnts = ddict(lambda: [set() for _ in range(hops+2)])

        smallSubgraphEnts = ddict(set)
        idx = 0
        while idx < len(self.parts) and len(self.parts[idx]) < maxEntsInSub:
            root = random.choice(list(self.parts[idx]))
            while (idx < len(self.parts)) and (len(smallSubgraphEnts[root]) + len(self.parts[idx]) < maxEntsInSub):
                smallSubgraphEnts[root] |= self.parts[idx]
                unExpandEnts -= self.parts[idx]
                idx += 1

        choices = {e for e in unExpandEnts if selNeis(e)}
        while len(choices) != 0:
            print(f"1.init {len(choices)}", end='     \r')
            choice = list(choices)
            random.shuffle(choice)
            for ent in choice:
                if degreeAve/2 <= degree[ent] <= 2*degreeAve:   #这里为什么这样？
                    break
            else:
                ent = choice[0]
            subgraphEnts[ent][0] |= {ent}
            subgraphEnts[ent][1] |= {ent}
            expend = set()
            for hop in range(1, hops+1):
                for h in subgraphEnts[ent][hop]:
                    if h not in unExpandEnts:
                        continue
                    unExpandEnts -= {h}
                    choices -= {h}
                    expend |= {h}
                    for t in self.neighborsHT[h]:
                        if hop != 1 and (t in subgraphEnts[ent][0] or random.random() > shift(hop, len(subgraphEnts[ent][hop]))):
                            continue
                        subgraphEnts[ent][hop+1].add(t)
                        subgraphEnts[ent][0].add(t)
            if len(subgraphEnts[ent][hops+1])<degreeAve and len(subgraphEnts[ent][3]) < (degreeAve if hops==3 else degreeAve/2):  #这里的代码什么意思？
                unExpandEnts |= expend
                del subgraphEnts[ent]
        subLen = [len(x[0]) for x in subgraphEnts.values()]

        choices = {e for e in unExpandEnts}
        while len(choices) != 0:
            print(f"2.connect {len(choices)}", end='     \r')
            ent = random.choice((list(choices)))
            choices -= {ent}
            for hop in range(2, hops+1):
                for root in sorted(subgraphEnts.keys(), key = lambda x: len(subgraphEnts[x][0])):
                    if len(self.neighborsHT[ent] & subgraphEnts[root][hop])!=0:
                        break
                else:
                    continue
                subgraphEnts[root][hop+1] |= {ent}
                subgraphEnts[root][0] |= {ent}
                unExpandEnts -= {ent}
                if hops == 3 and hop == 2:
                    subgraphEnts[root][hop+2] |= self.neighborsHT[ent]
                    subgraphEnts[root][0] |= self.neighborsHT[ent]
        subLen = [len(x[0]) for x in subgraphEnts.values()]

        choices = {e for e in unExpandEnts} #297
        while len(choices) != 0:
            print(f"3.more_hop {len(choices)}", end='    \r')
            ent = random.choice((list(choices)))
            choices -= {ent}
            for root in sorted(subgraphEnts.keys(), key = lambda x: len(subgraphEnts[x][0])):
                if ent in subgraphEnts[root][0]:
                    break
            else:
                continue
            unExpandEnts -= {ent}
            subgraphEnts[root][-1] -= {ent}
            subgraphEnts[root][0] |= self.neighborsHT[ent]
        subLen = [len(x[0]) for x in subgraphEnts.values()]

        tryTimes = len(unExpandEnts) * 5 #267
        while len(unExpandEnts) != 0 and (tryTimes := tryTimes - 1) > 0:
            print(f"4.remain {len(unExpandEnts)}-{tryTimes}", end='    \r')
            ent = random.choice((list(unExpandEnts)))
            for root in sorted(subgraphEnts.keys(), key = lambda x: len(subgraphEnts[x][0])):
                if len(self.neighborsHT[ent] & subgraphEnts[root][0]) != 0:
                    break
            else:
                continue
            unExpandEnts -= {ent}
            subgraphEnts[root][-1] -= {ent}
            subgraphEnts[root][0] |= self.neighborsHT[ent] | {ent}
        subLen = [len(x[0]) for x in subgraphEnts.values()]

        if self.para.del_exceed:  #这行代码的作用
            for root in subgraphEnts:
                delCnt = len(subgraphEnts[root][0]) - maxEntsInSub
                if delCnt > 0:
                    delEnts = list(subgraphEnts[root][-1])
                    random.shuffle(delEnts)
                    delEnts = delEnts[:delCnt]
                    subgraphEnts[root][0] -= set(delEnts)

        ht2r = ddict(set)
        for h,r,t in self.data['train']:
            ht2r[(h,t)].add(r)
        allHTs = {ht for ht in ht2r.keys()}
        subgraph = {root:subs[0] for root, subs in subgraphEnts.items()} #63
        subgraph.update({k:v for k,v in smallSubgraphEnts.items() if len(v)!=0}) # 63+3 =66
        self.subgraph = dict()
        for root, subs in subgraph.items():
            hts = torch.tensor(list(subs)).view(-1, 1) # (59, 1)
            cnt = hts.shape[0]
            hts = torch.cat([hts.repeat(cnt, 1), hts.repeat(1, cnt).view(-1, 1)], dim=-1) #[3481, 2]
            hts = {tuple(ht) for ht in hts.tolist()} & allHTs #267
            self.subgraph[root] = torch.LongTensor([(ht[0],r,ht[1]) for ht in hts for r in ht2r[ht]])  #边的构造

        subTripLen = [hrt.shape[0] for hrt in self.subgraph.values()]
        return (max(subTripLen),max(subLen),len(self.subgraph),len(unExpandEnts))

    # def extractSubgraph_ORI(self, hops, dataset):
    #     self.subgraph = dict()
    #     self.unseenEnts = set(range(self.num_ent))
    #     degree = math.ceil(len(self.data['train']) / self.num_ent) * 2
    #     print(f"{degree=}")
    #     probability = [0, 1.0 / math.ceil(degree / 2), 1.0 / math.ceil(degree / 4)]
    #     best = 0
    #     subLen = []
    #     subEnts = []
    #     while len(self.unseenEnts) != 0:
    #         bestchoice = [x for x in self.unseenEnts if (degree <= sum((len(ee) for ee in self.neighbors[x].values())) <= 2*degree)]
    #         if len(bestchoice) != 0:
    #             ent = random.choice(bestchoice)
    #             best += 1
    #         else:
    #             ent = random.choice(list(self.unseenEnts))
    #         if ent in self.neighbors:
    #             self.subgraph[ent], nents = self.getSubgraph(ent, hops, probability)
    #             subLen.append(self.subgraph[ent].shape[0])
    #             subEnts.append(nents)
    #             print(f'{len(self.unseenEnts)}/{self.num_ent}, best={best}    ', end='\r')
    #             self.unseenEnts -= {ent}
    #     print(f"\n#subgraphs={len(self.subgraph)}, best={best}, meanLen={np.mean(subLen):.2f}, maxLen={max(subLen)}, minLen={min(subLen)}")
    #     print(f"meanEnt={np.mean(subEnts):.2f}, maxEnt={max(subEnts)}, minEnt={min(subEnts)}")
    #     subTripLen = [hrt.shape[0] for hrt in self.subgraph.values()]
    #     return (max(subTripLen),max(subLen),len(self.subgraph))

    # def getSubgraph(self, head, hops, probability):
    #     nodes = [head]
    #     relation = []
    #     length = [0]
    #     for jump, pb in zip(range(hops), probability):
    #         length.append(len(nodes))
    #         for idx in range(length[jump], length[jump+1]):
    #             parent = nodes[idx]
    #             if parent not in self.unseenEnts:
    #                 continue
    #             self.unseenEnts -= {parent}
    #             for r in self.neighbors[parent]:
    #                 for t in self.neighbors[parent][r]:
    #                     if random.random() < pb:
    #                         continue
    #                     relation.append((parent, r, t))
    #                     if t not in nodes:
    #                         nodes.append(t)

    #     subgraph = set()
    #     for h, r, t in relation:
    #         subgraph.add((h, r, t) if r < self.num_rel else (t, r-self.num_rel, h))

    #     return torch.LongTensor(list(subgraph)), length[-1]


class HeadDataSet(Dataset):
    def __init__(self, database: DataBase, mode: str, para: object) -> None:
        self.num_ent = database.num_ent
        self.num_rel = database.num_rel
        self.subgraph = database.subgraph
        self.hr_t = {h: {r: np.array(t) for r, t in rt.items()} for h, rt in database.neighbors.items()}
        self.topred = para.topred
        self.mode = mode
        if mode == 'train':
            self.heads = [h for h in self.subgraph.keys() if self.subgraph[h].shape[0] >= 1/self.topred] #这里为什么要加一个限制呢,经过计算这里的限制没有用
        else:
            self.heads = list(self.subgraph.keys())
        self.padLen = para.padLen
        self.edge_index = torch.zeros((2, self.padLen * 2))
        self.edge_type = torch.zeros(self.padLen * 2)
        self.seen_ents = torch.zeros(self.padLen * 2)
        self.seen_rels = torch.zeros(self.padLen * 2)
        self.toPredTensor = torch.zeros((self.padLen, 3))

    def __len__(self) -> int:
        return len(self.heads)

    def __getitem__(self, head):
        triples = self.subgraph[head] #[109, 3]
        idx = torch.randperm(triples.shape[0])
        allTriples = triples[idx]
        if self.mode == 'train':
            triples = allTriples[:triples.shape[0]*4//5] #[87, 3]

        edge_index = torch.cat([triples[:, [0, 2]], triples[:, [2, 0]]], dim=0).t() #[2, 1386]  [2, 174] [h, t]
        edge_type = torch.cat([triples[:, 1], triples[:, 1]+self.num_rel], dim=0) #[1386]

        ents, rels = edge_index[0, :].unique(), edge_type.unique() #82, 24  #118, 22

        perdHTs = torch.cat([ents.view(-1, 1).repeat(1, len(ents)).reshape(-1, 1),  #[6724, 2] #82*82 [13924, 2]  所有的(h,t)可能性
                             ents.view(-1, 1).repeat(len(ents), 1)], dim=-1)

        if self.mode == 'train':
            perdHTsLab = (allTriples[:, [0, 2]].unsqueeze(0) == perdHTs.unsqueeze(1)).all(dim=-1).any(dim=-1) #[1936] #这行代码什么意思
            subLab = (triples[:, [0, 2]].unsqueeze(0) == perdHTs.unsqueeze(1)).all(dim=-1).any(dim=-1) #[1936]
            return edge_index, edge_type, perdHTs, perdHTsLab, subLab
        return edge_index, edge_type, perdHTs

    def __iter__(self):
        random.shuffle(self.heads)
        for head in self.heads:
            yield self.__getitem__(head)


class CompGCNDataSet(Dataset):
    def __init__(self, database: DataBase, mode: str, para: object) -> None:
        self.triples = database.data[mode]
        self.sr2o = database.sr2o if mode == "train" else database.sr2o_all
        self.smooth = para.lbl_smooth
        self.num_ent = database.num_ent
        self.num_rel = database.num_rel

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx):
        h,r,t = self.triples[idx]
        triple, label = torch.LongTensor((h,r,t)), torch.LongTensor(self.sr2o[(h,r)])
        trp_label = torch.zeros((self.num_ent))
        trp_label[label] = 1.0

        return triple, trp_label.bool()
