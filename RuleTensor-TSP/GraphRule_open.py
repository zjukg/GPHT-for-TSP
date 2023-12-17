import argparse
import os
from os.path import join
from collections import defaultdict
import random
import pickle
from tqdm import tqdm
import numpy as np
import torch
import time
from data_pre import data_pretrain
from math import sqrt
class Data():
    def __init__(self, data_path, percent) -> None:
        with open(f'{data_path}/e2i_sel.ple', 'rb') as e2i, open(f'{data_path}/r2i_sel.ple', 'rb') as r2i:
            self.e2id = pickle.load(e2i) 
            self.r2id = pickle.load(r2i) 
            self.ents = [x.strip() for x in self.e2id.keys()]
            self.rels = [x.strip() for x in self.r2id.keys()]

            self.pos_rels = len(self.rels)
            rels = self.rels + ['inv_'+x for x in self.rels] + ['<slf>']
            self.id2r = {i: rels[i] for i in range(len(rels))}
            self.id2e = {i: self.ents[i] for i in range(len(self.ents))}

        self.data = {}
        for split in ['train', 'valid', 'test']:
            with open(f'{data_path}/{percent}_{split}.txt') as f:
                data = [item.strip().split('\t') for item in f.readlines()]
                self.data[split] = {(self.e2id[h], self.r2id[r], self.e2id[t]) for h, r, t in data}
        self.htPair = {'valid': {(h,t) for split in ['train', 'valid'] for h,_,t in self.data[split]}} 
        self.htPair['test'] = self.htPair['valid'] | {(h,t) for h,_,t in self.data['test']} 

        self.nx = {e: defaultdict(list) for e in range(len(self.id2e))}
        indices = [[] for _ in range(self.pos_rels)]
        values = [[] for _ in range(self.pos_rels)]
        for h, r, t in self.data['train']:
            indices[r].append((h, t))
            values[r].append(1)
            self.nx[h][t].append(r)
            self.nx[t][h].append(r+self.pos_rels)
        indices = [torch.LongTensor(x).T for x in indices] 
        values = [torch.FloatTensor(x) for x in values]   
        size = torch.Size([len(self.ents), len(self.ents)])
        self.rel_mat = [torch.sparse.FloatTensor(indices[i], values[i], size).coalesce() for i in range(self.pos_rels)]
        self.rel_mat.append(torch.sparse.FloatTensor(torch.LongTensor(
            [[i, i] for i in range(len(self.ents))]).T, torch.ones(len(self.ents)), size).coalesce())

    def statistics(self, rule2rel):
        self.rule2rel = rule2rel
        train_rule2rel = [(h, rule2rel[r], t) for (h, r, t) in self.data['train']]
        self.triplets = set(train_rule2rel + self.data['valid'] + self.data['test'])

        self.hr_t, self.hr_c = defaultdict(set), defaultdict(int)
        self.tr_h, self.tr_c = defaultdict(set), defaultdict(int)
        for h, r, t in train_rule2rel:
            self.hr_t[(h, r)].add(t)
            self.tr_h[(t, r)].add(h)
            self.hr_c[(h, r)] += 1
            self.tr_c[(t, r)] += 1
        init_cnt = 3
        for cnt in [self.hr_c, self.tr_c]:
            for key in cnt:
                cnt[key] += init_cnt
        self.hr_t = {key: np.array(list(val)) for key, val in self.hr_t.items()}
        self.tr_h = {key: np.array(list(val)) for key, val in self.tr_h.items()}

    def getinfo(self):
        return len(self.ents), len(self.rels)


class GraphRule:
    def __init__(self, args, dataset):
        super().__init__()
        self.args = args
        self.data = dataset
        self.rule_len = args.rule_len #3
        
        self.save = f'RuleTensor-TSP/EXPS/{args.dataset}/{args.percent}-J{args.rule_len}_hc{args.hc_thr}-sc{args.sc_thr}_{os.uname().nodename}_{time.strftime(r"%Y%m%d-%H:%M")}'
        os.mkdir(self.save)
        self.device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
        self.entity_num, self.rel_num = self.data.getinfo() 
        self.id2e = self.data.id2e 
        self.id2r = self.data.id2r 
        self.nx = self.data.nx 
        self.topK = 3
        self.mat1 = [x.clone().to(self.device).to_dense().to_sparse() for x in self.data.rel_mat]
        self.mat2 = []

        self.rule_set = {}
        start_time = time.time()
        print("get rules...")
        while self.sampleRules() and time.time()-start_time<60:
            print(f'\r get {len(self.rule_set)} rules', end='')
        print("done")
        self.allAddPath = 0
        self.bar_format = '{desc}{percentage:3.0f}-{total_fmt}|{bar}|[{elapsed}<{remaining}{postfix}]'
        self.epoch = 0

    def __del__(self):
        pass

    def negRel(self, rel):
        if rel < self.data.pos_rels:
            return rel + self.data.pos_rels
        if rel < self.data.pos_rels * 2:
            return rel - self.data.pos_rels
        return rel

    def sampleRules(self):
        def rand(lis):
            if len(lis) == 0:
                return False
            if len(lis) == 1:
                return lis[0]
            return lis[random.randint(0, len(lis)-1)]
        cnt = 0
        sam = 0
        while sam < 1000:
            rule = []
            node = [rand(list(self.id2e.keys()))] #从实体中随机选择一个
            for __ in range(self.rule_len): # 3
                next = rand(list(self.nx[node[-1]].items()))  #t, r
                if next == False:
                    break
                node.append(next[0])      
                rule.append(rand(next[1])) 
                if node[-1] in self.nx[node[0]]:
                    tmp = self.nx[node[0]][node[-1]]  
                else:
                    continue
                if (len(rule) == 1 and len(tmp) > 1) or (len(rule) > 1 and tmp):  #
                    sam += 1
                    if len(rule) == 1 and len(tmp)>1:
                        tmp.remove(rule[0])
                    head = rand(tmp)   #7
                    if self.id2r[head] == '<slf>':
                        continue
                    if head < self.data.pos_rels:
                        rule = tuple(rule + [head])
                    else:
                        rule = tuple([self.negRel(x) for x in reversed(rule)] + [self.negRel(head)]) 
                    if rule not in self.rule_set:
                        self.rule_set[rule] = -1
                        cnt += 1
                    break
        return cnt

    def ruleInitial(self, rel_num, rule_len):
        tmp = [[] for _ in range(rule_len + 2)]
        tmp[0] = [[]]

        for leng in range(rule_len):
            for item in tmp[leng]:
                for r in range(rel_num):
                    tmp[leng+1].append(item+[r])

        for item in tmp[leng+1]:
            for r in range(rel_num//2):
                chk = [x for x in item if x != rel_num-1]
                chk += [rel_num-1 for _ in range(rule_len - len(chk))]
                tmp[leng+2].append(chk+[r])
        self.rule_set = {tuple(item): -1 for item in tmp[-1]}
        return self.rule_set

    def getMat(self, mat, ind):
        posRel = self.data.pos_rels
        if ind < posRel:
            return mat[ind]
        if ind < posRel * 2:
            return mat[ind - posRel].t()
        return mat[-1]

    def qCalConf(self, mat, rule_set):
        calcCache = [[None] for _ in range(self.rule_len)]
        self.calPath = defaultdict(list)
        for rule in tqdm(rule_set, desc=f"{self.epoch}-calcConfidence", ncols=60, bar_format=self.bar_format, leave=False):
            if 0 <= rule_set[rule] < 0.1:
                continue

            result = self.getMat(mat, rule[0]).coalesce()  #rule : (6, 15, 4, 3)
            for i in range(1, len(rule)-1):
                if calcCache[i][0] == rule[:i+1]:
                    result = calcCache[i][1]
                else:
                    result = torch.sparse.mm(result, self.getMat(mat, rule[i])).to_dense().to_sparse().coalesce()
                    calcCache[i] = [rule[:i+1], result]
           
            num_result = sum(result.values().bool())
            if num_result > 0:
                num_true = result * self.getMat(mat, rule[-1]).bool().float() #这里相当于sup(rule)
                num_true = num_true.values()
                num_true[num_true > 1] = 1
                num_true = sum(num_true)
                SC = num_true / num_result   
                HC = num_true / self.getMat(mat, rule[-1]).values().bool().float().sum()
                rule_set[rule] = SC.item()
                if (1 > rule_set[rule] > self.args.sc_thr) and (HC.item() > self.args.hc_thr):
                    self.calPath[rule[-1]].append((rule, result.bool().float()))
        return rule_set

    def updateMat(self, mat, rule_set): 
        mat2 = [x.clone() for x in mat]
        cnt_all = 0
        for headRule, results in tqdm(rule_set.items(), desc=f"{self.epoch}-updateMat", ncols=60, bar_format=self.bar_format, leave=False):
            add = torch.sparse_coo_tensor(torch.tensor([[], []]), [], mat[0].shape).to(self.device)
            cnt = torch.sparse_coo_tensor(torch.tensor([[], []]), [], mat[0].shape).to(self.device)
            for rule, result in results:
                tmp = (result - result * self.getMat(mat, headRule).bool().float()).coalesce()
                cnt = (cnt + tmp.bool().float()).coalesce()
                add = (add + self.rule_set[rule] * tmp).coalesce()
            add = add.to_dense().to_sparse().coalesce()
            cnt = cnt.to_dense().to_sparse().coalesce()
            tmp = torch.sparse_coo_tensor(cnt.indices(), 1.0/cnt.values(), cnt.shape).to(self.device)
            mat2[headRule] = (mat2[headRule] + add * tmp).to_dense().to_sparse().coalesce()
            cnt_all += len(cnt.values())
        return mat2, cnt_all

    def updateMaxMat(self, mat, rule_set):
        mat2 = [x.clone() for x in mat]
        cnt_all = 0
        for headRule, results in tqdm(rule_set.items(), desc=f"{self.epoch}-updateMat", ncols=60, bar_format=self.bar_format, leave=False):
            results = sorted(results, key=lambda x: -self.rule_set[x[0]])
            for rule, result in results:
                tmp = (result - result * self.getMat(mat2, headRule).bool().float()).coalesce()
                mat2[headRule] = (mat2[headRule] + self.rule_set[rule] * tmp).to_dense().to_sparse().coalesce()
                cnt_all += int(sum(tmp.values()))

        return mat2, cnt_all

    def toDataset(self, mat):
        with open(f'{self.save}/{self.args.percent}-{self.args.sc_thr}.txt', 'w') as f:
            for head in range(len(mat) - 1):
                mat[head] = mat[head].coalesce()
                ind = mat[head].indices()
                val = mat[head].values()
                for x, y, z in zip(ind[0].tolist(), ind[1].tolist(), val.tolist()):
                    f.write(f'{self.id2e[x]}\t{self.id2r[head]}\t{self.id2e[y]}\t{z}\n')

    def runEpoch(self):
        self.epoch += 1      
        self.rule_set = self.qCalConf(self.mat1, self.rule_set)   #规则的筛选
        self.mat2, addPath = self.updateMaxMat(self.mat1, self.calPath)
        self.allAddPath += addPath
        print(f"valid rules:{len(self.calPath)}, adds:{addPath}, total adds:{self.allAddPath}")
        with open(f"{self.save}/train.log", "a") as log:
            log.write(f"valid rules:{len(self.calPath)},adds:{addPath}, total adds:{self.allAddPath}\n")
        return addPath >= (1 - self.args.sc_thr)*100 and self.epoch < 40

    def getNewTriple(self, mat, truthSet, r_dict, owl):
        newTriples = []
        rightCnt = 0
        mrr_right = 0
        mrr_false = 0
        for head in range(len(mat) - 1):
            other_r = r_dict[head]
            mat[head] = mat[head].coalesce()
            ind = mat[head].indices()
            val = mat[head].values()
            notZero = torch.where((0 < val) & (val < 1))[0]
            ind = ind[:, notZero]
            val = val[notZero]
            for h, t, w in zip(ind[0].tolist(), ind[1].tolist(), val.tolist()):
                
                correct = (h, head, t) in self.data.data[truthSet]
                newTriples.append((w, correct, (h, head, t)))
                mrr_right += correct
        newTriples.sort(key=lambda x: -x[0])
        allRanks = []
        rank = 0
        sumrank = 0
        for w, corr, _ in newTriples:
            rank +=1
            if corr:
                sumrank +=1/rank
                allRanks.append(rank)
            else:
                h, head, t = _
                for r in other_r:
                    if (h, r, t) in self.data.data[truthSet]:
                        mrr_false +=1
                        sumrank -=1/rank
        assert len(allRanks) == mrr_right
        mrr_right = max(0.1, mrr_right)
        mrr_len = mrr_right+mrr_false
        pre = (mrr_right/len(newTriples) + mrr_right/mrr_len)/2 if len(newTriples) and mrr_len else 0
        recall = sqrt(mrr_right / len(self.data.htPair[truthSet]))
        f1 = 2*pre*recall /(pre+recall) if (pre+recall) else -1 
        message = "{"
        message += '"split":"{}", "owl":"{}", "label":"J{}-{}-{}", "add":{}, "true":{}, "mrr_len":{}, "rank":{}, "len_test":{}, "pre":{}, "recall":{}, "f1":{}'.format(
                truthSet, owl, self.args.rule_len, self.args.hc_thr, self.args.sc_thr, len(newTriples), mrr_right, mrr_len, sumrank , len(self.data.htPair[truthSet]), pre, recall, f1)
        message += "}\n"
        with open(f'{self.save}/../metric-SP{self.args.percent}.txt', 'a') as metric:
            metric.write(message)

        rulesCnt = 0
        with open(f'{self.save}/{truthSet}.log', 'w') as rank:
            rank.write(message)
            useRules = set()
            for _, results in self.calPath.items():
                for rule, _ in results:
                    useRules.add(f'{",".join([self.id2r[x] for x in rule[:-1]])} -> {self.id2r[rule[-1]]}')
                    rulesCnt += 1
            rank.write('\n'.join(useRules))

            rank.write(f'\n\nweight\trank\ttriple\n')
            for ra, trip in zip(allRanks, filter(lambda x: x[1], newTriples)):
                w, _, (h, r, t) = trip
                rank.write(f'{w:.4f}\t{ra}\t-{self.id2e[h]},{self.id2r[r]},{self.id2e[t]}-\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=1007)
    parser.add_argument('-dataset', type=str, default='Wiki79k')
    parser.add_argument('-r_same', type=str, default='0.8_r')
    parser.add_argument('-percent', type=float, default=0.8)
    parser.add_argument('-rule_len', type=int, default=3)
    parser.add_argument('-sc_thr', type=float, default=0.50)
    parser.add_argument('-hc_thr', type=float, default=0.50)

    parser.add_argument('-gpu', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(40)
    r_dict = data_pretrain(args.dataset)
    base_data = Data(join('DATA', args.dataset, 'data'), args.percent)

    model = GraphRule(args, base_data)

    start = time.time()
    while model.runEpoch():
        model.mat1 = model.mat2
    print("use time", time.time() - start)
    with open(f"{model.save}/train.log", "a") as log:
            log.write(f"use time:{time.time() - start}")
    model.getNewTriple(model.mat2, 'valid', r_dict, False)
    model.getNewTriple(model.mat2, 'test', r_dict, False)
