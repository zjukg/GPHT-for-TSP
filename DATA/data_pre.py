from collections import defaultdict as ddict
import numpy as np
def data_pretrain(dataset):
    data = ddict(list)
    with open(f'DATA/{dataset}/data/entities.dict', 'r') as e2i:
        ent_lines = [line.strip().split() for line in e2i]
        ent2id = {e: int(i) for i, e in ent_lines}
    with open(f'DATA/{dataset}/data/relations.dict', 'r') as r2i:
        rel_lines = [line.strip().split() for line in r2i]
        rel2id = {r: int(i) for i, r in rel_lines}

    for split in ['train', 'valid', 'test']:
        for line in open(f'DATA/{dataset}/data_ori/{split}.txt'):
            sub, rel, obj = line.strip().split('\t')
            sub, rel, obj = ent2id[sub], rel2id[rel], ent2id[obj]
            data[split].append((sub, rel, obj))
    num_rel = len(rel2id) 
    all_triples = set(data['train'] + data['valid'] + data['test'])
    r_list = [i for i in range (num_rel*2)]
    r_to_r = dict.fromkeys(r_list, [])
    r_ht = ddict(set)
    r_dict = dict.fromkeys(r_list, 0)
    for h, r, t in all_triples:
        r_dict[r] +=1
        r_dict[r+num_rel] +=1
        r_ht[r].add((h, t))
        r_ht[r+num_rel].add(t,h)
    
    for j in range(num_rel*2):
        for k in range(j+1, num_rel*2):
            same_value = set(r_ht[j].values()) & set(r_ht[k].values())
            r_to_r[j].append(j)
            if len(same_value) == 0:
                continue
            if max(len(same_value)/r_dict[j], len(same_value)/r_dict[k]) > 0.8:
                r_to_r[j].append(k)

    return r_to_r