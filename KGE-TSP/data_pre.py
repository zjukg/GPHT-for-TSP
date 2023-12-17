from collections import defaultdict as ddict
import numpy as np
def data_pretrain(dataset, minconf):
    data = ddict(list)
    with open(f'DATA/{dataset}/data/entities.dict', 'r') as e2i:
        ent_lines = [line.strip().split() for line in e2i]
        ent2id = {e: int(i) for i, e in ent_lines}
    with open(f'DATA/{dataset}/data/relations.dict', 'r') as r2i:
        rel_lines = [line.strip().split() for line in r2i]
        rel2id = {r: int(i) for i, r in rel_lines}

   
    for split in ['all']:
        for line in open(f'DATA/{dataset}/data_ori/{split}.txt'):
            #sub, rel, obj = line.strip().split('\t')
            sub, rel, obj = line.strip().split()
            sub, rel, obj = ent2id[sub], rel2id[rel], ent2id[obj]
            data[split].append((sub, rel, obj))
    num_rel = len(rel2id) 
    all_triples = set(data['all'])
    r_list = [i for i in range (num_rel*2)]
    r_ht = ddict(set)
    r_dict = dict.fromkeys(r_list, 0)
    for h, r, t in all_triples:
        r_dict[r] +=1
        r_dict[r+num_rel] +=1
        r_ht[r].add((h, t))
        r_ht[r+num_rel].add((t,h))
    r_to_r = {}
    for j in range(num_rel*2):
        r_to_r[j] = []
        for k in range(num_rel*2):
            same_value = set(r_ht[j]) & set(r_ht[k])
            if max(len(same_value)/r_dict[j], len(same_value)/r_dict[k]) <minconf:
                r_to_r[j].append(k)

    return r_to_r
