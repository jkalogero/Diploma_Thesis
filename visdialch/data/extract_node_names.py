from cv2 import split
import h5py
import json
import numpy as np
from collections import Counter, OrderedDict
from tqdm import tqdm
from multiprocessing import Pool


ext_vocab_file = '/home/jkalogero/KBGN-Implementation/data/ext_vocab.json'
DATA_DIR = '/home/jkalogero/KBGN-Implementation/data/'
grounded = {
    'train': DATA_DIR + 'train_grounded.json',
    'val': DATA_DIR + 'val_grounded.json',
    'test': DATA_DIR + 'test_grounded.json'
}
adj_lists = {
    'train': DATA_DIR + 'train_adj_list.h5',
    'val': DATA_DIR + 'val_adj_list.h5',
    'test': DATA_DIR + 'test_adj_list.h5'
}



def load_resources(cpnet_vocab_path):
    global concept2id, id2concept

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

def extract(data):
    img_id, split = data
    with h5py.File(adj_lists[split], "r") as f:
    
        if split == 'test':
            allkeys = list(f[img_id].keys())
            keys = [int(k) for k in allkeys if k !='original_limit']
            last = str(max(keys))
            res= [id2concept[n] for n in f[img_id][last]['concepts'] if n != 0]
        else:
            res= [id2concept[n] for n in f[img_id]['9']['concepts'] if n != 0]
    return res

def main():
    load_resources('/home/jkalogero/KBGN-Implementation/data/cpnet/pad_concept.txt')
    concepts = []


    # samples
    # img_ids = {
    #     'train': ['378466', '332243', '378461', '287140', '575029','525211'],
    #     'val': ['185565', '284024', '574189', '148816', '88394'],
    #     'test': ['568676', '171555', '104707', '302663', '452138']
    # }
    img_ids = {}

    for split in grounded.keys():
        # get all ids
        with open(grounded[split]) as gr:
            img_ids[split] = list(json.load(gr).keys())
        
        datalist = [(i,split) for i in img_ids[split]]
        with Pool() as p:
            concepts = concepts  + [_ for _ in tqdm(p.imap(extract, datalist), total=len(img_ids))]
        
            
    cnt = 0
    for i in concepts:
        cnt+=len(i)
    print(cnt)
    # convert
    final_c = list([_c for el in concepts for _c in el])
    print('len(c) = ',len(final_c))

    concepts_set= set(final_c)
    print(len(concepts_set))
    res = {c:1 for c in concepts_set}

    with open(ext_vocab_file, 'w') as f:
        json.dump(res, f)

if __name__ == '__main__':
    main()
