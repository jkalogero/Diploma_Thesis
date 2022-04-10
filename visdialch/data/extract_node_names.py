from cv2 import split
import h5py
import json
import numpy as np
from collections import Counter, OrderedDict
from tqdm import tqdm
from multiprocessing import Pool

def load_resources(cpnet_vocab_path):
    global concept2id, id2concept

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    

# input_files = ['../../data/adj_train_paths.h5', '../../data/adj_val_paths.h5', '../../data/adj_test_paths.h5']
input_files = ['/home/jkalogero/KBGN-Implementation/data/train_adj_list.h5', '/home/jkalogero/KBGN-Implementation/data/val_adj_list.h5', '/home/jkalogero/KBGN-Implementation/data/test_adj_list.h5']
# input_files = ['/home/jkalogero/KBGN-Implementation/data/debug_adj.h5']

ext_vocab_file = '/home/jkalogero/KBGN-Implementation/data/ext_vocab.json'

load_resources('/home/jkalogero/KBGN-Implementation/data/cpnet/pad_concept.txt')
concepts = np.array([])


for fil in input_files:
    split = fil.split('/')[-1].split('_')[0]
    test= split == 'test'
    with h5py.File(fil, "r") as f:
        # with Pool() as p:
        #     concepts_set.update(set([_ for _ in p.imap(execute, f)]))
        for image_id in tqdm(f):
            # print(image_id)
            if test:
                allkeys = list(f[image_id].keys())
                keys = [int(k) for k in allkeys if k !='original_limit']
                last = str(max(keys))
                concepts= np.append(concepts, np.array([n for n in f[image_id][last]['concept_names'] if n != '<PAD>']))
            else:
                concepts= np.append(concepts, np.array([n for n in f[image_id]['9']['concept_names'] if n != '<PAD>']))
            # break
        with open('/home/jkalogero/KBGN-Implementation/data/'+split+'_save.npy', 'wb') as f:
            np.save(f,concepts)

concepts_set= set(concepts)
print(len(concepts_set))
res = {c:1 for c in concepts_set}

with open(ext_vocab_file, 'w') as f:
    json.dump(res, f)
# global concept_embs, relation_embs
# concept_embs = np.load(concept_emb_path) # could already exist?
# relation_embs = np.load(rel_emb_path) # could already exist?