import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import json


DATA_DIR = '/home/jkalogero/KBGN-Implementation/data/'
adj_lists = {
    'train': DATA_DIR + 'train_adj_list.h5',
    'val': DATA_DIR + 'val_adj_list.h5',
    'test': DATA_DIR + 'test_adj_list.h5',
    'debug': DATA_DIR + 'debug_adj.h5'
}


files = [DATA_DIR + 'debug_adj.h5', DATA_DIR + 'debug_adj.h5']

grounded = {
    # 'train': DATA_DIR + 'train_grounded.json',
    'val': DATA_DIR + 'val_grounded.json',
    # 'test': DATA_DIR + 'test_grounded.json'
}
global_counter = {split:np.array([0 for _ in range(35)]) for split in grounded.keys()}
output_files = {split: DATA_DIR +split+'_stats.npy' for split in grounded.keys()}

def _count_relations(data):
    counter = np.array([0 for _ in range(35)])
    img_id, split = data
    with h5py.File(adj_lists[split], "r") as f:
        for _round in f[img_id]:
            adj_list = np.array(f[str(img_id)][_round]['adj_list'])
            for i, l in enumerate(adj_list):
                rel = i // 45
                counter[rel] += len([el for el in l if el])
    
    return counter

def count_relations():
    img_ids = {}

    for split in grounded.keys():
        # get all ids
        with open(grounded[split]) as gr:
            img_ids[split] = list(json.load(gr).keys())

        datalist = [(i,split) for i in img_ids[split]]
        # datalist = [378466,('debug')]
        
        with Pool() as p:
            global_counter[split] = global_counter[split]  + np.array([_ for _ in tqdm(p.imap(_count_relations, datalist, 50), total=len(img_ids), desc='Counting')])


if __name__ == '__main__':
    

    # for f in files:
    count_relations()

    for split,counter in global_counter.items():
        total = np.sum(counter, axis=0)        
        np.save(output_files[split], total)
        print(f'Saved statistics in {output_files[split]}')