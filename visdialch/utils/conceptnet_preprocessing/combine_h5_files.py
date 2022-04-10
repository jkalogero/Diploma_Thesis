import h5py
import glob
from tqdm import tqdm

def link_file(filename, target_file):
    keys = []
    # get list of keys for current file
    with h5py.File(filename) as f:
        keys = list(f.keys())

    h = h5py.File(target_file, 'a')
    for key in keys:

        h[key] = h5py.ExternalLink(filename,key)
    
    h.close()



if __name__ == '__main__':

    all_files = glob.glob('/home/jkalogero/KBGN-Implementation/data/train_adj_list_part_*.h5')
    target_file = 'data/train_adj_list.h5'

    for _f in tqdm(all_files):
        link_file(_f, target_file)

    # check if all keys present
    with h5py.File(target_file) as f:
        all_keys = list(f.keys())
    
    # only for train split
    print('All OK: ', len(all_keys) == 123287)