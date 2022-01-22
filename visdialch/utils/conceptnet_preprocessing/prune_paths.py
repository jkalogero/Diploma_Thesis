import multiprocessing
from tqdm import tqdm
import json
from multiprocessing import Pool


def _prunePaths(data_list):

    
    img_id, paths, pairs_scores, threshold = data_list

    original_len = 0
    pruned_len = 0
    for pair, pair_scores in zip(paths, pairs_scores):
        original_paths = pair['edges']
        if original_paths is not None:
            pruned_paths = [p for p, score in zip(original_paths, pair_scores) if score >= threshold]
            
            original_len += len(original_paths)
            pruned_len += len(pruned_paths)
            assert len(original_paths) >= len(pruned_paths)
            pair['edges'] = pruned_paths
    print("original_len: {}   pruned_len: {}   keep_rate: {:.4f}".format(original_len, pruned_len, pruned_len / original_len))
    return (img_id, paths)
    
    
def prunePaths(paths, path_scores, output_path, threshold, verbose=True):
    print(f'Pruning paths for {paths}...')
    
    
    with open(paths, 'r', encoding='utf-8') as f_paths, \
            open(path_scores, 'r', encoding='utf-8') as scores:
        
        paths = json.load(f_paths)
        scores = json.load(scores)
        data_list = [(img_id, v, scores[img_id], threshold) for img_id, v in paths.items()]

        with Pool() as p:
            res = {k:v for (k,v) in tqdm(p.imap(_prunePaths,data_list), total=len(paths), desc='Prunig paths...')}

        
        with open(output_path, 'w+') as fout:
            fout.write(json.dumps(res))

    print(f'Pruned paths saved to {output_path}.\n')
