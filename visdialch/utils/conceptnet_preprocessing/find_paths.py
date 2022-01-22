# from loaders import load_matcher, load_cpnet_vocab, load_cpnet, load_resources
import random
import numpy as np
import json
from multiprocessing import Pool
from tqdm import tqdm
import time
from .conceptnet import merged_relations
from spacy.matcher import Matcher
import networkx as nx



def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)

def getEdge(source: str, target: str):
    global cpnet
    rel_list = cpnet[source][target]  # list of dicts
    seen = set()
    # get unique values from rel_list
    res = [r['rel'] for r in rel_list.values() if r['rel'] not in seen and (seen.add(r['rel']) or True)]
    return res


def getPath(source: str, target: str, ifprint=False):
    """
    Find paths for a pair of two concepts, source and target.
    """
    global cpnet, cpnet_simple, concept2id, id2concept, relation2id, id2relation

    s = concept2id[source]
    t = concept2id[target]
    if source == 'male' and target == 'person':
        print('\n\n\ns: \n', s, '\nt:', t, '\n')

    if s not in cpnet_simple.nodes() or t not in cpnet_simple.nodes():
        return


    all_paths = []
    try:
        for p in nx.shortest_simple_paths(cpnet_simple, source=s, target=t):
            if source == 'male' and target == 'person':
                print("ALKSJDLKAJSKDJ")
                print('\n\n\npath: \n', p, '\n\n')
            if len(p) > 4 or len(all_paths) >= 100:  # top 100 paths
                break
            if len(p) >= 2:  # skip paths of length 1
                all_paths.append(p)
    except nx.exception.NetworkXNoPath:
        pass
    pf_res = []
    for p in all_paths:
        # print([id2concept[i] for i in p])
        rl = []
        for src in range(len(p) - 1):
            src_concept = p[src]
            tgt_concept = p[src + 1]

            rel_list = getEdge(src_concept, tgt_concept)
            rl.append(rel_list)
            if ifprint:
                rel_list_str = []
                for rel in rel_list:
                    if rel < len(id2relation):
                        rel_list_str.append(id2relation[rel])
                    else:
                        rel_list_str.append(id2relation[rel - len(id2relation)] + "*")
                print(id2concept[src_concept], "----[%s]---> " % ("/".join(rel_list_str)), end="")
                if src + 1 == len(p) - 1:
                    print(id2concept[tgt_concept], end="")
        if ifprint:
            print()
        if source == 'male' and target == 'person':
            print('\n\n\tMALE PERSON will add to edges\n', {"path": p, "rel": rl}, '\n\n')
        pf_res.append({"path": p, "rel": rl})
    
        if source == 'male' and target == 'person':
            print('\n\n\tpf_res\n', pf_res, '\n\n')
    return pf_res



def _findPaths(datalist):

    img_id, data= datalist
    paths = []

    for c1 in data[-1]:
        for c2 in data[-1]:
            if not c1 == c2:
                rel = getPath(c1, c2)
                if c1 == 'male' and c2 == 'person':
                    print('\n\n\tMALE PERSON will add to edges\n', {'source': c1, 'target': c2, 'edges': rel}, '\n\n')
                paths.append({'source': c1, 'target': c2, 'edges': rel})
    return (img_id, paths)    


def findPaths(grounded_path, cpnet_vocab_path, cpnet_graph_path, output_path, random_state=0):
    """
    Find paths between concepts on ConceptNet.

    Parameters
    ----------

    grounded_path: str
        Path to file with the concepts for each round of each dialog. 
    """
    print(f'Generating paths for {grounded_path}...')
    random.seed(random_state)
    np.random.seed(random_state)

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet

    load_resources(cpnet_vocab_path) # could already exist?
    load_cpnet(cpnet_graph_path) # could already exist?
    
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    
    datalist = [(k, v) for k,v in data.items()]
    with Pool() as p:
        res = {k:v for (k,v) in tqdm(p.imap(_findPaths, datalist), total=len(data), desc='Finding paths...')}

    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(res))

    print(f'Paths saved to {output_path}')