from urllib.parse import scheme_chars
import networkx as nx
from .conceptnet import merged_relations
import json
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix
import pickle

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


def concepts2adj(node_ids):
    """
    Compute the adj matrix given a set of nodes.
    The adj matrix will have shape: RxNxN
    """
    global id2relation
    cids = np.array(node_ids, dtype=np.int32) #list of nodes
    n_rel = len(id2relation) #number of relations
    n_nodes = cids.shape[0] #number of nodes
    # initialize the adjacency matrix
    adj = np.zeros((n_rel, n_nodes, n_nodes), dtype=np.uint8)
    # for each pair of nodes
    for s in range(n_nodes):
        for t in range(n_nodes):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c): # if edge exists
                # cpnet[s_c][t_c] is a dict with edge attributes
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    # cids += 1  # note!!! index 0 is reserved for padding
    # save as a sparse coo matrix
    adj = coo_matrix(adj.reshape(-1, n_nodes))
    return adj, cids

def _generateAdj(data_list):
    """
    Generates adj matrices and also return concepts ids.
    """
    img_id, data = data_list

    res = []
    for _round in data: # for each round
        all_concepts = set(_round)
        common_neighbours = set()
        # print("\n\nROUND CONCEPTS:") #delete
        # for c1 in _round:#delete
            # print(id2concept[c1])#delete
        
        # For each pair of the grounded concepts of current round
        for c1 in _round:
            for c2 in _round:
                if c1!=c2 and c1 in cpnet_simple.nodes and c2 in cpnet_simple.nodes:
                    # cpnet_simple[c1] will get a list of the neighbours of c1
                    # common_neighbours is the intersection of the neighbours of c1 and c2
                    common_neighbours |= set(cpnet_simple[c1]) & set(cpnet_simple[c2])
    
        # Remove from the common_neighbours the grounded concepts
        common_neighbours = common_neighbours - all_concepts
        # Schema graph is just a set of all the nodes
        schema_graph = sorted(all_concepts) + sorted(common_neighbours)
        adj, concepts = concepts2adj(schema_graph)
        res.append({'adj':adj, 'c':concepts})
    
    return (img_id, res)
    
    

def generateAdj(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path):
    """
    Generate adjacency matrices in COO format (R*N, N), where
    R: number of relations, N: number of nodes

    Parameters
    ----------

    grounded_path: str
        Path to file with the concepts for each round of each dialog. 
    """

    print(f'Generating adjacency matrices data for {grounded_path}...\n')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    load_resources(cpnet_vocab_path)
    print("Will load cpnet.")
    load_cpnet(cpnet_graph_path)
    print("Finished loading cpnet.")

    with open(grounded_path, 'r', encoding='utf-8') as fin:
        grounded_concepts = json.load(fin)
    
    
    data_list = [(img_id, [[concept2id[c] for c in _round] for _round in dialog]) \
        for img_id, dialog in grounded_concepts.items()]


    with Pool() as p:
        res = {k:v for (k,v) in tqdm(p.imap(_generateAdj,data_list,50), total=len(grounded_concepts), desc='Generating adj matrices..')}
    
    # print(res)
    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'Adj matrices saved to {output_path}.\n')