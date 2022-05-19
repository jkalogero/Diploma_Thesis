from urllib.parse import scheme_chars
import networkx as nx
from .conceptnet import merged_relations
import json
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix
import pickle
import h5py
from scipy import spatial
from collections import defaultdict, Counter
from operator import itemgetter

inv_relations = ['inv_'+el for el in merged_relations]
full_relations = merged_relations + inv_relations
MAX_NODES = 45
MAX_EDGES=100

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




def createAdjList(schema_graph, adj_dict, original_concepts, num_rel=17, add_inverse=True, add_padding=True):
    """
    The adjacency list will have shape: (num_rel x MAX_NODES,MAX_EDGES), or (2xnum_rel x MAX_NODES,MAX_EDGES)
    if add_inverse = True.
    For each node, its list contains the index of the neighbouring node in schema_graph.

    Parameters:
    ===========
    schema_graph: List<int>
        list of nodes
    
    adj_dict: Dict
        Keys are the source node c_id and values the tuple (rel_id, target node t_id)
    
    original_concepts: Set<int>
        Set of the original concepts.

    
    add_inverse: bool
        Add inverse relations. This will convert num_rel -> 2*num_rel
    
    add_padding: bool
        Add padding until `MAX_EDGES`. Padding value: "<PAD>".
    """
    
    # initialize list
    n_rel = 2*num_rel if add_inverse else num_rel
    adj_list = [[] for _ in range((n_rel+1)*MAX_NODES)] # add one for the self-relation

    
    list_original_concepts = list(original_concepts)
    # define an order for the nodes
    indexes = {v:k for k,v in enumerate(list_original_concepts)}
    # print(indexes)
    # cnt=0
    # counter = defaultdict(int)
    for node in adj_dict: # for each node
        if node in original_concepts: # if original
            for r,n in adj_dict[node]: # add all of its neighbours to itself
                idx = indexes[node] + r*MAX_NODES # compute idx
                if len(adj_list[idx]) < MAX_EDGES:
                    adj_list[idx].append(n) # add node
                    # counter[r]+=1
                    # cnt+=1
        # add inverse
        for r,n in adj_dict[node]: 
            if n in original_concepts: # if original
                idx = indexes[n] + (r+num_rel)*MAX_NODES # compute idx
                if len(adj_list[idx]) < MAX_EDGES:
                    adj_list[idx].append(node) # add node
                    # cnt+=1
                    # counter[r]+=1
    
    
    for idx,node in enumerate(list_original_concepts):
        # add self relation as a new relation
        self_rel_id=34 # last rel id

        self_i = self_rel_id*MAX_NODES + idx
        adj_list[self_i].append(node)


    # padding
    if add_padding:
        for idx,_ in enumerate(adj_list):
            adj_list[idx] += [0 for _ in range(MAX_EDGES-len(adj_list[idx]))]
    # print('ADJ LIST:\n')
    # for idx,l in enumerate(adj_list):
    #     print('node: ', idx % 45, ' relation: ', idx// 45)
    #     print(l)
    return np.array(adj_list)





def score_triple(h,rel,t):
    res = (1 + 1 - spatial.distance.cosine(rel, t - h)) / 2
    return res


def newNode(node, original_nodes, extra_nodes):
    return (node not in original_nodes) and (node not in extra_nodes)

def concepts2adj(node_ids, original, limit):
    """
    Compute the adj list given a set of nodes.
    The adj list will have shape: RxN,E

    Parameters:
    ===========
    node_ids: list
        List of all nodes [original,neighbours]
    original: list
        Mask of length = len(node_ids): [True, True, ..., False, False]
    limit: int
        Number of original nodes and index of first False in max
    """

    global id2relation, relation_embs, concept_embs, threshold

    # print("\n\nSCHEMA GRAPH: ", len(node_ids), ' nodes') #delete
    
    
    cids = np.array(node_ids, dtype=np.int32) #list of nodes

    # initialize shema graph only with the originals
    # we need a list to keep the order
    new_schema_graph = np.array(node_ids[:limit], dtype=np.int32)
    # and a set for quick checking if already visited the node
    original_nodes_set = set(new_schema_graph)
    
    
    extra_nodes = set()
    
    n_rel = len(id2relation) #number of relations
    n_nodes = cids.shape[0] #number of original nodes
    # initialize the adjacency matrix
    # adj = np.zeros((n_rel, n_nodes, n_nodes), dtype=np.uint8)
    
    # adj_dict has c_id (source) as key and a list of tuples (rel_id, c_id (target)) as value
    adj_dict = defaultdict(list)
    # cnt = 0
    # min_score = 10
    # max_score = -10
    for s in range(n_nodes):
        for t in range(n_nodes): # for each pair of nodes
            if original[s] or original[t]: # with at least one original
                s_c, t_c = cids[s], cids[t]
                # print('\n\n Node: ', id2concept[s_c] , ' with ', id2concept[t_c])
                if cpnet.has_edge(s_c, t_c): # if edge exists
                    # cpnet[s_c][t_c] is a dict with edge attributes
                    for e_attr in cpnet[s_c][t_c].values():
                        if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel: #if it a relation of our interest
                            score = score_triple(concept_embs[s_c], relation_embs[e_attr['rel']], concept_embs[t_c])
                            # if over the threshold or both original nodes
                            if score > threshold or (original[s] and original[t]):
                                # skip if new node occurs but the upper limit of nodes is reached
                                # if not ((newNode(s_c, original_nodes_set, extra_nodes) or newNode(t_c, original_nodes_set, extra_nodes)) and (len(extra_nodes) >= MAX_NODES - len(original_nodes_set))):
                                # if not originals add to extra nodes set
                                if s_c not in original_nodes_set:
                                    extra_nodes.add(s_c)
                                if t_c not in original_nodes_set:
                                    extra_nodes.add(t_c)
                                
                                adj_dict[s_c].append((e_attr['rel'],t_c, score))
                                # print(id2concept[s_c], ' --> ', id2relation[e_attr['rel']], ' --> ', id2concept[t_c], '| score = ', score)
                                # cnt +=1

    # cids += 1  # note!!! index 0 is reserved for padding
    # print('cnt in adj_dict = ', cnt)
    for key,value in adj_dict.items():
        _sorted = sorted(value, key=itemgetter(2), reverse=True) # sort based on score
        adj_dict[key] = [(el[0],el[1]) for el in _sorted[:MAX_EDGES]] # keep only the first
    
    if extra_nodes:
        new_schema_graph = np.append(new_schema_graph, list(extra_nodes))
    
    # concept_names = [id2concept[c] for c in new_schema_graph]
    # print("NEW SCHEMA GRAPH: ", len(new_schema_graph), ' nodes. Increase of ', len(new_schema_graph)/or_len) #delete

    # pad schema graph - FIRST SLICE AND PAD AND THEN CREATE ADJ LIST BASED ON THE PADDED SCHEMA GRAPH
    # new_schema_graph = new_schema_graph[:MAX_NODES]
    # new_schema_graph = np.pad(new_schema_graph, (0,MAX_NODES - len(new_schema_graph)))
    adj_list = createAdjList(new_schema_graph, adj_dict, original_nodes_set)

    
    return adj_list, new_schema_graph

def _generateAdj(data_list):
    """
    Find common neighbours in graph and create adjacency list.
    """
    img_id, data = data_list
    res = []
    for _round in data: # for each round
        # keep only MAX_NODES nodes in each round
        # These nodes will be the rows of the adj_list
        if len(_round) > MAX_NODES:
            # keep only the last (most recent nodes)
            _round = _round[len(_round)-MAX_NODES:]

        original_concepts = set(_round) # all original concepts
        common_neighbours = set()
        
        # For each pair of the grounded concepts of current round
        for c1 in _round:
            for c2 in _round:
                if c1!=c2 and c1 in cpnet_simple.nodes and c2 in cpnet_simple.nodes:
                    # cpnet_simple[c] will get a list of the neighbours of c
                    # common_neighbours is the intersection of the neighbours of c1 and c2
                    common_neighbours |= set(cpnet_simple[c1]) & set(cpnet_simple[c2])
    
        # Remove from the common_neighbours the grounded concepts
        common_neighbours = common_neighbours - original_concepts
        # Schema graph is just a list of all the nodes
        schema_graph = sorted(original_concepts) + sorted(common_neighbours)
        # Get a mask for original and extra nodes. True if original.
        original_mask = np.array([i<len(original_concepts) for i in range(len(schema_graph))])
        # construct adjacency list
        adj_list, concepts = concepts2adj(schema_graph, original_mask, len(original_concepts))
        
        res.append({'adj_list':adj_list, 'concepts': concepts})


    return (img_id, res)
    
    

def generateAdj(grounded_path, cpnet_graph_path, cpnet_vocab_path,concept_emb_path, rel_emb_path, prune_threshold, output_path, part=None):
    """
    Generate adjacency matrices in COO format (R*N, N), where
    R: number of relations, N: number of nodes

    Parameters
    ----------

    grounded_path: str
        Path to file with the concepts for each round of each dialog. 
    """

    print(f'Generating adjacency matrices data for {grounded_path}...\n')
    print('Will save to ', output_path)
    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    # concept2id and id2concept refer to the line in which each node is present in cpnet_vocab file
    load_resources(cpnet_vocab_path)
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        grounded_concepts = json.load(fin)
    
    global concept_embs, relation_embs
    concept_embs = np.load(concept_emb_path)
    relation_embs = np.load(rel_emb_path)

    global threshold
    threshold = prune_threshold

    
    data_list = [(img_id, [[concept2id[c] for c in _round] for _round in dialog]) \
        for img_id, dialog in grounded_concepts.items()]
    if part:
        if part =='1':
            data_list = data_list[:3000]  
        elif part =='2':
            data_list = data_list[3000:6000]  
        else:
            data_list = data_list[6000:]
    # sample = ['378466', '332243', '378461', '287140', '575029','525211']
    # sample = ['378466']
    # data_list = [(img_id, [[concept2id[c] for c in _round] for _round in dialog]) \
    #     for img_id, dialog in grounded_concepts.items() if img_id in sample]
    

    print("Will load cpnet.")
    load_cpnet(cpnet_graph_path)
    print("Finished loading cpnet.")



    with Pool() as p:
        res = {k:v for (k,v) in tqdm(p.imap(_generateAdj,data_list,50), total=len(data_list), desc='Generating adj matrices..')}
    

    h = h5py.File(output_path,'w')
    for k,v in tqdm(res.items()):
        grp = h.create_group(k)
        for idx,_round in enumerate(v):
            subgrp = grp.create_group(str(idx))
            data = np.array(_round['adj_list'], dtype=object)
            subgrp.create_dataset('adj_list', data=data.astype(np.int64), chunks=True)
            subgrp.create_dataset('concepts', data=_round['concepts'].astype(np.int64), chunks=True)
    h.close()

    print(f'Adj matrices saved to {output_path}.\n')