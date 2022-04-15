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
from collections import defaultdict
from operator import itemgetter

inv_relations = ['inv_'+el for el in merged_relations]
full_relations = merged_relations + inv_relations
max_nodes = 45

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




def createAdjList(schema_graph, adj_dict, original_concepts, max_nodes = 45,max_edges=45, num_rel=17, add_inverse=True, add_padding=True):
    """
    The adjacency list will have shape: (num_rel x max_nodes,max_edges), or (2xnum_rel x max_nodes,max_edges)
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

    max_edges: int
        Number of relations for each node in the adjacency list
    
    add_inverse: bool
        Add inverse relations. This will convert num_rel -> 2*num_rel
    
    add_padding: bool
        Add padding until `max_edges`. Padding value: "<PAD>".
    """
    
    # initialize list
    n_rel = 2*num_rel if add_inverse else num_rel
    adj_list = [[] for _ in range(n_rel*max_nodes)]


    _schema = {v:i for i,v in enumerate(schema_graph)}

    # iterate schema_graph to keep the order of the nodes
    for idx,node in enumerate(schema_graph):
        # print('node ', id2concept[node])
        for neighbour in adj_dict[node]:
            # add in list only if it is original
            if node in original_concepts:
                i = neighbour[0]*max_nodes + idx
                if len(adj_list[i]) < max_edges:
                    # adj_list[i].append(str(id2concept[neighbour[1]]))
                    adj_list[i].append(neighbour[1])

            if add_inverse and neighbour[1] in original_concepts:
                # id of inverse relation of r: r + num_rel
                i_reverse = _schema[neighbour[1]]+(neighbour[0]+num_rel)*max_nodes
                if i_reverse >= len(adj_list):
                    print('\n_schema[neighbour[1]]+(neighbour[0]+num_rel)*max_nodes: ', _schema[neighbour[1]],' + (',neighbour[0],'+',num_rel,')*',max_nodes,'\n'
                    ,'i_reverse = ', i_reverse
                    ,'\nlen(original_concepts) = ', len(original_concepts))
                if len(adj_list[i_reverse]) < max_edges:
                    # adj_list[i_reverse].append(str(id2concept[node]))
                    adj_list[i_reverse].append(node)
        
    for idx,node in enumerate(schema_graph):
        # add self relation using `isa` relation type.. rel_id=5
        if node and node in original_concepts: # if not pad
            isa_rel_id=5
            self_i = isa_rel_id*max_nodes + idx
            adj_list[self_i].append(node)
            # adj_list[self_i].append(str(id2concept[node]))


    # padding
    if add_padding:
        for idx,_ in enumerate(adj_list):
            adj_list[idx] += [0 for _ in range(max_edges-len(adj_list[idx]))]
    return np.array(adj_list)





def score_triple(h,rel,t):
    res = (1 + 1 - spatial.distance.cosine(rel, t - h)) / 2
    return res


def newNode(node, original_nodes, extra_nodes):
    return (node not in original_nodes) and (node not in extra_nodes)

def concepts2adj(node_ids, original, limit, max_edges = 45):
    """
    Compute the adj list given a set of nodes.
    The adj list will have shape: RxN,E
    """
    global id2relation, relation_embs, concept_embs, threshold

    # print("\n\nSCHEMA GRAPH: ", len(node_ids), ' nodes') #delete
    
    
    cids = np.array(node_ids, dtype=np.int32) #list of nodes

    # initialize shema graph only with the originals
    # we need a list to keep the order
    new_schema_graph = np.array(node_ids[:limit], dtype=np.int32)
    # and a set for quick checking if already visited the node
    new_schema_graph_set = set(new_schema_graph)
    
    
    extra_nodes = set()
    
    n_rel = len(id2relation) #number of relations
    n_nodes = cids.shape[0] #number of nodes
    # initialize the adjacency matrix
    # adj = np.zeros((n_rel, n_nodes, n_nodes), dtype=np.uint8)
    
    # adj_dict has c_id (source) as key and a list of tuples (rel_id, c_id (target)) as value
    adj_dict = defaultdict(list)
    # cnt = 0
    # min_score = 10
    # max_score = -10
    for s in range(n_nodes):
        for t in range(n_nodes): # for each pair of nodes in initial schema graph
            if original[s] or original[t]:
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
                                # if not ((newNode(s_c, new_schema_graph_set, extra_nodes) or newNode(t_c, new_schema_graph_set, extra_nodes)) and (len(extra_nodes) >= max_nodes - len(new_schema_graph_set))):
                                # if not originals add to extra nodes set
                                if s_c not in new_schema_graph_set:
                                    extra_nodes.add(s_c)
                                if t_c not in new_schema_graph_set:
                                    extra_nodes.add(t_c)
                                
                                adj_dict[s_c].append((e_attr['rel'],t_c, score))
                                    # min_score = min(min_score, score)
                                    # max_score = max(max_score, score)
                                    # adj[e_attr['rel']][s][t] = 1
                                # print(id2concept[s_c], ' --> ', id2relation[e_attr['rel']], ' --> ', id2concept[t_c], '| score = ', score)
                                    # cnt +=1

    # cids += 1  # note!!! index 0 is reserved for padding
    
    for key,value in adj_dict.items():
        _sorted = sorted(value, key=itemgetter(2), reverse=True) # sort based on score
        adj_dict[key] = [(el[0],el[1]) for el in _sorted[:max_edges]] # keep only the first
    
    if extra_nodes:
        new_schema_graph = np.append(new_schema_graph, list(extra_nodes))
    
    # concept_names = [id2concept[c] for c in new_schema_graph]
    # print("NEW SCHEMA GRAPH: ", len(new_schema_graph), ' nodes. Increase of ', len(new_schema_graph)/or_len) #delete

    # pad schema graph - FIRST SLICE AND PAD AND THEN CREATE ADJ LIST BASED ON THE PADDED SCHEMA GRAPH
    # new_schema_graph = new_schema_graph[:max_nodes]
    # new_schema_graph = np.pad(new_schema_graph, (0,max_nodes - len(new_schema_graph)))
    adj_list = createAdjList(new_schema_graph, adj_dict, new_schema_graph_set, max_edges=max_edges)

    
    return adj_list, new_schema_graph

def _generateAdj(data_list):
    """
    Find common neighbours in graph and create adjacency list.
    """
    img_id, data = data_list
    res = []
    for _round in data: # for each round
        if len(_round) > max_nodes:
            # print('Exceeded on image: ', img_id)
            print('Max nodes = ', max_nodes, 'len(_round) = ', len(_round))
            _round = _round[len(_round)-max_nodes:] # keep the last 
            assert len(_round) == max_nodes
            print("Asserted that exceeding node limit was handled.")
        all_concepts = set(_round)
        # print('Grounded: ')
        # for c in all_concepts:
        #     print(id2concept[c], end='\t')
        common_neighbours = set()
        # print("\n\nROUND CONCEPTS:") #delete
        # for c1 in all_concepts:#delete
        #     print(id2concept[c1])#delete
        
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
        # Get the limit between original and extra nodes
        arange = np.arange(len(schema_graph))
        original_mask = arange < len(all_concepts)
        adj_list, concepts = concepts2adj(schema_graph, original_mask, len(all_concepts))
        
        res.append({'adj_list':adj_list, 'concepts': concepts})
    
    
    return (img_id, res)
    
    

def generateAdj(grounded_path, cpnet_graph_path, cpnet_vocab_path,concept_emb_path, rel_emb_path, prune_threshold, output_path):
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
    
    # sample = ['378466', '332243', '378461', '287140', '575029','525211']
    # data_list = [(img_id, [[concept2id[c] for c in _round] for _round in dialog]) \
    #     for img_id, dialog in grounded_concepts.items() if img_id in sample]
    
    # print('data_list = ', data_list)

    print("Will load cpnet.")
    load_cpnet(cpnet_graph_path)
    print("Finished loading cpnet.")



    with Pool() as p:
        res = {k:v for (k,v) in tqdm(p.imap(_generateAdj,data_list,80), total=len(grounded_concepts), desc='Generating adj matrices..')}
    

    h = h5py.File(output_path,'w')
    for k,v in tqdm(res.items()):
        # print('Creating group for image: ', k)
        grp = h.create_group(k)
        for idx,_round in enumerate(v):
            subgrp = grp.create_group(str(idx))
            # string_dt = h5py.special_dtype(vlen=str)
            data = np.array(_round['adj_list'], dtype=object)
            # concept_names = np.array(_round['concepts'], dtype=object)
            # print(type(_round['adj_list'][0]))
            # print(type(_round['adj_list'][0][0]))
            subgrp.create_dataset('adj_list', data=data.astype(np.int64), chunks=True)
            # subgrp.create_dataset('adj_list', data=data, dtype=string_dt)
            # subgrp.create_dataset('concepts', data=concept_names, dtype=string_dt)
            subgrp.create_dataset('concepts', data=_round['concepts'].astype(np.int64), chunks=True)
            # subgrp.create_dataset('original_mask', data=_round['original_mask'].astype(np.int64), chunks=True)
            # subgrp.create_dataset('shape', data=_round['adj'].shape, chunks=True)
        # grp.create_dataset('original_limit', data=[_round['original_limit'] for _round in v], chunks=True)
    h.close()
    print(f'Adj matrices saved to {output_path}.\n')