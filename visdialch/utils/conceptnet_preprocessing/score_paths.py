from tqdm import tqdm
from multiprocessing import Pool
import json
import numpy as np
from scipy import spatial
from .conceptnet import merged_relations


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}

def score_triple(h, t, r, flag):
    res = -10
    for i in range(len(r)):
        if flag[i]:
            temp_h, temp_t = t, h
        else:
            temp_h, temp_t = h, t
        # result  = (cosine_sim + 1) / 2
        res = max(res, (1 + 1 - spatial.distance.cosine(r[i], temp_t - temp_h)) / 2)
    return res


def score_triples(concept_id, relation_id, debug=False):
    global relation_embs, concept_embs, id2relation, id2concept
    concept = concept_embs[concept_id]
    relation = []
    flag = []
    for i in range(len(relation_id)):
        embs = []
        l_flag = []

        if 0 in relation_id[i] and 17 not in relation_id[i]:
            relation_id[i].append(17)
        elif 17 in relation_id[i] and 0 not in relation_id[i]:
            relation_id[i].append(0)
        if 15 in relation_id[i] and 32 not in relation_id[i]:
            relation_id[i].append(32)
        elif 32 in relation_id[i] and 15 not in relation_id[i]:
            relation_id[i].append(15)

        for j in range(len(relation_id[i])):
            if relation_id[i][j] >= 17:
                embs.append(relation_embs[relation_id[i][j] - 17])
                l_flag.append(1)
            else:
                embs.append(relation_embs[relation_id[i][j]])
                l_flag.append(0)
        relation.append(embs)
        flag.append(l_flag)

    res = 1
    for i in range(concept.shape[0] - 1):
        h = concept[i]
        t = concept[i + 1]
        score = score_triple(h, t, relation[i], flag[i])
        res *= score

    if debug:
        print("Num of concepts:")
        print(len(concept_id))
        to_print = ""
        for i in range(concept.shape[0] - 1):
            h = id2concept[concept_id[i]]
            to_print += h + "\t"
            for rel in relation_id[i]:
                if rel >= 17:
                    # 'r-' means reverse
                    to_print += ("r-" + id2relation[rel - 17] + "/  ")
                else:
                    to_print += id2relation[rel] + "/  "
        to_print += id2concept[concept_id[-1]]
        print(to_print)
        print("Likelihood: " + str(res) + "\n")

    return res


def _scorePaths(datalist):
    img_id, data = datalist
    statement_scores = []
    for pair in data:
        paths = pair["edges"]
        if paths is not None:
            path_scores = []
            for path in paths:
                assert len(path["path"]) > 1
                score = score_triples(concept_id=path["path"], relation_id=path["rel"], debug=True)
                path_scores.append(score)
            statement_scores.append(path_scores)
        else:
            statement_scores.append(None)
    return (img_id, statement_scores)


def scorePaths(raw_paths_path, concept_emb_path, rel_emb_path, cpnet_vocab_path, output_path):
    print(f'Scoring paths for {raw_paths_path}...')
    
    global concept2id, id2concept, relation2id, id2relation
    load_resources(cpnet_vocab_path)

    global concept_embs, relation_embs
    concept_embs = np.load(concept_emb_path) # could already exist?
    relation_embs = np.load(rel_emb_path) # could already exist?

    
    with open(raw_paths_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    data_list = [(k,v) for k,v in data.items()]
    with Pool() as p:
        res = {k:v for (k,v)  in tqdm(p.imap(_scorePaths, data_list), total=len(data), desc='Scoring paths...')}

    with open(output_path, 'w', encoding='utf-8') as fout:    
        fout.write(json.dumps(res))

    print(f'Path scores saved to {output_path}')
    print()
