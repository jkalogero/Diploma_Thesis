from spacy.matcher import Matcher
import networkx as nx
import json

from conceptnet import merged_relations



def load_matcher(nlp, pattern_path):
    """
    Load the file with the patterns.
    """
    with open(pattern_path, "r", encoding="utf8") as fin:
        all_patterns = json.load(fin)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in all_patterns.items():
        matcher.add(concept, None, pattern)
    return matcher


def load_cpnet_vocab(cpnet_vocab_path):
    """
    Load the file with the ConceptNet vocab.
    """
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]
    return cpnet_vocab


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
