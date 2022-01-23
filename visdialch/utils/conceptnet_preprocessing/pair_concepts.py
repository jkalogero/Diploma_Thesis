import random
import numpy as np
import json
from multiprocessing import Pool
from tqdm import tqdm
import time
from spacy.matcher import Matcher
import spacy
from .loaders import load_matcher
import nltk


def lemmatize(nlp, concept):
    doc = nlp(concept.replace("_", " "))
    lcs = set()
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs

def prune(data, cpnet_vocab_path):
    # reload cpnet_vocab
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]

    pruned_data = []
    for _round in data:
        prune = []
        for c in _round:
            if c[-2:] == "er" and c[:-2] in _round:
                continue
            if c[-1:] == "e" and c[:-1] in _round:
                continue
            have_stop = False
            # remove all concepts having stopwords, including hard-grounded ones
            for t in c.split("_"):
                if t in nltk_stopwords:
                    have_stop = True
            if not have_stop and c in cpnet_vocab:
                prune.append(c)
        
        
        pruned_data.append(prune)
    return pruned_data


def _pairConcepts(data_list):

    start = time.time()
    img_id, dialog, cpnet_vocab_path, pattern_path, concat = data_list

    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    matcher = load_matcher(nlp, pattern_path)
    # CPNET_VOCAB = load_cpnet_vocab(cpnet_vocab_path) #for hard ground...
    print('-- load time pair: ', time.time()- start)
    blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be",
                 "yes", "no"
                 ])
    
    paired_concepts = []
    for i, _round in enumerate(dialog):   
        _round_t = ' '.join(_round)

        doc = nlp(_round_t)
        matches = matcher(doc)
        mentioned_concepts = set()
        span_to_concepts = {}

        for match_id, start, end in matches:
            span = doc[start:end].text  # the matched span
            original_concept = nlp.vocab.strings[match_id]
            original_concept_set = set()
            original_concept_set.add(original_concept)

            if len(original_concept.split("_")) == 1:
                # tag = doc[start].tag_
                # if tag in ['VBN', 'VBG']:

                original_concept_set.update(lemmatize(nlp, nlp.vocab.strings[match_id]))

            if span not in span_to_concepts:
                span_to_concepts[span] = set()

            span_to_concepts[span].update(original_concept_set)

        for span, concepts in span_to_concepts.items():
            concepts_sorted = list(concepts)
            # print("span: ", span)
            # print("concept_sorted: ", concepts_sorted)
            concepts_sorted.sort(key=len)

            # mentioned_concepts.update(concepts_sorted[0:2])

            shortest = concepts_sorted[0:3]

            for c in shortest:
                if c in blacklist:
                    continue

                # a set with one string like: set("like_apples")
                lcs = lemmatize(nlp, c)
                intersect = lcs.intersection(shortest)
                if len(intersect) > 0:
                    mentioned_concepts.add(list(intersect)[0])
                else:
                    mentioned_concepts.add(c)

            # if a mention exactly matches with a concept
            # exact_match = set([concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()])
            # assert len(exact_match) < 2
            # mentioned_concepts.update(exact_match)

        if len(mentioned_concepts) == 0:
            print("No concepts added for this round.")
            print(f'img_id = {img_id}, round = {_round}') 
            
        
        mentioned_concepts = sorted(list(mentioned_concepts))
        # res[img_id].append(mentioned_concepts)
        paired_concepts.append(mentioned_concepts)

    paired_concepts = prune(paired_concepts, cpnet_vocab_path)

    # if concatenate
    if concat:
        concatenated_history = []
        concatenated_history.append(paired_concepts[0])
        for i in range(1, len(paired_concepts)):
            concatenated_history.append([])
            for j in range(i + 1):
                concatenated_history[i].extend(paired_concepts[j])
        paired_concepts = concatenated_history

        # check if can do better
        for idx,_r in enumerate(paired_concepts):
            paired_concepts[idx] = list(set(_r))
    
    return (img_id, paired_concepts)



def pairConcepts(input_path, cpnet_vocab_path, pattern_path, output_path, concat=True, num_processes=1, debug=False):
    """
    Pair the entities present in the dialog/image with the
    entities found in ConceptNet.

    Parameters
    ----------
    input_path: str
        Path to the dataset's tokenized file.

    cpnet_vocab_path: str
        Path to ConceptNet's vocab file.

    pattern_path: str
        Path to the file containing the matching patterns.
    
    output_path: str
        Path to the resulted file.
    
    num_processes: int
        Path to the file containing the matching patterns.
    
    debug: bool
        Use only five examples if True.
    """
    
    global nltk_stopwords
    nltk.download('stopwords', quiet=True)
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    
    with open(input_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    
    # if debug and len(data) > 2:
    #     print("Debug with big file.")
    #     data = {k: data[k] for k in list(data.keys())[:2]}
    
    res = {}
    # # Multiprocessing
    data_list = [(k,v,cpnet_vocab_path, pattern_path, concat) for k,v in data.items()]
    with Pool() as p:
        res = {k:v for (k,v) in tqdm(p.imap(_pairConcepts, data_list), total = len(data), desc='Pairing concepts...')}
        
    
    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(res))

    print(f'grounded concepts saved to {output_path}.\n')
