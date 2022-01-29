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

def hard_ground(nlp, sent, cpnet_vocab, all_concepts):
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab and t.lemma_ not in all_concepts:
            res.add(t.lemma_)
    sent = " ".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    
    if len(res) == 0:
        for t in doc:
            if t.text in cpnet_vocab and t.text not in all_concepts:
                res.add(t.text)
    try:
        assert len(res) > 0
    except Exception:
        print(f"For {sent}, concept not found in hard grounding.")
    return res

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
    all_concepts = set()
    for i, _round in enumerate(dialog):   
        # print('round ', i)
        _round_t = ' '.join(_round)

        doc = nlp(_round_t)
        matches = matcher(doc) #find mathces in cpnet of the tokens in doc
        # print("matches = ", matches)
        mentioned_concepts = set()
        span_to_concepts = {} # dict that matches each token of the doc to a nubmer or concepts in cpnet

        for match_id, start, end in matches: #for each match
            span = doc[start:end].text  # the matched span (the token in doc whose match we are examining)
            original_concept = nlp.vocab.strings[match_id] # get the string representation of the match
            original_concept_set = set()
            original_concept_set.add(original_concept)

            if len(original_concept.split("_")) == 1:
                # tag = doc[start].tag_
                # if tag in ['VBN', 'VBG']:
                # print('BEFORE LEMMATIZING', original_concept_set)
                original_concept_set.update(lemmatize(nlp, nlp.vocab.strings[match_id]))
                # print('AFTER LEMMATIZING', original_concept_set)

            # print("IN MATCHES span: ", span)
            # print("IN MATCHES concept_sorted: ", original_concept_set)

            if span not in span_to_concepts:
                span_to_concepts[span] = set()

            span_to_concepts[span].update(original_concept_set)

        for span, concepts in span_to_concepts.items():
            concepts_sorted = list(concepts)
            concepts_sorted.sort(key=len)
            
            shortest = concepts_sorted[0:3]

            for c in shortest:
                if c in blacklist:
                    continue

                lcs = lemmatize(nlp, c)
                intersect = lcs.intersection(shortest)
                # print("intersect = ", intersect)
                if len(intersect) > 0 and list(intersect)[0] not in all_concepts:
                    mentioned_concepts.add(list(intersect)[0])
                    all_concepts.add(list(intersect)[0])
                    break
                # elif c not in all_concepts:
                    print('Adding: ', c)
                #     mentioned_concepts.add(c)
                #     all_concepts.add(c)
                #     break

            # if a mention exactly matches with a concept
            # exact_match = set([concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()])
            # assert len(exact_match) < 2
            # mentioned_concepts.update(exact_match)

        if len(mentioned_concepts) == 0:
            mentioned_concepts = hard_ground(nlp, _round_t, cpnet_vocab_path, all_concepts)
                        
        
        mentioned_concepts = sorted(list(mentioned_concepts))
        # print("mentioned_concepts = ", mentioned_concepts,'\n\n')
        # res[img_id].append(mentioned_concepts)
        paired_concepts.append(mentioned_concepts)
        # print(paired_concepts)

    paired_concepts = prune(paired_concepts, cpnet_vocab_path)
    # print(paired_concepts)

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
