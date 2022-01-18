import os
import argparse
from multiprocessing import cpu_count
import getpass
from typing import List
import numpy as np
from tqdm import tqdm
import spacy #takes long
from spacy.matcher import Matcher
import en_core_web_sm
import json
import nltk

from conceptnet import extract_english, construct_graph
from grounding import create_matcher_patterns

EOS_TOK = "</S>"
UNK_TOK = '<UNK>'
PAD_TOK = '<PAD>'
SEP_TOK = '<SEP>'

EXTRA_TOKS = [EOS_TOK, UNK_TOK, PAD_TOK, SEP_TOK]

# ==================================================================================
# Paths
# ==================================================================================
username = getpass.getuser()
WORKING_DIR = os.getcwd()
DATA_DIR = '/home/'+username+'/KBGN-Implementation/data/'


splits = ['train', 'val', 'test']

dataset_paths = {
    'train': DATA_DIR + 'visdial_1.0_train.json',
    'val': DATA_DIR + 'visdial_1.0_val.json',
    'test': DATA_DIR + 'visdial_1.0_test.json'
}

dataset_tokenized_paths = {
    'train': DATA_DIR + 'visdial_1.0_train_tokenized.json',
    'val': DATA_DIR + 'visdial_1.0_val_tokenized.json',
    'test': DATA_DIR + 'visdial_1.0_test_tokenized.json'
}

conceptnet_csv_file = DATA_DIR + 'cpnet/conceptnet-assertions-5.6.0.csv'
conceptnet_vocab_file = DATA_DIR + 'cpnet/concept.txt'
conceptnet_en_file = DATA_DIR + 'cpnet/conceptnet.en.csv'
conceptnet_patterns = DATA_DIR + 'cpnet/matcher_patterns.json'
conceptnet_unpruned_graph = DATA_DIR + 'cpnet/conceptnet.en.unpruned.graph'
conceptnet_pruned_graph = DATA_DIR + 'cpnet/conceptnet.en.pruned.graph'

glove_file = DATA_DIR + 'glove.6B.300d.txt'
glove_npy = DATA_DIR + 'glove.6B.300d.npy'
glove_vocab = DATA_DIR + 'glove.vocab'

numberbatch_file = DATA_DIR + 'transe/numberbatch-en-19.08.txt'
numberbatch_npy = DATA_DIR + 'transe/nb.npy'
numberbatch_vocab = DATA_DIR + 'transe/nb.vocab'
numberbatch_concept_npy = DATA_DIR + 'transe/concept.nb.npy'

grounded = {
    'train': DATA_DIR + 'train_grounded.json',
    'val': DATA_DIR + 'val_grounded.json',
    'test': DATA_DIR + 'test_grounded.json'
}



nltk.download('stopwords', quiet=True)
nltk_stopwords = nltk.corpus.stopwords.words('english')

def files_exist(files: List):
    """
    Function that checks if all the files listed exist.
    """
    return all([os.path.isfile(_file) for _file in files])


def load_embeddings(path, embedding='glove', add_special_tokens=EXTRA_TOKS, random_state=0):
    vocab = []
    embeddings = None
    cnt = sum(1 for _ in open(path, 'r', encoding='utf-8')) # number of lines in file
    
    with open(path, "r", encoding="utf8") as f:
        if embedding == 'numberbatch': # skip line if numberbatch
            f.readline() 
        
        for i, line in tqdm(enumerate(f), total=cnt):
            elements = line.strip().split(" ")
            word = elements[0].lower()
            vec = np.array(elements[1:], dtype=float)
            vocab.append(word)
            if embeddings is None:
                embeddings = np.zeros((cnt, len(vec)), dtype=np.float64)
            embeddings[i] = vec

    np.random.seed(random_state)
    n_special = 0 if add_special_tokens is None else len(add_special_tokens)
    add_vectors = np.random.normal(np.mean(embeddings), np.std(embeddings), size=(n_special, embeddings.shape[1]))
    embeddings = np.concatenate((embeddings, add_vectors), 0)
    vocab += add_special_tokens
    return vocab, embeddings


def embeddings2npy(emb_path, output_npy_path, output_vocab_path, embedding='glove'):
    """
    Extract the embeddings GloVe or Numberbatch to .npy files and the vocab
    files.

    Parameters
    ----------
    emb_path: str
        Path to the embeddings initial file.
    
    output_npy_path: str
        Path to the resulted .npy file.
    
    output_vocab_path: str
        Path to the resulted .vocab file.
    
    embedding: str
        The name of the embedding {'glove', 'numberbatch'}
        to be used for determining wether to skip the first 
        line or not.
    """
    
    print(f'binarizing {embedding} embeddings...')

    vocab, vectors = load_embeddings(emb_path, embedding=embedding)
    np.save(output_npy_path, vectors)
    with open(output_vocab_path, "w", encoding='utf-8') as fout:
        for word in vocab:
            fout.write(word + '\n')

    print(f'Binarized {embedding} embeddings saved to {output_npy_path}')
    print(f'{embedding} vocab saved to {output_vocab_path}\n')


def load_pretrained_embeddings(emb_npy_path, emb_vocab_path, vocab_path, verbose=True, save_path=None):
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            vocab.append(line.strip())
    load_vectors_from_npy_with_vocab(emb_npy_path=emb_npy_path, emb_vocab_path=emb_vocab_path, vocab=vocab, verbose=verbose, save_path=save_path)


def load_vectors_from_npy_with_vocab(emb_npy_path, emb_vocab_path, vocab, verbose=True, save_path=None):
    with open(emb_vocab_path, 'r', encoding='utf-8') as fin:
        emb_w2idx = {line.strip(): i for i, line in enumerate(fin)}
    emb_emb = np.load(emb_npy_path)
    vectors = np.zeros((len(vocab), emb_emb.shape[1]), dtype=float)
    oov_cnt = 0
    for i, word in enumerate(vocab):
        if word in emb_w2idx:
            vectors[i] = emb_emb[emb_w2idx[word]]
        else:
            oov_cnt += 1
    if verbose:
        print(len(vocab))
        print('embedding oov rate: {:.4f}'.format(oov_cnt / len(vocab)))
    if save_path is None:
        return vectors
    np.save(save_path, vectors)

def tokenize_dataset_file(dialog_path, output_path, concat=False, debug=False):
    """
    Tokenize the dialogs and create json files with key the image_id
    and values the dialogs.

    Parameters
    ----------
    dialog_path: str
        Path to the dataset's dialogs file.

    output_path: str
        Path to the resulted JSON file.
        Format: 
            keys: image_id
            values: dialog
    
    concat: bool
        If True the history will contain concatenated QA pairs from 
        previous rounds.
    """

    print(f'Tokenizing {dialog_path} file.')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    cnt = sum(1 for _ in open(dialog_path, 'r'))

    tokens = {}
    
    with open(dialog_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
        split = data['split']
        dialogs = data['data']['dialogs']
        if debug:
            dialogs = dialogs[:5]
        answers = data['data']['answers']
        questions = data['data']['questions']
    

    for dialog in tqdm(dialogs, total=cnt, desc='tokenizing'):

        history = [[tokenize_sentence_spacy(nlp, dialog['caption'])]] \
            + [[tokenize_sentence_spacy(nlp, questions[_round['question']])] \
            + ([tokenize_sentence_spacy(nlp, answers[_round['answer']])] if 'answer' in _round.keys() else []) \
            for _round in dialog['dialog']]

        
        # if concatenate
        if concat:
            concatenated_history = []
            concatenated_history.append([dialog['caption']])
            for i in range(1, len(history)):
                concatenated_history.append([])
                for j in range(i + 1):
                    concatenated_history[i].extend(history[j])
            history = concatenated_history

        tokens[dialog['image_id']] = history
        
    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(tokens))

def tokenize_sentence_spacy(nlp, sent):
    tokens = ' '.join([tok.text.lower() for tok in nlp(sent)])
    return tokens

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

def prune(data, cpnet_vocab_path):
    # reload cpnet_vocab
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]

    prune_data = {}
    for item in tqdm(data, total=len(data), desc='Prunning...'):
        prune_data[item] = []
        for _round in data[item]:
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
            
            # item["qc"] = prune
            prune_data[item].append(prune)
    return prune_data

def lemmatize(nlp, concept):
    doc = nlp(concept.replace("_", " "))
    lcs = set()
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs

def pairConcepts(input_path, cpnet_vocab_path, pattern_path, output_path, num_processes=1, debug=False):
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
    
    with open(input_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    
    if debug and len(data) > 2:
        print("Debug with big file.")
        data = {k: data[k] for k in list(data.keys())[:2]}
    
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    matcher = load_matcher(nlp, pattern_path)
    CPNET_VOCAB = load_cpnet_vocab(cpnet_vocab_path) #for hard ground...

    blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])

    res = {}
    for img_id, dialog in tqdm(data.items(), total=len(data), desc='Pairing...'):
        # dialog is a list of all the rounds
        res[img_id] = []
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
                exact_match = set([concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()])
                # print("exact match: ", exact_match)
                assert len(exact_match) < 2
                mentioned_concepts.update(exact_match)

            if len(mentioned_concepts) == 0:
                print("AAAAAA\n\n"*13) # panic
                break
                
            
            mentioned_concepts = sorted(list(mentioned_concepts))
            res[img_id].append(mentioned_concepts)
    print("before\n",res)
    res = prune(res, cpnet_vocab_path)
    print("after\n",res)

    # check_path(output_path)
    with open(output_path, 'w', encoding='utf-8') as fout:
        # for dic in res:
        fout.write(json.dumps(res) + '\n')

    print(f'grounded concepts saved to {output_path}.\n')
            
        

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clear', 
        action="store_true", 
        default=False, 
        help='Extract everything from scratch.')

    parser.add_argument(
        '--debug', 
        action="store_true",
        default=False, 
        help='Use only a 5 examples, for debugging reasons.')

    args = parser.parse_args()

    # Print args.
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))



    # GloVe
    if not files_exist([glove_npy, glove_vocab]) or args.clear:
        embeddings2npy(glove_file, glove_npy, glove_vocab, 'glove')
    
    # Numberbatch
    if not files_exist([numberbatch_npy, numberbatch_vocab]) or args.clear:
        embeddings2npy(numberbatch_file, numberbatch_npy , numberbatch_vocab, 'numberbatch')
    
    # Extract English relations from ConceptNet and vocab 
    # if files don't exist
    if not files_exist([conceptnet_en_file, conceptnet_vocab_file]) or args.clear:
        extract_english(conceptnet_csv_file, conceptnet_en_file, conceptnet_vocab_file)
    
    # Load pretrained embeddings
    if not files_exist([numberbatch_concept_npy]):
        load_pretrained_embeddings(
            numberbatch_npy, 
            numberbatch_vocab, 
            conceptnet_vocab_file, 
            False,
            numberbatch_concept_npy)

    # Construct Graph Unpruned
    if not files_exist([conceptnet_unpruned_graph, conceptnet_pruned_graph]) or args.clear:
        construct_graph(
            conceptnet_en_file, 
            conceptnet_vocab_file, 
            conceptnet_unpruned_graph, 
            False)
        # Construct Graph Pruned
        construct_graph(
            conceptnet_en_file, 
            conceptnet_vocab_file, 
            conceptnet_pruned_graph, 
            True)

    # Create patterns for matching dataset entities with ConceptNet
    # entities.
    if not files_exist([conceptnet_patterns]) or args.clear:
        create_matcher_patterns(conceptnet_vocab_file, conceptnet_patterns)


    # ==================================================================================
    # Preprocess dataset files.
    # ==================================================================================
    # Tokenize and create files with keys the image_id
    if not files_exist([dataset_tokenized_paths[split] for split in splits]) or args.clear:
        for split in splits:
            tokenize_dataset_file(
                    dataset_paths[split], 
                    dataset_tokenized_paths[split], 
                    debug=args.debug)

    # Pair the dataset entities with the ConceptNet entities
    if not files_exist([grounded[split] for split in splits]) or args.clear:
        for split in splits:
            pairConcepts(dataset_tokenized_paths[split],
                conceptnet_vocab_file, 
                conceptnet_patterns, 
                grounded[split],
                debug=args.debug)
    


if __name__ == '__main__':
    main()