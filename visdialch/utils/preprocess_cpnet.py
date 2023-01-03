import os
import argparse
from multiprocessing import cpu_count
import getpass
# from tkinter import dialog
from typing import List
import numpy as np
from tqdm import tqdm
import spacy #takes long
from spacy.matcher import Matcher
import en_core_web_sm
import json
import random
import time
from multiprocessing import Pool

from conceptnet_preprocessing.conceptnet import extract_english, construct_graph
from conceptnet_preprocessing.grounding import create_matcher_patterns
from conceptnet_preprocessing.loaders import load_matcher
from conceptnet_preprocessing.tokenizing import tokenizeDatasetFile
from conceptnet_preprocessing.pair_concepts import pairConcepts
from conceptnet_preprocessing.find_paths import findPaths
from conceptnet_preprocessing.score_paths import scorePaths
from conceptnet_preprocessing.prune_paths import prunePaths
from conceptnet_preprocessing.generate_adj import generateAdj


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
DATA_DIR = '/home/'+username+'/Diploma_Thesis/data/'


splits = ['test']
# splits = ['train']

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
pad_conceptnet_vocab_file = DATA_DIR + 'cpnet/pad_concept.txt'
conceptnet_en_file = DATA_DIR + 'cpnet/conceptnet.en.csv'
pad_conceptnet_en_file = DATA_DIR + 'cpnet/pad_conceptnet.en.csv'
conceptnet_patterns = DATA_DIR + 'cpnet/matcher_patterns.json'
conceptnet_unpruned_graph = DATA_DIR + 'cpnet/conceptnet.en.unpruned.graph'
pad_conceptnet_unpruned_graph = DATA_DIR + 'cpnet/pad_conceptnet.en.unpruned.graph'
conceptnet_pruned_graph = DATA_DIR + 'cpnet/conceptnet.en.pruned.graph'
pad_conceptnet_pruned_graph = DATA_DIR + 'cpnet/pad_conceptnet.en.pruned.graph'

glove_file = DATA_DIR + 'glove.6B.300d.txt'
glove_npy = DATA_DIR + 'glove.6B.300d.npy'
glove_vocab = DATA_DIR + 'glove.vocab'

numberbatch_file = DATA_DIR + 'transe/numberbatch-en-19.08.txt'
numberbatch_npy = DATA_DIR + 'transe/nb.npy'
numberbatch_vocab = DATA_DIR + 'transe/nb.vocab'
numberbatch_concept_npy = DATA_DIR + 'transe/concept.nb.npy'
pad_numberbatch_concept_npy = DATA_DIR + 'transe/pad_concept.nb.npy'

transe_ent = DATA_DIR + 'transe/glove.transe.sgd.ent.npy'
transe_rel = DATA_DIR + 'transe/glove.transe.sgd.rel.npy'
pad_transe_ent = DATA_DIR + 'transe/pad_glove.transe.sgd.ent.npy'
pad_transe_rel = DATA_DIR + 'transe/pad_glove.transe.sgd.rel.npy'

grounded = {
    'train': DATA_DIR + 'train_grounded.json',
    'val': DATA_DIR + 'val_grounded.json',
    'test': DATA_DIR + 'test_grounded.json'
}

concepts_paths = {
    'train': DATA_DIR + 'train_paths.json',
    'val': DATA_DIR + 'val_paths.json',
    'test': DATA_DIR + 'test_paths.json'
}

scored_paths = {
    'train': DATA_DIR + 'scored_train_paths.json',
    'val': DATA_DIR + 'scored_val_paths.json',
    'test': DATA_DIR + 'scored_test_paths.json'
}

pruned_concepts_paths = {
    'train': DATA_DIR + 'pruned_train_paths.json',
    'val': DATA_DIR + 'pruned_val_paths.json',
    'test': DATA_DIR + 'pruned_test_paths.json'
}

sub_graphs_adj = {
    'train': DATA_DIR + 'train_adj_list.h5',
    'val': DATA_DIR + 'val_adj_list.h5',
    'test': DATA_DIR + 'test_adj_list.h5'
}

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


    parser.add_argument(
        '--prune_threshold', 
        action="store_true",
        default=0.45,
        help='Threshold for pruning edges.')

    parser.add_argument(
        '--n', default='0',
        help='Number of grounded file.')

    args = parser.parse_args()

    if args.n != '0':
        # grounded['train'] = DATA_DIR + 'train_grounded_part_'+str(args.n.split('_')[0])+'.json'
        sub_graphs_adj['test'] = DATA_DIR + 'test_adj_list_part_'+str(args.n)+'.h5'

    # Print args.
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))



    # GloVe
    if not files_exist([glove_npy, glove_vocab]):
        embeddings2npy(glove_file, glove_npy, glove_vocab, 'glove')
    
    # Numberbatch
    if not files_exist([numberbatch_npy, numberbatch_vocab]):
        embeddings2npy(numberbatch_file, numberbatch_npy , numberbatch_vocab, 'numberbatch')
    
    # Extract English relations from ConceptNet and vocab 
    # if files don't exist
    if not files_exist([conceptnet_en_file, conceptnet_vocab_file]):
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
    if not files_exist([conceptnet_unpruned_graph, conceptnet_pruned_graph]):
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
    if not files_exist([conceptnet_patterns]):
        create_matcher_patterns(conceptnet_vocab_file, conceptnet_patterns)

    # if args.debug:
    #     global splits
    #     splits = ['train']
    #     sub_graphs_adj['train'] = '/home/'+username+'/Diploma_Thesis/data/debug_adj.h5'

    # ==================================================================================
    # Preprocess dataset files.
    # ==================================================================================
    # Tokenize and create files with keys the image_id
    if (not files_exist([dataset_tokenized_paths[split] for split in splits]) or args.clear) and False:
        for split in splits:
            tokenizeDatasetFile(
                    dataset_paths[split], 
                    dataset_tokenized_paths[split], 
                    debug=args.debug)

    # Pair the dataset entities with the ConceptNet entities
    if (not files_exist([grounded[split] for split in splits]) or args.clear) and False:
        start_time = time.time()
        for split in splits:
            pairConcepts(
                dataset_tokenized_paths[split],
                conceptnet_vocab_file, 
                conceptnet_patterns, 
                grounded[split],
                debug=args.debug)
        print("--- Completed concept pairing in %s seconds. ---" % (time.time() - start_time))

    

    if (not files_exist([concepts_paths[split] for split in splits]) or args.clear) and False:
        start_time = time.time()
        for split in splits:
            findPaths(
                grounded[split],
                conceptnet_vocab_file,
                conceptnet_pruned_graph,
                concepts_paths[split])
        print("--- Completed path exploring in %s seconds. ---" % (time.time() - start_time))
    

    if not (files_exist([scored_paths[split] for split in splits]) or args.clear) and False:
        start_time = time.time()
        for split in splits:
            scorePaths(
                concepts_paths[split],
                transe_ent,
                transe_rel,
                conceptnet_vocab_file,
                scored_paths[split]
                )
        print("--- Completed path scoring in %s seconds. ---" % (time.time() - start_time))
    
    if not (files_exist([pruned_concepts_paths[split] for split in splits]) or args.clear) and False:
        start_time = time.time()
        for split in splits:
            prunePaths(
                concepts_paths[split],
                scored_paths[split],
                pruned_concepts_paths[split],
                grounded[split],
                args.prune_threshold
                )
        print("--- Completed path pruning in %s seconds. ---" % (time.time() - start_time))
    
    if not files_exist([sub_graphs_adj[split] for split in splits]) or args.clear or args.debug:
        start_time = time.time()
        for split in splits:
            generateAdj(
                grounded[split],
                pad_conceptnet_pruned_graph,
                pad_conceptnet_vocab_file,
                pad_transe_ent,
                transe_rel,
                args.prune_threshold,
                sub_graphs_adj[split],
                args.n
                # args.n.split('_')[1]
                )
        print("--- Completed generating subgraphs in %s seconds. ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
