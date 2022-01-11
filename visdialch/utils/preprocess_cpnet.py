import os
import argparse
from multiprocessing import cpu_count
import getpass
import numpy as np
from tqdm import tqdm
import spacy
import json

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


jsonl_paths = {
    'train_jsonl': DATA_DIR + 'train.jsonl', 
    'val_jsonl': DATA_DIR + 'val.jsonl',
    'test_jsonl': DATA_DIR + 'test.jsonl'
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
numberbatch_npy = DATA_DIR + 'transe/nb.npy',
numberbatch_vocab = DATA_DIR + 'transe/nb.vocab',
numberbatch_concept_npy = DATA_DIR + 'transe/concept.nb.npy'



def load_embeddings(path, embedding=False, add_special_tokens=None, random_state=0):
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


def embeddings2npy(emb_path, output_npy_path, output_vocab_path, embedding):
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

def tokenize_statement_file(statement_path, output_path):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    nrow = sum(1 for _ in open(statement_path, 'r'))
    with open(statement_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, total=nrow, desc='tokenizing'):
            data = json.loads(line)
            for statement in data['statements']:
                tokens = [tok.text.lower() for tok in nlp(statement['statement'])]
                # tokens = [tok.text.lower() for tok in nlp(sent)]
                fout.write(' '.join(tokens) + '\n')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', type=bool, default=False, help='Extract everything from scratch.')

    args = parser.parse_args()

    # Print args.
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))


# print(jsonl_paths['train_jsonl'])
# print(jsonl_paths['val_jsonl'])
# print(jsonl_paths['test_jsonl'])

# Create jsonl files if they do not exist
# for file in jsonl_paths:
#     if not os.path.isfile(jsonl_paths[file]):
#         pass


    # GloVe
    if not os.path.isfile(glove_npy) and not os.path.isfile(glove_vocab) or args.clear:
        embeddings2npy(glove_file, glove_npy, glove_vocab)
    
    # Numberbatch
    if not os.path.isfile(numberbatch_npy) and not os.path.isfile(numberbatch_vocab) or args.clear:
        embeddings2npy(numberbatch_file, numberbatch_npy , numberbatch_vocab, True)
    
    # Extract English relations from ConceptNet and vocab 
    # if files don't exist
    if not os.path.isfile(conceptnet_en_file) and not os.path.isfile(conceptnet_vocab_file) or args.clear:
        extract_english(conceptnet_csv_file, conceptnet_en_file, conceptnet_vocab_file)
    
    # Load pretrained embeddings
    load_pretrained_embeddings(
        numberbatch_npy, 
        numberbatch_vocab, 
        conceptnet_vocab_file, 
        False,
        numberbatch_concept_npy)

    # Construct Graph Unpruned
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
    create_matcher_patterns(conceptnet_vocab_file, conceptnet_patterns)


    # ==================================================================================
    # Preprocess dataset files.
    # ==================================================================================

    


if __name__ == '__main__':
    main()