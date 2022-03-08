import numpy as np
from numpy import random
import getpass
from vocabulary import Vocabulary
import yaml
from tqdm import tqdm
import json

# For the VisDial Vocabulary there were:
#   118 OOV for GloVe embeddings
#   0 OOV for ELMo embeddings
#   611 OOV for GloVe embeddings

username = getpass.getuser()
DATA_DIR = '/home/'+username+'/KBGN-Implementation/data/'

def embeddings2npy(input_path, output_path, dataset_vocabulary, name):
    embedding = {} # dict with keys the word and values the values from embedding file.
    with open(input_path, "r") as emb_file:
        if not name == 'elmo':
            if name == 'numberbatch':
                next(emb_file)
            for line in emb_file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embedding[word] = vector
        else:
            embedding = json.load(emb_file)
    
    KAT = []
    for key in embedding.keys():
        keylist = [key]
        token = dataset_vocabulary.to_indices(keylist) # list of idxs for the given words
        key_and_token = keylist + token
        KAT.append(key_and_token)
    emb_token = {} # dict with keys the idx of the word and values the embedding of the word from embedding file.
    for item in KAT:
        emb_token[item[1]] = embedding[item[0]]

    emb_list = [] # list with the embedding of the corresponding word as in vocab list
    cnt=0
    for i in range(len(dataset_vocabulary)):
        if i in emb_token.keys():
            emb_list.append(emb_token[i])
        else:
            randArray = random.random(size=(1, 300)).tolist()
            emb_list.append(randArray[0])
            cnt+=1
    print('In ', name, ' there were ', cnt, 'OOV.')
    # save emb_list
    np.save(output_path, emb_list)

def main():

    config = yaml.load(open('/home/'+username+'/KBGN-Implementation/configs/default.yml'))

    dataset_vocabulary = Vocabulary(
        config["dataset"]["word_counts_json"], 
        min_count=config["dataset"]["vocab_min_count"])

    emb_path_list = [
        (config["dataset"]["glovepath"], DATA_DIR+'glove_visdial.json', 'glove'),
        (config["dataset"]["elmopath"], DATA_DIR+'elmo_visdial.json', 'elmo'),
        (config["dataset"]["numberbatchpath"], DATA_DIR+'numberbatch_visdial.json', 'numberbatch')]
    for (pth, name, is_numb) in tqdm(emb_path_list):
        embeddings2npy(pth, name, dataset_vocabulary, is_numb)
if __name__ == '__main__':
    main()