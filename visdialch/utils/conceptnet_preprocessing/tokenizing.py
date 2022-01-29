import json
import spacy
from multiprocessing import Pool
from tqdm import tqdm

def tokenize_sentence_spacy(nlp, sent):
    tokens = [tok.text.lower() for tok in nlp(sent)]
    return tokens

def _tokenizeDatasetFile(datalist):

    dialog, answers, questions,nlp = datalist

    

    history = [tokenize_sentence_spacy(nlp, dialog['caption']) + tokenize_sentence_spacy(nlp, questions[dialog['dialog'][0]['question']])] \
        + [tokenize_sentence_spacy(nlp, questions[dialog['dialog'][idx+1]['question']]) \
        + (tokenize_sentence_spacy(nlp, answers[_round['answer']]) if 'answer' in _round.keys() else []) \
        for idx,_round in enumerate(dialog['dialog'][:-1])]

    
    return (dialog['image_id'], history)

def tokenizeDatasetFile(dialog_path, output_path, concat=False, debug=False):
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

    sample = 5

    print(f'Tokenizing {dialog_path} file.')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

    
    with open(dialog_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
        
        dialogs = data['data']['dialogs']
        if debug:
            dialogs = dialogs[:sample]
        answers = data['data']['answers']
        questions = data['data']['questions']
    
    data_list = [(d,answers,questions,nlp) for d in dialogs]

    with Pool() as p:
        res = {k:v for (k,v) in tqdm(p.imap(_tokenizeDatasetFile,data_list, 80), total=len(dialogs), desc='Tokenizing...')}
        
    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(res))

