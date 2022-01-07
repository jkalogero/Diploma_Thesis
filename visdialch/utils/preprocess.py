import os
import argparse
from multiprocessing import cpu_count
import getpass

username = getpass.getuser()
paths = {
    'train_jsonl': '/home/'+username+'/KBGN-Implementation/data/train.jsonl', 
    'val_jsonl': '/home/'+username+'/KBGN-Implementation/data/val.jsonl',
    'test_jsonl': '/home/'+username+'/KBGN-Implementation/data/test.jsonl'
}
print(paths['train_jsonl'])
print(paths['val_jsonl'])
print(paths['test_jsonl'])

for file in paths:
    if not os.path.isfile(paths[file]):
        pass