import json
import numpy as np
import getpass

username = getpass.getuser()
DATA_DIR = '/home/'+username+'/KBGN-Implementation/data/'
data_path = DATA_DIR + 'visdial_1.0_train.json'

with open(data_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

dialogs = data['data']['dialogs']
key_list = [d['image_id'] for d in dialogs]
n_notebooks = 18
colab_size = 6000
vm_size = len(key_list) - n_notebooks*colab_size
print(colab_size)
print(vm_size)
print(len(key_list))
print(colab_size)


subsets = {k+2:set(key_list[vm_size+k*colab_size:vm_size+(k+1)*colab_size]) for k in np.arange(n_notebooks)}
subsets[1] = key_list[:vm_size]
# for k in subsets:
#     print(k,'\t',len(subsets[k]))

suffix = 'subset_train_part_'
postfix = '.json'

for k in subsets:
    subset = data.copy()
    new_dialogs = [dialog for dialog in dialogs if dialog['image_id'] in subsets[k]]
    subset['data']['dialogs'] = new_dialogs
    with open(suffix+str(k)+postfix,'w+') as fout:
        fout.write(json.dumps(subset))
    print("Saved subset to ", suffix+str(k)+postfix)