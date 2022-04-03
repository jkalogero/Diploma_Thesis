import h5py


def link_file(filename, target_file):
    keys = []
    # get list of keys for current file
    with h5py.File(filename) as f:
        keys = list(f.keys())
    print(keys)

    h = h5py.File(target_file, 'a')
    for key in keys:

        h[key] = h5py.ExternalLink(filename,key)
    
    h.close()



if __name__ == '__main__':

    all_files = ['file'+str(i)+'.h5' for i in range(1,10)] #change

    for _f in all_files:
        link_file(_f)

    # check if all keys present
    with h5py.File('all.h5') as f:
        all_keys = list(f.keys())
    print(len(all_keys))