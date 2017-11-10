#%% Initialization and function definition


import os
from PIL import Image
import numpy as np
import glob  # utility to iterate on the system paths
import csv  # read the ground truth file [label - value]


def list_files(parent, ext='', rec=False):
    regex = '/**/*' if rec else '/*'
    fnames = [f for f in glob.iglob(parent + regex + ext, recursive=rec)]
    return fnames


def list_dirs(parent):
    # extract folder information
    dirinfo = os.listdir(parent)
    # create complete file names
    dnames = [os.path.join(parent, x) for x in dirinfo]
    # extract only directories
    dnames = [x for x in dnames if os.path.isdir(x)]
    return dnames


def load_depth_images(folder, verbose=True):

    class_paths = list_dirs(folder)  # complete subfolder paths
    nb_classes = len(class_paths)

    nb_samples = len(list_files(folder, ext='', rec=True)) 
    cols = 224
    rows = 224
    target_shape = (cols, rows)  # PIL format

    data = {}
    # each image must be a RGB-8bit image
    imshape = target_shape + (3,)

    images = np.zeros((nb_samples,) + imshape, dtype=np.uint8)
    values = np.zeros((nb_samples,), dtype=np.int32)
    labels = []


    #%% Load images


    print('Loading images...')
    count = 0
    for cid, cpath in enumerate(class_paths):  # iterate on the classes
        class_name = os.path.basename(cpath)
        labels.append(class_name)
        fpaths = list_files(cpath, ext='')
        if verbose:
            print('>', class_name)
        for fid, fpath in enumerate(fpaths):
            try:
                # load and resize the image
                image = Image.open(fpath)
                if image.size != target_shape:
                    image = image.resize(target_shape, Image.NEAREST)
                image = np.array(image)
                images[count] = image.astype(np.uint8)
                values[count] = cid
                count += 1
            except Exception as e:
                print('Skipping', fpath, 'because', e)

    if count > 0:
        images = images[:count]
        values = values[:count]
    nb_samples = count

    return images, values, labels


def save_dataset(filename, data):

    print('Saving', filename)

    variables = {}
    variables['images'] = data[0]
    variables['values'] = data[1]
    variables['labels'] = data[2]

    import scipy.io as sio
    sio.savemat(filename, variables, do_compression=True)


#%%


if __name__ == '__main__':

    folders = ['train', 'test']

    dest_folder = './bin_data/'
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for folder in folders:
        print('Loading', folder, 'set')
        data = load_depth_images('raw_data/' + folder, verbose=False)
        save_dataset(dest_folder + folder + '.mat', data)
