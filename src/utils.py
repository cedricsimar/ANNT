
from settings import Settings
import numpy as np
import shutil
from distutils.dir_util import copy_tree
import os

def reshape_mnist_images(mnist_images):

    new_images = []
    
    for i in range(len(mnist_images)):
        new_images.append(np.reshape(mnist_images[i], (Settings.INPUT_SHAPE[0], Settings.INPUT_SHAPE[1])))
    
    return (np.array(new_images))


def clean_folder(folder_path):

    try:
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
    except Exception as e:
        print(e)
    

def copy_files_from_to(src, dest):
    copy_tree(src, dest, verbose=0)

