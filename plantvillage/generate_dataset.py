import cv2
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool
import tqdm
import random
import errno
import shutil

def make_dir(path):
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

folders = ['Bacterial_spot', 'healthy', 'Early_blight', 'Two-spotted_spider_mite', 'Yellow_Leaf_Curl_Virus']
images = []
background_folder = 'complex_background'
filter_folder = 'filters'

# What I want is :
# Generate images with all images, but only label those that are in "folders"


if __name__ == "__main__":
    