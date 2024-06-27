import loopclosuretoolbox.dbow as dbow
import cv2
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import pickle
import data_utils
import utils
import json
import pandas as pd

from matplotlib import pyplot as plt

import os

def train_on_descriptors():

    database_folder1 = "/home/gvasserm/dev/aicv_amr_ws/results_lc4large_map_def/"
    database_folder2 = "/home/gvasserm/dev/aicv_amr_ws/results_ptk4map/"

    descriptor_files1 = data_utils.find_images(database_folder1,["*.yml"])
    descriptor_files2 = data_utils.find_images(database_folder2,["*.yml"])

    descriptor_files_all = descriptor_files1 + descriptor_files2

    training_features = []

    for f in tqdm(descriptor_files_all):
        descriptors = utils.load_descriptors(f)
        if descriptors is not None:
                training_features.append(descriptors)

    
    k = 10; l=6
    scoring = dbow.ScoringType.L1_NORM
    voc = dbow.Vocabulary(k, l, dbow.WeightingType.TF_IDF, scoring)

    voc.create(training_features)
    voc.save(f"./config/mapping_ptk_lc4_gftt_{k}_{l}.yaml", True)

    return


def train_on_images():

    training_features = []

    database_folder = "/home/gvasserm/dev/aicv_amr_ws/results_img/"
    image_files = sorted(data_utils.find_images(database_folder), key=utils.sort_key)

    # add entries to Database
    for image in tqdm(image_files):
        keypoints, descriptors = utils.get_descriptors_orb(image) # user's implementation
        if descriptors is not None:
            training_features.append(descriptors)

    
    k = 10; l = 6
    scoring = dbow.ScoringType.L1_NORM
    voc = dbow.Vocabulary(k, l, dbow.WeightingType.TF_IDF, scoring)
    voc.create(training_features)
    voc.save(f"./config/mapping_semi_static_ptk_orb_{k}_{l}.yaml", True)

    return


if __name__ == '__main__':
    train_on_descriptors()

    