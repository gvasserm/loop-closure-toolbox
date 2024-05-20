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

from test_dbow3 import run_on_data

def train_on_descriptors(descriptor_files_all, voc_name, k=10, l=6):

    training_features = []

    for f in tqdm(descriptor_files_all):
        descriptors = utils.load_descriptors(f)
        if descriptors is not None:
                training_features.append(descriptors)

    
    scoring = dbow.ScoringType.L1_NORM
    voc = dbow.Vocabulary(k, l, dbow.WeightingType.TF_IDF, scoring)

    voc.create(training_features)
    voc.save(f"./config/{voc_name}_{k}_{l}.yaml", True)

    return


def setup_train_dataset():
     
    database_folder1 = "/home/gvasserm/dev/aicv_amr_ws/results_gftt_default_ptk/"
    database_folder2 = "/home/gvasserm/dev/aicv_amr_ws/results_gftt_dbow_ptk2/"

    descriptor_files1 = data_utils.find_images(database_folder1,["*.yml"])
    descriptor_files2 = data_utils.find_images(database_folder2,["*.yml"])

    descriptor_files_all = descriptor_files1 + descriptor_files2

    return descriptor_files_all

def setup_test_dataset():
    return data_utils.find_images("data",["*.yml"])

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

def test_all():
    descriptor_files_all = setup_test_dataset()

    voc_name = "test_gftt"
    k=10
    l=5
    
    train_on_descriptors(descriptor_files_all, voc_name, k=k, l=l)

    sdbow = pd.read_csv(f"data/225.csv").values
    fids = sdbow[:,0].astype(np.int32)

    fpath_queries = data_utils.find_images("data",["*.yml"])
    d = {int(f.split('/')[-1].split("desc")[-1].split('.')[0]): f for f in fpath_queries}
    fpath_queries = [d[i] for i in fids]

    voc = dbow.Vocabulary(f"./config/{voc_name}_{k}_{l}.yaml")
    fpath_key = f"data/desc225.yml"
    run_on_data(voc, fpath_key, fpath_queries, plot=True)


if __name__ == '__main__':
    #descriptor_files_all = setup_train_dataset()
    test_all()