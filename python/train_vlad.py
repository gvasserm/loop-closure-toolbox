# Python API of LCD Toolbox
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
import vlad

import os

def train_on_descriptors():
    database_folder1 = "/home/gvasserm/dev/aicv_amr_ws/results_gftt_default_ptk/"
    database_folder2 = "/home/gvasserm/dev/aicv_amr_ws/results_gftt_dbow_ptk2/"

    descriptor_files1 = data_utils.find_images(database_folder1,["*.yml"])
    descriptor_files2 = data_utils.find_images(database_folder2,["*.yml"])

    descriptor_files_all = descriptor_files1 + descriptor_files2

    training_features = []

    for f in tqdm(descriptor_files_all):
        descriptors = utils.load_descriptors(f)
        if descriptors is not None:
                training_features.append(descriptors)


    vlad_d = vlad.VLAD(k=256, n_vocabs=1, norming="RN", lcs=True).fit(training_features)

    f = vlad_d._extract_vlads(training_features)

    return


if __name__ == '__main__':
    train_on_descriptors()