# Python API of LCD Toolbox
import loopclosuretoolbox.dbow as dbow
import loopclosuretoolbox.vlad as vlad

import cv2
import utils
import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import shutil

def run_on_data(voc, fpath_key, fpath_queries, plot=False):
    db = dbow.Database(voc, False)

    fids = [i for i in range(len(fpath_queries))]

    # add entries to Database
    for file_pathq in fpath_queries:
        descriptors = utils.load_descriptors(file_pathq)

        if descriptors is not None:
            db.add(descriptors)
        else:
            descriptors = np.zeros((2000, 32))
            db.add(descriptors)
    
    qdescriptors = utils.load_descriptors(fpath_key)
    
    results = db.query(qdescriptors, -1)
    res = {r[0]: r[1] for r in results}
    res1 = []
    for i, fid in enumerate(fids):
        if i in res:
            res1.append([fid, res[i]])
        else:
            res1.append([fid, 0])

    res = np.asarray(res1)

    if plot:
        plt.plot(res[:,0], res[:,1], '-ro')
        plt.legend(['dbow'])
        plt.show()
    return res

def copy_files(source_files, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for source_path in tqdm(source_files):
        target_path = os.path.join(target_dir, source_path.split('/')[-1])

        if os.path.isdir(source_path):
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


def create_test():
    frameID = 225

    dir_path = "/home/gvasserm/dev/aicv_amr_ws/results_gftt_default_ptk/"
    sdbow = pd.read_csv(f"data/{frameID}.csv").values
    fids = sdbow[:,0].astype(np.int32)

    file_paths = []
    for fid in fids:
        file_paths.append(f"{dir_path}/desc{fid}.yml")
    
    copy_files(file_paths, "data")
    return



if __name__ == '__main__':
    create_test()
    #process_ptk()