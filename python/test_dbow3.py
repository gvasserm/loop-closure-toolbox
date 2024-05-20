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

def flip_image(image, flip_cams=[0,3]):

    camIDs = [0,1,2,3]
    cams = [[0, 640], [640, 1280], [1280, 1720], [1720, 2560]]

    for cam in camIDs:
        if cam in flip_cams:
            image[:,cams[cam][0]:cams[cam][1]] = cv2.flip(image[:,cams[cam][0]:cams[cam][1]], 0)

    return image

def test(voc, frameID, plot=False):
    db = dbow.Database(voc, False)
    dir_path = "/home/gvasserm/dev/aicv_amr_ws/results_gftt_default_ptk/"
   
    sdbow = pd.read_csv(f"{dir_path}/{frameID}.csv").values
    fids = sdbow[:,0].astype(np.int32)

    # add entries to Database
    for fid in fids:
        file_pathd = f"{dir_path}/desc{fid}.yml"
        descriptors = utils.load_descriptors(file_pathd)

        if descriptors is not None:
            db.add(descriptors)
        else:
            descriptors = np.zeros((2000, 32))
            db.add(descriptors)
    
    file_path = f"{dir_path}/desc{frameID}.yml"
    qdescriptors = utils.load_descriptors(file_path)
    
    results = db.query(qdescriptors, -1)
    res = {r[0]: r[1] for r in results}
    res1 = []
    for i, fid in enumerate(fids):
        if i in res:
            res1.append([fid, res[i]])
        else:
            res1.append([fid, 0])

    res = np.asarray(res1)

    sdbow[:,1] = (sdbow[:,1]-np.min(sdbow[:,1]))/(np.max(sdbow[:,1]) - np.min(sdbow[:,1]))
    res[:,1] = (res[:,1]-np.min(res[:,1]))/(np.max(res[:,1]) - np.min(res[:,1]))
    if plot:
        plt.plot(res[:,0], res[:,1], '-ro')
        plt.plot(sdbow[:,0], sdbow[:,1], '-b')
        plt.legend(['dbow', 'default'])
        plt.show()
    return res

def process_ptk():

    dir_path = '/home/gvasserm/dev/aicv_amr_ws/results_gftt_default_ptk/'
    loop_closers = pd.read_csv(f"{dir_path}/loop_closure.csv", header=None).values
    frameIDs = [int(l[0].split(' ')[1]) for l in loop_closers]
    
    res_all = {}

    out_dir = f'{dir_path}/dbow3_dot1/' 
    voc = dbow.Vocabulary("./config/mapping_semi_static_ptk_gftt_10_6.yaml")
    
    os.makedirs(out_dir, exist_ok=True)
    for frameID in tqdm(frameIDs):
        res = test(voc, frameID)
        df = pd.DataFrame(res)
        df.to_csv(f'{out_dir}{frameID}.csv' , index=False)
        res_all[frameID] = res
    
    return

if __name__ == '__main__':
    test()
    #process_ptk()