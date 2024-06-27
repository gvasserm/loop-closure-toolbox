# Python API of LCD Toolbox
import loopclosuretoolbox.dbow as dbow

import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from sklearn.metrics import precision_score, recall_score, accuracy_score

def query_frame(voc, dir_path, kID, plot=False):
    db = dbow.Database(voc, False)

    sdbow = pd.read_csv(f"{dir_path}/{kID}.csv").values
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
    
    file_path = f"{dir_path}/desc{kID}.yml"
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

def benchmark(voc_path, dir_path):
    voc = dbow.Vocabulary(voc_path)

    lcdata = pd.read_csv(f"{dir_path}/loop_closure.csv").values
    lcdata = np.asarray([[int(l[0].split(' ')[0]), int(l[0].split(' ')[1])] for l in lcdata])
    keyID_ = lcdata[:,1].astype(np.int32)
    queryID_ = lcdata[:,0].astype(np.int32)

    # add entries to Database
    predicted_indices = []
    ground_truth_indices = []
    for k, q in tqdm(zip(keyID_, queryID_)):
        res = query_frame(voc, dir_path, k, plot=False)
        max_index = np.argmax(res[:, 1])
        qpred = res[max_index,0]
        predicted_indices.append(int(qpred))
        ground_truth_indices.append(q)

    # Calculate precision, recall, and accuracy
    precision = precision_score(ground_truth_indices, predicted_indices, average='macro')
    recall = recall_score(ground_truth_indices, predicted_indices, average='macro')
    accuracy = accuracy_score(ground_truth_indices, predicted_indices)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")

    return res

if __name__ == '__main__':

    voc_path = "config/mapping_ptk_lc4_gftt_10_6.yaml"
    dir_path = "/home/gvasserm/data/AMRLoopClosureData/results_ptk4map/"
    benchmark(voc_path, dir_path)