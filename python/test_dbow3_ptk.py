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

def get_descriptors(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = flip_image(image)
    #plt.imshow(image)
    #plt.show()
    # Initialize ORB detector
    orb = cv2.ORB_create(2000)

    #print(f"ORB descriptor EdgeThreshold={orb.getEdgeThreshold()} MaxFeatures={orb.getMaxFeatures()} FastThreshold={orb.getFastThreshold()} FirstLevel={orb.getFirstLevel()} NLevels={orb.getNLevels()} PatchSize={orb.getPatchSize()} ScaleFactor={orb.getScaleFactor()} ScoreType={orb.getScoreType()} WTA_K={orb.getWTA_K()}")
    # Detect keypoints
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors, image


def select_top_keypoints(keypoints, descriptors):

    indexed_keypoints = list(zip(keypoints, descriptors))
    indexed_keypoints.sort(key=lambda x: x[0].response, reverse=True)

    # Select the top 500 keypoints and their descriptors
    top_keypoints, top_descriptors = zip(*indexed_keypoints[:400])

    return top_keypoints, top_descriptors

def deserializeKeyPointsFromSimpleFormat(filepath):
    keypoints = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 7:
                x, y, size, angle, response, octave = map(float, parts[:6])
                class_id = int(parts[6])
                kp = cv2.KeyPoint(x=x, y=y, size=size, angle=angle, response=response, octave=int(octave), class_id=class_id)
                keypoints.append(kp)
    return keypoints

def load_descriptors(file_path):
    # Create a FileStorage object for reading
    file_storage = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    
    # Read the descriptors
    descriptors = file_storage.getNode("desc").mat()
    
    # Release the file
    file_storage.release()
    
    return descriptors


def draw_keypoints(img, keypoints, size=4, color=(0, 0, 255)):
    """
    
    """
    out_im = copy.deepcopy(img)
    if len(out_im.shape) == 2:
        out_im = cv2.cvtColor(out_im, cv2.COLOR_GRAY2RGB)
    # Draw matches
    for kp in keypoints:
        # Get the matching keypoints in both images
        pt = tuple(np.round(kp.pt).astype(int))
        
        # Draw the keypoints
        cv2.circle(out_im, pt, size, color, -1, cv2.LINE_AA)
    
    return out_im


def test(voc, frameID, plot=False):
    # # create Vocabulary instance from file path
    #voc = dbow.Vocabulary("./config/orbvoc.dbow3")
    #voc = dbow.Vocabulary("./config/mapping_semi_static_ptk_orbD_10_6.yaml")
    #voc = dbow.Vocabulary("./config/mapping_semi_static_ptk_gftt_10_6.yaml")

    # create Vocabulary instance from file path
    db = dbow.Database(voc, False)
    #db = vlad.Database(voc)

    #dir_path = "/home/gvasserm/dev/aicv_amr_ws/results_orb_default_ptk/"
    dir_path = "/home/gvasserm/dev/aicv_amr_ws/results_gftt_default_ptk/"
    #query_images = sorted(utils.load_images_from_folder(dir_path, full_path=False), key=utils.sort_key)
    #query_images = {int(im.split('.')[0].replace('im', '')): im for im in query_images}
    
    sdbow = pd.read_csv(f"{dir_path}/{frameID}.csv").values
    fids = sdbow[:,0].astype(np.int32)

    # add entries to Database
    for fid in fids:
        #imname = query_images[fid]
        #keypoints, descriptors, img = get_descriptors(os.path.join(dir_path, imname)) # user's implementation
        file_pathd = f"{dir_path}/desc{fid}.yml"
        file_pathk = f"{dir_path}/kps{fid}.csv"
        descriptors = load_descriptors(file_pathd)
        #keypoints = deserializeKeyPointsFromSimpleFormat(file_pathk)
        #keypoints, descriptors = select_top_keypoints(keypoints, descriptors)

        #orb = cv2.ORB_create(2000)

        # image = cv2.imread(os.path.join(dir_path, imname))
        # keypoints1, descriptors1= orb.compute(image, keypoints_)
        
        # im1 = draw_keypoints(img, keypoints, size=4, color=(0, 0, 255))
        # im1 = draw_keypoints(im1, keypoints_, size=2, color=(255, 0, 0))

        # plt.imshow(im1)
        # plt.show()

        #keypoints, descriptors = select_top_keypoints(keypoints, descriptors)
        if descriptors is not None:
            db.add(descriptors)
            #bv = voc.transform(descriptors)
            #db.add(bv)

        else:
            descriptors = np.zeros((500, 32))
            #bv = voc.transform(descriptors)
            db.add(descriptors)
            #db.add(bv)
    
    #keypoints, qdescriptors = get_descriptors(os.path.join(dir_path, query_images[frameID])) # user's implementation
    #keypoints, qdescriptors = select_top_keypoints(keypoints, qdescriptors)
    #bv = voc.transform(descriptors)
    file_path = f"{dir_path}/desc{frameID}.yml"
    qdescriptors = load_descriptors(file_path)
    #keypoints = deserializeKeyPointsFromSimpleFormat(file_pathk)
    #keypoints, descriptors = select_top_keypoints(keypoints, descriptors)
    results = db.query(qdescriptors, -1)
    #results = db.query(bv, -1)

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
        #plt.savefig('raw.jpg')
        plt.show()

    return res

def draw_cpp_results():

    frameID = 224
    dir_path = "/home/gvasserm/dev/aicv_amr_ws/results_orb_old_default_ptk/"
    sdbow = pd.read_csv(f"{dir_path}/{frameID}.csv").values
    res = pd.read_csv("/home/gvasserm/dev/loop-closure-toolbox/results.csv").values
    #res = np.sort(res, axis=0)

    sdbow[:,1] = (sdbow[:,1]-np.min(sdbow[:,1]))/(np.max(sdbow[:,1]) - np.min(sdbow[:,1]))
    res[:,1] = (res[:,1]-np.min(res[:,1]))/(np.max(res[:,1]) - np.min(res[:,1]))

    plt.plot(res[:,0], res[:,1], '-ro')
    plt.plot(sdbow[:,0], sdbow[:,1], '-b')
    plt.legend(['dbow', 'default'])
    plt.show()

def process_ptk():

    dir_path = '/home/gvasserm/dev/aicv_amr_ws/results_gftt_default_ptk/'
    loop_closers = pd.read_csv(f"{dir_path}/loop_closure.csv", header=None).values
    frameIDs = [int(l[0].split(' ')[1]) for l in loop_closers]

    res_all = {}

    out_dir = f'{dir_path}/dbow3_dot1/' 
    voc = dbow.Vocabulary("./config/mapping_semi_static_ptk_gftt_dot1_10_6.yaml")
    
    os.makedirs(out_dir, exist_ok=True)
    for frameID in tqdm(frameIDs):
        res = test(voc, frameID)
        df = pd.DataFrame(res)
        df.to_csv(f'{out_dir}{frameID}.csv' , index=False)
        res_all[frameID] = res
    
    return

if __name__ == '__main__':
    #test()
    #draw_cpp_results()
    process_ptk()