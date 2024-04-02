# Python API of LCD Toolbox
import loopclosuretoolbox.dbow as dbow
#import loopclosuretoolbox.vlad as vlad
import cv2
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import pickle

import os

def sort_key(filename):
    # Extract the base name (e.g., "1" from "1.jpg")
    basename = os.path.splitext(filename)[0].split('/')[-1]
    # Convert to integer for correct numeric sorting
    return int(basename.split('_')[-1])

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add or remove file extensions as needed
            img_path = os.path.join(folder, filename)
            images.append(img_path)
    return images

def get_descriptors(img_path):
    if type(img_path) == str:
        image = cv2.imread(img_path)
    else:
        image = img_path
    # Initialize ORB detector
    orb = cv2.ORB_create(2000)
    # Detect keypoints
    keypoints = orb.detect(image, None)
    # Compute descriptors
    keypoints, descriptors = orb.compute(image, keypoints)

    return keypoints, descriptors

def train():

    # # or create Vocabulary later
    k = 10; l=5
    #k = 6; l = 6
    voc = dbow.Vocabulary(k, l, dbow.WeightingType.TF_IDF, dbow.ScoringType.L1_NORM) # k=10, l=5
    training_features = []
    # voc_created.create(training_features)

    query_images = sorted(load_images_from_folder("/home/gvasserm/dev/aicv_amr_ws/results_img/"), key=sort_key)
    # add entries to Database
    for image in tqdm(query_images):
        keypoints, descriptors = get_descriptors(image) # user's implementation
        if descriptors is not None:
            training_features.append(descriptors)
    
    voc.create(training_features)
    voc.save(f"./config/mapping_{k}_{l}.yaml", True)

def train_4movs():

    # # or create Vocabulary later
    k = 10; l=5
    #k = 6; l = 6
    voc = dbow.Vocabulary(k, l, dbow.WeightingType.TF_IDF, dbow.ScoringType.L1_NORM) # k=10, l=5
    training_features = []
    # voc_created.create(training_features)

    cap0 = cv2.VideoCapture("out0.mp4")
    cap1 = cv2.VideoCapture("out1.mp4")
    cap2 = cv2.VideoCapture("out2.mp4")
    cap3 = cv2.VideoCapture("out3.mp4")

    num_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))

    # add entries to Database
    for frame_idx in tqdm(range(num_frames)):
    #while cap0.isOpened():
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        if not ret0 or not ret1 or not ret2 or not ret3:
            break

        if frame_idx % 30 == 0:
            frame = np.hstack((frame0, frame1, frame2, frame3))

            cv2.imshow("lkjl", frame)
            cv2.waitKey()

            keypoints, descriptors = get_descriptors(frame) # user's implementation
            if descriptors is not None:
                training_features.append(descriptors)

    cap0.release()
    cap1.release()
    cap2.release()
    cap3.release()
    print('\nfinished')

    with open('training_features.pkl', 'wb') as file: 
        # A new file will be created 
        pickle.dump(training_features, file) 
    
    voc.create(training_features)
    voc.save(f"./config/mapping_syn_{k}_{l}.yaml", True)
    print('\nfinished creating voc')

def train_from_pickle():

    k = 10; l=5
    #k = 6; l = 6
    voc = dbow.Vocabulary(k, l, dbow.WeightingType.TF_IDF, dbow.ScoringType.L1_NORM) # k=10, l=5

    with open('training_features.pkl', 'rb') as file: 
        # A new file will be created 
        training_features = pickle.load(file) 

    training_features = training_features[0:100]
    
    voc.create(training_features)
    voc.save(f"./config/mapping_syn_{k}_{l}.yaml", True)
    print('\nfinished creating voc')


if __name__ == '__main__':
    #train()
    train_4movs()
    #train_from_pickle()