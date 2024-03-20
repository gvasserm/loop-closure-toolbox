# Python API of LCD Toolbox
import loopclosuretoolbox.dbow as dbow
#import loopclosuretoolbox.vlad as vlad
import cv2
from tqdm import tqdm

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
    image = cv2.imread(img_path)
    # Initialize ORB detector
    orb = cv2.ORB_create(2000)
    # Detect keypoints
    keypoints = orb.detect(image, None)
    # Compute descriptors
    keypoints, descriptors = orb.compute(image, keypoints)

    return keypoints, descriptors

def train():

    # # or create Vocabulary later
    #k = 10; l=5
    k = 6
    l = 6
    voc = dbow.Vocabulary(k, l, dbow.WeightingType.TF_IDF, dbow.ScoringType.L1_NORM) # k=10, l=5
    training_features = []
    # voc_created.create(training_features)

    query_images = sorted(load_images_from_folder("/home/gvasserm/dev/aicv_amr_ws/results/"), key=sort_key)
    # add entries to Database
    for image in tqdm(query_images):
        keypoints, descriptors = get_descriptors(image) # user's implementation
        if descriptors is not None:
            training_features.append(descriptors)
    
    voc.create(training_features)
    voc.save(f"./config/mapping_{k}_{l}.yaml", True)

if __name__ == '__main__':
    train()