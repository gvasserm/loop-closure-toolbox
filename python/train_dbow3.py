# Python API of LCD Toolbox
import loopclosuretoolbox.dbow as dbow
#import loopclosuretoolbox.vlad as vlad
import cv2

import os

def sort_key(filename):
    # Extract the base name (e.g., "1" from "1.jpg")
    basename = os.path.splitext(filename)[0].split('/')[-1]
    # Convert to integer for correct numeric sorting
    return int(basename)

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
    orb = cv2.ORB_create(500)
    # Detect keypoints
    keypoints = orb.detect(image, None)
    # Compute descriptors
    keypoints, descriptors = orb.compute(image, keypoints)

    return keypoints, descriptors

def train():

    # # or create Vocabulary later
    voc = dbow.Vocabulary(10, 5, dbow.WeightingType.TF_IDF, dbow.ScoringType.L1_NORM) # k=10, l=5
    training_features = []
    # voc_created.create(training_features)

    query_images = sorted(load_images_from_folder("/home/gvasserm/dev/rtabmap/data/samples"), key=sort_key)
    # add entries to Database
    for image in query_images:
        keypoints, descriptors = get_descriptors(image) # user's implementation
        if descriptors is not None:
            training_features.append(descriptors)
    
    voc.create(training_features)
    voc.save("./config/test_rgb_10_5.yaml", True)

if __name__ == '__main__':
    train()