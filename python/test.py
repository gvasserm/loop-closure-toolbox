# Python API of LCD Toolbox
import loopclosuretoolbox.dbow as dbow
import loopclosuretoolbox.vlad as vlad
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
    orb = cv2.ORB_create()
    # Detect keypoints
    keypoints = orb.detect(image, None)
    # Compute descriptors
    keypoints, descriptors = orb.compute(image, keypoints)

    return keypoints, descriptors

# # create Vocabulary instance from file path
voc = dbow.Vocabulary("./config/sthereo_07_rgb_4_3.yaml")

# # or create Vocabulary later
# voc_created = dbow.Vocabulary(4,3) # k=4, l=3
# training_features = ... # list of local descriptors to be used for train
# voc_created.create(training_features)

# create Vocabulary instance from file path
db = dbow.Database(voc, False, 0)

query_images = sorted(load_images_from_folder("/home/gvasserm/dev/rtabmap/data/samples"), key=sort_key)
# add entries to Database
for image in query_images:
    keypoints, descriptors = get_descriptors(image) # user's implementation
    if descriptors is not None:
        db.add(descriptors)


for idx, query_image in enumerate(query_images):
    keypoints, qdescriptors = get_descriptors(query_image) # user's implementation
    if qdescriptors is not None:
        results = db.query(qdescriptors, 10) # retrieve best 5 results

    # result is a tuple of (entry_id, score)
    for result in results:
        print(f'image: {idx}, idx: {result[0]}, score: {result[1]}')




