import os
import cv2
import copy
import numpy as np

def sort_key(filename, id_split='im'):
    # Extract the base name (e.g., "1" from "1.jpg")
    basename = os.path.splitext(filename)[0].split('/')[-1]
    # Convert to integer for correct numeric sorting
    return int(basename.split(id_split)[-1])

def load_images_from_folder(folder, full_path=True):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add or remove file extensions as needed
            if full_path:
                img_path = os.path.join(folder, filename)
            else:
                img_path = filename
            images.append(img_path)
    return images


def load_descriptors(file_path):
    # Create a FileStorage object for reading
    file_storage = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    
    # Read the descriptors
    descriptors = file_storage.getNode("desc").mat()
    
    # Release the file
    file_storage.release()
    
    return descriptors

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


def get_descriptors_orb(img_path):
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


def flip_image(image, flip_cams=[0,3]):

    camIDs = [0,1,2,3]
    cams = [[0, 640], [640, 1280], [1280, 1720], [1720, 2560]]

    for cam in camIDs:
        if cam in flip_cams:
            image[:,cams[cam][0]:cams[cam][1]] = cv2.flip(image[:,cams[cam][0]:cams[cam][1]], 0)

    return image