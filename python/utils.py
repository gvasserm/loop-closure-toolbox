import os
import cv2

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