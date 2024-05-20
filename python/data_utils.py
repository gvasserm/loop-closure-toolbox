import os
from glob import glob
import re
from pathlib import Path

def extract_number(filename):
    # Extract the numerical part of the filename
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # In case there's no number in the filename

def find_images(directory, extensions=['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']):
    # Define image file extensions
    images = []

    # Walk through directory
    for root, dirs, files in os.walk(directory):
        for extension in extensions:
            # Use glob to find files matching the extension
            images.extend(glob(os.path.join(root, extension)))
    
    #images = sorted(images, key=extract_number)
    images = sorted(images)
    return images

def read_images_paths(dataset_folder):
    """Find images within 'dataset_folder'. If the file
    'dataset_folder'_images_paths.txt exists, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over large folders might be slow.
    
    Parameters
    ----------
    dataset_folder : str, folder containing JPEG images
    
    Returns
    -------
    images_paths : list[str], paths of JPEG images within dataset_folder
    """
    
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")
    
    file_with_paths = Path(dataset_folder) / "images_paths.txt"
    if os.path.exists(file_with_paths):
        print(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [dataset_folder + "/" + path for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(f"Image with path {images_paths[0]} "
                                    f"does not exist within {dataset_folder}. It is likely "
                                    f"that the content of {file_with_paths} is wrong.")
    else:
        print(f"Searching test images in {dataset_folder} with glob()")
        #images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        images_paths = find_images(dataset_folder)
        if len(images_paths) == 0:
            raise FileNotFoundError(f"Directory {dataset_folder} does not contain any JPEG images")
    return images_paths