import numpy as np
import os

if False:
    directory = '/home/gvasserm/dev/rtabmap/data/samples/'
    padding = 3  # Number of digits in the renamed files

    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add other image extensions if needed
            # Extract the base name and extension
            basename, extension = os.path.splitext(filename)
            
            # Check if basename is numeric and pad it
            if basename.isnumeric():
                new_filename = basename.zfill(padding) + extension
                
                # Construct the full old and new file paths
                old_filepath = os.path.join(directory, filename)
                new_filepath = os.path.join(directory, new_filename)
                
                # Rename the file
                os.rename(old_filepath, new_filepath)
                print(f'Renamed "{filename}" to "{new_filename}"')



dataset_directory = "/home/gvasserm/dev/VPR-Bench/datasets/corridor/"
ground_truth_info=np.load(dataset_directory+'ground_truth_new.npy',allow_pickle=True)
ground_truth_info

