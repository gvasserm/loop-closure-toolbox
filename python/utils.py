import os

def sort_key(filename):
    # Extract the base name (e.g., "1" from "1.jpg")
    basename = os.path.splitext(filename)[0].split('/')[-1]
    # Convert to integer for correct numeric sorting
    return int(basename.split('_')[-1])

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