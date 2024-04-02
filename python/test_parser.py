import yaml
import numpy as np

def load_keypoints_from_yaml(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    keypoints = data['keypointsA']

    # Convert loaded keypoints to a list of cv2.KeyPoint objects
    cv_keypoints = []
    for kp in keypoints:
        # Assuming the YAML file format matches the one described previously
        point = kp['pt']
        size = kp['size']
        angle = kp['angle']
        response = kp['response']
        octave = kp['octave']
        class_id = kp['class_id']

        # Create a KeyPoint-like structure, adjust as necessary for your usage
        # OpenCV's KeyPoint class is not directly available in Python, but you can emulate it or directly use it if needed
        cv_kp = {
            'pt': np.array(point),
            'size': size,
            'angle': angle,
            'response': response,
            'octave': octave,
            'class_id': class_id
        }
        cv_keypoints.append(cv_kp)

    return cv_keypoints

# Example usage
filename = "/home/gvasserm/dev/rtabmap/out.yaml"
keypoints = load_keypoints_from_yaml(filename)
for kp in keypoints:
    print(kp)