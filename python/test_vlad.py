# Python API of LCD Toolbox
import loopclosuretoolbox.vlad as vlad
import cv2
import utils

def get_descriptors(img_path):
    image = cv2.imread(img_path)
    # Initialize ORB detector
    orb = cv2.ORB_create()
    # Detect keypoints
    keypoints = orb.detect(image, None)
    # Compute descriptors
    keypoints, descriptors = orb.compute(image, keypoints)

    return keypoints, descriptors

def test():
    db = vlad.Database("./config/test_rgb_8_4.yaml")
    #db = vlad.Database("./config/orbvoc.dbow3")

    query_images = sorted(utils.load_images_from_folder("/home/gvasserm/dev/rtabmap/data/samples"), key=utils.sort_key)
    # add entries to Database
    for image in query_images:
        keypoints, descriptors = get_descriptors(image) # user's implementation
        if descriptors is not None:
            db.add(descriptors)


    pscore = db.compute_pairwise_score()
    # cv2.imshow('pscore', pscore)
    # cv2.waitKey(0)

    for idx, query_image in enumerate(query_images):
        keypoints, qdescriptors = get_descriptors(query_image) # user's implementation
        if qdescriptors is not None:
            results = db.query(qdescriptors, 20) # retrieve best 5 results

        # result is a tuple of (entry_id, score)
        for result in results:
            print(f'image: {idx}, idx: {result[0]}, score: {result[1]}')

if __name__ == '__main__':
    test()
