# Python API of LCD Toolbox
import loopclosuretoolbox.dbow as dbow
import cv2
import utils
import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

def get_descriptors(img_path):
    image = cv2.imread(img_path)
    # Initialize ORB detector
    orb = cv2.ORB_create()
    # Detect keypoints
    keypoints = orb.detect(image, None)
    # Compute descriptors
    keypoints, descriptors = orb.compute(image, keypoints)

    return keypoints, descriptors

def test_likelihood2():

    orig = pd.read_csv("/home/gvasserm/dev/rtabmap/results/orig_41.csv").values
    new = pd.read_csv("/home/gvasserm/dev/rtabmap/results/new_41.csv").values

    plt.plot(orig[:,0], orig[:,1], 'b', label='orig')
    plt.plot(new[:,0], new[:,1], 'r',  label='dbow3')

    plt.xlabel('frame')
    plt.ylabel('similarity')
    plt.legend()

    plt.show()

    return


def test_likelihood():

    l1 = [
    0,
    0.00221442897,
    0.00631062873,
    0.0528649986,
    0.111102439,
    0.0157463159,
    0.0294886846,
    0.0172728747,
    0.0037086932,
    0.00109040202,
    0.0139884157,
    0,
    0.00676934142,
    0.0180716105,
    0.0206441693,
    0.0222150274,
    0.0282951407,
    0.00891886093,
    0.0112095382,
    0.0243066512,
    0.00190337386,
    0.00915507507,
    0.00751958042,
    0.00511058606,
    0.00891865231,
    0.0011555755]

    l2= [
    0,
    0.0163583681,
    0.0295074414,
    0.0679838881,
    0.0732325092,
    0.0223484002,
    0.0345624201,
    0.0446537174,
    0.0437642559,
    0.048365742,
    0.0486412533,
    0,
    0.0367015526,
    0.0596003532,
    0.024829343,
    0.0572041757,
    0.0509382337,
    0.0382721908,
    0.0242173392,
    0.0687983558,
    0.0406152382,
    0.0593217611,
    0.0343320966,
    0.0308367275,
    0.0471370742,
    0.0442986712]


    plt.figure()
    plt.plot(l1, 'r')
    plt.plot(l2, 'b')
    plt.show()

    return

def benchmark():
    from performance_comparison import performance_comparison
    dataset_name = 'sample'
    dataset_directory = ''
    vpr_techniques = ['DBoW3']
    query_all = {'DBoW3': []}
    ground_truth_info = {'DBoW3': []}
    retrieved_all = {'DBoW3': []}
    scores_all = {'DBoW3': []}
    encoding_time_all = {'DBoW3': 1.0}
    matching_time_all = {'DBoW3': 1.0}
    all_retrievedindices_scores_allqueries_dict = {'DBoW3': []}
    descriptor_shape_dict = {'DBoW3': [(512), 'float32']}

    pred_ = json.load(open("./results/pred.json", "r"))
    gt_ = json.load(open("/home/gvasserm/dev/rtabmap/data/gt.json", "r"))

    gt = {}
    pred = {}
    for k in gt_.keys():
        if k in pred_:
            pred[k] = pred_[k]
            gt[k] = gt_[k]

    retrieved_all = {'DBoW3': {k: pred[k][0] for k in pred.keys()}}
    query_all =  {'DBoW3': [{k for k in pred.keys()}]}
    scores_all =  {'DBoW3': {k: pred[k][1] for k in pred.keys()}}

    all_retrievedindices_scores_allqueries_dict = scores_all

    ground_truth_info = gt

    performance_comparison(dataset_name, 
                           dataset_directory, 
                           vpr_techniques, 
                           ground_truth_info,
                           query_all, 
                           retrieved_all, 
                           scores_all, 
                           encoding_time_all, 
                           matching_time_all, 
                           all_retrievedindices_scores_allqueries_dict,
                           descriptor_shape_dict)

    return

def test():
    # # create Vocabulary instance from file path
    #voc = dbow.Vocabulary("./config/test_rgb_8_4.yaml")
    #voc = dbow.Vocabulary("./config/config/orb_slam_10_5.yaml")
    
    voc = dbow.Vocabulary(10, 5)
    voc.load("./config/orbvoc.dbow3")

    # create Vocabulary instance from file path
    db = dbow.Database(voc, False, 0)

    dir_path = "/home/gvasserm/dev/rtabmap/data/samples"
    query_images = sorted(utils.load_images_from_folder(dir_path, full_path=False), key=utils.sort_key)
    dictq = {i: q for i, q in enumerate(query_images)}
    dictqi = {value: key for key, value in dictq.items()}
    # add entries to Database
    for image in query_images[:40]:
        keypoints, descriptors = get_descriptors(os.path.join(dir_path, image)) # user's implementation
        if descriptors is not None:
            db.add(descriptors)
        else:
            descriptors = np.zeros((500, 32)).astype(np.uint8)
            db.add(descriptors)

    pscore = db.compute_pairwise_score()
    # cv2.imshow('pscore', pscore)
    # cv2.waitKey(0)

    res = {}
    for idx, query_image in enumerate([query_images[41]]):
        keypoints, qdescriptors = get_descriptors(os.path.join(dir_path, query_image)) # user's implementation
        
        results = []
        score = []
        
        if qdescriptors is not None:
            results = db.query(qdescriptors, -1) # retrieve best 5 results
            score = [r[1] for r in results if dictq[r[0]]!=query_image]
            result = [dictq[r[0]] for r in results if dictq[r[0]]!=query_image]
            res[query_image] = (result, score)

        # result is a tuple of (entry_id, score)
        #for result in results:
        #    print(f'image: {idx}, idx: {result[0]}, score: {result[1]}')
            
        ind = {i: dictqi[r] for i, r in enumerate(result)}

        import matplotlib.pyplot as plt
        for key, value in ind.items():
            plt.plot(value, score[key], 'b.')
        plt.show()

    json.dump(res, open("/home/gvasserm/dev/loop-closure-toolbox/results/pred.json", 'w'))

    return

if __name__ == '__main__':
    #test()
    #benchmark()
    #test_likelihood()
    test_likelihood2()
