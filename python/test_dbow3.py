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
    orb = cv2.ORB_create(2000)
    # Detect keypoints
    keypoints = orb.detect(image, None)
    # Compute descriptors
    keypoints, descriptors = orb.compute(image, keypoints)

    return keypoints, descriptors

def test_likelihood():

    ind = 50
    data_orig = json.load(open("/home/gvasserm/dev/rtabmap/results/orig_maps_" + str(ind) +".json", "r"))
    data_new = json.load(open("/home/gvasserm/dev/rtabmap/results/new_maps_"+ str(ind) +".json", "r"))

    data_orig = {d[0]: {d_[0]: d_[1] for d_ in d[1]}  for d in data_orig}
    data_new = {d[0]: {d_[0]: d_[1] for d_ in d[1]}  for d in data_new}

    data_new_ = {}
    q = data_new[ind]
    for k, d in data_new.items():
        if k!=ind:
            data_new_[k] = {}
            for ki in d.keys():
                if ki in q:
                  data_new_[k][ki] = data_new[k][ki]


    origc = {k: len(d) for k, d in data_orig.items()}
    newc = {k: len(d) for k, d in data_new_.items()}

    x, y  = np.asarray(list(origc.keys())), np.asarray(list(origc.values()))
    #y = y/np.max(y)
    plt.plot(x, y, '-b*', label='orig')
    x, y  = np.asarray(list(newc.keys())), np.asarray(list(newc.values()))
    #y = y/np.max(y)
    plt.plot(x, y, '-ro',  label='dbow3')

    plt.xlabel('frame')
    plt.ylabel('similarity')
    plt.legend()

    plt.show()
    
    new = pd.read_csv("/home/gvasserm/dev/rtabmap/results/orig_41.csv").values
    orig = pd.read_csv("/home/gvasserm/dev/rtabmap/results/new_41.csv").values

    new[:,1] = new[:,1]/np.max(new[:,1])
    orig[:,1] = orig[:,1]/np.max(orig[:,1])
    
    plt.plot(orig[:,0], orig[:,1], '-b*', label='orig')
    plt.plot(new[:,0], new[:,1], '-ro',  label='dbow3')

    plt.xlabel('frame')
    plt.ylabel('similarity')
    plt.legend()

    plt.show()

    return

def test_like():

    ind = 181

    new = pd.read_csv(f"/home/gvasserm/dev/aicv_amr_ws/results/new_{ind}.csv").values
    orig = pd.read_csv(f"/home/gvasserm/dev/aicv_amr_ws/results/orig_{ind}.csv").values

    new[:,1] = new[:,1]/np.max(new[:,1])
    orig[:,1] = orig[:,1]/np.max(orig[:,1])
    
    # plt.plot(orig[:,0], orig[:,1], '-b*', label='orig')
    # plt.plot(new[:,0], new[:,1], '-ro',  label='dbow3')

    plt.plot(orig[:,0], orig[:,1], '-b*', label='dbow3')
    plt.plot(new[:,0], new[:,1], '-ro',  label='orig')

    plt.xlabel('frame')
    plt.ylabel('similarity')
    plt.legend()

    plt.show()

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
    
    #voc = dbow.Vocabulary(10, 5)
    #voc.load("./config/orbvoc.dbow3")
    #voc.load("./config/test_rgb_10_5.yaml")
    voc = dbow.Vocabulary("./config/mapping_10_5.yaml")

    # create Vocabulary instance from file path
    db = dbow.Database(voc, False)

    dir_path = "/home/gvasserm/dev/aicv_amr_ws/results/"
    query_images = sorted(utils.load_images_from_folder(dir_path, full_path=False), key=utils.sort_key)
    dictq = {i: q for i, q in enumerate(query_images)}
    dictqi = {value: key for key, value in dictq.items()}
    # add entries to Database
    for image in query_images[:200]:
        keypoints, descriptors = get_descriptors(os.path.join(dir_path, image)) # user's implementation
        if descriptors is not None:
            db.add(descriptors)
        else:
            descriptors = np.zeros((500, 32)).astype(np.uint8)
            db.add(descriptors)

    #pscore = db.compute_pairwise_score()
    #cv2.imshow('pscore', pscore)
    #cv2.waitKey(0)

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
    test()
    #test_voc()
    #benchmark()
    #test_likelihood()
    #test_like()
