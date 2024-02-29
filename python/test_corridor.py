import numpy as np
import os
import re
import json

import utils


text = """
 #1 ptime(0.042718s) STM(1) WM(0) hyp(0) value(0.00)
 #2 ptime(0.040906s) STM(2) WM(0) hyp(0) value(0.00)
 #3 ptime(0.046203s) STM(3) WM(0) hyp(0) value(0.00)
 #4 ptime(0.038729s) STM(4) WM(0) hyp(0) value(0.00)
 #5 ptime(0.058972s) STM(5) WM(0) hyp(0) value(0.00)
 #6 ptime(0.023531s) STM(6) WM(0) hyp(0) value(0.00)
 #7 ptime(0.024470s) STM(7) WM(0) hyp(0) value(0.00)
 #8 ptime(0.081536s) STM(8) WM(0) hyp(0) value(0.00)
 #9 ptime(0.035435s) STM(9) WM(0) hyp(0) value(0.00)
 #10 ptime(0.028937s) STM(10) WM(0) hyp(0) value(0.00)
 #11 ptime(0.005288s) STM(10) WM(1) hyp(0) value(0.00)
 #12 ptime(0.020425s) STM(10) WM(2) hyp(2) value(0.00)
 #13 ptime(0.033225s) STM(10) WM(3) hyp(1) value(0.00)
 #14 ptime(0.029220s) STM(10) WM(4) hyp(2) value(0.00)
 #15 ptime(0.030389s) STM(10) WM(5) hyp(2) value(0.00)
 #16 ptime(0.131155s) STM(10) WM(6) hyp(6) value(0.00)
 #17 ptime(0.021211s) STM(10) WM(7) hyp(4) value(0.00)
 #18 ptime(0.028018s) STM(10) WM(8) hyp(3) value(0.00)
 #19 ptime(0.027897s) STM(10) WM(9) hyp(3) value(0.00)
 #20 ptime(0.030364s) STM(10) WM(10) hyp(3) value(0.00)
 #21 ptime(0.033576s) STM(10) WM(11) hyp(9) value(0.00)
 #22 ptime(0.027160s) STM(10) WM(12) hyp(10) value(0.00)
 #23 ptime(0.031867s) STM(10) WM(13) hyp(10) value(0.00)
 #24 ptime(0.029677s) STM(10) WM(14) hyp(7) value(0.00)
 #25 ptime(0.036602s) STM(10) WM(15) hyp(6) value(0.00)
 #26 ptime(0.040422s) STM(10) WM(16) hyp(7) value(0.00)
 #27 ptime(0.033414s) STM(10) WM(17) hyp(15) value(0.00)
 #28 ptime(0.035815s) STM(10) WM(18) hyp(7) value(0.00)
 #29 ptime(0.034741s) STM(10) WM(19) hyp(10) value(0.00)
 #30 ptime(0.255145s) STM(10) WM(20) hyp(13) value(0.00)
 #31 ptime(0.034038s) STM(10) WM(21) hyp(13) value(0.00)
 #32 ptime(0.033621s) STM(10) WM(22) hyp(10) value(0.00)
 #33 ptime(0.034359s) STM(10) WM(23) hyp(22) value(0.00)
 #34 ptime(0.033144s) STM(10) WM(24) hyp(13) value(0.00)
 #35 ptime(0.034741s) STM(10) WM(25) hyp(21) value(0.00)
 #36 ptime(0.034687s) STM(10) WM(26) hyp(16) value(0.00)
 #37 ptime(0.035407s) STM(10) WM(27) hyp(22) value(0.00)
 #38 ptime(0.037119s) STM(10) WM(28) hyp(16) value(0.00)
 #39 ptime(0.035353s) STM(10) WM(29) hyp(22) value(0.00)
 #40 ptime(0.033711s) STM(10) WM(30) hyp(25) value(0.00)
 #41 ptime(0.035082s) STM(10) WM(31) hyp(2) value(0.15) LOOP 41->2
 #42 ptime(0.038611s) STM(10) WM(32) hyp(3) value(0.15) LOOP 42->3
 #43 ptime(0.039474s) STM(10) WM(33) hyp(4) value(0.16) LOOP 43->4
 #44 ptime(0.037575s) STM(10) WM(34) hyp(4) value(0.13) LOOP 44->4
 #45 ptime(0.040530s) STM(10) WM(35) hyp(6) value(0.00)
 #46 ptime(0.028388s) STM(10) WM(36) hyp(5) value(0.13) LOOP 46->5
 #47 ptime(0.030162s) STM(10) WM(37) hyp(8) value(0.15) LOOP 47->8
 #48 ptime(0.037452s) STM(10) WM(38) hyp(9) value(0.17) LOOP 48->9
 #49 ptime(0.033447s) STM(10) WM(39) hyp(10) value(0.21) LOOP 49->10
 #50 ptime(0.021184s) STM(10) WM(40) hyp(10) value(0.24) LOOP 50->10
 #51 ptime(0.004947s) STM(10) WM(41) hyp(0) value(0.00)
 #52 ptime(0.035724s) STM(10) WM(42) hyp(12) value(0.25) LOOP 52->12
 #53 ptime(0.037038s) STM(10) WM(43) hyp(12) value(0.22) LOOP 53->12
 #54 ptime(0.039018s) STM(10) WM(44) hyp(14) value(0.17) LOOP 54->14
 #55 ptime(0.038738s) STM(10) WM(45) hyp(15) value(0.20) LOOP 55->15
 #56 ptime(0.036559s) STM(10) WM(46) hyp(16) value(0.28) LOOP 56->16
 #57 ptime(0.034008s) STM(10) WM(47) hyp(17) value(0.36) LOOP 57->17
 #58 ptime(0.487057s) STM(10) WM(48) hyp(18) value(0.43) LOOP 58->18
 #59 ptime(0.032952s) STM(10) WM(49) hyp(19) value(0.35) LOOP 59->19
 #60 ptime(0.039122s) STM(10) WM(50) hyp(20) value(0.29) LOOP 60->20
 #61 ptime(0.036248s) STM(10) WM(51) hyp(21) value(0.29) LOOP 61->21
 #62 ptime(0.053625s) STM(10) WM(52) hyp(22) value(0.46) LOOP 62->22
 #63 ptime(0.036832s) STM(10) WM(53) hyp(23) value(0.59) LOOP 63->23
 #64 ptime(0.037578s) STM(10) WM(54) hyp(24) value(0.76) LOOP 64->24
 #65 ptime(0.039886s) STM(10) WM(55) hyp(25) value(0.79) LOOP 65->25
 #66 ptime(0.039080s) STM(10) WM(56) hyp(26) value(0.82) LOOP 66->26
 #67 ptime(0.038881s) STM(10) WM(57) hyp(27) value(0.84) LOOP 67->27
 #68 ptime(0.037985s) STM(10) WM(58) hyp(28) value(0.80) LOOP 68->28
 #69 ptime(0.035991s) STM(10) WM(59) hyp(30) value(0.76) LOOP 69->30
 #70 ptime(0.039921s) STM(10) WM(60) hyp(31) value(0.81) LOOP 70->31
 #71 ptime(0.038600s) STM(10) WM(61) hyp(32) value(0.79) LOOP 71->32
 #72 ptime(0.040804s) STM(10) WM(62) hyp(33) value(0.78) LOOP 72->33
 #73 ptime(0.043892s) STM(10) WM(63) hyp(34) value(0.75) LOOP 73->34
 #74 ptime(0.036722s) STM(10) WM(64) hyp(35) value(0.79) LOOP 74->35
 #75 ptime(0.037974s) STM(10) WM(65) hyp(36) value(0.76) LOOP 75->36
 #76 ptime(0.040891s) STM(10) WM(66) hyp(36) value(0.79) LOOP 76->36
 #77 ptime(0.039778s) STM(10) WM(67) hyp(38) value(0.82) LOOP 77->38
 #78 ptime(0.040119s) STM(10) WM(68) hyp(39) value(0.83) LOOP 78->39
 #79 ptime(0.039035s) STM(10) WM(69) hyp(39) value(0.66) LOOP 79->39
 #80 ptime(0.031206s) STM(10) WM(70) hyp(39) value(0.41) LOOP 80->39
 #81 ptime(0.041935s) STM(10) WM(71) hyp(2) value(0.41) LOOP 81->2
 #82 ptime(0.040186s) STM(10) WM(72) hyp(42) value(0.40) LOOP 82->42
 #83 ptime(0.040892s) STM(10) WM(73) hyp(43) value(0.41) LOOP 83->43
 #84 ptime(0.040710s) STM(10) WM(74) hyp(4) value(0.46) LOOP 84->4"
"""
 
loop_data = re.findall(r"LOOP (\d+)->(\d+)", text)
loop_data = [[int(d[0])-1, int(d[1])-1] for d in loop_data]
ground_truth_info = {}

images = sorted(utils.load_images_from_folder("/home/gvasserm/dev/rtabmap/data/samples", full_path=False), key=utils.sort_key)

for i in range(len(loop_data)):
    q = loop_data[i][0]
    k = loop_data[i][1]
    
    if images[q] in ground_truth_info:
        ground_truth_info[images[q]].append(images[k])
    else:
        ground_truth_info[images[q]] = [images[k]]

    if images[k] in ground_truth_info:
        ground_truth_info[images[k]].append(images[q])
    else:
        ground_truth_info[images[k]] = [images[q]]

    json.dump(ground_truth_info, open("/home/gvasserm/dev/rtabmap/data/gt.json", 'w'))


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

