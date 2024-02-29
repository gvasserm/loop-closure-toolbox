#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Originally created on Thu Mar  5 12:05:56 2020

@author: mubariz
"""

from evaluate_vpr_techniques import evaluate_vpr_techniques
from performance_comparison import performance_comparison
        
def exec_eval_mode(dataset_name, dataset_directory,precomputed_directory,VPR_techniques, save_descriptors, scale_percent):
        
    print('Evaluation Mode 0')
    query_all, retrieved_all, scores_all, encoding_time_all, matching_time_all, all_retrievedindices_scores_allqueries_dict, descriptor_shape_dict=evaluate_vpr_techniques(dataset_directory,precomputed_directory,VPR_techniques,save_descriptors, scale_percent)   #Evaluates all VPR techniques currently available in the framework on specified dataset. 
    performance_comparison(dataset_name, dataset_directory, VPR_techniques,query_all, retrieved_all, scores_all, encoding_time_all, matching_time_all, all_retrievedindices_scores_allqueries_dict, descriptor_shape_dict)

    