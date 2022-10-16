# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 15:41:50 2022

@author: gary
"""

import os
import h5py
import numpy as np

from utils import video_summarization as tools


if __name__ == '__main__':
    
    h5_path = os.path.join('./datasets','cosum_goo3DRes_shot_center.h5')
    dataset = h5py.File(h5_path, 'r')
    
    score_folder = './save_score'
    score_list = os.listdir(score_folder)
    score_list = [file for file in score_list if os.path.isfile(os.path.join(score_folder,file))]
    
    
    video_name = score_list[10]
    test_score_path = os.path.join(score_folder, video_name)
    video_pred_score = np.load(test_score_path, allow_pickle=True)
    
    video_name = video_name.split('.')[0]
    video = dataset[video_name]
    
    gt_summary = video['gt_summary'][...]
    gt_summary = np.reshape(gt_summary, (1, gt_summary.shape[0]))
    segment = video['segmentation'][...]
    mAP_5 = tools.mean_average_precision(video_pred_score, gt_summary, segment, 5)
    mAP_15 = tools.mean_average_precision(video_pred_score, gt_summary, segment, 15)
    
