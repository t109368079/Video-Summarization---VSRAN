# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:26:07 2022

@author: Yuuki Misaki
"""

import os
import h5py
import numpy as np
from sklearn.metrics import average_precision_score as mAP

from utils import video_summarization as tools


def create_human_summary(anno_score, segment, length):
    seg_scores = []
    for seg in segment:
        start = seg[0]
        end = seg[1]
        
        seg_scores.append(np.mean(anno_score[start:end+1]))
    seg_scores = np.array(seg_scores)
    n_select = int(length*segment.shape[0])
    arg_sort = np.argsort(seg_scores)
    
    shot_summary = np.zeros(segment.shape[0])
    for i in range(segment.shape[0]-1,-1,-1):
        index = arg_sort[i]
        shot_summary[index] = 1
        if np.sum(shot_summary) >= n_select:
            break
        else:
            pass
                
    return shot_summary

def create_summary(shot_score, length=0.5):
    n_segment = shot_score.shape[0]
    shot_summary = np.zeros(n_segment)
    summary_length = int(length*n_segment)
    soted_index = np.argsort(shot_score)
    
    for i in range(n_segment-1, n_segment-summary_length-1,-1):
        index = soted_index[i]
        shot_summary[index] = 1
    return shot_summary
        
    

def create_shot_score(scores, segment):
    shot_scores = np.zeros(segment.shape[0])
    for i, seg in enumerate(segment):
        start = seg[0]
        end = seg[1]
        segment_scores = scores[start:end]
        
        average_seg_score = np.mean(segment_scores)
        shot_scores[i] = average_seg_score
    return shot_scores


score_path = './datasets/ydata-tvsum50-anno.tsv'
table_path = './datasets/Index_Name_Table_TVSum.txt'

table = tools.read_index_table(table_path)
score = tools.read_tvsum_score(score_path)

ref_dataset_path = './datasets/tvsum_goo3DRes_shot_mean.h5'
ref_dataset = h5py.File(ref_dataset_path, 'r')



#%%

video_list = list(score.keys())
top5_list = []
top15_list = []
for video_name in video_list:
    video_index = table[video_name]
    video = ref_dataset[video_index]
    segment = video['segmentation'][...]
    n_segment = segment.shape[0]
    
    video_anno_scores = score[video_name]
    video_anno_scores = np.array(video_anno_scores)
    gt_score = np.mean(video_anno_scores,axis=0)
    shot_score = create_shot_score(gt_score, segment)
    shot_summary = create_summary(shot_score, length=0.5)
    
    top5_map = 0
    top15_map = 0
    for anno_score in video_anno_scores:
        anno_shot_score = create_shot_score(anno_score, segment)
        anno_shot_score -= anno_shot_score.min()
        anno_shot_score /= anno_shot_score.max()
        
        sorted_index = np.argsort(anno_shot_score)
        
        top5_index = [sorted_index[i] for i in range(n_segment-1, n_segment-6,-1)]
        top15_index = [sorted_index[i] for i in range(n_segment-1, n_segment-16,-1)]
        
        top5_score = anno_shot_score[top5_index]
        top15_score = anno_shot_score[top15_index]
        
        top5_summary = shot_summary[top5_index]
        top15_summary = shot_summary[top15_index]
        
        top5_tmp = mAP(top5_summary, top5_score)
        top15_tmp = mAP(top15_summary, top15_score)
        
        top5_tmp = 0 if np.isnan(top5_tmp) else top5_tmp
        top15_tmp = 0 if np.isnan(top15_tmp) else top15_tmp
        
        top5_map += top5_tmp
        top15_map += top15_tmp
    top5_map /= 20
    top15_map /= 20
    
    top5_list.append(top5_map)
    top15_list.append(top15_map)

#%% 

Average_top5 = sum(top5_list)/50
Average_top15 = sum(top15_list)/50
print(f'Top5: {Average_top5}')
print(f'Top15: {Average_top15}')
    
        
    
    
    
#     sorted_index = np.argsort(human_shot_score)
#     top5_index = [sorted_index[i] for i in range(n_segment-1, n_segment-6,-1)]
#     top15_index = [sorted_index[i] for i in range(n_segment-1, n_segment-16,-1)]
    
#     top5_human_score = human_shot_score[top5_index]
#     top15_human_score = human_shot_score[top15_index]
    
#     top5_map = 0
#     top15_map = 0
#     for i, anno_score in enumerate(video_anno_scores):
#         anno_summary = create_human_summary(anno_score, segment, length=0.5)
#         video_anno_summary[i] = anno_summary
        
#         top5_temp = mAP(anno_summary[top5_index], top5_human_score)
#         top15_temp = mAP(anno_summary[top15_index], top15_human_score)
        
#         top5_temp = 0 if np.isnan(top5_temp) else top5_temp
        
#         top5_map += top5_temp
#         top15_map += top15_temp
#     top5_map /= 20
#     top15_map /= 20
    
#     top5_list.append(top5_map)
#     top15_list.append(top15_map)

# #%% 
# print(f'')
    
    
    
        

human_summary = 0



