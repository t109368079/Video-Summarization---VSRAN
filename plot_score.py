# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:01:20 2022

@author: gary
"""

import os
import h5py
import numpy as np

from matplotlib import pyplot as plt
from utils import tools

def color_coding(pred_summary, gt_summary):
  c=[]
  for x,y in zip(pred_summary, gt_summary):
    if x == 0 and y == 0:   #True Negative 這個shot既不再預測中亦不再實際摘要中
      color = 'red'
    elif x == 1 and y == 0:   #False Positive
      color = 'orange'
    elif x == 0 and y == 1:   #False Negative
      color = 'yellow'
    else:                # True Positive
      color = 'green'
    c.append(color)

  return c


def plot_keyshot(frame_seg_score, gt, summary, gt_summary, cp, video_index, mode):
    fig = plt.figure(figsize = (20, 9))
    nf = len(summary)
    plt.bar(x=list(range(1, nf+1)), height=frame_seg_score, width=1, color=color_coding(summary, gt_summary))
    for s in cp:
        plt.axvline(x=s[1], color='skyblue')
    plt.plot(list(range(0, nf)), gt, linewidth=3, color='black', linestyle='dashed')
    plt.xlabel("Segments")
    plt.ylabel("Number of Pick")
    plt.title("{} Video-".format(mode)+str(video_index))
    plt.show()
    
    plt.clf()

def normalize(score):
  score -= score.min()
  score /= score.max()

  return score

def upsample_shot(shot_level, segment, nframe):
    frame_level = np.zeros(nframe)
    for i in range(len(shot_level)):
        s = shot_level[i]
        seg = segment[i]
        frame_level[seg[0]:seg[1]+1] = s
    return frame_level

if __name__ == '__main__':
    root = './'
    key_shot_selection = 'greedy'
    ref_dataset_path = os.path.join(root,'datasets','tvsum_goo3DRes_shot_mean.h5')
    save_score_dir = os.path.join(root,'save_score')
    
    ref_dataset = h5py.File(ref_dataset_path, 'r')
    video_list = list(ref_dataset.keys())
    video_index = 8
    video_name = f'video_{video_index}'
    
    video = ref_dataset[video_name]
    n_frame = video['number_of_frames'][...]
    segment = video['segmentation'][...]
    
    gt_score = video['gt_score'][...]
    gt_summary = video['selected_shot'][...]
    
    gt_score_frame = upsample_shot(gt_score, segment, n_frame)
    gt_score_frame = normalize(gt_score_frame)
    gt_summary_frame = upsample_shot(gt_summary, segment, n_frame)
    
    pred_shot_score_path = os.path.join(save_score_dir,f'{video_name}.npy')
    pred_shot_score = np.load(pred_shot_score_path, allow_pickle=True)
    # pred_shot_score = normalize(pred_shot_score)
    pred_summary_frame = tools.generate_summary_shot(pred_shot_score,segment,key_shot_selection)
    pred_frame_score = upsample_shot(pred_shot_score, segment, n_frame)
    
    plot_keyshot(pred_frame_score, gt_score_frame, pred_summary_frame, gt_summary_frame, segment, video_index, key_shot_selection)
