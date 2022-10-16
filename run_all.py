# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 10:04:08 2022

@author: Yuuki Misaki
"""

import os
import h5py 
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


from utils.config import ConfigParser 
from utils.My_Dataset import My_VideoSummarizationDataset
from utils.kfold_split import read_splits, generate_splits
from utils.combination import generate_comb

from trainer import Trainer
from models.model import My_Model


def write_report(text, path):
    with open(path, 'w') as fw:
        fw.write(text)
    return

def run(splits, config, dataset):
    print(config.report_text)
    # Generate config text
    report_text = config.report_text+'\n'
    report_text += f'Maximum relative distance: {config.max_relative_distance}\n'
    report_text += f'Number of stacks: {config.n_stack}\n'
    
    # Training
    video_shot_score = {}
    average_fscore = 0
    average_kendall = 0
    average_spearman = 0
    for split_index, split in enumerate(splits):
        print(f'Fold {split_index+1}/{len(splits)}')
        
        train_index = split['train']
        train_index = [i-1 for i in train_index]
        valid_index = split['validate']
        valid_index = [i-1 for i in valid_index]
        
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(valid_index)
        
        train_dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler)
        valid_dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=valid_sampler)
        
        model = My_Model(config)
        trainer = Trainer(model, train_dataloader, valid_dataloader, config, split_index)
        trainer.run()
        
        video_shot_score.update(trainer.get_best_valid_video_seg_score())
        average_fscore += trainer.bsf_fscore
        average_kendall += abs(trainer.kendall_bsf_fscore)
        average_spearman += abs(trainer.spearman_bsf_fscore)
        report_text += f"Split: {split_index}, Best F-score: {trainer.bsf_fscore}, Kendall: {abs(trainer.kendall_bsf_fscore)}, Spearman: {abs(trainer.spearman_bsf_fscore)}"
    
    average_fscore /= len(splits)
    average_kendall /= len(splits)
    average_spearman /= len(splits)
    report_text +=f"Average F1-score: {average_fscore}"
    print(f"Average F1-score: {average_fscore}")
    print(f"Average Kendall: {average_kendall}")
    print(f"Average Spearman: {average_spearman}")
    write_report(report_text, config.report_path)

if __name__ == "__main__":
    # Read Config
    config_path = './config.yaml'
    config = ConfigParser(config_path)
    
    # Load dataset
    h5_dataset = h5py.File(config.dataset_path, 'r')
    dataset = My_VideoSummarizationDataset(h5_dataset, config)
    
    # Read or Generate Split
    indexes = list(range(len(dataset)))
    if os.path.isfile(config.split_file_path):
        splits = read_splits(config.split_file_path)
        print("Read Split: ",end='')
    else:
        splits = generate_splits(indexes, 5, shuffle=True)
        print("Generate Split: ",end='')
    print(splits)
    
    
    gt_name_type = ['gt_score', 'selected_count']
    key_shot_select_type = ['greedy', 'knapsack']
    pe_type = ['spe']
    
    run_types = generate_comb([gt_name_type, key_shot_select_type, pe_type],['gt_name', 'key_shot_selection', 'pe'])
    
    for run_type in run_types:
        config.gt_name = run_type['gt_name']
        config.key_shot_selection = run_type['key_shot_selection']
        config.pe = run_type['pe']
        if run_type['gt_name'] == 'gt_score':
            gt_show = 'GT'
        else:
            gt_show = 'CA'
        
        config.report_text = f"{config.dataset_name} Standard Center Motion & Static Shot Feature using {run_type['key_shot_selection']} and {run_type['gt_name']} {run_type['pe']}"
        config.report_path = f"./report_folder/{config.dataset_name}_result/Std_{config.dataset_name}_{config.shot_repre}_{gt_show}_{run_type['key_shot_selection']}_{run_type['pe']}.txt"
    
        run(splits, config, dataset)
