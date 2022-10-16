import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import h5py
import numpy as np
from scipy.stats import kendalltau as kendall
from scipy.stats import spearmanr as spearman
from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
import statistics

from models.model import My_Model


from utils.config import ConfigParser
import utils.video_summarization as utils
from utils.loss_metric import LossMetric
from utils.f1_score_metric import F1ScoreMetric
from utils.visualization import plot
from utils.My_Dataset import My_VideoSummarizationDataset
from utils.My_Dataset import Aug_VideoSummarizationDataset
from utils.CoSum_DataLoader import CoSum_VideoSummarizationDataset


class Trainer():

    def __init__(self, model, train_dataloader, validate_dataloader, config, aug_list=None, split_index=None):
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.aug_list = aug_list
        self.config = config
        self.split_index = split_index
        self.mode = self.config.mode
        self.device = torch.device(config.device)

        self.max_epochs = self.config.epoches
        self.current_epoch = None
        self.model = model.to(self.device)
        self.model.apply(self._initialize_weights)


        self.optimizer,self.scheduler = self._initialize_optimizer()
        self.criterion = nn.MSELoss()
        
        

        self.losses = LossMetric()       # training loss for each epoch
        self.f1_scores = F1ScoreMetric() # validation f1 score for each epoch
        self.kendall_record = F1ScoreMetric()
        self.spearman_record = F1ScoreMetric()
        self.train_loss_mean=[]
        self.valid_loss_mean=[]
        self.best_checkpoint = None
        self.video_shot_score_dict = {}
        self.bsf_fscore = 0
        self.kendall_bsf_fscore = 0
        self.spearman_bsf_fscore = 0
        return


    def _initialize_weights(self, module):
        if type(module) == nn.Linear:
            nn.init.xavier_uniform_(module.weight, gain = nn.init.calculate_gain('relu')) # gain = 1.414
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)
        return


    def _initialize_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.learning_rate, weight_decay = self.config.l2_regularization)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        return optimizer,scheduler


    def _augment(self):
        if self.aug_list is None:
            return
        self.model.train()
        for aug_dataloader in self.aug_list:
            for data, label in tqdm(aug_dataloader):
                data, label = utils.aug_preprocess(data, label)
                data = data[0].to(self.device), data[1].to(self.device)
                label = label.to(self.device)
                print(label)
                
                self.optimizer.zero_grad()
                output, _ = self.model(data)
                
                loss = self.criterion(output, label)
                loss.backward()
                # print(loss.item())
                
                self.optimizer.step()
        return

    # training loop
    def _train(self):
        self.model.train()
        self.train_loss_mean=[]
        for data, label, segmentation, summaries, key ,_ in tqdm(self.train_dataloader):
          data, label, segmentation, summaries, key = utils.preprocess(data, label, segmentation, summaries, key)
          data = data[0].to(self.device), data[1].to(self.device)
          label = label.to(self.device)
          
          self.optimizer.zero_grad()
          # 下面這行是原本的
          # output, _ = self.model(data, segmentation, indexes, self.mode)
          # 這是新的
          try:
              output,_ = self.model(data)
          except:
              # print(key)
              continue
          loss = self.criterion(output, label)
          loss.backward()
          self.train_loss_mean.append(loss.cpu().detach().item())
          self.optimizer.step()
          # self.scheduler.step(loss) 
          self.losses.update(self.current_epoch, key, loss.cpu().detach().numpy())
        return


    # validation loop
    def _validate(self):
        self.model.eval()
        self.valid_loss_mean=[]
        
        epoch_video_shot_score_dict = {}
        with torch.no_grad():
          for data, label, segmentation, summaries, key ,indexes in tqdm(self.validate_dataloader):
              data, label, segmentation, summaries, key = utils.preprocess(data, label, segmentation, summaries, key)

              data = data[0].to(self.device), data[1].to(self.device)
              label = label.to(self.device)
              # 這是原本的
              # output, attention_weights = self.model(data, segmentation, indexes,self.mode)
              # 這是新的
              try:
                  output, attention_weights = self.model(data)
              except:
                  continue
              loss = self.criterion(output, label)
              output = output[0].cpu().numpy()    # (1 x number of downsampled frames)
              label_np = label.detach().cpu().numpy()
              label_np = np.reshape(label_np, -1)
              kendall_coeff,_ = kendall(output, label_np)
              spearman_coeff, _ = spearman(output, label_np)
              attention_weights = attention_weights.cpu().numpy() # (number of downsampled frames x number of downsampled frames)
              self.valid_loss_mean.append(loss.item())
              
              if self.config.fixed_summary_length:
                  proportion = 0.15
              else:
                  proportion = utils.get_propotion(summaries)
              
              if True in np.isnan(output):
                  continue
              summary = utils.generate_summary(utils.get_frame_scores(output, segmentation), segmentation, self.config.key_shot_selection, proportion = proportion)
              shot_summary = utils.generate_summary_shot(output, segmentation, self.config.key_shot_selection, proportion=proportion) # key_shot_selection are either knapsack or greedy
              
              
              _, _, f1_score = utils.evaluate_summary(summary, summaries,self.config.eval_mode)
              _,_,f1_shot = utils.evaluate_summary(shot_summary, summaries)
              if f1_shot > f1_score:
                  f1_score = f1_shot
              self.f1_scores.update(self.current_epoch, key, f1_score)
              self.kendall_record.update(self.current_epoch, key, kendall_coeff)
              self.spearman_record.update(self.current_epoch, key, spearman_coeff)
              epoch_video_shot_score_dict.update({key: output})
        return epoch_video_shot_score_dict

    def get_best_valid_video_seg_score(self):
        return self.video_shot_score_dict
    
    # 每個epoch結束時存檔
    def _save_model(self, filename):
        save_path = f'{self.config.checkpoints_path}/{filename}'
        torch.save(self.model.state_dict(), save_path)
        self.best_checkpoint = filename
        return


    def _delete_model(self, filename):
        file_path = f'{self.config.checkpoints_path}/{filename}'
        if os.path.isfile(file_path):
            os.remove(file_path)
        return


    def _on_training_start(self):
        return


    def _on_epoch_start(self):
        return


    def _on_epoch_end(self, tmp_video_shot_score_dict):
        
        # self.writer.add_scalar('Loss/train',statistics.mean(self.train_loss_mean), self.current_epoch)
        # self.writer.add_scalar('Loss/validation',statistics.mean(self.valid_loss_mean), self.current_epoch)
 
        # self.scheduler.step(statistics.mean(self.train_loss_mean)) 
        mean_loss = self.losses.get_current_status()
        mean_f1_score, max_f1_score, is_max_f1_score_updated = self.f1_scores.get_current_status()
        mean_kendall, max_kendall, is_max_kendall = self.kendall_record.get_current_status()
        mean_spearman, max_spearman, _ = self.spearman_record.get_current_status()
        # self.writer.add_scalar('F1/validation',mean_f1_score, self.current_epoch)
        if self.config.is_verbose is True:
            output_format = 'Epoch {} \t train loss = {:.6f} \t library train loss = {:.6f} \t val F1 score (max) = {:.6f} ({:.6f}) \t Kendall (max) = {:.6f} ({:.6f}) \t Spearman (max) = {:.6f} ({:.6f})'
            print(output_format.format(self.current_epoch + 1, mean_loss, statistics.mean(self.train_loss_mean), mean_f1_score, max_f1_score, mean_kendall, max_kendall, mean_spearman, max_spearman))
            # print(self.f1_scores.get_epoch(self.current_epoch))
        if is_max_f1_score_updated:
            self.video_shot_score_dict = tmp_video_shot_score_dict
            self.bsf_fscore = mean_f1_score
            self.kendall_bsf_fscore = mean_kendall
            self.spearman_bsf_fscore = mean_spearman
            if self.best_checkpoint is not None:
                self._delete_model(self.best_checkpoint)
            filename = 'epoch_{}_{}_{:.4f}.pth'.format(self.split_index, self.current_epoch, mean_f1_score)
            # self._save_model(filename)
        if is_max_kendall:
            self.bsf_kendall = mean_kendall
        return


    def _on_training_end(self):
            
        save_path = f'{self.config.checkpoints_path}/losses_{self.split_index}.csv'
        save_path_valid = f'{self.config.checkpoints_path}/valid_losses_{self.split_index}.csv'
        # self.losses.get_epoch_means().to_csv(save_path)
        # self.losses.get_valid_epoch_means().to_csv(save_path_valid)
        # # files.download(save_path)
        # files.download(save_path_valid)
        return


    def run(self):
        self._on_training_start()
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            self._on_epoch_start()
            # self._augment()
            self._train()
            tmp = self._validate()
            if len(tmp) == 0:
                continue
            self._on_epoch_end(tmp)
        self._on_training_end()
        return


if __name__ == '__main__':
    
    config_path = './transfer_config.yaml'
    config = ConfigParser(config_path)
    
    aug_dataset_path = './datasets/YT_goo3DRes_shot_center.h5'
    h5_aug_dataset = h5py.File(aug_dataset_path, 'r')
    aug_dataset = Aug_VideoSummarizationDataset(h5_aug_dataset,config)
    indexes = [i for i in range(len(list(h5_aug_dataset.keys())))]
    sampler = SubsetRandomSampler(indexes)
    dataloader = DataLoader(aug_dataset, sampler=sampler)
    
    
    device = torch.device('cuda')
    model = My_Model(config).to(device)
    
    lossfun = nn.MSELoss()
    
    for data, label in tqdm(dataloader):
        test_label = label
        data, label = utils.aug_preprocess(data, label)
        
        data = data[0].to(device), data[1].to(device)
        label = label.to(device)
        
        output,_ = model(data)
        loss = lossfun(output, label)
        loss.backward()
        break
    
    
    # for data, label, segmentation, summaries, key ,indexes in tqdm(validate_dataloader):
         # data, label, segmentation, summaries, key = utils.preprocess(data, label, segmentation, summaries, key)
        
         # data = data[0].to(torch.device('cpu')), data[1].to(torch.device('cpu'))
         # label = label.to(torch.device('cpu'))
         # # 這是原本的
         # # output, attention_weights = self.model(data, segmentation, indexes,self.mode)
         # # 這是新的
         # try:
         #     output, attention_weights = model(data)
         # except:
         #     continue
         
         # output = output[0].cpu().numpy()    # (1 x number of downsampled frames)
         # label_np = label.detach().cpu().numpy()
         # label_np = np.reshape(label_np, -1)
         # kendall_coeff,_ = kendall(output, label_np)
         # spearman_coeff, _ = spearman(output, label_np)
         # attention_weights = attention_weights.cpu().numpy() # (number of downsampled frames x number of downsampled frames)
         # summary = utils.generate_summary(utils.get_frame_scores(output, segmentation), segmentation, 'knapsack', proportion = proportion)
         # break

