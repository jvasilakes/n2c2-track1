#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03-Mar-2020

author: fenia
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
from blank_net import BlankNet as target_model
from helpers.base_trainer import BaseTrainer
from helpers.io import *
from time import time
import numpy as np
import json
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from torch.cuda.amp import autocast, GradScaler
#

class Trainer(BaseTrainer):
    def __init__(self, config, device, iterators=None, vocabs=None):
        """
        Trainer object.
        Args:
            config (dict): model parameters
            iterators (dict): 'train' and 'test' iterators
        """
        self.config = config
        self.iterators = iterators
        self.vocabs = vocabs
        self.device = device

        self.model = None  # self.init_model(target_model)
        self.optimizer = None   # self.set_optimizer(self.model, opt=self.config['optimizer'])
        self.scheduler = None
        self.primary_metric = []  # on the validation set
        self.best_score = 0
        self.best_epoch = 0
        self.cur_patience = 0
        self.averaged_params = {}
        self.iterations = len(self.iterators['train'])
        self.save_path = os.path.join(self.config['model_folder'], 'bert.model')
        ##P
        self.scaler = GradScaler()
        ##E
    def optimise(self):
        """
        what is the purpose
        """
        print('Why I am inside optimise?')
        exit()
        return 
    
 

    def calculate_performance(self, epoch, tracker, time_, mode='train'):
        
#         print(tracker['gtruth'])
        y_true = np.concatenate(tracker['gtruth'])
        y_logits = np.vstack(tracker['logits'])
        y_pred = np.argmax(y_logits, axis=-1)
#         nclasses = len(self.vocabs['r_vocab'])
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        res = classification_report(y_true, y_pred, output_dict=True)
        perf = {'micro_f1':micro_f1, 'NoDisp_f1': res['0']['f1-score'], 'Disp_f1': res['1']['f1-score'], 'Und_f1': res['2']['f1-score']}
        pr, re, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        perf['macro'] = (pr,re, f1)
        tracker['total'] = np.mean(tracker['total'])
        print_performance(epoch, tracker, perf, time_, name=mode)
        
        return perf


    @staticmethod
    def init_tracker():
        return {'total': [], 'logits': [], 'gtruth': [], 'total_samples': 0}

    def run(self):
        """
        Main Training Loop.
        """
#         self.print_params2update(self.model)
        print_options(self.config)
        self._print_start_training()
        
        for epoch in range(1, self.config['epochs'] + 1):
            train_tracker, time_ = self.train_epoch(epoch)

            _ = self.calculate_performance(epoch, train_tracker, time_, mode='train')

            dev_tracker, time_ = self.eval_epoch(iter_name='dev')
            dev_perf = self.calculate_performance(epoch, dev_tracker, time_, mode='dev')

            self.primary_metric += [dev_perf['micro_f1']]

            stop = self.epoch_checking_larger(epoch, self.primary_metric[-1])
            print('current best epoch:', self.best_epoch)
            if stop:
                break
            print()

        self._print_end_training()

        print('Best epoch: {}\n'.format(self.best_epoch))
        return self.best_score

    def train_epoch(self, epoch):
        """
        Evaluate the model on the train set.
        """
        t1 = time()
        self.model = self.model.train()
        tracker = self.init_tracker()

        iterations = len(self.iterators['train'])
        for batch_idx, batch in enumerate(self.iterators['train']):
            step = ((epoch-1) * iterations) + batch_idx

            tracker['total_samples'] += len(batch['names'])
            
            for keys in batch.keys():
                if keys != 'names':
                    batch[keys] = batch[keys].to(self.device)

            ##P
#             self.model.zero_grad()
            if not self.config['autoscalling']:
                loss, logits = self.model(batch)  # forward pass
                loss = loss/ self.config['accumulate_batches']
                
                loss.backward()

                # gradient clipping
                if self.config['clip'] > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip'])

                if  (batch_idx + 1)%self.config['accumulate_batches']==0 or (batch_idx + 1) == len(self.iterators['train']):     
                    self.optimizer.step()   # update
                    self.scheduler.step()
                    self.model.zero_grad()
                    
            else: ## Autoscalling
                with autocast():
                    loss, logits = self.model(batch)  # forward pass
                    loss = loss/ self.config['accumulate_batches']
                    
                self.scaler.scale(loss).backward()

                if  (batch_idx + 1)%self.config['accumulate_batches']==0 or (batch_idx + 1) == len(self.iterators['train']):     
                    if self.config['clip'] > 0: # gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip'])
                    self.scaler.step(self.optimizer)   # update
                    self.scaler.update()
                    self.scheduler.step()  ## questionable, may need scaler
                    self.model.zero_grad()  


            tracker['logits'] += [logits.cpu().data.numpy()]
            tracker['gtruth'] += [batch['labels'].cpu().data.numpy()]
            tracker['total'] += [loss.item()]


            if batch_idx % self.config['log_interval'] == 0:
                print('Step {:<6}    LOSS = {:10.4f}'.format(step, loss.item()))

        t2 = time()
        return tracker, t2-t1

    def eval_epoch(self, iter_name='dev', final=False):
        """
        Evaluate the model on the test set.
        No backward computation is allowed.
        """
        t1 = time()
        self.model = self.model.eval()
        tracker = self.init_tracker()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.iterators[iter_name]):
                tracker['total_samples'] += len(batch['names'])

                for keys in batch.keys():
                    if keys != 'names':
                        batch[keys] = batch[keys].to(self.device)
                
                if not self.config['autoscalling']:
                    loss, logits = self.model(batch)  # forward pass
                    loss = loss/ self.config['accumulate_batches']
                        
                else: ## Autoscalling
                    with autocast():
                        loss, logits = self.model(batch)  # forward pass
                        loss = loss/ self.config['accumulate_batches']

                # collect logits & losses
                tracker['logits'] += [logits.cpu().data.numpy()]
                tracker['gtruth'] += [batch['labels'].cpu().data.numpy()]
                tracker['total'] += [loss.item()]

        t2 = time()
        return tracker, t2-t1

