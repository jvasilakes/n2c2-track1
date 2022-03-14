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
        self.vocabs = vocabs #they are inversed
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
        self.scaler = GradScaler()


        
    def calculate_stats(self, y_true, y_sigmoid, typ, keyword=''):
        y_pred =  np.array(y_sigmoid > self.config['threshold'], dtype=float)
        mi_f1 = f1_score(y_true, y_pred, average='micro')
        pr, re, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=1)
        return {'micro_f1':mi_f1, 'macro_pr':pr , 'macro_re':re, 'macro_f1':f1,'pred':y_pred}


    def calculate_performance(self, epoch, tracker, time_, mode='train'):
        tracker['total'] = np.mean(tracker['total'])
        perf = {}
        ##Disp counting
        y_true_ev = np.vstack([entry[1] for entry in tracker['events']])
        y_true_ac = np.vstack([entry[1] for entry in tracker['actions']])
        y_sigmoid_ac = np.vstack([entry[0] for entry in tracker['actions']])
        y_sigmoid_ev = np.vstack([entry[0] for entry in tracker['events']])
        true_dispo = y_true_ev[:,2] == 1
#         pred_dispo1 = np.sum(y_sigmoid_ac > self.config['threshold'], 1) > 0 #based on actions
        pred_dispo2 = y_sigmoid_ev[:,2] > self.config['threshold']
#         pred_dispo = np.logical_or(pred_dispo1, pred_dispo2) # they have sthe same shape but...
        total_dispo = np.logical_or(true_dispo, pred_dispo2)
        disp_count=(np.sum(true_dispo),np.sum(pred_dispo2),np.sum(total_dispo),len(true_dispo))
        print_start(epoch, tracker, mode, time_, disp_count)
                ##  Events
        perf['events'] = self.calculate_stats(y_true_ev, y_sigmoid_ev, 'events')
        print_performance(epoch, tracker, 'events', perf['events'], time_, mode)
                ## Actions All
        perf['actions_all'] = self.calculate_stats(y_true_ac, y_sigmoid_ac, 'actions','All')
        print_performance(epoch, tracker, 'actions', perf['actions_all'], time_, mode)
                ## Pred + True
        y_true_ac = np.vstack([entry[1] for entry in tracker['actions']])[total_dispo] 
        y_sigmoid_ac = np.vstack([entry[0] for entry in tracker['actions']])[total_dispo]
        perf['actions'] = self.calculate_stats(y_true_ac, y_sigmoid_ac, 'actions', 'Pred+True')
        print_performance(epoch, tracker, 'actions', perf['actions'], time_, mode)
                ####
        tracker['event_preds'] = perf['events']['pred']
        tracker['action_preds'] = perf['actions_all']['pred']
        return perf['events']

    @staticmethod
    def init_tracker():
        return {'total': [], 'events': [], 'actions': [], 'total_samples': 0, 'samples': [], 'event_preds': [], 'action_preds':[]}

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
            ##
            if self.best_epoch == epoch:
                print_preds(dev_tracker, self.iterators['dev'], self.config, epoch, 'dev')
#                 print_cases(dev_tracker['samples'], dev_tracker['preds'], self.iterators['dev'], self.config, epoch)
            ##
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
                if keys not in ['names', 'old_pos']:
                    batch[keys] = batch[keys].to(self.device)

            with autocast():
                eloss, aloss, esigmoid, asigmoid = self.model(batch)  # forward pass
                loss = self.config['event_weight'] * eloss + \
                           (1 - self.config['event_weight'])* aloss
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

            tracker['samples'] += [batch['indxs'].cpu().data.numpy()] 
            tracker['events'] += [(esigmoid.cpu().data.numpy(), batch['elabels'].cpu().data.numpy())]
            tracker['actions'] += [(asigmoid.cpu().data.numpy(), batch['alabels'].cpu().data.numpy())]
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
                    if keys not in ['names', 'old_pos']:
                        batch[keys] = batch[keys].to(self.device)
                
                with autocast():
                    eloss, aloss, esigmoid, asigmoid  = self.model(batch)  # forward pass
                    loss = self.config['event_weight'] * eloss + \
                        (1 - self.config['event_weight'])* aloss
                    loss = loss/ self.config['accumulate_batches']

                # collect sigmoid & losses
                tracker['samples'] += [batch['indxs'].cpu().data.numpy()] 
                tracker['events'] += [(esigmoid.cpu().data.numpy(), batch['elabels'].cpu().data.numpy())]
                tracker['actions'] += [(asigmoid.cpu().data.numpy(), batch['alabels'].cpu().data.numpy())]
                tracker['total'] += [loss.item()]

        t2 = time()
        return tracker, t2-t1

