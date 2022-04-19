#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03-Feb-2022

author: panos
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
from time import time
import datetime
import numpy as np
import os
import copy
from helpers.io import exp_name
from transformers import get_linear_schedule_with_warmup,AdamW

class BaseTrainer:
    def __init__(self, config, device, iterators, vocabs):
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
        self.monitor = {}
        self.best_score = 0
        self.best_epoch = 0
        self.cur_patience = 0
        self.optimizer = None
        self.averaged_params = {}

    @staticmethod
    def print_params2update(main_model):
        print('MODEL:')
        for p_name, p_value in main_model.named_parameters():
            if p_value.requires_grad:
                print('  {} --> Update'.format(p_name))
            else:
                print('  {}'.format(p_name))

    def init_model(self, some_model):
        main_model = some_model(self.config, self.vocabs, self.device)

        # GPU/CPU
        if self.config['device'] != -1:
            torch.cuda.set_device(self.device)
            main_model.to(self.device)
        return main_model

    def set_optimizer(self, main_model):
        optimizer = AdamW(main_model.parameters(),
                               lr=self.config['lr'],
                               weight_decay=self.config['weight_decay'])
        return optimizer
    
    def set_scheduler(self, optimizer, batch_steps):
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config['warmup_epochs']*batch_steps, 
            num_training_steps=self.config['epochs']*batch_steps)
        return scheduler

    @staticmethod
    def _print_start_training():
        print('\n======== START TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))

    @staticmethod
    def _print_end_training():
        print('\n======== END TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))

    def epoch_checking_larger(self, epoch, item):
        """
        Perform early stopping
        Args:
            epoch (int): current training epoch
        Returns (bool): stop or not
        """
        if item > self.best_score:  # improvement
            self.best_score = item
            if self.config['early_stop']:
                self.cur_patience = 0
                self.best_epoch = epoch
            print('Saving checkpoint')
            self.save_checkpoint()
        else:
            self.cur_patience += 1
            if not self.config['early_stop']:
                self.best_epoch = epoch

        if self.config['patience'] == self.cur_patience and self.config['early_stop']:  # early stop must happen
            self.best_epoch = epoch - self.config['patience']
            return True
        else:
            return False

    def save_checkpoint(self):
        torch.save({'model_params': self.model.state_dict(),
                    'vocabs': self.vocabs,
                    'best_epoch': self.best_epoch,
                    'best_score': self.best_score,
                    'optimizer': self.optimizer.state_dict()}, self.save_path)

    def load_checkpoint(self, model_folder):
        path = os.path.join(model_folder, 'bert.model')
        print('Loading model from path', path)
        checkpoint = torch.load(path)

        self.vocabs = checkpoint['vocabs']
        return checkpoint

    def assign_model(self, checkpoint):
        # Load checkpoint
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_params'].items() if
                           (k in model_dict) and (model_dict[k].shape == checkpoint['model_params'][k].shape)}

        print('Loading pre-trained model')
        for d in pretrained_dict.keys():
            print(' ', d)
        print()

        self.model.load_state_dict(pretrained_dict, strict=False)

        # self.model.events = self.vocabs['events']
        # self.model.actions = self.vocabs['actions']
        # self.model.r_vocab = self.vocabs['r_vocab']
        # vocabs = {'events': {}, 'actions': {}})
        # # freeze
        # if self.config['freeze_pretrained']:
        #     for p_name, p_value in self.model.named_parameters():
        #         if p_name in pretrained_dict:
        #             p_value.requires_grad = False

        self.model.to(self.device)

