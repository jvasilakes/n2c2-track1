#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03-Feb-2022

author: panos
"""
import numpy as np
import torch
import random
import argparse
from math import ceil
from torch import utils
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, AutoTokenizer 
from datasets import BertDataset, Collates
from trainer import Trainer
from blank_net import BlankNet as target_model
from helpers.io import *


def set_seed(seed):
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    
    config = load_config(args.config)
    setup(args, config)
    config['mode'] = args.mode
    print('Setting the seed to {}'.format(config['seed']))
    set_seed(config['seed'])
    config['model_folder'], config['exp'] = setup_log(config, mode=config['mode'], 
                                                      folder_name=config['exp_name'])
    device = torch.device("cuda:{}".format(config['device']) if config['device'] != -1 else "cpu")

    train_loader, val_loader, train_data = load_data(config)
    trainer = load_trainer(train_loader, val_loader, train_data, config, device)
    _ = trainer.run()
def setup(args, config):
    
    config['train_data'] = args.data + 'train_data.txt'
    config['dev_data'] = args.data + 'dev_data.txt'
    
def load_data(config):

#     tokenizer = PreTrainedTokenizerFast.from_pretrained('bert-base-uncased') ##P
#     tokenizer.pad_token = "[PAD]" ##P
#     tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') ##P
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#     tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    train_data_ = BertDataset(config['train_data'], config['max_sen_len'], mode='train', 
                     tokenizer = tokenizer, verbs= config['verbs'], dummy = False)
    print('Train data:',len(train_data_))
    train_loader_ = DataLoader(train_data_, batch_size=config['batch_size'],
                           shuffle=True,
                           collate_fn=Collates(),
                           num_workers=0)
    dev_data_ = BertDataset(config['dev_data'], config['max_sen_len'], mode='dev',
                          tokenizer = tokenizer, verbs= config['verbs'], dummy = False)
    print('Dev data:', len(dev_data_))
    
    dev_loader_ = DataLoader(dataset=dev_data_, batch_size=config['batch_size'],
                           shuffle=True,
                           collate_fn=Collates(),
                           num_workers=0)
    return train_loader_, dev_loader_, train_data_

def load_trainer(train_loader_, dev_loader_, train_data_, config, device):
    trainer = Trainer(config, device,
                      iterators={'train': train_loader_, 'dev': dev_loader_},
                      vocabs={'events': {v: k for k, v in train_data_.event_vocab.items()}, 'actions': {v: k for k, v in train_data_.action_vocab.items()}})

    trainer.model = trainer.init_model(target_model)
    trainer.optimizer = trainer.set_optimizer(trainer.model)
    batch_steps = ceil(len(train_data_)/config['batch_size'])
    trainer.scheduler = trainer.set_scheduler(trainer.optimizer, batch_steps)
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'test'])
    parser.add_argument('--data', type=str)
    args = parser.parse_args()
    main(args)