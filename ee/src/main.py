#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03-Feb-2022

author: panos
"""
from os.path import join
import numpy as np
import torch
import random
import argparse
from math import ceil
from torch import utils
from torch.utils.data import DataLoader
from transformers import AutoTokenizer #
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

def setup(args, config):
    config['mode'] = args.mode
    config['bert'] = args.bert
    config['use_verbs'] = not args.no_verbs
    if args.mode == 'train':
        config['train_data'] = config['data_dir'] + join(args.split, 'spacy/train_data.txt')
        config['dev_data'] = config['data_dir'] + join(args.split,'spacy/dev_data.txt')
    elif args.mode == 'predict':
        config['test_data'] = args.test_path

    # config['pred_dir'] = join(config['pred_dir'], args.bert+'_'+args.train_split)

    if args.model_folder: # all the results folder
        config['model_folder'] = args.model_folder
    else:
        config['model_folder'] = join(config['results_folder'], args.bert+'_'+args.split)

    # config['pred_data'] = args.outdir
    config['exp_name'] = args.bert


def main(args):
    
    config = load_config(args.config)
    setup(args, config)
    
    print('Setting the seed to {}'.format(config['seed']))
    set_seed(config['seed'])
    config['exp'] = setup_log(config, folder_name=config['model_folder'], mode=config['mode'] )
    device = torch.device("cuda:{}".format(config['device']) if config['device'] != -1 else "cpu")
    if args.mode == 'train':
        train_loader, val_loader, train_data = load_data(config)
        trainer = load_trainer(train_loader, val_loader, train_data, config, device)
        _ = trainer.run()
    elif args.mode =='predict': ## load model and test
        trainer = load_saved_model(config, device)
        print_options(config)
        tracker, time_ = trainer.eval_epoch(iter_name='test')
        trainer.predict(tracker)
        print_preds(tracker, trainer.iterators['test'], config, 0, 'test')
    print('Terminating-- Please look for predictions, log and saved model in', config['model_folder'])

def load_data(config):

#     tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    if config['bert'] == 'base':
        tokenizer = AutoTokenizer.from_pretrained('../bert_models/bert-base-uncased') ##P
        
    elif config['bert'] =='clinical':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    elif config['bert'] =='blue':
        tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")

    if config['mode'] == 'train':
        train_data_ = BertDataset(config['train_data'],mode='train',tokenizer = tokenizer,
                                  use_verbs= config['use_verbs'], config=config)
        print('Train data:',len(train_data_))
        train_loader_ = DataLoader(train_data_, batch_size=config['batch_size'],
                               shuffle=True,
                               collate_fn=Collates(),
                               num_workers=0)
        dev_data_ = BertDataset(config['dev_data'],mode='dev',tokenizer = tokenizer,
                                use_verbs= config['use_verbs'], config=config)
        print('Dev data:', len(dev_data_))

        dev_loader_ = DataLoader(dataset=dev_data_, batch_size=config['batch_size'],
                               shuffle=True,
                               collate_fn=Collates(),
                               num_workers=0)
        return train_loader_, dev_loader_, train_data_

    else:
        test_data_ = BertDataset(config['test_data'], mode='test', tokenizer=tokenizer,
                                use_verbs=config['use_verbs'], config=config)
        print('Test data:', len(test_data_))

        test_loader_ = DataLoader(dataset=test_data_, batch_size=config['batch_size'],
                                 shuffle=False,
                                 collate_fn=Collates(),
                                 num_workers=0)
        return [], test_loader_, []

def load_trainer(train_loader_, dev_loader_, train_data_, config, device):
    trainer = Trainer(config, device,
                      iterators={'train': train_loader_, 'dev': dev_loader_},
                      vocabs={'events': {v: k for k, v in train_data_.event_vocab.items()}, 'actions': {v: k for k, v in train_data_.action_vocab.items()}})

    trainer.model = trainer.init_model(target_model)
    trainer.optimizer = trainer.set_optimizer(trainer.model)
    batch_steps = ceil(len(train_data_)/config['batch_size'])
    trainer.scheduler = trainer.set_scheduler(trainer.optimizer, batch_steps)
    return trainer

def load_saved_model(config,  device, which=None):
    trainer = Trainer(config, device,
                      iterators={'train': [], 'val': [], 'test': []},
                      vocabs={'events': {}, 'actions': {}})
    checkpoint = trainer.load_checkpoint(config['model_folder'])
    vocabs = checkpoint['vocabs']
    _, test_loader_, _ = load_data(config)
    trainer.iterators['test'] = test_loader_
    trainer.iterations = len(test_loader_)
    trainer.model = trainer.init_model(target_model)
    trainer.optimizer = trainer.set_optimizer(trainer.model)
    trainer.assign_model(checkpoint)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,  required=True,)
    parser.add_argument('--mode', type=str,  required=True, choices=['train', 'predict'])
    parser.add_argument('--model_folder', type=str, help='Must include the bert type model specified')
    parser.add_argument('--test_path', type=str, help='Path to test_data.txt file')
    parser.add_argument('--split', type=str,  choices=['default', 'split0', 'split1', 'split2', 'split3', 'split4'])
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--bert', type=str,  required=True, choices=['base', 'clinical', 'blue'])
    parser.add_argument('--no_verbs', action='store_true', help='No verbs usgae')
    args = parser.parse_args()
    main(args)