#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03-Feb-2022

author: panos
"""

import re
import math
import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import random
import json
import heapq
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

max_tok_len = 300
class BertDataset(Dataset):

    def __init__(self, path, max_sen_len=None, mode='train', 
                 tokenizer = None, dummy = False):
        super().__init__()
        self.max_sen_len = max_sen_len
        self.mode = mode
        self.data = []
        self.tokenizer = tokenizer
        self.ev_vocab = {'NoDisposition':0,'Disposition':1, 'Undetermined':2}

        with open(path) as infile:
            for item in tqdm(infile, desc='Loading ' + self.mode.upper()):
                sample = json.loads(item)
                ## {text: ..., trig:{'s','e','name'}, events:[NoDispotion, Dispotition], fname}
                sentences = sample['text']
                pos = (sample['trig']['s'], sample['trig']['e'])
                
#                 labels = np.zeros((len(self.ev_vocab),), 'i')
                # Multi label
#                 for ev in sample['events']:
#                     labels[self.ev_vocab[ev]] = 1
                # Single label
                labels = self.ev_vocab[sample['events'][0]]
                ident =  sample['trig']['name'] +'/'+ sample['fname']
                ## Adding special token @ at before and after the entity
                sentences = sentences[:pos[0]] +'@ ' + sentences[pos[0]:pos[1]] + ' @' + sentences[pos[1]:]
                pos = [idx + 2 for idx in pos]
                ##
                tokenized = self.tokenizer.encode_plus(sentences, add_special_tokens = True, truncation= False, return_offsets_mapping=True, 
                                                       padding="max_length", max_length=max_tok_len, return_attention_mask=True, return_tensors='pt')
                tok_len = tokenized['input_ids'].size()[1]
                count = 0
                while tok_len > max_tok_len: 
#                     print('Unacceptable token length {} > {}.\n File {}, text {}'.format(tok_len, max_tok_len, sample['fname'], sample['text']))
                    per_throw = min(1 - max_tok_len/tok_len + 0.2, 0.8)
                    sentences, pos = self.sent_trunc(sentences, pos, per_throw) 
                    tokenized = self.tokenizer.encode_plus(sentences, add_special_tokens = True, truncation= False, return_offsets_mapping=True, 
                                                           padding="max_length", max_length=max_tok_len, return_attention_mask=True, return_tensors='pt')
                    tok_len = tokenized['input_ids'].size()[1]
                    count +=1
#                     print('tok_len: ',tok_len, 'count:', count)
                offsets = tokenized["offset_mapping"].squeeze() 
                m_tok = self.find_tokens(pos, offsets,sentences) #2 cases where it doesnt work. we should work with tokens
#                 out_tok = [m_tok[0] - 1]
                self.data.append([labels,  ident,  m_tok, tokenized['input_ids'].squeeze(), 
                                  tokenized['attention_mask'].squeeze(), 
                                  tokenized['token_type_ids'].squeeze()])

    def sent_trunc(self, sentences, pos, per):
        sent_len = len(sentences)
        win = math.ceil(((1- per) * sent_len)/2)
        st = max(pos[0] - win,0)
        en = min(pos[1]+win,sent_len)
        return sentences[st:en], (pos[0] - st, pos[1] - st)
        
    def find_tokens(self, pos, offsets,sentences):
        i, first, last = 0, -1, -1
        error = False
        while i<max_tok_len:
            if offsets[i][0] >= pos[0]:
                first = i
                break
            else:
                i +=1
        while i<max_tok_len:
            if offsets[i][1] >= pos[1]:
                last = i
                break
            else:
                i +=1
        if first ==-1 or last==-1 or i>=max_tok_len:
            print("Error first=%d last=%d i=%d",first,last,i)
        return first,last
    ## ~  
 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return_list = self.data[index]
        return return_list


class Collates():
    def __init__(self, batch_first=True):
        self.batch_first = batch_first

    def _collate(self, data):
        """
        allow dynamic padding based on the current batch
        """
        data = list(zip(*data))
        label, names = data[:2]
#         bs = sum(data[4], [])
        labels = torch.from_numpy(np.stack(label)).long()
        token_out = data[2]
        token_out = torch.from_numpy(np.stack(token_out)).long()
        bert_batch_seqs = pad_sequence(data[3], batch_first=True)
        bert_batch_mask = pad_sequence(data[4], batch_first=True)
        bert_batch_type = pad_sequence(data[5], batch_first=True)
        output = {'labels': labels, 'names': names, 'input_ids': bert_batch_seqs,
                  'mask':bert_batch_mask, 'type':bert_batch_type, 'tok_out': token_out}
        return output

    def __call__(self, batch):
        return self._collate(batch)

                    ## TESTING AND DEBUGGING
#  run individually as: 
#  python datasets.py --config ../configs/local.yaml 
from torch.utils.data import DataLoader
from os.path import isfile, join
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from helpers.io import load_config
import argparse
from time import sleep
def main(args):
    config = load_config(args.config)
#     tokenizer = PreTrainedTokenizerFast.from_pretrained('bert-base-uncased') ##P
#     tokenizer.pad_token = "[PAD]" ##P
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') ##P
    
    train_data_ = BertDataset(config['train_data'], config['max_sen_len'], mode='train', 
                 tokenizer = tokenizer, dummy = False)
    print('Train data:',len(train_data_))
    train_loader_ = DataLoader(train_data_, batch_size=config['batch_size'], 
                               shuffle=True, collate_fn=Collates(), num_workers=0)
    dev_data_ = BertDataset(config['dev_data'], config['max_sen_len'], mode='dev',
                          tokenizer = tokenizer, dummy = False)
    print('Dev data:', len(dev_data_))
    dev_loader_ = DataLoader(dataset=dev_data_, batch_size=config['batch_size'],
                           shuffle=True, collate_fn=Collates(), num_workers=0)
    ### Testing the quality of the data
    for batch_idx, batch in enumerate(train_loader_):
        print_batch(batch, tokenizer)
        sleep(10)
        
def print_batch(batch, tokenizer):
    print('Batch with {} samples \nNames of entities {}\nPositions of tokens {}'.format(len(batch['names']), batch['names'], batch['tok_out']))
    tokens, sentences = [], []
    for i in range(len(batch['names'])):
        ids = batch['input_ids'][i][batch['tok_out'][i][0]:batch['tok_out'][i][1]+1]
        tokens.append(tokenizer.convert_ids_to_tokens(ids))
        st_id = batch['input_ids'][i][batch['tok_out'][i][0] -1]
        en_id = batch['input_ids'][i][batch['tok_out'][i][1] +1]
        tok_out = tokenizer.convert_ids_to_tokens([st_id, en_id])
        if (tok_out[0] != '@' or tok_out[1] != '@'):
            print('Error, output token ', tok_out, ' != @')
        sents = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
        sentences.append(''.join(list(filter(lambda tok: tok != '[PAD]', sents))))
    print('Reconstructed output tokens', tokens)
    print('Reconstructed output sentences', sentences)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    main(args)


    