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
    
    def add_no_overlap(self, sents, ord_dict, idx):
        keys = list(ord_dict.keys())
        new_sents = sents
        offset = 0
        cur_st, cur_en = keys[0]
        cur_symb = ord_dict[(cur_st, cur_en)]
        for i in range(1, len(ord_dict)):
            st , en = keys[i]
            symb = ord_dict[(st,en)]
            if st < cur_en: #There is overlap with next
                if symb=='@': #keep next
#                     print('Overlap between verb:<%s> and ent:<%s> id:%s' % (sents[cur_st:cur_en], sents[st:en], idx))
                    cur_st, cur_en, cur_symb = st, en, symb
                elif cur_symb!='@': #keep next
                    print('Overlap between verbs?')
            elif st == cur_en + 1 and symb!='@' and cur_symb!='@':
                cur_en = en
                
            else: ## add to sent
                new_st, new_en = cur_st+offset, cur_en+offset
                new_sents = new_sents[:new_st] + cur_symb+ ' '+ \
                        new_sents[new_st:new_en] + ' '+cur_symb + new_sents[new_en:]
                if cur_symb =='@':
                    new_pos = (new_st+2, new_en+2)
                cur_st, cur_en, cur_symb = st, en, symb
                offset +=4
                
        new_st, new_en = cur_st+offset, cur_en+offset
        new_sents = new_sents[:new_st] + cur_symb+ ' '+ \
                new_sents[new_st:new_en] + ' '+cur_symb + new_sents[new_en:]
        if cur_symb =='@':
            new_pos = (new_st+2, new_en+2)
        return new_sents, new_pos
                
                   
    
 
    def __init__(self, path, max_sen_len=None, mode='train', 
                 tokenizer = None, verbs = False, dummy = False):
        super().__init__()
        self.max_sen_len = max_sen_len
        self.mode = mode
        self.data = []
        self.tokenizer = tokenizer
        self.event_vocab = {'NoDisposition':0,'Disposition':1, 'Undetermined':2}
        self.ievent_vocab = {v: k for k, v in self.event_vocab.items()}
        self.action_vocab = {'Start':0,'Stop':1,'Increase':2,'Decrease':3, 'OtherChange':4, 'UniqueDose':5, 'Unknown':6}
        self.iaction_vocab = {v: k for k, v in self.action_vocab.items()}
        with open(path) as infile:
            for item in tqdm(infile, desc='Loading ' + self.mode.upper()):
                sample = json.loads(item)
                ## {text: ..., trig:{'s','e','name'}, events:[NoDispotion, Dispotition], fname,
                ##  verbs: [{'t','st','en'}], action:{}}
                sentences = sample['text']
                orig_pos = sample['trig']['old_pos']
                pos = (sample['trig']['s'], sample['trig']['e'])
                ident =  sample['trig']['name'] +'/'+ sample['fname']
                ## Adding special tokens
                if verbs:
                    to_add = { (verb['st'],verb['en']): '#' for verb in sample['verbs']} 
                else: 
                    to_add = {}
                to_add[pos] = '@'
                od = OrderedDict(sorted(to_add.items()))
                sentences, pos = self.add_no_overlap(sentences, od, ident)
                event_labels = np.zeros((len(self.event_vocab),), 'i')
                for ev in sample['events']: # Multi label
                    event_labels[self.event_vocab[ev]] = 1
#                 labels = self.event_vocab[sample['events'][0]] # Single label
                
                action_labels = np.zeros((len(self.action_vocab),), 'i')
                for action in sample['actions']: # Multi label
                    action_labels[self.action_vocab[action]] = 1
                if np.sum(action_labels)>0 and event_labels[self.event_vocab['Disposition']]==0:
                    print('Big error, we have action label but no Disposition event', ident)
                elif np.sum(action_labels)==0 and event_labels[self.event_vocab['Disposition']]==1:
                    print('Big error, we have Disposition but no action label', ident)
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
                self.data.append([event_labels, action_labels,  ident, orig_pos, m_tok, tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()])

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
        return_list = [index] + self.data[index]
        return return_list


class Collates():
    def __init__(self, batch_first=True):
        self.batch_first = batch_first

    def _collate(self, data):
        """
        allow dynamic padding based on the current batch
        """
        data = list(zip(*data))
        indx, elabel, alabel, names, opos = data[:5]
        indx = torch.from_numpy(np.stack(indx)).long()
        elabels = torch.from_numpy(np.stack(elabel)).long()
        alabels = torch.from_numpy(np.stack(alabel)).long()
        token_out = data[5]
        token_out = torch.from_numpy(np.stack(token_out)).long()
        bert_batch_seqs = pad_sequence(data[6], batch_first=True)
        bert_batch_mask = pad_sequence(data[7], batch_first=True)
        bert_batch_type = pad_sequence(data[8], batch_first=True)
        output = {'indxs':indx, 'elabels':elabels, 'alabels':alabels, 'names':names, 
                  'old_pos':opos, 'input_ids':bert_batch_seqs,
                  'mask':bert_batch_mask, 'type':bert_batch_type, 'tok_out':token_out}
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
    config['train_data'] = args.data + 'train_data.txt'
    config['dev_data'] = args.data + 'dev_data.txt'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') ##P
    
    train_data_ = BertDataset(config['train_data'], config['max_sen_len'], mode='train', 
                 tokenizer = tokenizer, verbs=config['verbs'], dummy = False)
    print('Train data:',len(train_data_))
    train_loader_ = DataLoader(train_data_, batch_size=config['batch_size'], 
                               shuffle=True, collate_fn=Collates(), num_workers=0)
    dev_data_ = BertDataset(config['dev_data'], config['max_sen_len'], mode='dev',
                          tokenizer = tokenizer,verbs=config['verbs'], dummy = False)
    print('Dev data:', len(dev_data_))
    dev_loader_ = DataLoader(dataset=dev_data_, batch_size=config['batch_size'],
                           shuffle=True, collate_fn=Collates(), num_workers=0)
    ### Testing the quality of the data
    for batch_idx, batch in enumerate(train_loader_):
        print_batch(batch, tokenizer)
        sleep(10)
#     print(dev_loader_.dataset[1][3])
        
def print_batch(batch, tokenizer):
    print('Batch with {} samples \nNames of entities {} with opos {}'.format(len(batch['names']), batch['names'],batch['old_pos']))
    tokens, sentences = [], []
    for i in range(len(batch['names'])):
        ids = batch['input_ids'][i][batch['tok_out'][i][0]:batch['tok_out'][i][1]+1]
        tokens.append(tokenizer.convert_ids_to_tokens(ids))
        st_id = batch['input_ids'][i][batch['tok_out'][i][0] -1]
        en_id = batch['input_ids'][i][batch['tok_out'][i][1] +1]
        tok_out = tokenizer.convert_ids_to_tokens([st_id, en_id])
        if (tok_out[0] != '@' or tok_out[1] != '@'):
            print('Error, output token ', tok_out, ' != @')
        toks = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
        sentences.append(tok_to_sent(toks))
#     print('Reconstructed output tokens', tokens)
    print('Reconstructed output sentences', '\n'.join(sentences))
    sleep(20)
    
def tok_to_sent(tokens):
    cl_tok = list(filter(lambda tok: tok not in ['[PAD]', '[CLS]','[SEP]'], tokens))
    new_tok, cur_tok = [], cl_tok[0]
    for tok in cl_tok[1:]:
        if len(tok)>1 and tok[0:2]=='##':
            cur_tok += tok[2:]
        else:
            new_tok.append(cur_tok)
            cur_tok = tok
    new_tok.append(cur_tok)
    return ' '.join(new_tok)
        
                  
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--data', type=str)
    args = parser.parse_args()
    main(args)


    