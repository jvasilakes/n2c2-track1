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
import itertools

class BertDataset(Dataset):

    
 
    def __init__(self, path, mode='train', tokenizer = None, use_verbs = False, config=None):
        super().__init__()
        self.mode = mode
        self.data = []
        self.tokenizer = tokenizer
        self.event_vocab = {'NoDisposition':0, 'Undetermined':1, 'Disposition':2}
        self.ievent_vocab = {v: k for k, v in self.event_vocab.items()}
        self.action_vocab = {'Start':0,'Stop':1,'Increase':2,'Decrease':3, 'OtherChange':4, 'UniqueDose':5, 'Unknown':6}
        self.iaction_vocab = {v: k for k, v in self.action_vocab.items()}
        #
        self.max_seq_len = config['max_tok_len']
        self.max_pair_len = config['max_pair_len'] if use_verbs else 0
        self.max_ent_len = self.max_pair_len*2
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
                to_add = {(verb['st'],verb['en']): verb['t'] for verb in sample['verbs']} if use_verbs else {}
#                 to_add[pos] = '@'
                od = OrderedDict(sorted(to_add.items()))
                #sentences, pos = self.add_no_overlap(sentences, od, ident)
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
                ## Tokenizing
                words = sentences.split(' ')
                tokens = [self.tokenizer.tokenize(w) for w in words]
                subwords = [w for li in tokens for w in li]
                # maxL = max(maxL, len(subwords))
                subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)])) # [0, 1, 1, 2, 3, 3, ..]
                token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens)) # [0, 1, 3, 4,..] caution error if sents start with ' '
                st_ent = token2subword[pos[0]] 
                en_ent = token2subword[pos[1]] 
                ent_len = en_ent - st_ent
                left_len = st_ent
                right_len = len(subwords) - en_ent 
                sent_len = right_len + left_len + ent_len
                if sent_len > self.max_seq_len -4:
                    to_cut = sent_len - self.max_seq_len +4 
                    left_len, right_len = per_split(left_len, right_len, to_cut)
                offset_left = st_ent-left_len ## offset in sub tokens
                offset_right = en_ent+right_len
#                 target_tokens = subwords[st_ent-left_len:st_ent] + ['[unused0]'] + subwords[st_ent:en_ent] + ['[unused1]'] + subwords[en_ent:en_ent+right_len]
                target_tokens = subwords[st_ent-left_len:st_ent] + [config['ent_tok0']] + subwords[st_ent:en_ent] + [config['ent_tok1']] + subwords[en_ent:en_ent+right_len]
                target_tokens = [self.tokenizer.cls_token] + target_tokens[:self.max_seq_len-4] + [self.tokenizer.sep_token]
                assert(len(target_tokens) <= self.max_seq_len)
                verbs, filtered = [], 0
                for st, en in to_add.keys():
                    tok_st = token2subword[st] 
                    tok_en = token2subword[en]
                    if tok_st < offset_left or tok_en > offset_right :
                        filtered +=1
                        continue
                    mini_offset = 1
                    if tok_st >= en_ent:
                        mini_offset +=2
                    elif tok_en <= en_ent:
                        mini_offset +=0
                    else:   # there is overlap with entity
                        print('Verb {} is overlapping with entity fname {}'.format(to_add[(st,en)], ident))
                        continue 
                    new_st = tok_st - offset_left + mini_offset
                    new_en = tok_en -offset_left + mini_offset
                    verbs.append((new_st,new_en))
#                     verbs.append(target_tokens[new_st:new_en])
                m_tok = (left_len+2, left_len+2+ent_len)
#                 if target_tokens[m_tok[0]:m_tok[1]] != sample['trig']['name']:
#                     print('Ent {} != name {}'.format(target_tokens[m_tok[0]:m_tok[1]], sample['trig']['name']))
                ## Converting to ids
                input_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
                L = len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(input_ids))

                attention_mask = torch.zeros((self.max_ent_len+self.max_seq_len, self.max_ent_len+self.max_seq_len), dtype=torch.int64)
                attention_mask[:L, :L] = 1
                ## maybe shuffle verbs before
                verbs = verbs[:self.max_pair_len]  ## will be 0 if verbs not allowed
                verb_count = len(verbs)
                input_ids = input_ids + [3] * (len(verbs)) + [self.tokenizer.pad_token_id] * (self.max_pair_len - len(verbs))
                input_ids = input_ids + [4] * (len(verbs)) + [self.tokenizer.pad_token_id] * (self.max_pair_len - len(verbs)) # for debug 
                
                position_ids = list(range(self.max_seq_len)) + [0] * self.max_ent_len 
                token_type_ids = [0] * self.max_seq_len  + [0] * self.max_ent_len 
                
                for i, pos in enumerate(verbs):
                    w1 = i + self.max_seq_len
                    w2 = w1 + self.max_pair_len
                    position_ids[w1] = pos[0]
                    position_ids[w2] = pos[1] -1 # I think -1 necessary to point to true end
                    
                    for xx in [w1,w2]:
                        for yy in [w1,w2]:
                            attention_mask[xx,yy] =1
                        attention_mask[xx, :L] = 1
                    
                ## check if you need att_left or att_right
                
                self.data.append([event_labels, action_labels,  ident, orig_pos, m_tok, 
                                  verb_count, torch.tensor(input_ids), attention_mask,
                                  torch.tensor(token_type_ids), torch.tensor(position_ids)])

 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return_list = [index] + self.data[index]
        return return_list
    

def per_split(left_len, right_len, to_cut):
    orig_left, orig_right =  left_len, right_len
    if right_len>0 and left_len>0:
        per = right_len / (right_len+left_len) 
        right_cut = math.ceil(per*to_cut)
        left_cut = math.ceil((1-per)*to_cut)
    elif right_len==0:
        left_cut, right_cut = to_cut, 0
    else:
        right_cut, left_cut = to_cut, 0
    
    right_len -= right_cut
    left_len -= left_cut
    if right_len<0 or left_len<0:
        print('Huge error during split, (orig, cut, after) left ({},{},{}), right ({},{},{})'.format(
            orig_left, left_cut, left_len, orig_right, right_cut, right_len ))
        exit()
    return left_len, right_len

class Collates():
    def __init__(self, batch_first=True):
        self.batch_first = batch_first

    def _collate(self, data):
        """
        allow dynamic padding based on the current batch
        """
        data = list(zip(*data))
        indx, elabel, alabel, names, opos, token_out, verb_counts  = data[:7]
        indx = torch.from_numpy(np.stack(indx)).long()
        elabels = torch.from_numpy(np.stack(elabel)).long()
        alabels = torch.from_numpy(np.stack(alabel)).long()
        token_out = torch.from_numpy(np.stack(token_out)).long()
        verb_counts = torch.from_numpy(np.stack(verb_counts)).long()
        bert_batch_seqs = pad_sequence(data[7], batch_first=True)
        bert_batch_mask = pad_sequence(data[8], batch_first=True)
        bert_batch_type = pad_sequence(data[9], batch_first=True)
        bert_batch_pos = pad_sequence(data[10], batch_first=True)
        output = {'indxs':indx, 'elabels':elabels, 'alabels':alabels, 'names':names, 
                  'old_pos':opos, 'verb_counts': verb_counts, 
                  'input_ids':bert_batch_seqs,'mask':bert_batch_mask,
                  'type':bert_batch_type, 'tok_out':token_out, 'pos_ids':bert_batch_pos}
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
    config['bert'] = args.bert
    config['use_verbs'] = args.use_verbs
    
    if config['bert'] == 'base':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') ##P
    elif config['bert'] =='clinical':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    else:
        print('Invalid bert model')
        exit(1)
    
    train_data_ = BertDataset(config['train_data'], mode='train', tokenizer = tokenizer, 
                              use_verbs=config['use_verbs'], config=config)
    print('Train data:',len(train_data_))
    train_loader_ = DataLoader(train_data_, batch_size=config['batch_size'], 
                               shuffle=True, collate_fn=Collates(), num_workers=0)
    dev_data_ = BertDataset(config['dev_data'],  mode='dev',tokenizer = tokenizer, 
                            use_verbs=config['use_verbs'], config=config)
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
        en_id = batch['input_ids'][i][batch['tok_out'][i][1] ]
        tok_out = tokenizer.convert_ids_to_tokens([st_id, en_id])
        if (tok_out[0] != '[unused0]' or tok_out[1] != '[unused1]'):
            print('Error, output token ', tok_out, ' != @')
        toks = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
        sentences.append(tok_to_sent(toks))
#     print('Reconstructed output tokens', tokens)
    print('Reconstructed output sentences', '\n'.join(sentences))
    sleep(20)
    
def tok_to_sent(tokens):
    cl_tok = list(filter(lambda tok: tok not in [ '[PAD]','[CLS]','[SEP]'], tokens))
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
    parser.add_argument('--bert', type=str, choices=['base', 'clinical'])
    parser.add_argument('--use_verbs', action='store_true', help='Use verbs')
    args = parser.parse_args()
    main(args)


    