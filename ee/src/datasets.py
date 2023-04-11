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

    def __init__(self, path, mode='train', tokenizer=None, use_verbs=False, config=None):
        super().__init__()
        self.mode = mode
        self.data = []
        self.tokenizer = tokenizer
        self.event_vocab = {'NoDisposition': 0, 'Undetermined': 1, 'Disposition': 2}
        self.ievent_vocab = {v: k for k, v in self.event_vocab.items()}
        self.action_vocab = {'Start': 0, 'Stop': 1, 'Increase': 2, 'Decrease': 3, 'OtherChange': 4, 'UniqueDose': 5,
                             'Unknown': 6}
        self.iaction_vocab = {v: k for k, v in self.action_vocab.items()}
        #
        self.max_seq_len = config['max_tok_len']
        self.max_verbs = config['max_verbs'] if use_verbs else 0
        self.max_mark_len = self.max_verbs if config['single_marker'] else self.max_verbs * 2 # change here
        with open(path) as infile:
            for item in tqdm(infile, desc='Loading ' + self.mode.upper()):
                sample = json.loads(item)
                ## {text: ..., trig:{'s','e','name'}, events:[NoDispotion, Dispotition], fname,
                ##  verbs: [{'t','st','en'}], action:{}}
                sentences = sample['text']
                orig_pos = sample['trig']['orig_off']  # ['old_pos']

                pos = (sample['trig']['s'], sample['trig']['e'])
                # ident = sample['trig']['name'] +'/'+ sample['fname']
                ident = sample['trig']['orig_name'] + '/' + sample['fname']
                ## Adding special tokens
                to_add = {(verb['st'], verb['en']): (verb['t'], verb['type']) for verb in
                          sample['verbs']} if use_verbs else {} # and config['use_markers'] else {}
                #                 to_add[pos] = '@'
                od = {verb[0] for verb in OrderedDict(sorted(to_add.items()))}  # this is wrong now
                # sentences, pos = self.add_no_overlap(sentences, od, ident)
                event_labels = np.zeros((len(self.event_vocab),), 'i')
                for ev in sample['events']:  # Multi label
                    event_labels[self.event_vocab[ev]] = 1
                #                 labels = self.event_vocab[sample['events'][0]] # Single label

                action_labels = np.zeros((len(self.action_vocab),), 'i')
                for action in sample['actions']:  # Multi label
                    action_labels[self.action_vocab[action]] = 1
                if np.sum(action_labels) > 0 and event_labels[self.event_vocab['Disposition']] == 0:
                    print('Big error, we have action label but no Disposition event', ident)
                elif np.sum(action_labels) == 0 and event_labels[self.event_vocab['Disposition']] == 1:
                    print('Big error, we have Disposition but no action label', ident)
                ## Tokenizing
                words = sentences.split(' ')
                tokens = [self.tokenizer.tokenize(w) for w in words]
                subwords = [w for li in tokens for w in li]
                # maxL = max(maxL, len(subwords))
                subword2token = list(
                    itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))  # [0, 1, 1, 2, 3, 3, ..]
                token2subword = [0] + list(itertools.accumulate(
                    len(li) for li in tokens))  # [0, 1, 3, 4,..] caution error if sents start with ' '
                st_ent = token2subword[pos[0]]
                en_ent = token2subword[pos[1]]
                ent_len = en_ent - st_ent
                left_len = st_ent
                right_len = len(subwords) - en_ent
                sent_len = right_len + left_len + ent_len
                if sent_len > self.max_seq_len - 4:
                    to_cut = sent_len - self.max_seq_len + 4
                    left_len, right_len = per_split(left_len, right_len, to_cut)
                offset_left = st_ent - left_len  ## offset in sub tokens
                offset_right = en_ent + right_len
                #                 target_tokens = subwords[st_ent-left_len:st_ent] + ['[unused0]'] + subwords[st_ent:en_ent] + ['[unused1]'] + subwords[en_ent:en_ent+right_len]
                # if config['use_markers']:
                target_tokens = subwords[st_ent - left_len:st_ent] + [config['ent_tok0']] + subwords[st_ent:en_ent] + [
                                    config['ent_tok1']] + subwords[en_ent:en_ent + right_len]
                # else:
                #     target_tokens = subwords[st_ent - left_len:st_ent] + subwords[st_ent:en_ent] + subwords[
                #                                                                                    en_ent:en_ent + right_len]
                target_tokens = [self.tokenizer.cls_token] + target_tokens[:self.max_seq_len - 4] + [
                    self.tokenizer.sep_token]
                assert (len(target_tokens) <= self.max_seq_len)
                verbs, filtered = [], 0
                verb_s, verb_e = [], []
                for st, en in to_add.keys():
                    tok_st = token2subword[st]
                    tok_en = token2subword[en]
                    if tok_st < offset_left or tok_en > offset_right:
                        filtered += 1
                        continue
                    mini_offset = 1
                    if tok_st >= en_ent:
                        mini_offset += 2
                    elif tok_en <= en_ent:
                        mini_offset += 0
                    else:  # there is overlap with entity
                        print('Verb {} is overlapping with entity fname {}'.format(to_add[(st, en)][0], ident))
                        continue
                    new_st = tok_st - offset_left + mini_offset
                    new_en = tok_en - offset_left + mini_offset

                    typ_s, typ_e = verb_tag(to_add[(st, en)][1], config['use_verb_categories'], config['single_marker'])
                    if config['only_typed_verbs'] and typ_s == 3:  # basically skip the untypped
                        continue
                    verbs.append((new_st, new_en))
                    verb_s.append(typ_s)
                    verb_e.append(typ_e)
                # Add CLS if no verbs
                if len(verbs) == 0 and use_verbs: # and config['use_markers']:
                    verbs.append((0, 1))
                    verb_s.append(3)
                    verb_e.append(4)

                # if config['use_markers']:
                m_tok = (left_len + 2, left_len + 2 + ent_len)
                # else:
                #     m_tok = (left_len + 1, left_len + 1 + ent_len)
                #

                ## sort verbs according to entity distanse
                # if config['sort_verbs']:
                #     sort_indx = sorted(range(len(verbs)), key=lambda k: min(abs(verbs[k][1] - m_tok[0]), abs(m_tok[1] - verbs[k][0])))
                #     verbs = [verbs[i] for i in sort_indx]
                #     verb_s = [verb_s[i] for i in sort_indx]
                #     verb_e = [verb_e[i] for i in sort_indx]
                ## Converting to ids
                input_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
                L = len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(input_ids))

                attention_mask = torch.zeros((self.max_mark_len + self.max_seq_len, self.max_mark_len + self.max_seq_len),
                                             dtype=torch.int64)
                attention_mask[:L, :L] = 1
                ## maybe shuffle verbs before
                verbs = verbs[:self.max_verbs]  ## will be 0 if verbs not allowed
                verb_s = verb_s[:self.max_verbs]  # [3] * (len(verbs))
                verb_e = verb_e[:self.max_verbs]  # [4] * (len(verbs))
                verb_count = len(verbs)

                if config['single_marker']:
                    input_ids = input_ids + verb_s + [self.tokenizer.pad_token_id] * (self.max_verbs - len(verbs))

                else:
                    # ver 1
                    # verb_seq = [x for z in zip(verb_s, verb_e) for x in z]
                    # input_ids = input_ids + verb_seq + [self.tokenizer.pad_token_id] * (2*(self.max_verbs - len(verbs)))
                    # ver 2
                    input_ids = input_ids + verb_s + [self.tokenizer.pad_token_id] * (self.max_verbs - len(verbs))
                    input_ids = input_ids + verb_e + [self.tokenizer.pad_token_id] * (self.max_verbs - len(verbs))

                # let me change it a bit
                position_ids = list(range(self.max_seq_len)) + [0] * self.max_mark_len
                token_type_ids = [0] * self.max_seq_len + [0] * self.max_mark_len

                for i, pos in enumerate(verbs):
                    if config['single_marker']:
                        w1 = self.max_seq_len + i
                        attention_mask[w1, w1] = 1
                        attention_mask[w1, :L] = 1
                        position_ids[w1] = pos[0]
                    else:
                        # ver 1
                        # w1 = self.max_seq_len + i*2
                        # w2 = w1 + 1 # self.max_verbs
                        # ver 2
                        w1 = self.max_seq_len + i
                        w2 = w1 + self.max_verbs
                        #
                        position_ids[w1] = pos[0]
                        position_ids[w2] = pos[1] - 1  # I think -1 necessary to point to true end

                        for xx in [w1, w2]:
                            for yy in [w1, w2]:
                                attention_mask[xx, yy] = 1
                            attention_mask[xx, :L] = 1



                ## check if you need att_left or att_right

                self.data.append([event_labels, action_labels, ident, orig_pos, m_tok,
                                  verb_count, torch.tensor(input_ids), attention_mask,
                                  torch.tensor(token_type_ids), torch.tensor(position_ids)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return_list = [index] + self.data[index]
        return return_list


def verb_tag(tag_list, use_verb_categories, single):
    if not single:
        return (3, 4)
    if not use_verb_categories:
        return (3, 3)
    if re.search("A2.1", tag_list) and len(tag_list) < 30: # best 30
        return (4, 4)
    elif re.search("A2.1", tag_list) and len(tag_list) >= 30 : # 39
        return (5, 5)
    elif re.search("T2", tag_list) and len(tag_list) < 30:
        return (6, 6)
    elif re.search("T2", tag_list) and len(tag_list) >= 30:
        return (7, 7)
    else:
        return (3, 3)


def per_split(left_len, right_len, to_cut):
    orig_left, orig_right = left_len, right_len
    if right_len > 0 and left_len > 0:
        per = right_len / (right_len + left_len)
        right_cut = math.ceil(per * to_cut)
        left_cut = math.ceil((1 - per) * to_cut)
    elif right_len == 0:
        left_cut, right_cut = to_cut, 0
    else:
        right_cut, left_cut = to_cut, 0

    right_len -= right_cut
    left_len -= left_cut
    if right_len < 0 or left_len < 0:
        print('Huge error during split, (orig, cut, after) left ({},{},{}), right ({},{},{})'.format(
            orig_left, left_cut, left_len, orig_right, right_cut, right_len))
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
        indx, elabel, alabel, names, opos, token_out, verb_counts = data[:7]
        indx = torch.from_numpy(np.stack(indx)).long()
        elabels = torch.from_numpy(np.stack(elabel)).long()
        alabels = torch.from_numpy(np.stack(alabel)).long()
        token_out = torch.from_numpy(np.stack(token_out)).long()
        verb_counts = torch.from_numpy(np.stack(verb_counts)).long()
        bert_batch_seqs = pad_sequence(data[7], batch_first=True)
        bert_batch_mask = pad_sequence(data[8], batch_first=True)
        bert_batch_type = pad_sequence(data[9], batch_first=True)
        bert_batch_pos = pad_sequence(data[10], batch_first=True)
        output = {'indxs': indx, 'elabels': elabels, 'alabels': alabels, 'names': names,
                  'old_pos': opos, 'verb_counts': verb_counts,
                  'input_ids': bert_batch_seqs, 'mask': bert_batch_mask,
                  'type': bert_batch_type, 'tok_out': token_out, 'pos_ids': bert_batch_pos}
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
from helpers.io import setup


def main(args):
    config = load_config(args.config)
    setup(args, config)
    val_loader_, dev_data = load_data(config)

    ### Testing the quality of the data
    for batch_idx, batch in enumerate(val_loader_):
        print_batch(batch, dev_data.tokenizer, config)
        sleep(10)


#     print(dev_loader_.dataset[1][3])

def print_batch(batch, tokenizer, config):
    print('Batch with {} samples \nNames of entities {} with opos {}'.format(len(batch['names']), batch['names'],
                                                                             batch['old_pos']))
    tokens, sentences = [], []
    unique_verbs = {}
    for i in range(len(batch['names'])):
        ids = batch['input_ids'][i][batch['tok_out'][i][0]:batch['tok_out'][i][1] + 1]
        # check if marked entity correctly
        tokens.append(tokenizer.convert_ids_to_tokens(ids))
        st_id = batch['input_ids'][i][batch['tok_out'][i][0] - 1]
        en_id = batch['input_ids'][i][batch['tok_out'][i][1]]
        tok_out = tokenizer.convert_ids_to_tokens([st_id, en_id])
        if (tok_out[0] != '[unused0]' or tok_out[1] != '[unused1]'):
            print('Error, output token ', tok_out, ' != @')
        # check the position of the verbs
        position_ids = batch['pos_ids'][i]
        st = config['max_tok_len']
        tags = tokenizer.convert_ids_to_tokens(batch['input_ids'][i][st:])
        verb_tok = []
        # for j in range(st, len(position_ids)):
        #     pos = position_ids[j]
        for j, pos in enumerate(position_ids[st:]):
            vid = batch['input_ids'][i][pos]
            tok = tokenizer.convert_ids_to_tokens([vid])
            if tok != ['[CLS]'] and len(tags[j]) > 0 and tok[0] not in unique_verbs:
                unique_verbs[tok[0]] = tags[j]
                print(tok[0], tags[j])

        #
        toks = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
        sentences.append(tok_to_sent(toks))
        # print(tok_to_sent(toks))
        # print(verb_tok)
    #     print('Reconstructed output tokens', tokens)
    print('Reconstructed output sentences', '\n'.join(sentences))
    sleep(20)


def tok_to_sent(tokens):
    cl_tok = list(filter(lambda tok: tok not in ['[PAD]', '[CLS]', '[SEP]'], tokens))
    new_tok, cur_tok = [], cl_tok[0]
    for tok in cl_tok[1:]:
        if len(tok) > 1 and tok[0:2] == '##':
            cur_tok += tok[2:]
        else:
            new_tok.append(cur_tok)
            cur_tok = tok
    new_tok.append(cur_tok)
    return ' '.join(new_tok)


def load_data(config):
    #     tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    if config['bert'] == 'base':
        tokenizer = AutoTokenizer.from_pretrained('../bert_models/bert-base-uncased')  ##P

    elif config['bert'] == 'clinical':
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    elif config['bert'] == 'blue':
        tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")

    if config['mode'] == 'train':

        dev_data_ = BertDataset(config['dev_data'], mode='dev', tokenizer=tokenizer,
                                use_verbs=config['use_verbs'], config=config)
        print('Dev data:', len(dev_data_))

        dev_loader_ = DataLoader(dataset=dev_data_, batch_size=config['batch_size'],
                                 shuffle=True,
                                 collate_fn=Collates(),
                                 num_workers=0,
                                 drop_last=False)
        return dev_loader_, dev_data_

    else:
        test_data_ = BertDataset(config['test_data'], mode='test', tokenizer=tokenizer,
                                 use_verbs=config['use_verbs'], config=config)
        print('Test data:', len(test_data_))

        test_loader_ = DataLoader(dataset=test_data_, batch_size=config['batch_size'],
                                  shuffle=False,
                                  collate_fn=Collates(),
                                  num_workers=0,
                                  drop_last=False)
        return [], test_loader_, []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, )
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'])
    parser.add_argument('--model_folder', type=str, help='Must include the bert type model specified')
    parser.add_argument('--test_path', type=str, help='Path to test_data.txt file')
    parser.add_argument('--split', type=str)
    parser.add_argument('--bert', type=str, required=True, choices=['base', 'clinical', 'blue'])
    parser.add_argument('--approach', type=str, required=True, choices=['LCM', 'LCM_attention', 'LCM_no_mtl', 'types', 'types_attention', 'baseline', 'baseline_mtl'])
    args = parser.parse_args()
    main(args)
