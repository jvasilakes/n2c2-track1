1#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/02/2022

author: panos
"""
import numpy as np
import torch
from torch import nn, torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, AutoModel
from transformers import logging
from pooler import TokenPooler
# logging.set_verbosity_warning()
# logging.set_verbosity_error()

class BlankNet(nn.Module):
    """
    Model architecture
    """
    def __init__(self, config, vocabs, device):
        """
        Args:
            params (dict): model parameters
            vocab (class): class structure with word vocabulary
            device (int): gpu or cpu (-1) device
        """
        super().__init__()
        self.device = device
        self.config = config
        self.hidden_dim = config['hidden_dim']
        logging.set_verbosity_warning()
        logging.set_verbosity_error()
        
        if config['bert'] == 'base':
            self.lang_encoder = BertModel.from_pretrained('bert-base-uncased') 
        elif config['bert'] =='clinical':
            self.lang_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif config['bert'] =='blue':
            self.lang_encoder = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
        else:
            print('Invalid BERT model')
            exit(1)

        
    
        #attention_probs_dropout_prob =  config['output_dropout']
        
        self.shared = nn.Sequential(
            nn.Linear(config['intermediate_dim'], config['hidden_dim']), ##This may change
            nn.ReLU(),
            nn.Dropout(p=config['dropout']))
        self.event_classifier = nn.Linear(config['hidden_dim'], len(vocabs['events']))
        self.action_classifier = nn.Linear(config['hidden_dim'], len(vocabs['actions']))
    # task loss
        self.sigm = nn.Sigmoid()
        self.loss_fnt = nn.BCEWithLogitsLoss()
        self.pooler = TokenPooler(config['pool_fn'], config, device)
        

    def specific_tokens(self, enc_seq, tokens_ent):
        idx = torch.arange(enc_seq.size(0)).to(self.device)
        st_token = tokens_ent[:, 0] - 1
        st_embs = enc_seq[idx, st_token]
        en_token = tokens_ent[:, 1] + 1
        en_embs = enc_seq[idx, en_token]
        return torch.cat((st_embs, en_embs), dim=-1)

    
    def forward(self, batch):
        ## P
        bert_batch = {'input_ids':batch['input_ids'], 'attention_mask':batch['mask'], 
                      'token_type_ids': batch['type'], 'position_ids': batch['pos_ids'] }
        tokens_ent = batch['tok_out']
        outputs = self.lang_encoder(**bert_batch)
        enc_out = outputs['last_hidden_state']
#         enc_out = outputs['pooler_output'] ## [CLS] -> sentence encoding
        h1 = self.specific_tokens(enc_out, tokens_ent)
        # h2 = self.pooling(enc_out, batch['verb_counts'])
        if self.config['use_verbs']:
            h2 = self.pooler(enc_out, batch['verb_counts'], h1[:, :self.hidden_dim])
            h = torch.cat((h1, h2), dim=-1)
        else:
            h = h1

        shared = self.shared(h) 
        event_logits = self.event_classifier(shared)
        action_logits = self.action_classifier(shared)
        outputs = (self.sigm(event_logits),self.sigm(action_logits),)
        ### 2 different approaches
        loss_event, loss_action = self.event_first(event_logits,batch['elabels'],
                                              action_logits, batch['alabels'], outputs)
#         loss_event, loss_action = self.action_first(event_logits,batch['elabels'], 
#                                               action_logits, batch['alabels'])

        outputs = (loss_event,loss_action,) + outputs

        return outputs
    
    def event_first(self, elogits, elabels, alogits, alabels,outputs):
        loss_event = self.loss_fnt(elogits, elabels.float())
        ## For action loss we have to think what negative samples to present
        true_dispo = elabels[:, 2] ==1 ##Disposition
        pred_dispo = outputs[0][:,2] > self.config['threshold'] ##Pred disposition
        indxs = true_dispo
        if torch.sum(indxs) > 0 and not self.config['no_mtl']:
            loss_action = self.loss_fnt(alogits[indxs], alabels[indxs].float())
        else:
            loss_action = torch.zeros(1)[0].to(self.device)
        
        return loss_event, loss_action
    
#     def action_first(self, elogits, elabels, alogits, alabels):
#         loss_action = self.loss_fnt(alogits, alabels.float())
#         tmp = elabels#[:, :2]
#         loss_event = self.loss_fnt(elogits, tmp.float())
        
#         return loss_event, loss_action