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
        if config['freeze_pretrained']:
            for param in self.lang_encoder.parameters():
                param.requires_grad = False
        
        self.shared = nn.Sequential(
            nn.Linear(config['intermediate_dim'], config['hidden_dim']), ##This may change
            nn.ReLU(),
            nn.Dropout(p=config['dropout']))
        self.event_classifier = nn.Linear(config['hidden_dim'], len(vocabs['events']))
        self.action_classifier = nn.Linear(config['hidden_dim'], len(vocabs['actions']))
    # task loss
        self.sigm = nn.Sigmoid()
        self.loss_fnt = nn.BCEWithLogitsLoss()
        

    def specific_tokens(self, enc_seq, tokens_ent):
        idx = torch.arange(enc_seq.size(0)).to(self.device)
        st_token = tokens_ent[:,0] -1 #will it work?
        st_embs = enc_seq[idx, st_token]
        en_token = tokens_ent[:,1] + 1 #will it work?
        en_embs = enc_seq[idx, en_token]
        return torch.cat((st_embs, en_embs), dim=-1)
    
    def pooling(self, enc_seq, verb_count):
        max_pairs = self.config['max_pair_len']
        s_levi = self.config['max_tok_len'] - 2* max_pairs
        w_ids = torch.arange(0, enc_seq.shape[1]).repeat(enc_seq.shape[0],1).to(self.device)
        verbs = verb_count.unsqueeze(1)
        ss_levi = torch.tensor(s_levi).repeat(verbs.shape).to(self.device)
        se_levi =  ss_levi + verbs
        es_levi = ss_levi + max_pairs
        ee_levi = es_levi + verbs
        _, ss, se, es, ee = torch.broadcast_tensors(w_ids, ss_levi, se_levi, es_levi, ee_levi)
        indx_s = torch.logical_and(w_ids >=ss, w_ids < se)
        indx_e = torch.logical_and(w_ids >= es, w_ids <ee)
        indx = torch.logical_or(indx_s,indx_e)
        ## Do whatever pooling you want with the booleab with
                    # mean 
#         return self.mean_pool(indx, enc_seq)
                    # mean 2
#         emb_s = self.mean_pool(indx_s, enc_seq)  
#         emb_e = self.mean_pool(indx_e, enc_seq)
#         return torch.cat((emb_s, emb_e), dim=-1)
                    # max 
        return self.max_pool(indx, enc_seq)
                    # max 2
#         emb_s = self.max_pool(indx_s, enc_seq)  
#         emb_e = self.max_pool(indx_e, enc_seq)
#         return torch.cat((emb_s, emb_e), dim=-1) 
              
    def mean_pool(self, indx, enc_seq):
        numerator = torch.matmul(indx.float().unsqueeze(1), enc_seq)
        denominator = torch.sum(indx.float().unsqueeze(1), dim=-1).unsqueeze(-1)
        zeros_indx = denominator == 0
        denominator[zeros_indx] += 1
        return torch.div(numerator, denominator).squeeze(1) 
    
    def max_pool(self, indx, enc_seq):
        x = enc_seq.clone()
        x.masked_fill_(~indx.unsqueeze(-1), 0)
        return torch.max(x,1)[0]
    
    def forward(self, batch):
        ## P
        bert_batch = {'input_ids':batch['input_ids'], 'attention_mask':batch['mask'], 
                      'token_type_ids': batch['type'], 'position_ids': batch['pos_ids'] }
        tokens_ent = batch['tok_out']
        outputs = self.lang_encoder(**bert_batch)
        enc_out = outputs['last_hidden_state']
#         enc_out = outputs['pooler_output'] ## [CLS] -> sentence encoding
        h1 = self.specific_tokens(enc_out, tokens_ent)
        h2 = self.pooling(enc_out, batch['verb_counts'])
        h = torch.cat((h1, h2), dim=-1)
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
        if torch.sum(indxs) > 0 :
            loss_action = self.loss_fnt(alogits[indxs], alabels[indxs].float())
        else:
            loss_action = torch.zeros(1)[0].to(self.device)
        
        return loss_event, loss_action
    
#     def action_first(self, elogits, elabels, alogits, alabels):
#         loss_action = self.loss_fnt(alogits, alabels.float())
#         tmp = elabels#[:, :2]
#         loss_event = self.loss_fnt(elogits, tmp.float())
        
#         return loss_event, loss_action