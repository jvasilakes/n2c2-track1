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
#         self.lang_encoder = BertModel.from_pretrained('bert-base-uncased') # hidden_dropout_prob=config['input_dropout']) 

        self.lang_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
        #attention_probs_dropout_prob =  config['output_dropout']  
        if config['freeze_pretrained']:
            for param in self.lang_encoder.parameters():
                param.requires_grad = False
        
        self.shared = nn.Sequential(
            nn.Linear(2*config['enc_dim'], config['enc_dim']), ##This may change
            nn.ReLU(),
            nn.Dropout(p=config['dropout']))
        self.event_classifier = nn.Linear(config['enc_dim'], len(vocabs['events']))
        self.action_classifier = nn.Linear(config['enc_dim'], len(vocabs['actions']))
    # task loss
        self.sigm = nn.Sigmoid()
        self.loss_fnt = nn.BCEWithLogitsLoss()
#         self.loss_fnt = nn.BCELoss()
#         self.loss_fnt = nn.CrossEntropyLoss()
        

    def specific_tokens(self, enc_seq, tokens_ent):
        idx = torch.arange(enc_seq.size(0)).to(self.device)
        st_token = tokens_ent[:,0] -1 #will it work?
        st_embs = enc_seq[idx, st_token]
        en_token = tokens_ent[:,1] + 1 #will it work?
        en_embs = enc_seq[idx, en_token]
        return torch.cat((st_embs, en_embs), dim=-1)
        
        

    def forward(self, batch):
        ## P
        bert_batch = {'input_ids':batch['input_ids'], 'attention_mask':batch['mask'], 
                      'token_type_ids': batch['type'] }
        tokens_ent = batch['tok_out']
        outputs = self.lang_encoder(**bert_batch)
        enc_out = outputs['last_hidden_state']
#         enc_out = outputs['pooler_output'] ## [CLS] -> sentence encoding
#         h = self.average_tokens(enc_out, tokens_ent)  # contextualised representations of args
        h = self.specific_tokens(enc_out, tokens_ent)
        shared = self.shared(h) 
        event_logits = self.event_classifier(shared) # float() ??
        action_logits = self.action_classifier(shared)
        outputs = (self.sigm(event_logits),self.sigm(action_logits),)
#         if batch['labels'] is not None:
        loss_event = self.loss_fnt(event_logits, batch['elabels'].float())
        ## For action loss we have to think what negative samples to present
        true_dispo = batch['elabels'][:, 1] ==1 ##Disposition
        pred_dispo = outputs[0][:,1] > self.config['threshold'] ##Pred disposition
        indxs = true_dispo
#         loss_action = self.loss_fnt(action_logits, batch['alabels'].float())
        if torch.sum(indxs) > 0 :
            loss_action = self.loss_fnt(action_logits[indxs], batch['alabels'][indxs].float())
        else:
            loss_action = torch.zeros(1)[0].to(self.device)
        outputs = (loss_event,loss_action,) + outputs

        return outputs