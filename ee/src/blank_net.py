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
from transformers import BertModel


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
        self.lang_encoder = BertModel.from_pretrained('bert-base-uncased',
                                                      hidden_dropout_prob=config['input_dropout']) 
        #attention_probs_dropout_prob =  config['output_dropout']  
        if config['freeze_pretrained']:
            for param in self.lang_encoder.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(2*config['enc_dim'], config['enc_dim']), ##This may change
            nn.ReLU(),
            nn.Dropout(p=config['output_dropout']),
            nn.Linear(config['enc_dim'], len(vocabs['r_vocab']))
        )
    # task loss
        self.loss_fnt = nn.CrossEntropyLoss()
        
    def average_tokens(self, enc_seq, mentions):
        """
        Merge tokens into mentions;
        Find which tokens belong to a mention (based on start-end ids) and average them
        """
        start1, end1, w_ids1 = torch.broadcast_tensors(mentions[:, 0].unsqueeze(-1),
                                                       mentions[:, 1].unsqueeze(-1),
                                                       torch.arange(0, enc_seq.shape[1]).unsqueeze(0).to(self.device))

        index_t1 = (torch.ge(w_ids1, start1) & torch.le(w_ids1, end1)).float().to(self.device).unsqueeze(1)

        arg = torch.div(torch.matmul(index_t1, enc_seq), torch.sum(index_t1, dim=2).unsqueeze(-1)).squeeze(1)  # avg
      
        assert torch.sum(torch.isnan(arg)) ==0 , 'Problem locating tokens'
        return arg
    
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
        logits = self.classifier(h)
        outputs = (logits,)
#         if batch['labels'] is not None:
        loss = self.loss_fnt(logits.float(), batch['labels'])
        outputs = (loss,) + outputs

        return outputs