1#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/02/2022

author: panos
"""
import numpy as np
import torch
from torch import nn, torch
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def register(name):
    def assign_name(func):
        func._tagged = name
        return func
    return assign_name

class TokenPooler(nn.Module):
    """
    Pools an encoded input sequence over the hidden dimension
    according to a list of indices that indicate the position
    of the levitated markers
    """
    def __init__(self, pool_fn, config, device):
        super().__init__()
        self.hidden_dim = config['enc_dim']
        self.outsize = self.hidden_dim
        self.max_verbs = config['max_verbs']
        self.max_mark_len = self.max_verbs if config['single_marker'] else self.max_verbs * 2  # change here
        self.s_levi = config['max_tok_len'] - self.max_mark_len
        self.device = device
        self.single = config['single_marker']
        try:
            self.pooler = self.pooler_functions[pool_fn]
            self.pool_fn = pool_fn
        except KeyError:
            raise ValueError(f"Unknown pool function '{pool_fn}'")
        self.insize = self.hidden_dim
        self.alignment_model = nn.Linear(2 * self.insize, 1)
        self.output_layer = nn.Sequential(
            nn.Linear(self.insize, self.insize),
            nn.Tanh())

    def forward(self, hidden, verb_count, subject_hidden):
        if self.single:
            token_mask = self.create_token_mask_single(hidden, verb_count)
        else:
            token_mask = self.create_token_mask(hidden, verb_count)
        # masked_hidden = hidden * token_mask
        pooled = self.pooler(token_mask, hidden, subject_hidden)
        transformed = self.output_layer(pooled)
        return transformed

    def create_token_mask_single(self, hidden, verb_count):

        w_ids = torch.arange(0, hidden.shape[1]).repeat(hidden.shape[0], 1).to(self.device)
        verbs = verb_count.unsqueeze(1)
        start_levi = torch.tensor(self.s_levi).repeat(verbs.shape).to(self.device)
        end_levi = start_levi + verbs # change here
        _, ss, se = torch.broadcast_tensors(w_ids, start_levi, end_levi)
        token_mask = torch.logical_and(w_ids >= ss, w_ids < se)  # batch x max_seq_len
        return token_mask

    def create_token_mask(self, hidden, verb_count):

        w_ids = torch.arange(0, hidden.shape[1]).repeat(hidden.shape[0], 1).to(self.device)
        verbs = verb_count.unsqueeze(1)
        ss_levi = torch.tensor(self.s_levi).repeat(verbs.shape).to(self.device)
        se_levi = ss_levi + verbs
        es_levi = ss_levi + self.max_verbs
        ee_levi = es_levi + verbs
        _, ss, se, es, ee = torch.broadcast_tensors(w_ids, ss_levi, se_levi, es_levi, ee_levi)
        mask_starting = torch.logical_and(w_ids >= ss, w_ids < se)
        mask_ending = torch.logical_and(w_ids >= es, w_ids < ee)
        token_mask = torch.logical_or(mask_starting, mask_ending)
        return token_mask
    @property
    def pooler_functions(self):
        if "_pooler_registry" in self.__dict__.keys():
            return self._pooler_registry
        else:
            self._pooler_registry = {}
            for name in dir(self):
                var = getattr(self, name)
                if hasattr(var, "_tagged"):
                    registry_name = var._tagged
                    self._pooler_registry[registry_name] = var
            return self._pooler_registry

    @register("max")
    def max_pool(self, mask, hidden, subject_hidden):
        x = hidden.clone()
        x.masked_fill_(~mask.unsqueeze(-1),  -math.inf) # 0
        pooled = torch.max(x, 1)[0]
        return torch.nan_to_num(pooled) # In case all are -inf

    @register("mean")
    def mean_pool(self, mask, hidden, subject_hidden):
        numerator = torch.matmul(mask.float().unsqueeze(1), hidden)
        denominator = torch.sum(mask.float().unsqueeze(1), dim=-1).unsqueeze(-1)
        zeros_indx = denominator == 0
        denominator[zeros_indx] += 1
        return torch.div(numerator, denominator).squeeze(1)

    @register("attention-softmax")
    def softmax_pooler(self, mask, hidden, subject_hidden):
        projection_fn = torch.nn.Softmax(dim=0)
        return self.generic_attention_pooler(
            mask, hidden, subject_hidden, projection_fn)

    def generic_attention_pooler(self, mask, hidden, subject_hidden, projection_fn):
        """
        Implements attention between a "subject" span and one or more "object"
        spans.
        projection_fn is a function which maps the attention scores to the
            simplex. E.g., softmax.
            subject hidden size = batch x hidden_dim
        """
        subject_hidden_rep = subject_hidden.unsqueeze(1).repeat(1, hidden.size(1), 1)  # projecting to batch x seq x hidden_dim
        subject_hidden_rep = subject_hidden_rep.masked_fill_(~mask.unsqueeze(-1),  0) # applied mask
        masked = hidden.clone()
        masked.masked_fill_(~mask.unsqueeze(-1),  0) # batch x max_seq_len x hidden dim
        alignment_inputs = torch.cat((subject_hidden_rep, masked), dim=2)
        attention_scores = self.alignment_model(alignment_inputs)
        batch_size = masked.size(0)
        # normalize over the levitated markers
        attention_weights = torch.zeros_like(attention_scores, dtype=torch.float32) # apparently needs explicit casting else torch.float16
        attn_mask = mask[:, :].bool()
        for ex_i in range(batch_size):
            masked_scores = torch.masked_select(attention_scores[ex_i],
                                                attn_mask[ex_i].unsqueeze(1))
            probs = projection_fn(masked_scores)
            attention_weights[ex_i][attn_mask[ex_i]] = probs.unsqueeze(1)
        # scale the levitated marker representations by the attention_weights
        # and sum over the levitated markers
        pooled = (masked * attention_weights).sum(1)
        return pooled

