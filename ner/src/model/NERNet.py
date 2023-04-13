# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


from transformers import BertModel, BertPreTrainedModel

class NERSpanModel(BertPreTrainedModel):
    def __init__(self, config, params):
        super(NERSpanModel, self).__init__(config)

        self.params = params
        self.ner_label_limit = params["ner_label_limit"]
        self.thresholds = params["ner_threshold"]
        self.num_entities = params["mappings"]["nn_mapping"]["num_entities"]        
        self.max_span_width = params["max_span_width"] 
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)        
        self.entity_classifier = nn.Linear(config.hidden_size * 3, self.num_entities)                               
       
        self.register_buffer(
                "label_ids",
                torch.tensor(
                    params["mappings"]["nn_mapping"]["mlb"].classes_, dtype=torch.uint8
                ),
            )
       
        # self.apply(self.init_bert_weights)
        self.params = params

    def get_span_embeddings(self, embeddings, all_token_masks, nn_entity_masks, device):
        '''
        Enumerate all possible spans and return their corresponding embeddings
        span embeddings = start word emb + end word emb + mean of all word emb
        '''
        flattened_token_masks = all_token_masks.flatten()  # (B * S, )
        flattened_embedding_indices = torch.arange(
            flattened_token_masks.size(0), device=device
        ).masked_select(
            flattened_token_masks
        )  # (all_actual_tokens, )
        # bert
        flattened_embeddings = torch.index_select(
            embeddings.view(-1, embeddings.size(-1)), 0, flattened_embedding_indices
        )  # (all_actual_tokens, H)

        span_starts = (
            torch.arange(flattened_embeddings.size(0), device=device)
                .view(-1, 1)
                .repeat(1, self.max_span_width)
        )  # (all_actual_tokens, max_span_width)

        flattened_span_starts = (span_starts.flatten())  # (all_actual_tokens * max_span_width, )

        span_ends = span_starts + torch.arange(self.max_span_width, device=device).view(1, -1)  # (all_actual_tokens, max_span_width)

        flattened_span_ends = (span_ends.flatten())  # (all_actual_tokens * max_span_width, )

        sentence_indices = (
            torch.arange(embeddings.size(0), device=device)
                .view(-1, 1)
                .repeat(1, embeddings.size(1))
        )  # (B, S)

        flattened_sentence_indices = sentence_indices.flatten().masked_select(
            flattened_token_masks
        )  # (all_actual_tokens, )

        span_start_sentence_indices = torch.index_select(
            flattened_sentence_indices, 0, flattened_span_starts
        )  # (all_actual_tokens * max_span_width, )

        span_end_sentence_indices = torch.index_select(
            flattened_sentence_indices,
            0,
            torch.min(
                flattened_span_ends,
                torch.ones(
                    flattened_span_ends.size(),
                    dtype=flattened_span_ends.dtype,
                    device=device,
                )
                * (span_ends.size(0) - 1),
            ),
        )  # (all_actual_tokens * max_span_width, )

        candidate_mask = torch.eq(
            span_start_sentence_indices,
            span_end_sentence_indices,  # Checking both indices is in the same sentence
        ) & torch.lt(
            flattened_span_ends, span_ends.size(0)
        )  # (all_actual_tokens * max_span_width, )

        flattened_span_starts = flattened_span_starts.masked_select(
            candidate_mask
        )  # (all_valid_spans, )

        flattened_span_ends = flattened_span_ends.masked_select(
            candidate_mask
        )  # (all_valid_spans, )

        span_start_embeddings = torch.index_select(
            flattened_embeddings, 0, flattened_span_starts
        )  # (all_valid_spans, H)

        span_end_embeddings = torch.index_select(
            flattened_embeddings, 0, flattened_span_ends
        )  # (all_valid_spans, H)

        # For computing embedding mean
        mean_indices = flattened_span_starts.view(-1, 1) + torch.arange(
            self.max_span_width, device=device
        ).view(
            1, -1
        )  # (all_valid_spans, max_span_width)

        mean_indices_criteria = torch.gt(
            mean_indices, flattened_span_ends.view(-1, 1).repeat(1, self.max_span_width)
        )  # (all_valid_spans, max_span_width)

        mean_indices = torch.min(
            mean_indices, flattened_span_ends.view(-1, 1).repeat(1, self.max_span_width)
        )  # (all_valid_spans, max_span_width)

        span_mean_embeddings = torch.index_select(
            flattened_embeddings, 0, mean_indices.flatten()
        ).view(
            *mean_indices.size(), -1
        )  # (all_valid_spans, max_span_width, H)

        coeffs = torch.ones(
            mean_indices.size(), dtype=embeddings.dtype, device=device
        )  # (all_valid_spans, max_span_width)

        coeffs[mean_indices_criteria] = 0

        span_mean_embeddings = span_mean_embeddings * coeffs.unsqueeze(
            -1
        )  # (all_valid_spans, max_span_width, H)

        span_mean_embeddings = torch.sum(span_mean_embeddings, dim=1) / torch.sum(
            coeffs, dim=-1
        ).view(
            -1, 1
        )  # (all_valid_spans, H)

        combined_embeddings = torch.cat(
                (
                    span_start_embeddings,
                    span_mean_embeddings,
                    span_end_embeddings,
                    # span_width_embeddings,
                ),
                dim=1,
            )  # (all_valid_spans, H * 3 + distance_dim)
        #split the combined embeddings to the batch: batch_size x num of span x 768
        all_span_masks = (nn_entity_masks > -1) # (B, max_spans)
        sentence_sections = all_span_masks.sum(dim=-1).cumsum(dim=-1)  # (B, )
        sentence_sections = sentence_sections.detach().cpu().numpy()[:-1]
       
        starts = span_start_embeddings.detach().cpu().numpy() 
        ends = span_end_embeddings.detach().cpu().numpy() 
        means = span_mean_embeddings.detach().cpu().numpy() 
       
        batch_start = np.split(starts, sentence_sections)
        batch_ends = np.split(ends, sentence_sections)
        batch_means = np.split(means, sentence_sections)

        return combined_embeddings, (batch_start, batch_ends, batch_means)
    
    def forward(self,            
            all_ids,
            all_token_masks,
            all_attention_masks,
            all_entity_masks,
            all_span_labels,                
    ):
        device = all_ids.device                
        # ! REDUCE
        # embeddings = self.dropout(embeddings)  # (B, S, H) (B, 128, 768)
        all_span_masks = (all_entity_masks > -1) # (B, max_spans) --> skip padding ones
        valid_masks = all_entity_masks[all_span_masks] > 0  # (all_valid_spans, ) -> skip invalid
            
        outputs = self.bert(input_ids=all_ids, attention_mask=all_attention_masks)
        org_embeddings = outputs.last_hidden_state
        feature_vector, _ = self.get_span_embeddings(org_embeddings, all_token_masks, all_entity_masks, device)
        gold_labels = all_span_labels[all_span_masks][valid_masks]  # (all_valid_spans, num_entities)       
        
        entity_preds = self.entity_classifier(feature_vector)  
        all_preds = torch.sigmoid(entity_preds)  # (all_valid_spans, num_entities)     
        # Clear values at invalid positions        
        all_preds[~valid_masks, : ] = 0        
        # Compute entity loss
        entity_loss = F.binary_cross_entropy_with_logits(
                    # entity_preds[all_entity_masks], all_span_labels[all_entity_masks]
                    entity_preds[valid_masks], gold_labels
                )
        
        _, all_preds_top_indices = torch.topk(all_preds, k=self.ner_label_limit, dim=-1)
        # Convert binary value to label ids
        all_preds = (all_preds > self.thresholds) * self.label_ids
        all_preds = torch.gather(all_preds, dim=1, index=all_preds_top_indices)
            
        return entity_loss, all_preds