import copy
from unicodedata import category
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

from model.NERNet import NERSpanModel
from model.NERNetRoberta import NERSpanRobertaModel

cpu_device = torch.device("cpu")
Term = namedtuple('Term', ['id2term', 'term2id', 'id2label'])

class SpanNER(nn.Module):
    """
    Network architecture
    """

    def __init__(self, params):
        super(SpanNER, self).__init__()

        device = params['device']
        if 'roberta' in params['encoder_config_name']:
            self.NER_layer = NERSpanRobertaModel.from_pretrained(params['bert_model'], params=params, return_dict=False)
        else:
            self.NER_layer = NERSpanModel.from_pretrained(params['bert_model'], params=params)
        
        self.entity_id = 1

        self.device = device
        self.params = params

    def forward(self, batch_input):  
        nn_bert_tokens, nn_token_mask, nn_attention_mask, \
                nn_span_indices, nn_span_labels, nn_entity_masks = batch_input

        ner_loss, e_preds = self.NER_layer(                    
                all_ids=nn_bert_tokens,
                all_token_masks=nn_token_mask,
                all_attention_masks=nn_attention_mask,
                all_entity_masks=nn_entity_masks,
                all_span_labels=nn_span_labels,                 
        )        
        
        e_preds = e_preds.detach().cpu().numpy() 
        ner_preds = {}
        all_span_masks = (nn_entity_masks > -1) # (B, max_spans)
        sentence_sections = all_span_masks.sum(dim=-1).cumsum(dim=-1)  # (B, )
        sentence_sections = sentence_sections.detach().cpu().numpy()[:-1]
        # Pred of each span
        e_preds = np.split(e_preds.astype(int), sentence_sections)
        e_preds = [pred.flatten() for pred in e_preds]
        ner_preds['preds'] = e_preds
        entity_idx = self.entity_id
        span_terms = []        
        for span_preds in e_preds:
            doc_spans = Term({},{},{})            
            for pred_idx, label_id in enumerate(span_preds):
                if label_id > 0:                    
                    term = "T" + str(entity_idx)
                    doc_spans.id2term[pred_idx] = term
                    doc_spans.term2id[term] = pred_idx
                    entity_idx += 1
            span_terms.append(doc_spans)

        self.entity_id = entity_idx
        
        ner_preds['loss'] = ner_loss
        ner_preds['terms'] = span_terms

        return ner_preds
    
