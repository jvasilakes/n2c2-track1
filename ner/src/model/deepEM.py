import copy
from unicodedata import category

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

from model.NERNet import NERSpanModel
from model.NERNetRoberta import NERSpanRobertaModel

from utils import utils

cpu_device = torch.device("cpu")


class DeepEM(nn.Module):
    """
    Network architecture
    """

    def __init__(self, params):
        super(DeepEM, self).__init__()

        device = params['device']
        if 'roberta' in params['encoder_config_name']:
            self.NER_layer = NERSpanRobertaModel.from_pretrained(params['bert_model'], params=params, return_dict=False)
        else:
            self.NER_layer = NERSpanModel.from_pretrained(params['bert_model'], params=params)
        
        self.trigger_id = -1

        self.device = device
        self.params = params

    def process_ner_output(self, span_terms, max_span_labels, nn_span_indices,
                           ner_loss, e_preds, e_golds, sentence_sections, span_masks, embeddings,
                           ):
        """Process NER output to prepare for training relation and event layers"""

        # entity output
        ner_preds = {}

        # ! Note that these below lines run on CPU
        sentence_sections = sentence_sections.detach().cpu().numpy()[:-1]
        all_span_masks = span_masks.detach() > 0
        
        # Embedding of each span
        embeddings = torch.split(embeddings, torch.sum(all_span_masks, dim=-1).tolist())
      
        # Pred of each span
        e_preds = np.split(e_preds.astype(int), sentence_sections)
        e_preds = [pred.flatten() for pred in e_preds]
        ner_preds['preds'] = e_preds

        e_golds = np.split(e_golds.astype(int), sentence_sections)
        e_golds = [gold.flatten() for gold in e_golds]
        ner_preds['golds'] = e_golds
        ner_preds['gold_terms'] = copy.deepcopy(span_terms)

        replace_term = True        

        if self.params["ner_predict_all"]:          

            if replace_term:
                for items in span_terms:
                    items.term2id.clear()
                    items.id2term.clear()

                # Overwrite triggers
                if self.trigger_id == -1:
                    self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

                trigger_idx = self.trigger_id + 1
                for sentence_idx, span_preds in enumerate(e_preds):
                    for pred_idx, label_id in enumerate(span_preds):
                        if label_id > 0:
                            term = "T" + str(trigger_idx)

                            # check trigger
                            if label_id in self.params['mappings']['nn_mapping']['trTypes_Ids']:
                                term = "TR" + str(trigger_idx)

                            span_terms[sentence_idx].id2term[pred_idx] = term
                            span_terms[sentence_idx].term2id[term] = pred_idx
                            trigger_idx += 1

                self.trigger_id = trigger_idx
        else:
            if replace_term:
                # Overwrite triggers
                if self.trigger_id == -1:
                    self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

                trigger_idx = self.trigger_id + 1
                for sentence_idx, span_preds in enumerate(e_preds):
                    # Update gold labels

                    # store gold entity index (a1)
                    a1ent_set = set()

                    for span_idx, span_term in span_terms[sentence_idx].id2term.items():

                        if span_term != "O" and not span_term.startswith("TR") and span_preds[span_idx] != 255:

                            # but do not replace for entity in a2 files
                            span_label = span_terms[sentence_idx].id2label[
                                span_idx]  # entity type, e.g: Gene_or_gene_product
                            if span_label not in self.params['a2_entities']:
                                # replace for entity (using gold entity)
                                span_preds[span_idx] = e_golds[sentence_idx][span_idx]

                                # save this index to ignore prediction
                                a1ent_set.add(span_idx)

                    for pred_idx, label_id in enumerate(span_preds):
                        span_term = span_terms[sentence_idx].id2term.get(pred_idx, "O")

                        # if this entity in a1: skip this span
                        if pred_idx in a1ent_set:
                            continue

                        remove_span = False

                        # add prediction for trigger or entity a2
                        if label_id > 0:
                            term = ''

                            # check trigger
                            if label_id in self.params['mappings']['nn_mapping']['trTypes_Ids']:
                                term = "TR" + str(trigger_idx)

                            # is entity
                            else:
                                etype_label = self.params['mappings']['nn_mapping']['id_tag_mapping'][label_id]

                                # check this entity type in a2 or not
                                if etype_label in self.params['a2_entities']:
                                    term = "T" + str(trigger_idx)
                                else:
                                    remove_span = True

                            if len(term) > 0:
                                span_terms[sentence_idx].id2term[pred_idx] = term
                                span_terms[sentence_idx].term2id[term] = pred_idx
                                trigger_idx += 1

                        # null prediction
                        if label_id == 0 or remove_span:
                            # do not write anything
                            span_preds[pred_idx] = 0

                            # remove this span
                            if span_term.startswith("T"):
                                del span_terms[sentence_idx].id2term[pred_idx]
                                del span_terms[sentence_idx].term2id[span_term]

                    span_preds[span_preds == 255] = 0
                self.trigger_id = trigger_idx

        num_padding = max_span_labels * self.params["ner_label_limit"]

        e_preds = [np.pad(pred, (0, num_padding - pred.shape[0]),
                          'constant', constant_values=-1) for pred in e_preds]
        e_golds = [np.pad(gold, (0, num_padding - gold.shape[0]),
                          'constant', constant_values=-1) for gold in e_golds]

        e_preds = torch.tensor(e_preds, device=self.device)
        nn_span_labels = torch.tensor(e_golds, device=self.device)

        embeddings = [f.pad(embedding, (0, 0, 0, max_span_labels - embedding.shape[0]),
                            'constant', value=0) for embedding in embeddings]

        embeddings = torch.stack(embeddings)
        embeddings = embeddings.unsqueeze(dim=2).expand(-1, -1, self.params["ner_label_limit"], -1)
        embeddings = embeddings.reshape(embeddings.size(0), -1, embeddings.size(-1))

        # output for ner
        ner_preds['loss'] = ner_loss
        ner_preds['terms'] = span_terms
        ner_preds['span_indices'] = nn_span_indices
        
        ner_preds['nner_preds'] = e_preds.detach().cpu().numpy()

        return embeddings, e_preds, e_golds, nn_span_labels, ner_preds

   

    def forward(self, batch_input, n_epoch=0):

        # 1 - get input
        nn_tokens, nn_bert_tokens, nn_token_mask, nn_attention_mask, nn_span_indices, \
                nn_span_labels, nn_entity_masks, nn_trigger_masks, span_terms, \
                etypes, max_span_labels = batch_input

        reco_loss = None
        kld = None
        #this is for Nhung's NER joint VAE
        ner_loss, e_preds, e_golds, sentence_sections, span_masks, bert_embs, span_embeddings, \
                sentence_emb = self.NER_layer(                    
                all_ids=nn_bert_tokens,
                all_token_masks=nn_token_mask,
                all_attention_masks=nn_attention_mask,
                all_entity_masks=nn_entity_masks,
                all_trigger_masks=nn_trigger_masks,
                all_span_labels=nn_span_labels,                        
        )

        ner_span_embs, e_preds, _, nn_span_labels, ner_preds = self.process_ner_output(span_terms,
                                                                max_span_labels,
                                                                nn_span_indices,
                                                                ner_loss, e_preds,
                                                                e_golds,
                                                                sentence_sections,
                                                                span_masks, span_embeddings,                                                                
                                                                )

        return ner_preds, e_golds
    
