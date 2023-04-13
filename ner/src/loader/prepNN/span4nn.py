"""Prepare data with span-based for training networks."""

from tokenize import group
import numpy as np
import torch

def bert_tokenize(sw_sentence, split_line_text, params, tokenizer_encoder):
    sw_tokens = [token for token, *_ in sw_sentence]

    num_tokens = len(sw_tokens)

    token_mask = [1] * num_tokens

    shorten = False

    # Account for [CLS] and [SEP] tokens
    if num_tokens > params['max_seq'] - 2:
        num_tokens = params['max_seq'] - 2
        sw_tokens = sw_tokens[:num_tokens]
        token_mask = token_mask[:num_tokens]
        shorten = True

    input_ids = tokenizer_encoder.convert_tokens_to_ids(["[CLS]"] + sw_tokens + ["[SEP]"])

    token_mask = [0] + token_mask + [0]

    # ! Whether use value 1 for [CLS] and [SEP]
    attention_mask = [1] * len(input_ids)

    if not shorten and num_tokens < params['max_seq']: #padding before the levitated markers
        num_pads = params['max_seq'] - num_tokens - 2
        input_ids += [0] * num_pads
        token_mask += [0]* num_pads
        attention_mask += [0] * num_pads

    return   torch.tensor(input_ids, dtype=torch.long), \
        num_tokens, \
        token_mask, \
        torch.tensor(attention_mask,dtype=torch.long)


def get_batch_data(entities, valid_starts, sw_sentence, split_line_text, tokenizer_encoder, params):
    mlb = params["mappings"]["nn_mapping"]["mlb"]
    # num_labels = params["mappings"]["nn_mapping"]["num_labels"]

    max_entity_width = params["max_entity_width"]
    # max_trigger_width = params["max_trigger_width"]
    max_span_width = params["max_span_width"]

    # bert tokenizer
    input_ids, num_tokens, token_mask, attention_mask = bert_tokenize(sw_sentence,
                                                                    split_line_text,
                                                                    params,
                                                                    tokenizer_encoder)

    bert_token_length = num_tokens + 2

    # Generate spans here
    span_starts = np.tile(
        np.expand_dims(np.arange(num_tokens), 1), (1, max_span_width)
    )  # (num_tokens, max_span_width)

    span_ends = span_starts + np.expand_dims(
        np.arange(max_span_width), 0
    )  # (num_tokens, max_span_width)

    span_indices = []
    span_labels = []    
    entity_masks = []
    
    for span_start, span_end in zip(
            span_starts.flatten(), span_ends.flatten()
    ):
        if span_start >= 0 and span_end < num_tokens:
            span_label = []  # No label            

            entity_mask = 1           
            if span_end - span_start + 1 > max_entity_width:
                entity_mask = 0
            
            valid_span = True
            # Ignore spans containing incomplete words
            if params['predict'] != 1:
                if span_start not in valid_starts or (span_end + 1) not in valid_starts:
                    # Ensure that there is no entity label here              
                    assert (span_start, span_end) not in entities
                    entity_mask = 0                                   
                    valid_span = False

            if valid_span:
                if (span_start, span_end) in entities:
                    span_label = entities[(span_start, span_end)]
    
            span_label = mlb.transform([span_label])[-1]
            span_indices += [(span_start, span_end)] * params["ner_label_limit"]
            span_labels.append(span_label)            
            entity_masks.append(entity_mask)
    
    return {
        'bert_token': input_ids,
        'token_mask': token_mask,
        'attention_mask': attention_mask,
        'span_indices': span_indices,
        'span_labels': span_labels,
        'entity_masks': entity_masks,
        'bert_token_length': bert_token_length,       
    }

def get_nn_data(entitiess, valid_startss, sw_sentences, split_line_text_, tokenizer_encoder, params):
    samples = []

    # filter by sentence length
    dropped = 0

    # for idx, sw_sentence in enumerate(sw_sentences):    
    for idx, split_line_text in enumerate(split_line_text_):

        if len(split_line_text) < 1:
            dropped += 1
            continue

        if len(split_line_text.split()) > params['block_size']:
            dropped += 1
            continue

        sw_sentence = sw_sentences[idx]
        entities = entitiess[idx]        
        valid_starts = valid_startss[idx]

        sample = get_batch_data (entities, valid_starts, sw_sentence, split_line_text, tokenizer_encoder, params)                          
                                      
        samples.append(sample)

    print('max_seq', params['max_seq'])

    bert_tokens = [sample["bert_token"] for sample in samples]
    all_token_masks = [sample["token_mask"] for sample in samples]
    all_attention_masks = [sample["attention_mask"] for sample in samples]
    all_span_indices = [sample["span_indices"] for sample in samples]
    all_span_labels = [sample["span_labels"] for sample in samples]
    all_entity_masks = [sample["entity_masks"] for sample in samples]
    bert_token_lengths = [sample["bert_token_length"] for sample in samples]
    
    print("dropped sentences: ", dropped)

    return {        
        'bert_tokens': bert_tokens,
        'token_mask': all_token_masks,
        'attention_mask': all_attention_masks,
        'span_indices': all_span_indices,
        'span_labels': all_span_labels,
        'entity_masks': all_entity_masks,
        'bert_token_lengths': bert_token_lengths,
        
    }


