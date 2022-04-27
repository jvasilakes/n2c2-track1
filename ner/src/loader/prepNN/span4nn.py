"""Prepare data with span-based for training networks."""

import numpy as np
from collections import namedtuple
import torch

Term = namedtuple('Term', ['id2term', 'term2id', 'id2label'])


def get_span_index(
        span_start,
        span_end,
        max_span_width,
        max_sentence_length,
        index,
        limit
):
    assert span_start <= span_end
    assert index >= 0 and index < limit
    assert max_span_width > 0
    assert max_sentence_length > 0

    max_span_width = min(max_span_width, max_sentence_length)
    invalid_cases = max(
        0, span_start + max_span_width - max_sentence_length - 1
    )
    span_index = (
            (max_span_width - 1) * span_start
            + span_end
            - invalid_cases * (invalid_cases + 1) // 2
    )
    return span_index * limit + index


def text_decode(sens, tokenizer):
    """Decode text from subword indices using pretrained bert"""

    ids = [id for id in sens]
    orig_text = tokenizer.decode(ids, skip_special_tokens=True)

    return orig_text


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

    ids = tokenizer_encoder.convert_tokens_to_ids(["[CLS]"] + sw_tokens + ["[SEP]"])

    # decode the text if shorten by max sequence
    if shorten:
        orig_text = text_decode(ids, tokenizer_encoder)
    else:
        orig_text = split_line_text

    token_mask = [0] + token_mask + [0]

    # ! Whether use value 1 for [CLS] and [SEP]
    attention_mask = [1] * len(ids)

    return sw_tokens, ids, num_tokens, token_mask, attention_mask, orig_text


def bert_gpt2_tokenize(split_line_text, tokenizer_decoder):
    # tokenized_text0 = tokenizer_encoder.convert_tokens_to_ids(tokenizer_encoder.tokenize(split_line_text))
    # tokenized_text0 = tokenizer_encoder.add_special_tokens_single_sentence(tokenized_text0)
    # tokenized_text0_length = len(tokenized_text0)
    tokenized_text1 = tokenized_text1_length = None
    if tokenizer_decoder != None:
        gpt2_bos_token = tokenizer_decoder.convert_tokens_to_ids(["<BOS>"])
        gpt2_eos_token = tokenizer_decoder.convert_tokens_to_ids(["<EOS>"])
        tokenized_text1 = tokenizer_decoder.convert_tokens_to_ids(tokenizer_decoder.tokenize(split_line_text))
        tokenized_text1 = tokenizer_decoder.add_special_tokens_single_sentence(tokenized_text1)
        tokenized_text1 = gpt2_bos_token + tokenized_text1 + gpt2_eos_token
        tokenized_text1_length = len(tokenized_text1)

    return tokenized_text1, tokenized_text1_length

# def get_priors (span_text, priors_tokenizer, priors_encoder, max_length):
    # with torch.no_grad():
    #     tokenized = priors_tokenizer.batch_encode_plus(span_text, add_special_tokens=True, 
    #                     truncation=True, max_length=max_length, 
    #                     padding="max_length", return_tensors='pt')
    
    #     priors = priors_encoder(tokenized)
    # span_tok = priors_tokenizer.tokenize(span_text)
    # token_mask = [1] * len(span_tok)
    # token_id = priors_tokenizer.convert_tokens_to_ids(["[CLS]"] + span_tok + ["[SEP]"])
    # token_mask = [0] + token_mask + [0]
    # attention_mask = [1] * len(token_id)
    # with torch.no_grad():
    #     output1, output2 = priors_encoder(torch.tensor(token_id, dtype=torch.long))
    #                                     # token_type_ids=torch.tensor(token_mask, dtype=torch.long), 
    #                                     # attention_mask=torch.tensor(attention_mask, dtype=torch.long))
    
    # return priors


def get_batch_data(fid, entities, terms, valid_starts, sw_sentence, words, words_id, sub_to_word, split_line_text,
                   tokenizer_encoder, events_map,
                   params):
    mlb = params["mappings"]["nn_mapping"]["mlb"]
    # num_labels = params["mappings"]["nn_mapping"]["num_labels"]

    max_entity_width = params["max_entity_width"]
    max_trigger_width = params["max_trigger_width"]
    max_span_width = params["max_span_width"]

    # bert tokenizer
    sw_tokens, bert_tokens, num_tokens, token_mask, attention_mask, orig_text = bert_tokenize(sw_sentence,
                                                                                              split_line_text,
                                                                                              params,
                                                                                              tokenizer_encoder)

    # bert and gpt2 tokenizers
    # TODO: there may be a bug here: if the num_tokens are not matched between the two tokenized outputs (check later)
    # tokenized_text1, tokenized_text1_length = bert_gpt2_tokenize(orig_text, tokenizer_decoder)
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
    span_labels_match_rel = []
    entity_masks = []
    trigger_masks = []
    span_terms = Term({}, {}, {})

    for span_start, span_end in zip(
            span_starts.flatten(), span_ends.flatten()
    ):
        if span_start >= 0 and span_end < num_tokens:
            span_label = []  # No label
            span_term = []
            span_label_match_rel = 0

            entity_mask = 1
            trigger_mask = 1

            if span_end - span_start + 1 > max_entity_width:
                entity_mask = 0
            if span_end - span_start + 1 > max_trigger_width:
                trigger_mask = 0

            # Ignore spans containing incomplete words
            valid_span = True
            if not (params['predict'] == 1 and (params['pipelines'] and params['pipe_flag'] != 0)):
                if span_start not in valid_starts or (span_end + 1) not in valid_starts:
                    # Ensure that there is no entity label here
                    if not (params['predict'] == 1 and (params['pipelines'] and params['pipe_flag'] != 0)):
                        # TODO: temporarily comment to fix bug, check again
                        assert (span_start, span_end) not in entities

                        entity_mask = 0
                        trigger_mask = 0
                        valid_span = False

            if valid_span:
                if (span_start, span_end) in entities:
                    span_label = entities[(span_start, span_end)]
                    span_term = terms[(span_start, span_end)]                    
                    # check if term can create relation in gold
                    # for idx, term in enumerate(span_term):
                    #     if term not in params['map_entities_without_relations']:
                    #         span_label_match_rel = 1
                    #         break

            #create spans to input for the decoder
            real_start = sub_to_word[span_start]
            real_end = sub_to_word[span_end]
            span_org = words_id[real_start:real_end+1]
            span_source = [params['mappings']['word_map']['<SOS>']] + span_org
            span_target = span_org + [params['mappings']['word_map']['<EOS>']] 
            span_length = len(span_source) 
                 
            # assert len(span_label) <= params["ner_label_limit"], "Found an entity having a lot of types"
            if len(span_label) > params["ner_label_limit"]:
                print('over limit span_label', span_term)

            # For multiple labels
            for idx, (_, term_id) in enumerate(
                    sorted(zip(span_label, span_term), reverse=True)[:params["ner_label_limit"]]):
                span_index = get_span_index(span_start, span_end, max_span_width, num_tokens, idx,
                                            params["ner_label_limit"])

                span_terms.id2term[span_index] = term_id
                span_terms.term2id[term_id] = span_index

                # add entity type
                term_label = params['mappings']['nn_mapping']['id_tag_mapping'][span_label[0]]
                span_terms.id2label[span_index] = term_label

            span_label = mlb.transform([span_label])[-1]

            span_indices += [(span_start, span_end)] * params["ner_label_limit"]
            span_labels.append(span_label)
            span_labels_match_rel.append(span_label_match_rel)
            entity_masks.append(entity_mask)
            trigger_masks.append(trigger_mask)
    
    return {
        # 'sw_tokens': sw_tokens,
        'bert_token': bert_tokens,
        'token_mask': token_mask,
        'attention_mask': attention_mask,
        'span_indices': span_indices,
        'span_labels': span_labels,
        'entity_masks': entity_masks,
        'trigger_masks': trigger_masks,
        'span_terms': span_terms,
        'bert_token_length': bert_token_length,       
    }


def get_nn_data(fids, entitiess, termss, valid_startss, sw_sentences, wordss, word_idss, sub_to_words,
                split_line_text_, tokenizer_encoder, events_map, params):
    samples = []

    # filter by sentence length
    dropped = 0

    # for idx, sw_sentence in enumerate(sw_sentences):
    span_priors = None
    for idx, split_line_text in enumerate(split_line_text_):

        fid = fids[idx]

        if len(split_line_text) < 1:
            dropped += 1
            continue

        if len(split_line_text.split()) > params['block_size']:
            dropped += 1
            continue

        sw_sentence = sw_sentences[idx]
        sub_to_word = sub_to_words[idx]
        words_id = word_idss[idx]
        words = wordss[idx]

        entities = entitiess[idx]
        terms = termss[idx]
        valid_starts = valid_startss[idx]

        sample = get_batch_data(fid, entities, terms, valid_starts, sw_sentence, words, words_id, sub_to_word,
                                split_line_text,
                                tokenizer_encoder,                                
                                events_map, params)       
        samples.append(sample)

    print('max_seq', params['max_seq'])

    bert_tokens = [sample["bert_token"] for sample in samples]
    all_token_masks = [sample["token_mask"] for sample in samples]
    all_attention_masks = [sample["attention_mask"] for sample in samples]
    all_span_indices = [sample["span_indices"] for sample in samples]
    all_span_labels = [sample["span_labels"] for sample in samples]
    all_entity_masks = [sample["entity_masks"] for sample in samples]
    all_trigger_masks = [sample["trigger_masks"] for sample in samples]
    all_span_terms = [sample["span_terms"] for sample in samples]

    # bert_tokens = [sample["bert_token"] for sample in samples]
    # gpt2_tokens = [sample["gpt2_token"] for sample in samples]
    bert_token_lengths = [sample["bert_token_length"] for sample in samples]
    # gpt2_token_lengths = [sample["gpt2_token_length"] for sample in samples]

    

    print("dropped sentences: ", dropped)

    return {
        # 'tokens': all_tokens,
        'bert_tokens': bert_tokens,
        'token_mask': all_token_masks,
        'attention_mask': all_attention_masks,
        'span_indices': all_span_indices,
        'span_labels': all_span_labels,
        'entity_masks': all_entity_masks,
        'trigger_masks': all_trigger_masks,
        'span_terms': all_span_terms,
        # 'gpt2_tokens': gpt2_tokens,
        'bert_token_lengths': bert_token_lengths,
        # 'gpt2_token_lengths': gpt2_token_lengths,       
    }
