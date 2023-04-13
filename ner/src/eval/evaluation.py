import time
import os
import torch
from tqdm import tqdm

from utils import utils
from utils.utils import _humanized_time, write_annotation_file
from collections import defaultdict


def eval(model, eval_dir, result_dir, eval_dataloader, eval_data,
            params, epoch=0,
        ):
    fidss = []
    ent_anns = []

    # Evaluation phase
    model.eval()

    t_start = time.time()

    for step, batch in enumerate(
            tqdm(eval_dataloader, desc="Iteration", leave=False)
    ):
        eval_data_ids = batch
        tensors = utils.get_tensors(eval_data_ids, eval_data, params)   

        nn_bert_tokens, nn_token_mask, nn_attention_mask, \
                nn_span_indices, nn_span_labels, nn_entity_masks = tensors
        
        fids = [
            eval_data["fids"][data_id] for data_id in eval_data_ids[0].tolist()
        ]
        offsets = [
            eval_data["offsets"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        words = [
            eval_data["words"][data_id] for data_id in eval_data_ids[0].tolist()
        ]
        sub_to_words = [
            eval_data["sub_to_words"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        subwords = [
            eval_data["subwords"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]

        with torch.no_grad():
            ner_out = model(tensors)

        fidss.append(fids)
        ent_ann = {'span_indices': nn_span_indices, 'ner_preds': ner_out['preds'], 'words': words,
                    'offsets': offsets, 'sub_to_words': sub_to_words, 'subwords': subwords,
                    'ner_terms': ner_out['terms']}            

        ent_anns.append(ent_ann)

        # Clear GPU unused RAM:
        if params['gpu'] >= 0:
            torch.cuda.empty_cache()

    n2c2_scores = estimate_ent(ref_dir=eval_dir,
                                    result_dir=result_dir,
                                    fids=fidss,
                                    ent_anns=ent_anns,                              
                                    params=params)
    # Print scores
    show_scores(params, n2c2_scores)
    
    if params['predict'] != 1: # train
        # saving models    
        if epoch > params['save_st_ep']:
            save_models(model, params, epoch, n2c2_scores)

    t_end = time.time()
    print('Elapsed time: {}\n'.format(_humanized_time(t_end - t_start)))
    # print()                                                                

    return n2c2_scores


def estimate_ent(ref_dir, result_dir, fids, ent_anns, params):
    """Evaluate entity performance using n2c2 script"""

    # generate brat prediction
    gen_annotation_ent(fids, ent_anns, params, result_dir)

    # calculate scores
    pred_dir = ''.join([result_dir, 'ent-ann/'])
    pred_scores_file = ''.join([result_dir, 'ent-scores-', params['ner_eval_corpus'], '.txt'])

    # run evaluation, output in the score file
    eval_performance(ref_dir, pred_dir, result_dir, pred_scores_file, params)

    # extract scores
    scores = extract_fscore(pred_scores_file)

    return scores

def get_entity_attrs(e_span_indice, words, offsets, sub_to_words):
    e_words = []
    e_offset = [-1, -1]
    curr_word_idx = -1
    for idx in range(e_span_indice[0], e_span_indice[1] + 1):
        if sub_to_words[idx] != curr_word_idx:
            e_words.append(words[sub_to_words[idx]])
            curr_word_idx = sub_to_words[idx]
        if idx == e_span_indice[0]:
            e_offset[0] = offsets[sub_to_words[idx]][0]
        if idx == e_span_indice[1]:
            e_offset[1] = offsets[sub_to_words[idx]][1]
    return ' '.join(e_words), (e_offset[0], e_offset[1])

def gen_annotation_ent(fidss, ent_anns, params, result_dir):
    """Generate entity prediction"""

    dir2wr = ''.join([result_dir, 'ent-ann/'])
    if not os.path.exists(dir2wr):
        os.makedirs(dir2wr)
    else:
        os.system('rm ' + dir2wr + '*.ann')

    # Initial ent+rel map
    map = defaultdict()
    for fids in fidss:
        for fid in fids:
            map[fid] = {}

    for xi, (fids, ent_ann) in enumerate(zip(fidss, ent_anns)):
        # Mapping entities
        entity_map = defaultdict()
        for xb, (fid) in enumerate(fids):
            span_indices = ent_ann['span_indices'][xb]
            ner_terms = ent_ann['ner_terms'][xb]
            ner_preds = ent_ann['ner_preds'][xb]
            words = ent_ann['words'][xb]
            offsets = ent_ann['offsets'][xb]
            sub_to_words = ent_ann['sub_to_words'][xb]

            entities = map[fid]
            check_offset = {}
            for x, pair in enumerate(span_indices):
                if pair[0].item() == -1: #skip the padding ones
                    break
                
                if ner_preds[x] > 0:                   
                    try:
                        e_id = ner_terms.id2term[x]
                        e_type = params['mappings']['rev_type_map'][
                            params['mappings']['nn_mapping']['tag2type_map'][ner_preds[x]]]
                        e_words, e_offset = get_entity_attrs(pair, words, offsets, sub_to_words)
                        if e_offset in check_offset: #in case we have different starts/ends subwords point to the same word
                            # print ("Duplicate predictions")
                            continue
                        check_offset[e_offset] = 1
                        # entity_map[(xb, (pair[0].item(), pair[1].item()))] = (
                        #     ner_preds[x], e_id, e_type, e_words, e_offset)
                        entity_map[(xb, x)] = (
                            ner_preds[x], e_id, e_type, e_words, e_offset)
                        entities[e_id] = {"id": e_id, "type": e_type, "start": e_offset[0], "end": e_offset[1],
                                          "ref": e_words}
                    except KeyError as error:
                        print('pred not map term', error, fid)
        

    for fid, ners in map.items():
        write_annotation_file(ann_file=dir2wr + fid + '.ann', entities=ners)

def show_scores(params, n2c2_scores):

    # print()
    print('-----EVALUATING BY N2C2 SCRIPT (FOR ENT & REL)-----\n')
    # print()
    print('STRICT_MATCHING:\n')
    print_scores('NER', n2c2_scores['NER'], 'st')
    # print()
    print('SOFT_MATCHING:\n')
    print_scores('NER', n2c2_scores['NER'], 'so')



def print_scores(k, v, stoso):
    if k == 'NER':
        print(
        k + "(MICRO): P/R/F1 = {:.02f}\t{:.02f}\t{:.02f} \n".format(
            v['micro'][stoso + '_p'], v['micro'][stoso + '_r'], v['micro'][stoso + '_f']), end="",
        )
    else:
        print(
        k + "(MICRO): P/R/F1 = {:.02f}\t{:.02f}\t{:.02f} , (MACRO): P/R/F1 = {:.02f}\t{:.02f}\t{:.02f}\n".format(
            v['micro'][stoso + '_p'], v['micro'][stoso + '_r'], v['micro'][stoso + '_f'],
            v['macro'][stoso + '_p'], v['macro'][stoso + '_r'], v['macro'][stoso + '_f']), end="",
    )
    # print()


def save_models(model, params, epoch, n2c2_scores):
    ner_fscore = n2c2_scores['NER']['micro']['st_f']
    is_save = False

    # Save models
    if params['save_model']:
        ner_model_path = params['model_dir']
        is_save = utils.handle_checkpoints(
            model=model.NER_layer, 
            checkpoint_dir=ner_model_path,
            params={
                'ner_learning_rate': params['ner_learning_rate'], 
                'max_span_width': params['max_span_width'], 
                'max_seq': params['max_seq'], 
                'seed': params['seed'],
                "filename": model.NER_layer.__class__.__name__,
                "epoch": epoch,
                "fscore": ner_fscore,
                'device': params['device'],
                'params_dir': params['params_dir'],
                'result_dir': params['result_dir']
            },            
            filter_func=utils.save_best_fscore,
            num_saved=1
        )
        if is_save:
            print("Saved model!")
    
    return is_save

def eval_performance(ref_dir, pred_dir, result_dir, pred_scores_file, params):
    # run evaluation script
    command = ''.join(
        ["python3 ", params['eval_script_path'], " ", ref_dir, " ", pred_dir, " > ", pred_scores_file])
    os.system(command)


def extract_fscore(path):
    file = open(path, 'r')
    lines = file.readlines()
    report = defaultdict()
    report['NER'] = defaultdict()
    report['REL'] = defaultdict()
   
    ent_or_rel = ''
    mi_or_mc = ''
    for line in lines:
        if '*' in line and 'Track' in line:
            ent_or_rel = 'NER'
            mi_or_mc = 'micro'
        elif '*' in line and 'RELATIONS' in line:
            ent_or_rel = 'REL'
        elif len(line.split()) > 0 and line.split()[0] == 'Drug':
            tokens = line.split()
            if len(tokens) > 8:
                strt_p, strt_r, strt_f, soft_p, soft_r, soft_f \
                    = tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6]
            else:
                strt_f, strt_r, strt_p, soft_f, soft_r, soft_p \
                    = tokens[-4], tokens[-5], tokens[-6], tokens[-1], tokens[-2], tokens[-3]
            if line.split()[1] == '(micro)':
                mi_or_mc = 'micro'
            elif line.split()[1] == '(macro)':
                mi_or_mc = 'macro'
            
            if mi_or_mc != '':
                report[ent_or_rel][mi_or_mc] = {'st_f': float(strt_f.strip()) * 100,
                                                'st_r': float(strt_r.strip()) * 100,
                                                'st_p': float(strt_p.strip()) * 100,
                                                'so_f': float(soft_f.strip()) * 100,
                                                'so_r': float(soft_r.strip()) * 100,
                                                'so_p': float(soft_p.strip()) * 100}

    return report
