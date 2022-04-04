import time
import os
import torch
from tqdm import tqdm

from eval.evalRE import estimate_ent
from scripts.pipeline_process import gen_ner_ann_files, gen_rel_ann_files
from utils import utils
from utils.utils import _humanized_time, debug

def eval(model, eval_dir, result_dir, eval_dataloader, eval_data,
            params, epoch=0,
            optimizer=None, global_steps=None):
    mapping_id_tag = params['mappings']['nn_mapping']['id_tag_mapping']
    rel_tp_tr, rel_fp_tr, rel_fn_tr = [], [], []

    fidss, wordss, offsetss, sub_to_wordss, span_indicess = [], [], [], [], []

    rel_anns = []
    ent_anns = []

    # Evaluation phase
    model.eval()
    # nner
    all_ner_preds, all_ner_golds, all_ner_terms = [], [], []

    t_start = time.time()

    for step, batch in enumerate(
            tqdm(eval_dataloader, desc="Iteration", leave=False)
    ):
        eval_data_ids = batch
        tensors = utils.get_tensors(eval_data_ids, eval_data, params)

        nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, \
        nn_entity_masks, nn_trigger_masks, _, etypes, _ = tensors
        
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
        gold_entities = [
            eval_data["entities"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]

        with torch.no_grad():
            if not params['predict']:
                ner_out, _ = model(tensors, epoch)
            else:
                ner_out, e_golds = model(tensors)
               
        ner_preds = ner_out['preds']
        if not params['predict']:  # Debug only
                # Case train REL only
            if params['skip_ner'] and params['rel_epoch'] >= (params['epoch'] - 1) and params['#use_gold_ner']:
                    ner_terms = ner_out['gold_terms']
                    ner_preds = ner_out['golds']
                # Case train EV only
            elif params['skip_ner'] and params['skip_rel'] and params['#use_gold_ner'] \
                        and params['use_gold_rel']:
                ner_terms = ner_out['gold_terms']
                ner_preds = ner_out['golds']
            else:
                ner_terms = ner_out['terms']
        else:
                # if params['gold_eval'] or params['pipelines'] or params['predict_rel']:
                #     if params['pipelines'] and params['pipe_flag'] == 0:
                #         ner_terms = ner_out['terms']
                #     else:
                #         ner_terms = ner_out['gold_terms']
                #         ner_preds = ner_out['golds']
                # else:
            ner_terms = ner_out['terms']

        all_ner_terms.append(ner_terms)

        for sentence_idx, ner_pred in enumerate(ner_preds):
            all_ner_golds.append(
                    [
                        (
                            sub_to_words[sentence_idx][span_start],
                            sub_to_words[sentence_idx][span_end],
                            mapping_id_tag[label_id],
                        )
                        for (
                                span_start,
                                span_end,
                            ), label_ids in gold_entities[sentence_idx].items()
                        for label_id in label_ids
                    ]
                )

            pred_entities = []
            for span_id, ner_pred_id in enumerate(ner_pred):
                span_start, span_end = nn_span_indices[sentence_idx][span_id]
                span_start, span_end = span_start.item(), span_end.item()
                if (ner_pred_id > 0
                            and span_start in sub_to_words[sentence_idx]
                            and span_end in sub_to_words[sentence_idx]
                    ):
                    pred_entities.append(
                            (
                                sub_to_words[sentence_idx][span_start],
                                sub_to_words[sentence_idx][span_end],
                                mapping_id_tag[ner_pred_id],
                            )
                    )
            all_ner_preds.append(pred_entities)

        fidss.append(fids)
        ent_ann = {'span_indices': nn_span_indices, 'ner_preds': ner_out['preds'], 'words': words,
                            'offsets': offsets, 'sub_to_words': sub_to_words, 'subwords': subwords,
                            'ner_terms': ner_terms}            

        ent_anns.append(ent_ann)

        wordss.append(words)
        offsetss.append(offsets)
        sub_to_wordss.append(sub_to_words)

      
        # Clear GPU unused RAM:
        if params['gpu'] >= 0:
            torch.cuda.empty_cache()

    
    if params['predict'] and params['pipelines']:
        if params['pipe_flag'] == 0:
            gen_ner_ann_files(fidss, ent_anns, params)
            return
        elif params['pipe_flag'] == 1:
            gen_rel_ann_files(fidss, ent_anns, None, params)
            return

     # n2c2: entity and relation
    n2c2_scores = estimate_ent(ref_dir=eval_dir,
                                result_dir=result_dir,
                                fids=fidss,
                                ent_anns=ent_anns,                              
                                params=params)
        # Print scores
    show_scores(params, n2c2_scores)
    
    # saving models
    if not params['predict']:
        if epoch > params['save_st_ep']:
            save_models(model, params, optimizer, global_steps, epoch, n2c2_scores)

    t_end = time.time()
    print('Elapsed time: {}\n'.format(_humanized_time(t_end - t_start)))
    # print()                                                                

    return n2c2_scores


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


def save_models(model, params, optimizer, global_steps, epoch, n2c2_scores):

    ner_fscore = n2c2_scores['NER']['micro']['st_f']

    if params['ner_epoch'] >= (params['epoch'] - 1):
        best_score = ner_fscore

    is_save = False

    # Save models
    if params['save_ner']:
        ner_model_path = params['model_dir']
        is_save = utils.handle_checkpoints(
            model=model.NER_layer, 
            checkpoint_dir=ner_model_path,
            params={
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

    if params['save_all_models']:
        deepee_model_path = params['model_dir']
        is_save = utils.handle_checkpoints(
            model=model,
            checkpoint_dir=deepee_model_path,
            params={
                "filename": "deepee_" + model.NER_layer.__class__.__name__,
                "epoch": epoch,
                "fscore": best_score,
                'device': params['device'],
                'params_dir': params['params_dir'],
                'result_dir': params['result_dir']
            },            
            filter_func=utils.save_best_fscore,
            num_saved=1
        )
        print("Saved all models")
       
    
    return is_save
