import torch
from torch.autograd import Variable
from tqdm import tqdm, trange

import time
import os
import pickle
import numpy as np
# from torch.utils.tensorboard import SummaryWriter

from eval.evaluation import eval
from utils import utils
from utils.utils import debug, _humanized_time
from utils.utils import extract_scores


def train(
        train_data_loader,
        dev_data_loader,
        train_data,
        dev_data,
        params,
        model,
        optimizer,
        scheduler=None,              
):

    # tb_writer = SummaryWriter(params['result_dir'])
    is_params_saved = False
    global_steps = 0

    gradient_accumulation_steps = params["gradient_accumulation_steps"]

    ner_prf_dev_str, ner_prf_dev_sof = [], []

    tr_batch_losses_ = []

    # create output directory for results
    result_dir = params['result_dir']
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Save params:
    if params['save_params']:
        if not is_params_saved:
            # saved_params_path = result_dir + params['task_name'] + '.param'
            saved_params_path = params['params_dir']
            with open(saved_params_path, "wb") as f:
                pickle.dump(params, f)
            print('SAVED PARAMETERS!')

    st_ep = 0

    model.zero_grad()

    params['best_epoch'] = 0
    scores = []
    for epoch in trange(st_ep, int(params["epoch"]), desc="Epoch"):
        if epoch < params['start_epoch']:
            continue
        # e_start = time.time()
        model.train()
        tr_loss = 0
        ner_loss =  0
        nb_tr_steps = 0
        
        # print()
        print(
            "====================================================================================================================")
        # print()
        debug(f"[1] Epoch: {epoch + 1}\n")

        # for mini-batches
        for step, batch in enumerate(
                tqdm(train_data_loader, desc="Iteration", leave=False)
        ):

            tr_data_ids = batch
            # batch_size = train_data_loader.batch_size         
            # e_start = time.time()
            tensors = utils.get_tensors(tr_data_ids, train_data, params)           
            
            ner_preds = model(tensors)
            
            # loss                
            total_loss = ner_preds['loss']
            
            if gradient_accumulation_steps > 1:
                total_loss /= gradient_accumulation_steps

            tr_loss += total_loss.item()
            nb_tr_steps += 1

            try:
                total_loss.backward()
            except RuntimeError as err:
                print('RuntimeError loss.backward(): ', err)

            if (step + 1) % params["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                global_steps += 1

                # Clear GPU unused RAM:
                if params['gpu'] >= 0:
                    torch.cuda.empty_cache()
            
        # save for batches
        ep_loss = tr_loss / nb_tr_steps
        # ner_loss = total_loss.item() / nb_tr_steps
        tr_batch_losses_.append(float("{0:.2f}".format(ep_loss)))
        # print()
        debug(f"[2] Train loss: {ep_loss}\n")
        debug(f"[3] Global steps: {global_steps}\n")           
        print("+" * 10 + "RUN EVALUATION" + "+" * 10)
       
        n2c2_scores = eval(
                model=model,
                eval_dir=params['dev_data'],
                result_dir=result_dir,
                eval_dataloader=dev_data_loader,
                eval_data=dev_data,                                
                params=params,
                epoch=epoch,               
        )

        if n2c2_scores is not None:
            # show scores
            show_results(n2c2_scores, ner_prf_dev_str, ner_prf_dev_sof)
            scores.append(n2c2_scores['NER']['micro']['st_f'])

            if max(scores) <= n2c2_scores['NER']['micro']['st_f']:
                params['best_epoch'] = epoch

        # Clear GPU unused RAM:
        if params['gpu'] >= 0:
            torch.cuda.empty_cache()
        
        if epoch > params['best_epoch'] + 10:
            debug(f"Early stop after 10 epoch from the best epoch")
            return

    return max(scores)
  

def show_results(n2c2_scores, ner_prf_dev_str, ner_prf_dev_sof):
    ner_prf_dev_str.append(
        [
            float("{0:.2f}".format(n2c2_scores['NER']['micro']['st_p'])),
            float("{0:.2f}".format(n2c2_scores['NER']['micro']['st_r'])),
            float("{0:.2f}".format(n2c2_scores['NER']['micro']['st_f'])),
        ]
    )

    # tb_writer.add_scalar('f1-score', n2c2_scores['NER']['micro']['st_f'])
    ner_prf_dev_sof.append(
        [
            float("{0:.2f}".format(n2c2_scores['NER']['micro']['so_p'])),
            float("{0:.2f}".format(n2c2_scores['NER']['micro']['so_r'])),
            float("{0:.2f}".format(n2c2_scores['NER']['micro']['so_f'])),
        ]
    )

    # ner
    _ = extract_scores('n2c2 ner strict (micro)', ner_prf_dev_str)
    extract_scores('n2c2 ner soft (micro)', ner_prf_dev_sof)
