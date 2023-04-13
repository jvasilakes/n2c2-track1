import os
import random

import time

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from model import training

from loader.prepData import prepdata
from loader.prepNN import mapping
from loader.prepNN import prep4nn

# from bert.optimization import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from model import SpanNER
from utils import utils

from utils.utils import load_bert_weights

from torch.nn import functional as F

def main():
    # check running time
    t_start = time.time()

    # set config path by command line
    inp_args = utils._parsing()
    config_path = inp_args['yaml']

    # set config path manually    
    # config_path = 'experiments/0/baseline/train-ner-vae.yaml'    

    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    parameters.update(inp_args)
   
    # print config
    utils._print_config(parameters, config_path)

    parameters['ner_learning_rate'] = float(parameters['ner_learning_rate'])
    
    if parameters['gpu'] >= 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cuda:" + str(parameters['gpu']) if torch.cuda.is_available() else "cpu")
        # torch.cuda.set_device(parameters['gpu'])
    else:
        device = torch.device("cpu")

    print('device', device)

    parameters['device'] = device

    # Fix seed for reproducibility
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(parameters['seed'])
    random.seed(parameters['seed'])
    np.random.seed(parameters['seed'])
    torch.manual_seed(parameters['seed'])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tokenizer_encoder = load_bert_weights(parameters)

    # Force predict = False
    parameters['predict'] = -1

    train_data = prepdata.prep_input_data(parameters['train_data'], parameters)
    dev_data = prepdata.prep_input_data(parameters['dev_data'], parameters)

    test_data = prepdata.prep_input_data(parameters['test_data'], parameters)

    # mapping
    parameters = mapping.generate_map(train_data, dev_data, test_data, parameters)  # add test data for mlee
    parameters['mappings']['nn_mapping'] = utils.gen_nn_mapping(parameters['mappings']['tag_map'],
                                                                parameters['mappings']['tag2type_map'],
                                                                parameters['trTypes_Ids'])

    train = prep4nn.data2network(train_data, 'train', tokenizer_encoder, parameters)
    dev = prep4nn.data2network(dev_data, 'val', tokenizer_encoder, parameters)

    if len(train) == 0:
        raise ValueError("Train set empty.")
    if len(dev) == 0:
        raise ValueError("Test set empty.")

    train_data = prep4nn.torch_data_2_network(cdata2network=train, tokenizer_encoder=tokenizer_encoder,
                                            params=parameters,
                                            do_get_nn_data=True)
    dev_data = prep4nn.torch_data_2_network(cdata2network=dev, tokenizer_encoder=tokenizer_encoder,                                                
                                            params=parameters,
                                            do_get_nn_data=True)

    trn_data_size = len(train_data['nn_data']['bert_tokens'])
    dev_data_size = len(dev_data['nn_data']['bert_tokens'])
    
    parameters['dev_data_size'] = dev_data_size

    train_data_ids = TensorDataset(torch.arange(trn_data_size))
    dev_data_ids = TensorDataset(torch.arange(dev_data_size))

    # shuffle
    train_sampler = RandomSampler(train_data_ids)

    train_dataloader = DataLoader(train_data_ids, sampler=train_sampler, batch_size=parameters['batchsize'])
    dev_sampler = SequentialSampler(dev_data_ids)
    dev_dataloader = DataLoader(dev_data_ids, sampler=dev_sampler, batch_size=parameters['batchsize'])

    # 2. model
    model = SpanNER.SpanNER(parameters)
    
    if parameters['start_epoch'] > 0:
        utils.handle_checkpoints(model=model,
                checkpoint_dir=parameters['model_dir'],
                params={
                    'device': device
                },
                resume=True)
   
    # 3. optimizer
    assert (
            parameters['gradient_accumulation_steps'] >= 1
    ), "Invalid gradient_accumulation_steps parameter, should be >= 1."

    parameters['batchsize'] //= parameters['gradient_accumulation_steps']

    num_train_steps = parameters['epoch'] * (
            (trn_data_size - 1) // (parameters['batchsize'] * parameters['gradient_accumulation_steps']) + 1)
    parameters['voc_sizes']['num_train_steps'] = num_train_steps

    model.to(device)

    # Prepare optimizer
    t_total = num_train_steps
    optimizer_grouped_parameters = utils.gen_optimizer_grouped_parameters(list(model.named_parameters()), "ner", parameters)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=parameters['ner_learning_rate'],
        correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=parameters['warmup_proportion']*t_total, num_training_steps=t_total) 
    
    training.train(train_data_loader=train_dataloader, dev_data_loader=dev_dataloader,
                train_data=train_data, dev_data=dev_data, params=parameters, model=model,
                optimizer=optimizer, 
                scheduler=scheduler,
        )
        
    print('TRAINING: DONE!')

    # calculate running time
    t_end = time.time()
    print('TOTAL RUNNING TIME: {}'.format(utils._humanized_time(t_end - t_start)))

    return


if __name__ == '__main__':
    main()
