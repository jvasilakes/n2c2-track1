import os
from os.path import exists, join
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import shutil




def setup_log(config, folder_name=None, mode='train'):
    """
    Setup .log file to record training process and results.
    Args:
        params (dict): model parameters
    Returns:
        model_folder (str): model directory
    """
    config['pred_dir'] = join(folder_name, 'predictions')
    if mode == 'train': #prediction on dev
        config['pred_dir'] = join(config['pred_dir'], 'dev')
    else:
        config['pred_dir'] = join(config['pred_dir'], 'test')
    experiment_name = exp_name(config)
    # model_folder = os.path.join(config['model_folder'], experiment_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    log_file = os.path.join(folder_name, mode + '.log')

    f = open(log_file, 'w')
    sys.stdout = Tee(sys.stdout, f)
    return experiment_name

def load_config(file):
    with open(file, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    return cfg


def exp_name(params):
    exp = []
    exp.append('bs={}'.format(params['batch_size']))
    exp.append('lr={}'.format(params['lr']))
    exp.append('wd={}'.format(params['weight_decay']))
    exp.append('c={}'.format(params['clip']))
    exp = '_'.join(exp)
    return exp

def humanized_time(second):
    """
    Args:
        second (float): time in seconds
    Returns: human readable time (hours, minutes, seconds)
    """
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)

def clean_tokenizer(tokens):
    clean_tokens = []
    cur_token = tokens[0]
    for i in range(len(tokens)-1):
        if len(tokens[i+1])>1 and tokens[i+1][0:2]=='##':
            cur_token += tokens[i+1][2:]
        else:
            clean_tokens.append(cur_token)
            cur_token = tokens[i+1]
    clean_tokens.append(cur_token)
    return clean_tokens


def print_single(f, no, category, pos, trig):
    
    # f.write('T{}\t{} {}\t{}\n'.format(no, category, pos, ' '.join(trig)))
    f.write('T{}\t{} {}\t{}\n'.format(no, category, pos, '/'.join(trig)))
    f.write('E{}\t{}:T{}\n'.format(no, category, no))

    return no + 1

def print_preds(tracker, loader, config, epoch, mode='dev'):
    print('-----Saving predictions for current epoch', epoch,'-----')
    dataset = loader.dataset
    samples = np.concatenate(tracker['samples'])
    event_pred = tracker['event_preds']
    action_pred = tracker['action_preds']
    ievent= dataset.ievent_vocab
    iaction = dataset.iaction_vocab
    file_dict = {}
    if exists(config['pred_dir']):
        shutil.rmtree(config['pred_dir'])
    os.makedirs(config['pred_dir'])

    for i, s in enumerate(samples):
        tmp = dataset[s][3].split('/')
        trig, fname = tmp[0:len(tmp)-1], tmp[-1]
        pos = dataset[s][4]

        if fname in file_dict:
            file_dict[fname].append((trig, pos, event_pred[i], action_pred[i]))
        else:
            file_dict[fname] = [(trig,  pos, event_pred[i], action_pred[i])]
    for fname, res_list in file_dict.items():
        with open(join(config['pred_dir'], fname+".ann"), "w") as tmp_file:
            # Move read cursor to the start of file.
            count = 0
            for trig, pos, e_preds, a_preds in res_list:
                if e_preds[0] == 1:
                    count = print_single(tmp_file, count, ievent[0], pos, trig) 
                if e_preds[1] == 1 or config['pipeline'] and not e_preds[0] and not e_preds[2]:
                    count = print_single(tmp_file, count, ievent[1], pos, trig)
                if e_preds[2] == 1: #Disposition
                    for j, pred in enumerate(a_preds):
                        if pred == 1:
                            _ = print_single(tmp_file, count, ievent[2], pos, trig)
                            tmp_file.write('A{}\tAction E{} {}\n'.format(count, count,iaction[j]))
                            count +=1
                    if np.sum(a_preds) == 0: ## meaning no action identified
                        count = print_single(tmp_file, count, ievent[2], pos, trig)
#                         print(fname)



                        
def print_cases(samples, preds, dev_loader, config, epoch):
    dataset = dev_loader.dataset
    tokenizer = dataset.tokenizer
    with open(config['dev_miss'],"w") as fout:
        for i, s in enumerate(samples):
            ids_tok = tokenizer.convert_ids_to_tokens(dataset[s][4])
            toks = list(filter(lambda tok: tok not in ['[PAD]','[SEP]','[CLS]'], ids_tok))
    #         print(toks)
            sent = ' '.join(clean_tokenizer(toks))
            pred_tuple = preds[i], dataset[s][1]
            eid = dataset[s][2]
            template = '{} {}| <<{}>>\n'
            fout.write(template.format(eid, pred_tuple, sent))
        fout.write("Epoch {}\n".format(epoch))
    
def print_start(epoch, state, mode, secs, disp_count):
    if mode == 'train' or mode =='test':
        print('---------- Epoch: {:02d} ----------'.format(epoch))    
    template = '\t{:<5} / LOSS = {:10.4f}  Time {}  Dispotion counts: {}/{}/{}/{}'
    print(template.format(mode.upper(), state['total'] if state['total'] else 0.0, 
                          humanized_time(secs), disp_count[0], disp_count[1],
                          disp_count[2], disp_count[3]))
    
def print_performance(epoch, state, typ, mon, secs, name='train', disp_count = None):

    if typ== 'events':    
        template = 'Events : Macro_Pr = {:.04f} | Macro_Re = {:.04f} | Macro_F1  = {:.04f} | Micro_Pr = {:.04f} | Micro_Re = {:.04f} | Micro_F1 = {:.04f} <<<'
    else:
        template = 'Actions: Macro_Pr = {:.04f} | Macro_Re = {:.04f} | Macro_F1  = {:.04f} | Micro_Pr = {:.04f} | Micro_Re = {:.04f} | Micro_F1 = {:.04f}'
    print(template.format(mon['macro_pr'],mon['macro_re'], mon['macro_f1'], mon['micro_pr'], mon['micro_re'], mon['micro_f1']))

class Tee(object):
    """
    Object to print stdout to a file.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f_ in self.files:
            f_.write(obj)
            f_.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f_ in self.files:
            f_.flush()

            
def print_options(params):
    print('''\nParameters:
            - Approach   {}
            - Marker single     {}\t Verbs             {}
            - MTL               {}\t Bert model        {}
            
            # optimise
            - batch_size        {}\t grad accumulation {}
            - Learning rate     {}\t Dropout           {}
            - Weight Decay      {}\t Gradient Clip     {}
            

            - Max seq len       {}\t Max verbs         {}
            - Epoch             {}\t Warmup Epochs     {}
            - Early stop        {}\t Patience          {}
            - Save folder       {}\t Mode              {}
            '''.format(
                    params['approach'],  params['single_marker'],
                    params['use_verbs'], not params['no_mtl'],
                    params['bert'], params['batch_size'],
                    params['accumulate_batches'], params['lr'],  params['dropout'],
                    params['weight_decay'], params['clip'],
                    params['max_tok_len'], params['max_verbs'],
                    params['epochs'], params['warmup_epochs'], 
                    params['early_stop'], params['patience'],
                    params['model_folder'], params['mode']
                    ))
    if params['mode'] == 'train':
        print('''Data:
                - Train/Dev         {}, {}
                '''.format(params['train_data'], params['dev_data']))
    elif params['mode'] == 'predict':
        print('''Data:
                - Test         {}
            '''.format(params['test_data']))

def plt_figure(self, epoch, classes, metric, y_title):

    plt.style.use('ggplot')
    figure(figsize=(10, 13), dpi=80)
    plt.barh(classes, metric)
    plt.gca().invert_yaxis()
    plt.title('Performance')
    plt.ylabel('Classes')
    plt.xlabel(y_title)
    plt.savefig('../figures/epoch'+str(epoch)+'.png', dpi=80, bbox_inches='tight')

def setup(args, config):
    config['mode'] = args.mode
    config['bert'] = args.bert
    config['approach'] = args.approach
    #
    config['use_verbs'] = not config['approach'] in ['baseline', 'baseline_mtl']

    config['no_mtl'] = config['approach'] in ['baseline', 'LCM_no_mtl']
    if config['no_mtl']:
        config['event_weight'] = 1.0

    config['use_verb_categories'] = args.approach == "types"
    config['only_typed_verbs'] = False  # args.only_typed_verbs

    ## single
    config['single_marker'] = args.approach == "types"
    config['pool_fn'] = 'attention-softmax' if config['approach'] in ['LCM_attention', 'types_attention'] else 'max'
    if not config['use_verbs']:
        config['max_verbs'] = 0
    elif config['single_marker']:
        config['max_verbs'] = 20
    else:
        config['max_verbs'] = 10
    if args.mode == 'train':
        config['train_data'] = config['data_dir'] + join(args.split, 'spacy/train_data.txt')
        config['dev_data'] = config['data_dir'] + join(args.split, 'spacy/dev_data.txt')
    elif args.mode == 'predict':
        config['test_data'] = args.test_path

    # config['pred_dir'] = join(config['pred_dir'], args.bert+'_'+args.train_split)
    config["intermediate_dim"] = 3 * config["hidden_dim"] if config['use_verbs'] else 2 * config["hidden_dim"]

    if args.model_folder: # all the results folder
        config['model_folder'] = args.model_folder
    else:
        config['model_folder'] = join(config['results_folder'], args.bert+'_'+args.split)

    # config['pred_data'] = args.outdir
    config['exp_name'] = args.bert



