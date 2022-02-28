import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def setup_log(params, folder_name=None, mode='train'):
    """
    Setup .log file to record training process and results.
    Args:
        params (dict): model parameters
    Returns:
        model_folder (str): model directory
    """
    if folder_name:
        model_folder = os.path.join(params['output_folder'], folder_name)
    else:
        model_folder = os.path.join(params['output_folder'], 'temp')

    experiment_name = exp_name(params)
    model_folder = os.path.join(model_folder, experiment_name)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    log_file = os.path.join(model_folder, mode + '.log')

    f = open(log_file, 'w')
    sys.stdout = Tee(sys.stdout, f)
    return model_folder, experiment_name

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
    
    f.write('T{}\t{} {}\t{}\n'.format(no, category, pos, trig))
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
    for i, s in enumerate(samples):
        tmp = dataset[s][3].split('/')
        trig, fname = tmp[0:len(tmp)-1],tmp[-1]
        pos = dataset[s][4]
        if fname in file_dict:
            file_dict[fname].append((trig, pos, event_pred[i], action_pred[i]))
            
        else:
            file_dict[fname] = [(trig,  pos, event_pred[i], action_pred[i])]      
            
    for fname, res_list in file_dict.items():
        with open(config['pred_dir']+mode+'/'+fname+".ann", "w") as tmp_file:
            # Move read cursor to the start of file.
            count = 0
            for trig, pos, e_preds, a_preds in res_list:
                if e_preds[0] == 1:
                    count = print_single(tmp_file, count, ievent[0], pos, trig) 
                if e_preds[2] == 1:
                    count = print_single(tmp_file, count, ievent[2], pos, trig)
                if e_preds[1] == 1: #Disposition
                    for j, pred in enumerate(a_preds):
                        if pred == 1:
                            _ = print_single(tmp_file, count, ievent[1], pos, trig)
                            tmp_file.write('A{}\tAction E{} {}\n'.format(count, count,iaction[j]))
                            count +=1
                    if np.sum(a_preds) == 0: ## meaning no action identified
                        count = print_single(tmp_file, count, ievent[1], pos, trig)
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
    
def print_performance(epoch, state, typ, mon, secs, name='train', disp_count = None):
    if name == 'train' and typ== 'events':
        print('---------- Epoch: {:02d} ----------'.format(epoch))
    if typ== 'events':    
        template = '\t{:<5} / LOSS = {:10.4f}  Time {}  Dispotion counts: {}/{}/{}/{}'
        print(template.format(name.upper(), state['total'] if state['total'] else 0.0, 
                              humanized_time(secs), disp_count[0], disp_count[1],
                              disp_count[2], disp_count[3]))
        template = 'Events : Macro_Pr = {:.04f} | Macro_Re = {:.04f} | Macro_F1  = {:.04f} | Micro_F1 = {:.04f} <<<'
    else:
        template = 'Actions: Macro_Pr = {:.04f} | Macro_Re = {:.04f} | Macro_F1  = {:.04f} | Micro_F1 = {:.04f}'
    print(template.format(mon['macro_pr'],mon['macro_re'], mon['macro_f1'], mon['micro_f1']))
#     template = '      | NoDisp_F1 = {:.04f} | Dispo_F1 = {:.04f} | Undet_F1  = {:.04f}'
#     print(template.format(monitor['NoDisp_f1'], monitor['Disp_f1'], monitor['Und_f1']))

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
            - Verbs             {}
            - batch_size        {}
            - grad accumulation {}
            - Learning rate     {}
            - Dropout           {}
            
            - Freeze Encoder    {}\tAutoscalling: {}  
            - Encoder Dim       {}\tLayers {}
            - Hidden dim        {}
            - Weight Decay      {}
            - Gradient Clip     {}
            - Max sen len       {}
            - Epoch             {}\tWarmup Epochs     {}
            - Early stop        {}\tPatience = {}
            - Train/Dev         {}, {}
            - Save folder       {}
            '''.format(
                    params['verbs'], params['batch_size'],  params['accumulate_batches'],
                    params['lr'],  params['dropout'], 
                    params['freeze_pretrained'], params['autoscalling'],
                    params['enc_dim'], params['enc_layers'], 
                    params['hidden_dim'], 
                    params['weight_decay'], params['clip'],
                    params['max_sen_len'],
                    params['epochs'], params['warmup_epochs'], 
                    params['early_stop'], params['patience'],
                    params['train_data'], params['dev_data'], params['output_folder'],
                    ))
def plt_figure(self, epoch, classes, metric, y_title):

    plt.style.use('ggplot')
    figure(figsize=(10, 13), dpi=80)
    plt.barh(classes, metric)
    plt.gca().invert_yaxis()
    plt.title('Performance')
    plt.ylabel('Classes')
    plt.xlabel(y_title)
    plt.savefig('../figures/epoch'+str(epoch)+'.png', dpi=80, bbox_inches='tight')
