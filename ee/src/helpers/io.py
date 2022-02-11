import os
import sys
import yaml
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

def print_performance(epoch, state, monitor, secs, name='train'):
    if name == 'train':
        print('---------- Epoch: {:02d} ----------'.format(epoch))
    template = '{:<5} |  LOSS = {:10.4f} | Time {}   | Micro_F1  = {:.04f} <<<'
    print(template.format(name.upper(), state['total'] if state['total'] else 0.0,
                          humanized_time(secs), monitor['micro_f1']))
    template = '      | NoDisp_F1 = {:.04f} | Dispo_F1 = {:.04f} | Undet_F1  = {:.04f}'
    print(template.format(monitor['NoDisp_f1'], monitor['Disp_f1'], monitor['Und_f1']))
    template = '      |  Macro_Pr = {:.04f} | Macro_Re = {:.04f} | Macro_F1  = {:.04f}'
    print(template.format(monitor['macro'][0],monitor['macro'][1], monitor['macro'][2]))

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
            - batch_size        {}
            - grad accumulation {}
            - Learning rate     {}
            - Dropout Input     {}\t Dropout Output {}
            
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
                    params['batch_size'],  params['accumulate_batches'],
                    params['lr'], params['input_dropout'], params['output_dropout'], 
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