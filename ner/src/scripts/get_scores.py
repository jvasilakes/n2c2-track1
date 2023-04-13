import os

experiments = [
    # "baseline", "baseline+i2b2_2019", 
    "biolm", "biolm+i2b2_2019", 
    "clinical_bert", "clinical_bert+i2b2_2019",
    # "bluebert", "bluebert+i2b2_2019"
    ]
folders = ["0", "1", "2", "3", "4", "data_v3"]
EXP_DIR = "experiments"

def get_metrics(fileLog):
    scores = {}
    with open(fileLog) as f1:
        lines = [s.strip() for s in f1.readlines()]
        flage, flags = False, False
        for line in lines:
            if line.startswith("STRICT_MATCHING:"):
                flage = True
            if line.startswith("SOFT_MATCHING:"):
                flags = True
            if line.startswith("NER(MICRO)") and flage:
                pos1 = line.index('=')
                metrics = line[pos1+1:].split('\t')
                scores['e_pre'] = str(metrics[0].strip())
                scores['e_re'] = str(metrics[1].strip())
                scores['e_f1'] = str(metrics[2].strip())
                flage = False
            elif line.startswith("NER(MICRO)") and flags:
                pos1 = line.index('=')                
                metrics = line[pos1+1:].split('\t')
                scores['s_pre'] = str(metrics[0].strip())
                scores['s_re'] = str(metrics[1].strip())
                scores['s_f1'] = str(metrics[2].strip())
                flags = False
    
    return scores

def get_n2c2_scores(filename):
    scores = {}
    with open(filename) as file:
        for line in file:
            line = line.strip()
            if line.startswith("Drug"):
                metrics = line.split('  ')
                if len(metrics) < 8:
                    print(line)
                scores['e_pre'] = metrics[1]
                scores['e_re'] = metrics[2]
                scores['e_f1'] = metrics[3]
                scores['s_pre'] = metrics[5]
                scores['s_re'] = metrics[6]
                scores['s_f1'] = metrics[7]

                # print(cols)
    return scores

def get_all_metrics(filename):  
    result = []
    for i in range(0, len(experiments),2):
        header = '\t'.join([experiments[i]] + ["Exact"]*3 + ["Lenient"]*3 + [' ']
                + [experiments[i+1]] + ["Exact"]*3 + ["Lenient"]*3 )
        result.append(header)
        header = '\t'.join([' '] + ["Precision", "Recall", "F1"] * 2 + [' '] + 
                [' '] + ["Precision", "Recall", "F1"] * 2)
        result.append(header)

        for split in folders:
            file1 = os.path.join(EXP_DIR, split, experiments[i], filename)
            scores1 = get_n2c2_scores(file1)
            file2 = os.path.join(EXP_DIR, split, experiments[i+1], filename)
            scores2 = get_n2c2_scores(file2)
            line = '\t'.join([split, scores1['e_pre'], scores1['e_re'], scores1['e_f1'],
                            scores1['s_pre'], scores1['s_re'], scores1['s_f1'], " ",
                            split, scores2['e_pre'], scores2['e_re'], scores2['e_f1'],
                            scores2['s_pre'], scores2['s_re'], scores2['s_f1']])
            result.append(line)

        result.append("\n")
    
    with open(filename + "_summary", "w") as f1:
        for r in result:
            f1.write(r + "\n")


def extract_fscore(path): 
    #this is copied from eval/evaluation.py
    file = open(path, 'r')
    lines = file.readlines()
    report = {}
    report['NER'] = {}
    report['REL'] = {}
   
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

def get_ensemble_scores():
    
    result = []
    for folder in ["ensemble-6", "ensemble-all"]:
        header = '\t'.join([folder] + ["Exact"]*3 + ["Lenient"]*3 )
        result.append(header)
        for split in ["0", "1", "2", "3", "4", "data_v3"]:        
            logFile = os.path.join(EXP_DIR, folder, 'result_'+ split + ".txt")
            scores1 = extract_fscore(logFile)
            line = '\t'.join([split, scores1['st_p'], scores1['st_r'], scores1['st_f'],
                            scores1['so_p'], scores1['so_r'], scores1['so_f']])
            result.append(line)
    
    with open("ensemble_exp_summary.tsv", "w") as f1:
        for r in result:
            f1.write(r + "\n")
                

# get_metrics("dev_baseline.log")
get_all_metrics('dev_result.txt')
get_all_metrics('test_result.txt')
# get_ensemble_scores()
