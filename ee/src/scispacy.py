import os
from os.path import isfile, join
import argparse
from glob import glob

import spacy
import scispacy  # noqa
from tqdm import tqdm
from helpers.io import *
import json
import re

def main(args):
    config = load_config(args.config)
    paths = [args.datadir+'train/', args.datadir +'dev/']
    nlp = spacy.load("en_core_sci_scibert")
#     nlp.max_length = 1000000 
    for path in paths:
        print(path)
        glob_path = join(path, "*.txt")
        glob_files = glob(glob_path)
        for fpath in tqdm(glob_files):
            with open(join(args.outdir+fpath.split('/')[-2], fpath.split('/')[-1].split('.')[0]+'_spacy.json'),"w") as fout:
                txt = open(fpath).read()
                doc = nlp(txt,disable = ['ner'])
                data = []
                sents = doc.sents
                sent_no = 0
                for sent in sents:
                    curr = 0
                    Verbs = []
                    clean_toks = [token for token in sent if clean_token(token.text)!='']
                    text_toks = [clean_token(token.text) for token in clean_toks]
                    base_sent = ' '.join(text_toks)
                    if base_sent ==',' or base_sent =='.': #special case
                        entry["_text"] += " "+ base_sent 
                        continue
                    elif base_sent =='': # mostly \n sent 
                        continue
                    for tok in clean_toks:
                        cl_tok = clean_token(tok.text)
                        end = curr + len(cl_tok)
                        if tok.pos_ =='VERB' and tok.tag_=='VBP':
                            print('cl tok <%s> is verb with tag <%s> ' % (cl_tok, tok.tag_))
                            Verbs.append({'t':cl_tok, 'st':curr, 'en':end})
                        curr = end + 1
                    for verb in Verbs:
                        if base_sent[verb['st']:verb['en']]!=verb['t']:
                            print('Error %s != %s in file %s' % (base_sent[verb['st']:verb['en']], verb['t'], fpath))
                    
                    entry =  {"sent_index": sent_no,  "_text": base_sent, "verbs": Verbs}
                    data.append(entry)
                    sent_no +=1
                            
                for entry in data:
                    fout.write(json.dumps(entry) + '\n')
                    
#             print(data)

spacy_ign = ["_","#","\n",'"',"@", '-']

def clean_token(s_tok):
    cl = s_tok.replace(" ","")
    for sym in spacy_ign:
        cl = cl.replace(sym,"")
    return cl
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--datadir', type=str)
    args = parser.parse_args()
    main(args)