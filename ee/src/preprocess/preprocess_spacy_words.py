#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03-Feb-2022

author: panos
"""

import os
import sys
import re
import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import json
from collections import OrderedDict


genia_splitter = os.path.join("../common", "geniass")
temp_dict = {'T':'entities','E':'events', 'Certainty':'certainty', 'Actor': 'actor', 
             'Action':'action', 'Temporality':'temporality', 'Negation':'negation'}
###############################
spacy_ign = ["_","#","\n",'"',"@",'-']

def spacy_replace(s):
    for special in spacy_ign:
        s = s.replace(special, '')
    return s
def parse_annotation_line(line):
    x = line.strip().split('\t')
    if x[0][0]=='T': # ( tr_id  (cl st end)  name)
        if len(x[1].split(';')) > 1: # ( tr_id  (cl st end; st end)  name)
            cl, st = x[1].split(' ')[:2]
            end =  x[1].split(' ')[-1]
        else:
            cl, st, end = x[1].split(' ')
        
        annot = {'id': x[0], 'name': '\t'.join(x[2:]),'st':int(st), 'end':int(end), 'cl':cl} #sent_no word_id bio
        return temp_dict['T'], annot
    
    elif x[0][0]=='E': # ( ev_id (cl:tr_id) )
        eid = x[0]
        cl, tr = x[1].split(':')
        annot = {'eid': eid, 'event-type':cl, 'tr':tr}
        return temp_dict['E'], annot
    elif x[0][0]=='A': # ( a_id type ev_id cat)
#         typ = x[1].split(' ')[0]
        typ, eid, cl = x[1].split(' ')
        annot = {'aid':x[0], 'type':typ, 'eid': eid,'cl':cl}
        return temp_dict[typ], annot
    else:
        print(x)
        raise Exception
        

def get_offsets(sentences):
    offsets = {0:  0}
    for i in range(len(sentences)-1):
        offsets[i+1] = offsets[i] + len(sentences[i]) + 1
    return offsets

def adjust_offsets(old_sents, new_sents, old_entities, f):
    """
    Adjust offsets based on tokenization
    Args:
        old_sents: (list) old, non-tokenized sentences
        new_sents: (list) new, tokenized sentences
        old_entities: (dic) entities with old offsets
    Returns:
        new_entities: (dic) entities with adjusted offsets
        abst_seq: (list) abstract sequence with entity tags
    """
    original = " ".join(old_sents)
    newtext = " ".join(new_sents)
    new_entities = []
    terms = {}
    for e in old_entities:
        start = e['st']
        end = e['end']
        old_pos = str(start)+' '+str(end)
        if (start, end) not in terms:
            terms[(start, end)] = [[start, end,  e['cl'], e['name'], e['id'], old_pos]]
        else:
            terms[(start, end)].append([start, end, e['cl'], e['name'], e['id'], old_pos])

    orgidx, newidx = 0, 0
    orglen, newlen = len(original), len(newtext)
    terms2 = terms.copy()
    while orgidx < orglen and newidx < newlen:
        # print(repr(original[orgidx]), orgidx, repr(newtext[newidx]), newidx)
        if original[orgidx] == newtext[newidx]:
            orgidx += 1
            newidx += 1
        elif original[orgidx] == "`" and newtext[newidx] == "'":
            orgidx += 1
            newidx += 1
        elif newtext[newidx] == '\n':
            newidx += 1
        elif original[orgidx] == '\n':
            orgidx += 1
        elif newtext[newidx] == ' ':
            newidx += 1
        elif original[orgidx] == ' ':
            orgidx += 1
        elif newtext[newidx] == '\t':
            newidx += 1
        elif original[orgidx] == '\t':
            orgidx += 1
        elif original[orgidx] in spacy_ign: 
            orgidx += 1
        elif newtext[newidx] in spacy_ign:
            newidx += 1

        else:
            print("Non-existent text in file %s: %d\t --> %s != %s " % (f, orgidx, repr(original[orgidx-10:orgidx+10]),
                                                             repr(newtext[newidx-10:newidx+10])))
            print('newtext', newtext)
            exit(0)
            

        starts = [key[0] for key in terms2.keys()]
        ends = [key[1] for key in terms2.keys()]

        if orgidx in starts:
            tt = [key for key in terms2.keys() if key[0] == orgidx]
            for sel in tt:
                for l in terms[sel]:
                    l[0] = newidx

        if orgidx in ends:
            tt2 = [key for key in terms2.keys() if key[1] == orgidx]
            for sel2 in tt2:
                for l in terms[sel2]:
                    if l[1] == orgidx:
                        l[1] = newidx

            for t_ in tt2:
                del terms2[t_]
        
    #   fix_sent_break
    newtext_break = "\n".join(new_sents)
    # check if test has issue with splittig entity
    sents_break = newtext_break.split('\n')
    ##
    cur = 0
    new_sent_range = []
    for s in sents_break:
        new_sent_range += [(cur, cur + len(s))]
        cur += len(s) + 1
        
    ent_sequences = []
    for ts in terms.values():
        for term in ts:
            new_ts = newtext_break[term[0]:term[1]].replace(" ", "").replace("\n", "")
            new_ts = spacy_replace(new_ts)
            old_ts = term[3].replace(" ", "").replace('\n', '')
            old_ts = spacy_replace(old_ts)
            if new_ts != old_ts:
                if new_ts.lower() == old_ts.lower():
                    tqdm.write('ENT_ID {}, Lowercase Issue: {} <-> {}'.format(term[4],new_ts, old_ts))
                    
                else:
                    tqdm.write('ENT_ID {}, Entities do not match: {} <-> {}'.format(term[4], new_ts, old_ts))
                exit()
                    
            # Find sentence number of each entity
            sent_no = []
            for s_no, sr in enumerate(new_sent_range):
                if set(np.arange(term[0], term[1])).issubset(set(np.arange(sr[0], sr[1]))):
                    sent_no += [s_no]

            assert (len(sent_no) == 1), '{} ({}, {}) -- {} -- {} <> {}'.format(sent_no, term[0], term[1],
                                                                               new_sent_range,
                                                                               newtext_break[term[0]:term[1]], term[3])
            new_entity = {'id':term[4],'name': newtext[term[0]:term[1]],'st':int(term[0]), 
                          'end':int(term[1]), 'cl':term[2], 'sent_no': sent_no[0], 'old_pos': term[5]}

            new_entities.append(new_entity)


    return new_entities, new_sents

### Convert dataset to Pubtator
filt = ["LICENSE","README"]

def check_overlap(ord_dict,f):
    prev_en = -1
    for i, (st,en) in enumerate(ord_dict):
        if st < prev_en:
            print("We have overlapping issue here",f)
            return True
        prev_en = en
    return False

def preprocess_spacy(data_path,folder_path, window):
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f not in filt]
    uniquefiles = list(set([f.split(".")[0] for f in onlyfiles if f.split(".")[0]]))
    uniquefiles.sort()

    ### Lets work with test 
    stats = {'entity_count':0, 'event_count': 0, 'event_type_count' :{'Disposition':0,'NoDisposition':0,'Undetermined':0},'unique_entity_count':0, 'no_sents':0, 'len_sents':0, 'avg_sent':0}
    with open(join(folder_path,data_path.split('/')[-2]+'_data.txt'),"w") as fout:
        with open(join(folder_path,data_path.split('/')[-2]+'_spacy_special.txt'),"w") as fspecial:
            for f in tqdm(uniquefiles[0:]):
                data = {'entities':[], 'events':[], 'certainty':[], 'actor':[],
                        'action':[], 'temporality':[], 'negation':[]}
                with open(join(data_path,f+".txt"),'r') as infile:
                    lines = infile.readlines()
                
                with open(join(data_path,f+'.ann'), 'r') as annfile:
                    for line in annfile.readlines():
                        cat, parsed = parse_annotation_line(line)
                        data[cat].append(parsed)
                        
                with open(join(folder_path+data_path.split('/')[-2], f+"_spacy.json"), 'r') as senfile:
                    new_sentences = [json.loads(line.strip()) for line in senfile]
                ## Processing
                orig_sentences = ''.join(lines).split('\n')
                split_sentences = [sent['_text'] for sent in new_sentences]
                new_entities, split_sentences = adjust_offsets(orig_sentences, split_sentences, data['entities'], f)
                data['entities'], data['sents']  = new_entities, split_sentences
                stats['no_sents'] += len(split_sentences)
                stats['len_sents'] += sum([len(s) for s in split_sentences])
                stats['entity_count'] += len(data["entities"])
                stats['event_count'] += len(data["events"])
                ###
                for ev in data['events']:
                    stats['event_type_count'][ev['event-type']] +=1
                unique = {}
                for ent in data['entities']:
                    if (ent['st'],ent['end']) in unique:
                        unique[(ent['st'],ent['end'])].append(ent)
                    else:
                        unique[(ent['st'],ent['end'])] = [ent]
                stats['unique_entity_count'] += len(unique.keys())
                overlap = check_overlap(OrderedDict(sorted(unique.items())),f)
                if overlap:
                    exit
                ###
                max_sent = len(split_sentences)
                offsets = get_offsets(split_sentences)
                ## Adding verb information
                sent_verbs = {}
                for sent in new_sentences:
                    sent_verbs[sent["sent_index"]] = sent["verbs"]
                for st,end in unique.keys(): # Each is gonna be a sample
                    entries = unique[(st,end)]
                    sent_no = entries[0]['sent_no']
                    sent_range = np.arange(start=max(0,sent_no - window), stop=min(max_sent,sent_no + window + 1))
                    txt = ' '.join(list(split_sentences[i] for i in sent_range))
                    offset = offsets[min(sent_range)]
                    ###
                    special = -1  ##special are the cases where the entity is concatenated with other chars
                    words = txt.split(' ')
                    original_words = txt.split(' ')
                    i, count = 0, 0
                    while count < st-offset  :
                        count += len(words[i]) + 1
                        i+=1
                    
                    if count > st-offset: ## we need to split
                        d = count - (st-offset)
                        l = len(words[i-1]) + 1
                        words.insert(i, words[i-1][l-d:])
                        words[i-1] = words[i-1][:l-d]
                        count -= d
                        special = count
                    new_st = i
                    while count < end-offset :
                        count += len(words[i]) + 1
                        i+=1
                    if count > end-offset+1: ## we need to split
                        d = count - (end-offset+1)
                        l = len(words[i-1])
                        words.insert(i, words[i-1][l-d:])
                        words[i-1] = words[i-1][:l-d] 
                        special = count
                    new_en = i
                    
                    sample = {'text': ' '.join(words)}
                    sample['trig'] = {'s': new_st, 'e':new_en, 'name': entries[0]['name'], 'old_pos': entries[0]['old_pos']}  
                    check (words, new_st, new_en, entries[0]['name'], original_words)
                        
                        
                    ids = [entry['id'] for entry in entries]
                    sample['events'] = [event['event-type'] for event in data['events'] if event['tr'] in ids]
                    sample['fname'] = f
                    ###
                    eids = [event['eid'] for event in data['events'] if event['tr'] in ids]
                    sample['actions'] =[action['cl'] for action in data['action'] if action['eid'] in eids]
                    if len(sample['actions'])> 1:
                        print('We got you ', f, eids, sample['actions'])
                    ###
                    sample['verbs'] = []
                    curr, off = 0, 0
                    verbs=[]
                    for sent_no in sent_range:
                        verbs_sent = sent_verbs[sent_no]
                        for verb in verbs_sent:
                            if special > 0 and curr + verb["st"]> special:
                                off =1
                            verbs.append({"t":verb["t"], "st":verb["st"]+curr+off, "en":verb["en"]+curr+off})
                        curr += len(split_sentences[sent_no]) + 1    
                            
                    i, count = 0, 0
                    for j in range(0, len(verbs)):
#                     while count < curr and j< len(verbs):
                        verb = verbs[j]
                        while count < verb['st']  :
                            count += len(words[i]) + 1
                            i+=1

                        new_st = i
                        while count < verb['en'] :
                            count += len(words[i]) + 1
                            i+=1
                        new_en = i
                        sample['verbs'].append({"t":verb["t"], "st":new_st, "en":new_en})
                        
                    for verb in sample['verbs']: #check
                        if ' '.join(words[verb['st']: verb['en']])!= verb['t']:
                            print('Problem {}  <> {} special {}'.format(' '.join(words[verb['st']: verb['en']]),verb['t'], special))
                    fout.write(json.dumps(sample) + '\n')
                    if len(entries) > 1 or len(sample['events']) > 1:
                        fspecial.write(json.dumps(sample) + '\n')
            stats['avg_sent'] = stats['len_sents'] /stats['no_sents']
    return stats


def check(words, new_st, new_en, name_str, original_words):

    if ' '.join(words[new_st: new_en])!= name_str:
        print(original_words)
        print(words)
        print(sample['text'])
        print(' '.join(words))
        print('start {} end {}'.format(new_st, new_en))                        
        print('Problem {}  <> {}'.format(' '.join(words[new_st: new_en]), name_str))
    
def main():
    # from preprocess import preprocess
    data_path = ["../data/train/","../data/dev/"] 
    preprocess_path = "../data/preprocessed/"

    stats = {}
    for i,path in enumerate(data_path):
        stats[path.split('/')[-2]] = preprocess(path, preprocess_path, 1)

    with open(join(preprocess_path,'stats.txt'),"w") as st_out:
         st_out.write(json.dumps(stats))
        
if __name__ == "__main__":
    main()