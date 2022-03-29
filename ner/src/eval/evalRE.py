import os
from collections import defaultdict

from utils.utils import write_lines


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



def estimate_ent(ref_dir, result_dir, fids, ent_anns, params):
    """Evaluate entity performance using n2c2 script"""

    # generate brat prediction
    gen_annotation_ent(fids, ent_anns, params, result_dir)

    # calculate scores
    pred_dir = ''.join([result_dir, 'ent-last/ent-ann/'])
    pred_scores_file = ''.join([result_dir, 'ent-last/ent-scores-', params['ner_eval_corpus'], '.txt'])

    # run evaluation, output in the score file
    eval_performance(ref_dir, pred_dir, result_dir, pred_scores_file, params)

    # extract scores
    scores = extract_fscore(pred_scores_file)

    return scores


def estimate_rel(ref_dir, result_dir, fids, ent_anns, rel_anns, params):
    """Evaluate entity and relation performance using n2c2 script"""

    # generate brat prediction
    gen_annotation(fids, ent_anns, rel_anns, params, result_dir)

    # calculate scores
    pred_dir = ''.join([result_dir, 'rel-last/rel-ann/'])
    pred_scores_file = ''.join([result_dir, 'rel-last/rel-scores-', params['ner_eval_corpus'], '.txt'])

    # run evaluation, output in the score file
    eval_performance(ref_dir, pred_dir, result_dir, pred_scores_file, params)

    # extract scores
    scores = extract_fscore(pred_scores_file)

    return scores

def gen_annotation_ent(fidss, ent_anns, params, result_dir):
    """Generate entity and relation prediction"""

    dir2wr = ''.join([result_dir, 'ent-last/ent-ann/'])
    if not os.path.exists(dir2wr):
        os.makedirs(dir2wr)
    else:
        os.system('rm ' + dir2wr + '*.ann')

    # Initial ent+rel map
    map = defaultdict()
    for fids in fidss:
        for fid in fids:
            map[fid] = {'ents': {}, 'rels': {}}

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

            entities = map[fid]['ents']
            # e_count = len(entities) + 1

            for x, pair in enumerate(span_indices):
                if pair[0].item() == -1:
                    break
                if ner_preds[x] > 0:
                    # e_id = 'T' + str(e_count)
                    # e_count += 1
                    try:
                        e_id = ner_terms.id2term[x]
                        e_type = params['mappings']['rev_type_map'][
                            params['mappings']['nn_mapping']['tag2type_map'][ner_preds[x]]]
                        if 'pipeline_entity_org_map' in params:
                            if e_id in params['pipeline_entity_org_map'][fid]:
                                e_words, e_offset = params['pipeline_entity_org_map'][fid][e_id]
                            else:
                                print(e_id)
                                e_words, e_offset = get_entity_attrs(pair, words, offsets, sub_to_words)
                        else:
                            e_words, e_offset = get_entity_attrs(pair, words, offsets, sub_to_words)
                        # entity_map[(xb, (pair[0].item(), pair[1].item()))] = (
                        #     ner_preds[x], e_id, e_type, e_words, e_offset)
                        entity_map[(xb, x)] = (
                            ner_preds[x], e_id, e_type, e_words, e_offset)
                        entities[e_id] = {"id": e_id, "type": e_type, "start": e_offset[0], "end": e_offset[1],
                                          "ref": e_words}
                    except KeyError as error:
                        print('pred not map term', error, fid)
        

    for fid, ners_rels in map.items():
        write_annotation_file(ann_file=dir2wr + fid + '.ann', entities=ners_rels['ents'])



def gen_annotation(fidss, ent_anns, rel_anns, params, result_dir):
    """Generate entity and relation prediction"""

    dir2wr = ''.join([result_dir, 'rel-last/rel-ann/'])
    if not os.path.exists(dir2wr):
        os.makedirs(dir2wr)
    else:
        os.system('rm ' + dir2wr + '*.ann')

    # Initial ent+rel map
    map = defaultdict()
    for fids in fidss:
        for fid in fids:
            map[fid] = {'ents': {}, 'rels': {}}

    for xi, (fids, ent_ann, rel_ann) in enumerate(zip(fidss, ent_anns, rel_anns)):
        # Mapping entities
        entity_map = defaultdict()
        for xb, (fid) in enumerate(fids):
            span_indices = ent_ann['span_indices'][xb]
            ner_terms = ent_ann['ner_terms'][xb]
            ner_preds = ent_ann['ner_preds'][xb]
            words = ent_ann['words'][xb]
            offsets = ent_ann['offsets'][xb]
            sub_to_words = ent_ann['sub_to_words'][xb]

            entities = map[fid]['ents']
            # e_count = len(entities) + 1

            for x, pair in enumerate(span_indices):
                if pair[0].item() == -1:
                    break
                if ner_preds[x] > 0:
                    # e_id = 'T' + str(e_count)
                    # e_count += 1
                    try:
                        e_id = ner_terms.id2term[x]
                        e_type = params['mappings']['rev_type_map'][
                            params['mappings']['nn_mapping']['tag2type_map'][ner_preds[x]]]
                        if 'pipeline_entity_org_map' in params:
                            if e_id in params['pipeline_entity_org_map'][fid]:
                                e_words, e_offset = params['pipeline_entity_org_map'][fid][e_id]
                            else:
                                print(e_id)
                                e_words, e_offset = get_entity_attrs(pair, words, offsets, sub_to_words)
                        else:
                            e_words, e_offset = get_entity_attrs(pair, words, offsets, sub_to_words)
                        # entity_map[(xb, (pair[0].item(), pair[1].item()))] = (
                        #     ner_preds[x], e_id, e_type, e_words, e_offset)
                        entity_map[(xb, x)] = (
                            ner_preds[x], e_id, e_type, e_words, e_offset)
                        entities[e_id] = {"id": e_id, "type": e_type, "start": e_offset[0], "end": e_offset[1],
                                          "ref": e_words}
                    except KeyError as error:
                        print('pred not map term', error, fid)
        if len(rel_ann) > 0:
            # Mapping relations
            pairs_idx = rel_ann['pairs_idx']
            rel_preds = rel_ann['rel_preds']
            # positive_indices = rel_ann['positive_indices']

            # if positive_indices:
            # pairs_idx_i = pairs_idx[0][positive_indices]
            # pairs_idx_j = pairs_idx[1][positive_indices]
            # pairs_idx_k = pairs_idx[2][positive_indices]
            # else:
            pairs_idx_i = pairs_idx[0]
            pairs_idx_j = pairs_idx[1]
            pairs_idx_k = pairs_idx[2]

            for x, i in enumerate(pairs_idx_i):
                relations = map[fids[i]]['rels']
                r_count = len(relations) + 1

                j = pairs_idx_j[x]
                k = pairs_idx_k[x]
                rel = rel_preds[x].item()
                role = params['mappings']['rev_rel_map'][rel].split(":")[1]
                # role = params['mappings']['rev_rtype_map'][rel]
                if role != 'Other':
                    # arg1s = entity_map[
                    #     (i.item(), (ent_ann['span_indices'][i][j][0].item(), ent_ann['span_indices'][i][j][1].item()))]
                    # arg2s = entity_map[
                    #     (i.item(), (ent_ann['span_indices'][i][k][0].item(), ent_ann['span_indices'][i][k][1].item()))]
                    try:
                        arg1s = entity_map[(i.item(), j.item())]
                        arg2s = entity_map[(i.item(), k.item())]

                        if int(params['mappings']['rev_rel_map'][rel].split(":")[0]) > int(
                                params['mappings']['rev_rel_map'][rel].split(":")[-1]):
                            arg1 = arg2s[1]
                            arg2 = arg1s[1]
                        else:
                            arg1 = arg1s[1]
                            arg2 = arg2s[1]
                        r_id = 'R' + str(r_count)
                        r_count += 1
                        relations[r_id] = {"id": r_id, "role": role,
                                           "left_arg": {"label": "Arg1", "id": arg1},
                                           "right_arg": {"label": "Arg2", "id": arg2}}
                    except KeyError as error:
                        print('error relation', fids[i], error)

                    # r_id = 'R' + str(r_count)
                    # r_count += 1
                    # relations[r_id] = {"id": r_id, "role": role,
                    #                    "left_arg": {"label": "Arg1", "id": arg2},
                    #                    "right_arg": {"label": "Arg2", "id": arg1}}

    for fid, ners_rels in map.items():
        write_annotation_file(ann_file=dir2wr + fid + '.ann', entities=ners_rels['ents'], relations=ners_rels['rels'])


def write_annotation_file(
        ann_file, entities=None, triggers=None, relations=None, events=None
):
    lines = []

    def annotate_text_bound(entities):
        for entity in entities.values():
            entity_annotation = "{}\t{} {} {}\t{}".format(
                entity["id"],
                entity["type"],
                entity["start"],
                entity["end"],
                entity["ref"],
            )
            lines.append(entity_annotation)

    if entities:
        annotate_text_bound(entities)

    if triggers:
        annotate_text_bound(triggers)

    if relations:
        for relation in relations.values():
            relation_annotation = "{}\t{} {}:{} {}:{}".format(
                relation["id"],
                relation["role"],
                relation["left_arg"]["label"],
                relation["left_arg"]["id"],
                relation["right_arg"]["label"],
                relation["right_arg"]["id"],
            )
            lines.append(relation_annotation)

    if events:
        for event in events.values():
            event_annotation = "{}\t{}:{}".format(
                event["id"], event["trigger_type"], event["trigger_id"]
            )
            for arg in event["args"]:
                event_annotation += " {}:{}".format(arg["role"], arg["id"])
            lines.append(event_annotation)

    write_lines(lines, ann_file)


def eval_performance(ref_dir, pred_dir, result_dir, pred_scores_file, params):
    # run evaluation script
    command = ''.join(
        ["python3 ", params['rel_eval_script_path'], " ", ref_dir, " ", pred_dir, " > ", pred_scores_file])
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
