from helpers.brat_reader import BratAnnotations
from helpers.io import print_single
import os
import argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

event_vocab = {'NoDisposition': 0, 'Undetermined': 1, 'Disposition': 2}
ievent_vocab = {v: k for k, v in event_vocab.items()}

# def max_voting_splits(folderIn, folderOut):
#     '''
#     '''
#     for subFolder in os.listdir(folderIn): #split
#         voters = os.listdir(os.path.join(folderIn, subFolder))
#         numberOfVoter = len(voters)
#         predictions = {}
#         for voter in voters:
#             if not os.path.exists(os.path.join(folderIn, subFolder, voter, 'predict-dev-org')):
#                 numberOfVoter -= 1
#                 continue
#             for file in os.listdir(os.path.join(folderIn, subFolder, voter, 'predict-dev-org')):
#                 file_predicts = {} #key: offset, value: number of votes
#                 span_texts = {} #key: offset, value: surface text
#                 if '.txt' in file: continue
#                 if file in predictions:
#                     file_predicts, span_texts = predictions[file]
#                 anns = BratAnnotations.from_file(os.path.join(folderIn, subFolder, voter, 'predict-dev-org', file))
#                 for span in anns.spans:
#                     start = span.start_index
#                     end = span.end_index
#                     if (start, end) not in file_predicts:
#                         file_predicts[(start, end)] = 1
#                         span_texts[(start, end)] = span.text
#                     else:
#                         file_predicts[(start, end)] += 1
#                 if file not in predictions:
#                     predictions[file] = (file_predicts, span_texts)
#         if not os.path.exists(os.path.join(folderOut, subFolder)):
#             os.mkdir(os.path.join(folderOut, subFolder))
#         for file in predictions:
#             annotations = {}
#             for offset in predictions[file][0]:
#                 if predictions[file][0][offset] >= numberOfVoter/2:
#                     annotations[offset] = predictions[file][1][offset]
#             results = get_longest_annotation(annotations)
#             with open (os.path.join(folderOut, file), 'w') as write:
#                 entityId = 1
#                 for (start,end) in results:
#                     write.write('T' + str(entityId) + '\tDrug ' + str(start)
#                                 + ' ' + str(end) + '\t' + results[(start,end)] + '\n')
#                     entityId += 1
filt = ["LICENSE", "README"]
etypes = ['NoDisposition', 'Undetermined', 'Disposition']

def ensemble_predict(pred_dir_list, ensemble_dir):
    if not os.path.exists(ensemble_dir):
        os.makedirs(ensemble_dir)

    ## is it possible to not include a file
    onlyfiles = [f for f in listdir(pred_dir_list[0]) if isfile(join(pred_dir_list[0], f)) and f not in filt]
    uniquefiles = list(set([f.split(".")[0] for f in onlyfiles if f.split(".")[0]]))
    uniquefiles.sort()
    predictions = {}

    for pred_dir in pred_dir_list:
        if not os.path.isdir(pred_dir):
            print("{} is not a valid directory".format(pred_dir))

    numberOfVoter = len(pred_dir_list)
    predictions = {}
    for f in tqdm(uniquefiles[0:]):
        predictions = {}
        span_texts = {}

        for pred_dir in pred_dir_list:
            pred_ann = BratAnnotations.from_file(join(pred_dir, f + ".ann"))

            local_pred = {}
            ## first of all, all the spans are the same
            # only depends how many times they are repeated

            for event in pred_ann.events: # this will increase the desposition recall
                span = event.span
                start, end = span.start_index, span.end_index
                span_texts[(start, end)] = [span.text]
                if (start, end) not in local_pred:
                    local_pred[(start, end)] = {'NoDisposition': 0, 'Undetermined': 0, 'Disposition': 0}
                local_pred[(start, end)][event.type] = 1

            for (start, end) in local_pred:
                if (start, end) not in predictions:
                    predictions[(start, end)] = {'NoDisposition': 0, 'Undetermined': 0, 'Disposition': 0}
                for etype in etypes:
                    predictions[(start, end)][etype] += local_pred[(start, end)][etype]  # 0 or 1

        final_predictions = {}
        for offset in predictions:
            final_predictions[offset] = [max(predictions[offset], key=predictions[offset].get)]

            # if offset not in final_predictions:
            #     final_predictions[offset] = ['Undetermined']
        count=0
        with open(join(ensemble_dir, f+".ann"), "w") as fensemble:
            for start, end in final_predictions.keys():
                for etype in final_predictions[(start,end)]:
                    count = print_single(fensemble, count, etype, str(start)+ ' ' + str(end), span_texts[(start, end)])

            # for span in pred_ann.spans:
            #     start = span.start_index
            #     end = span.end_index
            #     if (start, end) not in ent_pos:
            #         ent_pos[(start, end)] = 1
            #         span_texts[(start, end)] = span.text
            #     else:
            #         ent_pos[(start, end)] += 1
            # if len(gold_pos.keys()) > 0:
            #     pred_ann = BratAnnotations.from_file(join(pred_dir, f + ".ann"))
            #     for span in pred_ann.spans:
            #         start = span.start_index
            #         end = span.end_index
            #         if (start, end) not in gold_pos:
            #             print("error span not in golden")
            #         else:
            #             gold_pos[(start, end)] = -1
            #     for k,v in gold_pos.items():
            #         if v!= -1:
            #             print('Pos {},{} not in predictions for f {}'.format(k[0], k[1], f))
            #             exit()

 # for offset in predictions[file][0]:
#                 if predictions[file][0][offset] >= numberOfVoter/2:
#                     annotations[offset] = predictions[file][1][offset]
#             results = get_longest_annotation(annotations)
def main(args):
    ensemble_predict(args.pred_dir_list, args.ensemble_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir_list', nargs='+', help=' Prediction folders')
    parser.add_argument('--ensemble_out', type=str, help=' Prediction folders')
    args = parser.parse_args()
    main(args)
