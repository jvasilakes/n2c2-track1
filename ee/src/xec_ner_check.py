from helpers.brat_reader import BratAnnotations
import os
import argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

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


def check_ner(gold_dir, pred_dir):
    onlyfiles = [f for f in listdir(gold_dir) if isfile(join(gold_dir, f)) and f not in filt]
    uniquefiles = list(set([f.split(".")[0] for f in onlyfiles if f.split(".")[0]]))
    uniquefiles.sort()
    predictions = {}
    for f in tqdm(uniquefiles[0:]):
        gold_pos = {}
        span_texts = {}  # key: offset, value: surface text
        gold_ann = BratAnnotations.from_file(join(gold_dir, f + ".ann"))

        for span in gold_ann.spans:
            start = span.start_index
            end = span.end_index
            if (start, end) not in gold_pos:
                gold_pos[(start, end)] = 1
                span_texts[(start, end)] = span.text
            else:
                gold_pos[(start, end)] += 1
        if len(gold_pos.keys()) > 0:
            pred_ann = BratAnnotations.from_file(join(pred_dir, f + ".ann"))
            for span in pred_ann.spans:
                start = span.start_index
                end = span.end_index
                if (start, end) not in gold_pos:
                    print("error span not in golden")
                else:
                    gold_pos[(start, end)] = -1
            for k,v in gold_pos.items():
                if v!= -1:
                    print('Pos {},{} not in predictions for f {}'.format(k[0], k[1], f))
                    exit()


def main(args):
    check_ner(args.gold_dir, args.pred_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_dir', type=str, help='--indir')
    parser.add_argument('--pred_dir', type=str, help='--outdir')
    args = parser.parse_args()
    main(args)
