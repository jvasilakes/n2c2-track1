from brat_reader import BratAnnotations
import os
import argparse

def maxVoting(folderIn, folderOut):
    predictions = {}    
    for subFolder in os.listdir(folderIn): #split
        voters = os.listdir(os.path.join(folderIn, subFolder))
        numberOfVoter =  len(voters)
        for voter in voters:            
            if not os.path.exists(os.path.join(folderIn, subFolder, voter, 'predict-dev-org')):
                numberOfVoter -= 1
                continue
            for file in os.listdir(os.path.join(folderIn, subFolder, voter, 'predict-dev-org')):
                if '.txt' in file: continue
                file_predicts = {}
                span_texts = {}
                if file in predictions:
                    file_predicts, span_texts = predictions[file]
                anns = BratAnnotations.from_file(os.path.join(folderIn, subFolder, voter, 'predict-dev-org', file))
                for span in anns.spans:
                    start = span.start_index
                    end = span.end_index
                    if (start, end) not in file_predicts:
                        file_predicts[(start, end)] = 1
                        span_texts[(start, end)] = span.text
                    else:
                        file_predicts[(start, end)] += 1
                if file not in predictions:
                    predictions[file] = (file_predicts, span_texts)
        if not os.path.exists(os.path.join(folderOut, subFolder)):
            os.mkdir(os.path.join(folderOut, subFolder))
        for file in predictions:
            with open (os.path.join(folderOut, subFolder, file), 'w') as write:
                entityId = 1
                for (start,end) in predictions[file][0]:
                    if predictions[file][0][(start,end)] >= numberOfVoter/2:
                        write.write('T' + str(entityId) + '\tDrug ' + str(start) 
                                + ' ' + str(end) + '\t' + predictions[file][1][(start,end)] + '\n')
                        entityId += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help='--indir', default='ner/experiments')
    parser.add_argument('--outdir', type=str,help='--outdir', default='ner/ensemble')
    args = parser.parse_args()

    input_dir = getattr(args, 'indir')
    output_dir = getattr(args, 'outdir')

    maxVoting(input_dir, output_dir)

