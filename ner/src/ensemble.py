from brat_reader import BratAnnotations
import os
import argparse

def max_voting_splits(folderIn, folderOut):    
    '''
    '''
    for subFolder in os.listdir(folderIn): #split
        voters = os.listdir(os.path.join(folderIn, subFolder))
        numberOfVoter = len(voters)        
        predictions = {}    
        for voter in voters:            
            if not os.path.exists(os.path.join(folderIn, subFolder, voter, 'predict-dev-org')):
                numberOfVoter -= 1
                continue
            for file in os.listdir(os.path.join(folderIn, subFolder, voter, 'predict-dev-org')):
                file_predicts = {} #key: offset, value: number of votes
                span_texts = {} #key: offset, value: surface text
                if '.txt' in file: continue                
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
            annotations = {}
            for offset in predictions[file][0]:
                if predictions[file][0][offset] >= numberOfVoter/2:
                    annotations[offset] = predictions[file][1][offset]
            results = get_longest_annotation(annotations)
            with open (os.path.join(folderOut, subFolder, file), 'w') as write:
                entityId = 1
                for (start,end) in results:
                    write.write('T' + str(entityId) + '\tDrug ' + str(start) 
                                + ' ' + str(end) + '\t' + results[(start,end)] + '\n')
                    entityId += 1

def max_voting_test(folderIn, folderOut):    
    '''
    '''
    voters = os.listdir(folderIn)
    numberOfVoter = len(voters)        
    predictions = {}    
    for voter in voters:            
        if not os.path.exists(os.path.join(folderIn, voter, 'predict-test-org')):
            numberOfVoter -= 1
            continue
        for file in os.listdir(os.path.join(folderIn, voter, 'predict-test-org')):
            file_predicts = {}
            span_texts = {}
            if '.txt' in file: continue                
            if file in predictions:
                file_predicts, span_texts = predictions[file]
            anns = BratAnnotations.from_file(os.path.join(folderIn, voter, 'predict-test-org', file))
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
    if not os.path.exists(folderOut):
        os.mkdir(folderOut)
    for file in predictions:
        annotations = {}
        for offset in predictions[file][0]:
            if predictions[file][0][offset] >= numberOfVoter/2:
                annotations[offset] = predictions[file][1][offset]
        results = get_longest_annotation(annotations)
        with open (os.path.join(folderOut, file), 'w') as write:
            entityId = 1
            for (start,end) in results:
                write.write('T' + str(entityId) + '\tDrug ' + str(start) 
                            + ' ' + str(end) + '\t' + results[(start,end)] + '\n')
                entityId += 1
    
# def weighted_majority(folderIn, folderOut, folderGold, round):
def get_longest_annotation(annotations):
    '''
    annotations: key=offset of a span, value = text of the span
    '''
    start_offsets = {}
    for (start, end) in annotations:
        if start in start_offsets:
            start_offsets[start].append(end)
        else:
            start_offsets[start] = [end]
    results = {}
    for start in start_offsets:
        end = max(start_offsets[start])
        results[(start,end)] = annotations[(start,end)]
    
    return results


def main():
    '''
    to ensemble for different splits: input_dir should have the following structure:
        input_dir/*/model_name/predict-dev-org (*==split number)
    the folder 'predict-dev-org' contains the predictions with original offsets from the input .txt files
    the folder 'predict-dev-org' was created by running predict.py on the dev set

    to ensemble for the testing set: input_dir should have the structure of input_dir/model_name/predict-test-org
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir',type=str, help='--indir', default='experiments-i2b2')
    parser.add_argument('--outdir',type=str, help='--outdir', default='ensemble-all')
    parser.add_argument('--type',type=str, help='--type', default='dev')
    args = parser.parse_args()

    input_dir = getattr(args, 'indir')
    output_dir = getattr(args, 'outdir')
    type = getattr(args,'type')
    
    if type == 'dev':
        max_voting_splits(input_dir, output_dir)
    else:
        max_voting_test(input_dir, output_dir)

if __name__ == '__main__':
    main()