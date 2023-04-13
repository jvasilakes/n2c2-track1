# Convert files from brat annotated format to CoNLL format
#from: https://github.com/pranav-s/brat2CoNLL
from distutils.log import error
from os import listdir, path
from collections import namedtuple
import argparse


import scispacy
import spacy

nlp = spacy.load("en_core_sci_lg")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dir",
    dest="input_dir",
    type=str,
    default='test',
    help="Input directory where Brat annotations are stored",
)

parser.add_argument(
    "--output_file",
    dest="output_file",
    type=str,
    default='test.conll',
    help="Output file where CoNLL format annotations are saved",
)

class FormatConvertor:
    def __init__(self, input_dir: str, output_file: str):
        self.input_dir = input_dir
        self.output_file = output_file

        # self.input_dir = '/home/pranav/Dropbox (GaTech)/repos/brat2CoNLL/sample_input_data/'
        # self.output_file = '/home/pranav/Dropbox (GaTech)/repos/brat2CoNLL/sample_output_data/test.txt'

    def read_input(self, annotation_file: str, text_file: str):
        """Read the input BRAT files into python data structures
        Parameters
            annotation_file:
                BRAT formatted annotation file
            text_file:
                Corresponding file containing the text as a string
        Returns
            input_annotations: list
                A list of dictionaries in which each entry corresponds to one line of the annotation file
            text_string: str
                Input text read from text file
        """
        with open(text_file, 'r', encoding='UTF-8') as f:
            text_string = f.read()
        input_annotations = []
        print("Reading file ", annotation_file)
        # Read each line of the annotation file to a dictionary
        with open(annotation_file, 'r',encoding='UTF-8') as fi:
            for line in fi:
                annotation_record = {}
                if not line.startswith('T'): continue
                entry = line.split()                
                if len(entry) < 5:
                    raise Exception("Wrong format ", line)
                annotation_record["label"] = entry[1]
                annotation_record["start"] = int(entry[2])
                annotation_record["end"] = int(entry[3])
                annotation_record["text"] = ' '.join(entry[4:])
                input_annotations.append(annotation_record)
        # Annotation file need not be sorted by start position so sort explicitly. Can also be done using end position
        input_annotations = sorted(input_annotations, key=lambda x: x["start"])

        return input_annotations, text_string

    def clean_text(self, text):
        #if the whole text is space characters
        while ' ' in text:
            text = text.replace(' ', '')
        return text

    def parse_text(self):
        """Loop over all annotation files, and write tokens with their label to an output file"""
        file_pair_list = self.read_input_folder()

        with open(self.output_file, 'w',encoding='UTF-8') as fo:
            for file_pair in file_pair_list:                                
                annotation_file, text_file = file_pair.ann, file_pair.text
                input_annotations, text_string = self.read_input(annotation_file, text_file)
                text_string = text_string.replace('\n', ' ')
                # text_tokens = text_string.split()
                doc = nlp(text_string)                
                annotation_count = 0
                if len(input_annotations) == 0:
                    continue
                current_ann_start = input_annotations[annotation_count]["start"]
                current_ann_end = input_annotations[annotation_count]["end"]
                num_annotations = len(input_annotations)
                # current_index = 0
                for sent in doc.sents:                    
                    num_tokens = len(sent)
                    i = 0 # Initialize Token number                                    
                    while i < num_tokens:                       
                        # if current_index != current_ann_start:                           
                        if sent[i].idx != current_ann_start:
                            # if sent[i].idx > current_ann_start:
                            #     print("Differences: ", sent[i].idx - current_ann_start, sent[i])
                            #     return
                            # fo.write(f'{sent[i]} O\n')
                            fo.write('\t'.join([sent[i].text, sent[i].tag_,"O"]) + '\n')                            
                            #     current_index += len(sent[i].text)+1 if not sent[i].is_punct else len(sent[i].text)
                            # current_index += len(sent[i].text)+1 
                            i += 1
                        else:
                            label = input_annotations[annotation_count]["label"]
                            fo.write('\t'.join([sent[i].text, sent[i].tag_,"B-" + label]) + '\n')
                            # current_index += len(sent[i].text)+1 if not sent[i].is_punct else len(sent[i].text)
                            # current_index += len(sent[i].text)+1 
                            i += 1
                            while i < num_tokens and sent[i].idx <= current_ann_end and sent[i].idx + len(sent[i].text) <= current_ann_end:
                                # fo.write(f'{sent[i]} {label}\n')
                                # fo.write(sent[i].text + '\t' + label + '\n' )
                                fo.write('\t'.join([sent[i].text, sent[i].tag_,"I-" + label]) + '\n')
                                # current_index += len(sent[i].text)+1
                                # current_index += len(sent[i].text)+1 if not sent[i].is_punct else len(sent[i].text)
                                i += 1
                            annotation_count += 1
                            if annotation_count < num_annotations:
                                current_ann_start = input_annotations[annotation_count]["start"]
                                current_ann_end = input_annotations[annotation_count]["end"]
                    # current_index += 1
                    fo.write('\n') #end of a sentence


                # fo.write('\n') 
    
    # def write_output(self):
    #     """Read input from a single file and write the output"""
    #     input_annotations, text_string = self.read_input(annotation_file, text_file)
    #     self.parse_text(input_annotations, text_string)
    
    def read_input_folder(self):
        """Read multiple annotation files from a given input folder"""
        file_list = listdir(self.input_dir)
        annotation_files = sorted([file for file in file_list if file.endswith('.ann')])
        # print(annotation_files)
        file_pair_list = []
        file_pair = namedtuple('file_pair', ['ann', 'text'])
        # The folder is assumed to contain *.ann and *.txt files with the 2 files of a pair having the same file name
        for file in annotation_files:
            if file.replace('.ann', '.txt') in file_list:
                file_pair_list.append(file_pair(path.join(self.input_dir, file), path.join(self.input_dir, file.replace('.ann', '.txt'))))
            else:
                raise(f"{file} does not have a corresponding text file")
        
        return file_pair_list
            
if __name__ == '__main__':
    args = parser.parse_args()
    format_convertor = FormatConvertor(args.input_dir, args.output_file)
    format_convertor.parse_text()