import os
import argparse

"""
Convert annotations of n2c2 2010 to brat format
"""

#c="pain" 55:10 55:10||t="problem"||a="hypothetical"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--textdir", type=str,
                        default='auxiliary_data/n2c2_2010_concept_assertion_relation/partners/txt',  # noqa
                        help="Directory containing the original text file for i2b2 2009 annotations")
    parser.add_argument("--anndir", type=str,
                        default='auxiliary_data/n2c2_2010_concept_assertion_relation/partners/ast',  # noqa
                        help="Directory containing the files of n2c2 2009 annotations")
    parser.add_argument("--bratdir", type=str,
                        default='auxiliary_data/n2c2_2010_concept_assertion_relation/partners/ast_brat',  # noqa
                        help="Directory to save the annotations in brat format")
    return parser.parse_args()


def process_mention(annotation, offsets, orgiginal_text):
    start = annotation.index('"')
    end = annotation.index('"', start + 1)
    text = annotation[start+1:end].strip()
    str_offset = annotation[end+1:]
    mention_offsets = str_offset.strip().split(' ')
    if len(mention_offsets) != 2:
        print("Error in the format: ", annotation)
        return None
    else:           
        s_line_id, s_token_id = mention_offsets[0].split(':')        
        e_line_id, e_token_id = mention_offsets[1].split(':')      
        if int(s_line_id)-1 >= len(offsets) or int(e_token_id)-1 >= len(offsets):
            print("Errors in index of line " + s_line_id + "\n")
            print(annotation + "\t" + str(len(offsets[int(s_line_id)-1])))
            return None
        if int(s_token_id) >= len(offsets[int(s_line_id)-1]) or int(e_token_id) >= len(offsets[int(e_line_id)-1]):
            print("Errors in index of tokens " + annotation + "\n")
            print(annotation + "\t" + str(len(offsets[int(s_line_id)-1])))
            return None
        
        start = offsets[int(s_line_id)-1][int(s_token_id)][0] #line number start from 1
        end = offsets[int(e_line_id)-1][int(e_token_id)][1]
        ori_mention = orgiginal_text[start:end]        
        if s_line_id != e_line_id:
            ori_mention = ori_mention.replace('\n', ' ')            
        if (ori_mention.endswith('.') or ori_mention.endswith(';')) and ori_mention.lower() != text: 
            #in case the tokenisation is wrong, the original mention has a dot character at the end
            end = end - 1
            ori_mention = ori_mention[:-1]
        if ori_mention.lower() != text:
            print("Process mention: error in offsets " + annotation + "\n")
            print(ori_mention + "\t text: " + text)
            return None

        return (start, end, ori_mention)
       

def convert2brat(fileText, fileAnn, fileBrat):    
    with open(fileText) as fileread:        
        original_text = fileread.read()
        # lines = fileread.readlines()
        lines = original_text.split('\n')
        offsets = []
        global_position = 0        
        for id, line in enumerate(lines):
            sent_off = []
            tokens = line.split(' ')
            for tok_id, token in enumerate(tokens):                
                if '\t' in token:
                    sub_toks = token.split('\t')
                    for st in sub_toks:
                        check_token = original_text[global_position:global_position + len(st)]
                        assert st == check_token, "Process offset: error in offset " + line                    
                        sent_off.append((global_position, global_position + len(st), st))
                        global_position += len(st) + 1 # 1 for the space character or new line character    
                else:    
                    check_token = original_text[global_position:global_position + len(token)]
                    assert token == check_token, "Process offset: error in offset " + line                    
                    sent_off.append((global_position, global_position + len(token), token))
                    global_position += len(token) + 1 # 1 for the space character or new line character
            offsets.append(sent_off)
    
    with open(fileBrat, 'w') as filewrite:        
        with open(fileAnn) as fileread:
            att_id = 0
            for id, mention in enumerate(fileread):
                annotations = mention.strip().split('||')
                for annotation in annotations:
                    annotation = annotation.strip()
                    if annotation.startswith('c='):                        
                        medical = process_mention(annotation, offsets, original_text)                        
                    elif annotation.startswith('t='):
                        att_value = annotation[3:-1]
                        if medical != None:
                            filewrite.write("T" + str(id) + "\t" + att_value + " " + str(medical[0]) 
                                         + " " + str(medical[1]) + "\t" + medical[2] + "\n")
                    elif annotation.startswith('a='):
                        att_value = annotation[3:-1]                        
                        if att_value != 'nm':
                            if medical != None:
                                filewrite.write("A" + str(att_id) + '\tAssertion T' + str(id) + ' ' + att_value + '\n')                           
                                att_id += 1
                    

def convert2brat_gold(dirAnn, dirText, dirBrat):
    for file in os.listdir(dirAnn):       
        fileAnn = os.path.join(dirAnn, file)
        id = file[:-4]      
        print('Processing ', id)
        fileText = os.path.join(dirText, id + '.txt')
        fileBrat = os.path.join(dirBrat, id + '.ann')
        fileTextTarget = os.path.join(dirBrat, id + '.txt')
        os.system("cp " + fileText + " " + fileTextTarget)
        convert2brat(fileText, fileAnn, fileBrat)
       
def main(args):
    convert2brat_gold(args.anndir, args.textdir, args.bratdir)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)

