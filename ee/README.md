EE task - 14/3/2022

A basic method based on Matching the Blanks and An Improved Baseline for Sentence-level Relation Extraction.\
The model is extend with PL-markers, and after experiment, max-pooling (1 emb) produces the best results.\
Pl markers also has singificantly higher action performance.\

To test the dataset structure and see the returned batch run:
```
python datasets.py --config ../configs/local.yaml --data ../data/spacy/ --bert clinical --use_verbs 
```
You can change the parameters on the config file. 

## Training the model: 
```
python main.py --config ../configs/local.yaml --mode train --split default --bert blue 
```
Do not that there is a change compared to previous models and verbs are the dault option. To disable them use ```--no_verbs```.<br>

Alternatively you can use: 
```sh xec_n2c2_splits.sh <base,clinical,blue>```
That runs the specified model on all the splits.

## Evaluating the predictions:
To evalutate the predictions you have to give the two folders (gold, predicted) to the eval script:<br>
```
python main.py --config ../configs/local.yaml --mode predict --test_path ../data/default/spacy/dev_data.txt --bert blue --model_folder ../results/blue_default/
```
The results predictions will be under ```predictions/test/``` of the model folder specified.



