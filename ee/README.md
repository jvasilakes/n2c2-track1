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
```python main.py --config ../configs/local.yaml --mode train --train_data ../data/default/spacy/ --outdir ../evaluation/default/ --bert blue --model_folder ../saved_models/```
Do not that there is a change compared to previous models and verbs are the dault option. To disable them use ```--no_verbs```.<br>
## Testing the model:
```python main.py --config ../configs/local.yaml --mode predict --test_data ../data/default/spacy/dev_data.txt --outdir ../evaluation/default/ --bert blue --model_folder ../saved_models/blue/```
Model folder is the folder that contains the model. According to the parameters in config, the code will automatically load e.g. the ```bert.model``` file in the subfolder that matches the hyperparametrs (e.g. ```bs=32_lr=4e-05_wd=1e-06_c=10```)


Alternatively you can use: 
```sh xec_n2c2_splits.sh <base,clinical,blue>```

## Evaluating the predictions:
To evalutate the predictions you have to give the two folders (gold, predicted) to the eval script:<br>
```python eval_script_v3.py ../data/default/brat/dev/ ../predictions/test/```


