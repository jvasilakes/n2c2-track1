EE task - 19/3/2022

Currecnt version uses PLM along with 10 verbs. Concat with i2b2 does not seem to improve performance.<br>
You can change the parameters you want on the config file.<br>
If you want to run the model on different splits, you have to obtain the train_data/dev_data.txt that requires 2 steps if preprocessing.<br>
After getting Nhung's output I will adjust it to run automatically.

## Training the model: 
```sh xec_n2c2_train.sh <blue,base,clinical> <default/,split0-4>  <scispacy, make, > ``` <br>
Do note that there is a change compared to previous models and verbs are the default option. To disable them use ```--no_verbs```.<br>
The predictions (on dev), log and the saved model are saved at ```results/<model>_<split>/``` folder.

To run the model on all splits use: 
```sh xec_n2c2_splits.sh <train,test> <scispacy, make>```<br>
scispacy: rerun the whole preprocessing<br>
make: only make the training files<br>
nothing: assume train_data.txt and dev_data.txt exist.

## Evaluating the predictions:
```
python main.py --config ../configs/local.yaml --mode predict --test_path ../data/default/spacy/dev_data.txt --bert blue --model_folder ../results/blue_default/
```
The results predictions will be under ```predictions/test/``` of the model folder specified.
To evalutate the predictions you have to give the two folders (gold, predicted) to the eval script:<br>
```
python eval_script_v3.py ../data/default/brat/dev ../results/blue_default/predictions/test/
```

