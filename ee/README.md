## Configure  env
To install the required dependencies:
```
pip install -r requirements.txt
```
Note that we run the experiments using ```Python 3.10.4```

## Training the model: 
```
sh xec_n2c2_train.sh <blue, base, clinical> <default, split0-4>  <scispacy, make, run> <types, baseline_mtl, LCM> 
```
`scispacy`: rerun the whole preprocessing<br>
`make`: only make the training files<br>
`run`: assume train_data.txt and dev_data.txt exist.<br>
The predictions (on dev), log and the saved model are saved at ```results/<model>_<split>/``` folder.<br>
`types`: the typed model | `LCM`: levitated context markers | `baseline_mtl`: baseline model<br>
To run the model on all splits use:
```
sh xec_n2c2_splits.sh <blue, base, clinical> <scispacy, make, run> <types, baseline_mtl, LCM>
```

## Predicting with the model:
```
sh xec_n2c2_predict.sh <blue, base, clinical> <data_folder> <scispacy, run> 
```
This model predicts accoring to the trained ```bert.model``` which should be saved in ```results/<model>_<split>/```.<br>
All predictions are saved under ```results/<model>_<split>/predictions/test/```.<br>
If you want to predict directly with a model then:
```
python main.py --config ../configs/local.yaml --mode predict --test_path <path_to_data.txt> --bert blue --model_folder <path_to_saved_model_folder> --approach types
```
The resulting predictions will be under ```predictions/test/``` of the model folder specified.<br>
To evalutate the predictions you have to give the two folders (gold, predicted) to the eval script:<br>
```
python eval_script_v3.py <path_to_golden_labels_folder> <path_to_predicted_labels_folder>
```
