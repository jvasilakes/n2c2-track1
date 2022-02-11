EE task - 10/2/2022

A basic method based on Matching the Blanks and An Improved Baseline for Sentence-level Relation Extraction.

To test the dataset structure and see the returned batch run:
```
python datasets_v1.py --config ../configs/local.yaml
```
You can change the parameters on the config file. 

To run the model:

```
python main.py --config ../configs/local.yaml --mode train
```

Results:


---------- Epoch: 09 ----------
TRAIN |  LOSS =     0.0202 | Time 0h 01m 06s   | Micro_F1  = 0.9928 <<< \
      | NoDisp_F1 = 0.9954 | Dispo_F1 = 0.9960 | Undet_F1  = 0.9608\
      |  Macro_Pr = 0.9832 | Macro_Re = 0.9848 | Macro_F1  = 0.9840\
DEV   |  LOSS =     0.6529 | Time 0h 00m 02s   | Micro_F1  = 0.9129 <<<\
      | NoDisp_F1 = 0.9457 | Dispo_F1 = 0.8728 | Undet_F1  = 0.6897\
      |  Macro_Pr = 0.8849 | Macro_Re = 0.8066 | Macro_F1  = 0.8361\

* About %2 in the DEV are multi-labeled and my approach ingores though, so the F1 scores in an overestimation of the true task.

