EE task - 10/2/2022

A basic method based on Matching the Blanks and An Improved Baseline for Sentence-level Relation Extraction.

To test the dataset structure and see the returned batch run:
```
python datasets.py --config ../configs/local.yaml
```
You can change the parameters on the config file. 

To run the model:

```
python main.py --config ../configs/local.yaml --mode train
```

Results:


---------- Epoch: 09 ----------\
TRAIN |  LOSS =     0.0202 | Time 0h 01m 06s   | Micro_F1  = 0.9928 <<< \
      | NoDisp_F1 = 0.9954 | Dispo_F1 = 0.9960 | Undet_F1  = 0.9608\
      |  Macro_Pr = 0.9832 | Macro_Re = 0.9848 | Macro_F1  = 0.9840\
DEV   |  LOSS =     0.6529 | Time 0h 00m 02s   | Micro_F1  = 0.9129 <<<\
      | NoDisp_F1 = 0.9457 | Dispo_F1 = 0.8728 | Undet_F1  = 0.6897\
      |  Macro_Pr = 0.8849 | Macro_Re = 0.8066 | Macro_F1  = 0.8361\

* About %2 in the DEV are multi-labeled and my approach ingores though, so the F1 scores in an overestimation of the true task.

Running with the biomedicus:

---------- Epoch: 07 ----------\
TRAIN |  LOSS =     0.0477 | Time 0h 01m 12s   | Micro_F1  = 0.9833 <<<\
      | NoDisp_F1 = 0.9905 | Dispo_F1 = 0.9771 | Undet_F1  = 0.9283\
      |  Macro_Pr = 0.9680 | Macro_Re = 0.9627 | Macro_F1  = 0.9653\
DEV   |  LOSS =     0.6442 | Time 0h 00m 02s   | Micro_F1  = 0.8980 <<<\
      | NoDisp_F1 = 0.9375 | Dispo_F1 = 0.8636 | Undet_F1  = 0.5630\
      |  Macro_Pr = 0.8563 | Macro_Re = 0.7544 | Macro_F1  = 0.7880\
 
