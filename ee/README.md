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


### Bert-base: entities marked with '@', epoch: 06 (Best event f1)
| Set | Task | Macro Pr | Macro Re | Macro f1 | Micro f1 |
|-------|--------|----------|----------|----------|----------|
| Train | Event  |  0.9904  |  0.9847  |  0.9875  |  0.9939  |
| Train | Action |  0.9828  |  0.7812  |  0.8074  |  0.9781  |
|  Dev  | Event  |  0.8687  |  0.8613  |  0.8649  |>>0.9225<<|
|  Dev  | Action |  0.7986  |  0.7013  |  0.7033  |  0.7511  |

<!-- ---------- Epoch: 06 ----------
	TRAIN / LOSS =     0.0191  Time 0h 01m 05s  Dispotion counts: 1128/1117/1136/6125
Events : Macro_Pr = 0.9904 | Macro_Re = 0.9847 | Macro_F1  = 0.9875 | Micro_F1 = 0.9939 <<<
actions y_pred size (1136, 7) y_pred sum 1154.0
Actions: Macro_Pr = 0.9828 | Macro_Re = 0.7812 | Macro_F1  = 0.8074 | Micro_F1 = 0.9781
actions y_pred size (6125, 7) y_pred sum 1544.0
Actions: Macro_Pr = 0.8632 | Macro_Re = 0.7812 | Macro_F1  = 0.7385 | Micro_F1 = 0.8379
	DEV   / LOSS =     0.2051  Time 0h 00m 03s  Dispotion counts: 201/205/229/1010
Events : Macro_Pr = 0.8687 | Macro_Re = 0.8613 | Macro_F1  = 0.8649 | Micro_F1 = 0.9225 <<<
actions y_pred size (229, 7) y_pred sum 222.0
Actions: Macro_Pr = 0.7986 | Macro_Re = 0.7013 | Macro_F1  = 0.7033 | Micro_F1 = 0.7511
actions y_pred size (1010, 7) y_pred sum 326.0
Actions: Macro_Pr = 0.6179 | Macro_Re = 0.7013 | Macro_F1  = 0.6275 | Micro_F1 = 0.6081
 -->
