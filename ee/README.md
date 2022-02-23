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

### Clinical Bert: entities marked with '@', epoch: 04 (Best event f1)
| Set | Task | Macro Pr | Macro Re | Macro f1 | Micro f1 |
|-------|--------|----------|----------|----------|----------|
| Train | Event  |  0.9754  |  0.9709  |  0.9731  |  0.9873  |
| Train | Action |  0.9503  |  0.6560  |  0.6967  |  0.9355  |
|  Dev  | Event  |  0.8881  |  0.9235  |  0.9021  |  0.9397<<|
|  Dev  | Action |  0.7032  |  0.6455  |  0.6626  |  0.7624  |

<!-- ---------- Epoch: 04 ----------
	TRAIN / LOSS =     0.0401  Time 0h 01m 35s  Dispotion counts: 1128/1127/1158/6125
Events : Macro_Pr = 0.9754 | Macro_Re = 0.9709 | Macro_F1  = 0.9731 | Micro_F1 = 0.9873 <<<
actions y_pred size (1158, 7) y_pred sum 1147.0
Actions: Macro_Pr = 0.9503 | Macro_Re = 0.6560 | Macro_F1  = 0.6967 | Micro_F1 = 0.9355
actions y_pred size (6125, 7) y_pred sum 3512.0
Actions: Macro_Pr = 0.7311 | Macro_Re = 0.6560 | Macro_F1  = 0.5336 | Micro_F1 = 0.4636
	DEV   / LOSS =     0.1628  Time 0h 00m 02s  Dispotion counts: 201/187/213/1010
Events : Macro_Pr = 0.8881 | Macro_Re = 0.9235 | Macro_F1  = 0.9021 | Micro_F1 = 0.9397 <<<
actions y_pred size (213, 7) y_pred sum 205.0
Actions: Macro_Pr = 0.7032 | Macro_Re = 0.6455 | Macro_F1  = 0.6626 | Micro_F1 = 0.7624
actions y_pred size (1010, 7) y_pred sum 844.0
Actions: Macro_Pr = 0.5239 | Macro_Re = 0.6455 | Macro_F1  = 0.5211 | Micro_F1 = 0.3045
 -->

### Clinical Bert: entities marked with '@', epoch: 08 (Best action f1)
| Set | Task | Macro Pr | Macro Re | Macro f1 | Micro f1 |
|-------|--------|----------|----------|----------|----------|
| Train | Event  |  0.9989  |  0.9958  |  0.9974  |  0.9989  |
| Train | Action |  0.9989  |  0.8379  |  0.8467  |  0.9945  |
|  Dev  | Event  |  0.8988  |  0.8802  |  0.8892  |  0.9376  |
|  Dev  | Action |  0.7998  |  0.7132  |  0.7377  |  0.8073<<|

<!-- ---------- Epoch: 08 ----------
	TRAIN / LOSS =     0.0074  Time 0h 00m 48s  Dispotion counts: 1128/1125/1129/6125
Events : Macro_Pr = 0.9989 | Macro_Re = 0.9958 | Macro_F1  = 0.9974 | Micro_F1 = 0.9989 <<<
actions y_pred size (1129, 7) y_pred sum 1168.0
Actions: Macro_Pr = 0.9989 | Macro_Re = 0.8379 | Macro_F1  = 0.8467 | Micro_F1 = 0.9945
actions y_pred size (6125, 7) y_pred sum 2507.0
Actions: Macro_Pr = 0.7650 | Macro_Re = 0.8379 | Macro_F1  = 0.6869 | Micro_F1 = 0.6330
	DEV   / LOSS =     0.2013  Time 0h 00m 02s  Dispotion counts: 201/189/213/1010
Events : Macro_Pr = 0.8988 | Macro_Re = 0.8802 | Macro_F1  = 0.8892 | Micro_F1 = 0.9376 <<<
actions y_pred size (213, 7) y_pred sum 216.0
Actions: Macro_Pr = 0.7998 | Macro_Re = 0.7132 | Macro_F1  = 0.7377 | Micro_F1 = 0.8073
actions y_pred size (1010, 7) y_pred sum 386.0
Actions: Macro_Pr = 0.6109 | Macro_Re = 0.7132 | Macro_F1  = 0.6285 | Micro_F1 = 0.5809
 -->
### Bert-base: entities marked with '@', epoch: 06 (Best event f1)
| Set | Task | Macro Pr | Macro Re | Macro f1 | Micro f1 |
|-------|--------|----------|----------|----------|----------|
| Train | Event  |  0.9904  |  0.9847  |  0.9875  |  0.9939  |
| Train | Action |  0.9828  |  0.7812  |  0.8074  |  0.9781  |
|  Dev  | Event  |  0.8687  |  0.8613  |  0.8649  |  0.9225<<|
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
 
 ### Bert-base: entities marked with '@', epoch: 04 (Best action f1)
| Set | Task | Macro Pr | Macro Re | Macro f1 | Micro f1 |
|-------|--------|----------|----------|----------|----------|
| Train | Event  |  0.9658  |  0.9598  |  0.9628  |  0.9827  |
| Train | Action |  0.9198  |  0.6517  |  0.6891  |  0.9323  |
|  Dev  | Event  |  0.8608  |  0.7822  |  0.8161  |  0.9057  |
|  Dev  | Action |  0.8512  |  0.6853  |  0.7122  |  0.7773<<|
 
<!--  ---------- Epoch: 04 ----------
	TRAIN / LOSS =     0.0451  Time 0h 02m 37s  Dispotion counts: 1128/1119/1161/6125
Events : Macro_Pr = 0.9658 | Macro_Re = 0.9598 | Macro_F1  = 0.9628 | Micro_F1 = 0.9827 <<<
actions y_pred size (1161, 7) y_pred sum 1142.0
Actions: Macro_Pr = 0.9198 | Macro_Re = 0.6517 | Macro_F1  = 0.6891 | Micro_F1 = 0.9323
actions y_pred size (6125, 7) y_pred sum 1944.0
Actions: Macro_Pr = 0.7532 | Macro_Re = 0.6517 | Macro_F1  = 0.5790 | Micro_F1 = 0.6927
	DEV   / LOSS =     0.2188  Time 0h 00m 11s  Dispotion counts: 201/174/212/1010
Events : Macro_Pr = 0.8608 | Macro_Re = 0.7822 | Macro_F1  = 0.8161 | Micro_F1 = 0.9057 <<<
actions y_pred size (212, 7) y_pred sum 202.0
Actions: Macro_Pr = 0.8512 | Macro_Re = 0.6853 | Macro_F1  = 0.7122 | Micro_F1 = 0.7773
actions y_pred size (1010, 7) y_pred sum 298.0
Actions: Macro_Pr = 0.7323 | Macro_Re = 0.6853 | Macro_F1  = 0.6362 | Micro_F1 = 0.6332 -->
