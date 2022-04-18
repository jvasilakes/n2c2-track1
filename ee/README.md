EE task - 14/3/2022

A basic method based on Matching the Blanks and An Improved Baseline for Sentence-level Relation Extraction.\
The model is extend with PL-markers, and after experiment, max-pooling (1 emb) produces the best results.\
Pl markers also has singificantly higher action performance.\

To test the dataset structure and see the returned batch run:
```
python datasets.py --config ../configs/local.yaml --data ../data/spacy/ --bert clinical --use_verbs 
```
You can change the parameters on the config file. 

To run the model:

```
python main.py --config ../configs/local.yaml --mode train --data ../data/spacy/ --bert clinical --use_verbs 
```
### PL-marker with Clinical Bert: entities marked with 'unused0-1', verbs with 'unused2-3', max 10 verbs 
#### using max-pooling (1 emb) for the levitated verb markers
#### epoch: 05 (Best event and best action f1)
```
************************ Event Classification ************************
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
         Disposition  0.9727  0.8856  0.9271    0.9727  0.8856  0.9271
       Nodisposition  0.9569  0.9793  0.9680    0.9569  0.9793  0.9680
        Undetermined  0.8471  0.8276  0.8372    0.8471  0.8276  0.8372
                      ------------------------------------------------
     Overall (micro)  0.9505  0.9477  0.9491    0.9505  0.9477  0.9491
     Overall (macro)  0.9255  0.8975  0.9108    0.9255  0.8975  0.9108


*********************** Context Classification ***********************
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
              Action  0.8796  0.7602  0.8155    0.8796  0.7602  0.8155
```

Compared with mean-pool and also different embs (2) for start and end, but 1 emb with max performs best.<br>

Jake configuration performance:
```
Events : Macro_Pr = 0.9075 | Macro_Re = 0.8653 | Macro_F1  = 0.8851 | Micro_F1 = 0.9380 <<<
Actions: Macro_Pr = 0.5429 | Macro_Re = 0.6297 | Macro_F1  = 0.4502 | Micro_F1 = 0.2855
Actions: Macro_Pr = 0.7692 | Macro_Re = 0.6297 | Macro_F1  = 0.6180 | Micro_F1 = 0.7736
```

Across all splits:
```
Default: 
Events : Macro_Pr = 0.9255 | Macro_Re = 0.8975 | Macro_F1  = 0.9108 | Micro_F1 = 0.9491 <<<
Actions: Macro_Pr = 0.7780 | Macro_Re = 0.7665 | Macro_F1  = 0.7675 | Micro_F1 = 0.8322
Split0:
Events : Macro_Pr = 0.8157 | Macro_Re = 0.7143 | Macro_F1  = 0.7371 | Micro_F1 = 0.9249 <<<
Actions: Macro_Pr = 0.8172 | Macro_Re = 0.7503 | Macro_F1  = 0.7083 | Micro_F1 = 0.7805
Split1:
Events : Macro_Pr = 0.8788 | Macro_Re = 0.8459 | Macro_F1  = 0.8617 | Micro_F1 = 0.9389 <<<
Actions: Macro_Pr = 0.8745 | Macro_Re = 0.6728 | Macro_F1  = 0.6757 | Micro_F1 = 0.7821
Split2:
Events : Macro_Pr = 0.9096 | Macro_Re = 0.8885 | Macro_F1  = 0.8987 | Micro_F1 = 0.9504 <<<
Actions: Macro_Pr = 0.8613 | Macro_Re = 0.9238 | Macro_F1  = 0.8882 | Micro_F1 = 0.8746
Split3:
Events : Macro_Pr = 0.8744 | Macro_Re = 0.8858 | Macro_F1  = 0.8799 | Micro_F1 = 0.9312 <<<
Actions: Macro_Pr = 0.8420 | Macro_Re = 0.4580 | Macro_F1  = 0.4338 | Micro_F1 = 0.7929

```

| Set | Task | Macro Pr | Macro Re | Macro f1 | Micro f1 |
|-------|--------|----------|----------|----------|----------|
| Default | Event  | 0.9255  |  0.8975 |  0.9108  |  0.9491  |
| Default | Action | 0.7780  |  0.7665  |  0.7675  |  0.8322  |
| Split0 | Event  | 0.8157  |  0.7143  |  0.7371  |  0.9249 |
| Split0 | Action | 0.8172  |  0.7503  |  0.7083  |  0.7805  |
| Split1 | Event  | 0.8788 | 0.8459 | 0.8617 | 0.9389 |
| Split1 | Action | 0.8745 | 0.6728 | 0.6757 | 0.7821  |
| Split2 | Event  | 0.9096 | 0.8885 | 0.8987 | 0.9504 |
| Split2 | Action | 0.8613 | 0.9238 | 0.8882 | 0.8746  |

| Split3 | Event  | 0.8744 | 0.8858 | 0.8799 | 0.9312|
| Split3 | Action | 0.8420 | 0.4580 | 0.4338 | 0.7929 |


## Older results with solid markers

### Clinical Bert: entities marked with '@', epoch: 06 (Best event f1)

Results:
```
*********************** Medication Extraction ************************
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
                Drug  1.0000  0.9901  0.9950    1.0000  0.9901  0.9950


************************ Event Classification ************************
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
         Disposition  0.9418  0.8856  0.9128    0.9418  0.8856  0.9128
       Nodisposition  0.9658  0.9738  0.9698    0.9658  0.9738  0.9698
        Undetermined  0.8256  0.8161  0.8208    0.8256  0.8161  0.8208
                      ------------------------------------------------
     Overall (micro)  0.9493  0.9427  0.9460    0.9493  0.9427  0.9460
     Overall (macro)  0.9111  0.8918  0.9011    0.9111  0.8918  0.9011


*********************** Context Classification ***********************
                      ------- strict -------    ------ lenient -------
                      Prec.   Rec.    F(b=1)    Prec.   Rec.    F(b=1)
              Action  0.8201  0.7014  0.7561    0.8201  0.7014  0.7561
```
There is a discrepancy in the Action measures of the eval and my script.\
The main reason is that for my script I also count the action predictions of the golden events.\
But in an end-to-end setting that would be incorrect, since if we do not predict Disposition event, we wouldn't predict any actions.\
Howeve, I will leave this score for my script, to have a measure for the action predicting capabilities of the model.

<!-- We got you  330-04 ['E17', 'E18', 'E19'] ['Stop', 'Start', 'Stop'] -->
### Clinical Bert: entities marked with '@', epoch: 06 (Best event f1)
| Set | Task | Macro Pr | Macro Re | Macro f1 | Micro f1 |
|-------|--------|----------|----------|----------|----------|
| Train | Event  |  0.9986  |  0.9939  |  0.9962  |  0.9981  |
| Train | Action |  0.9947  |  0.7998  |  0.8223  |  0.9850  |
|  Dev  | Event  |  0.9111  |  0.8918  |  0.9011  |  0.9460<<|
|  Dev  | Action |  0.8445  |  0.6822  |  0.6845  |  0.7814  |

<!-- ---------- Epoch: 06 ----------
	TRAIN / LOSS =     0.0130  Time 0h 00m 51s  Dispotion counts: 1128/1127/1132/6125
Events : Macro_Pr = 0.9986 | Macro_Re = 0.9939 | Macro_F1  = 0.9962 | Micro_F1 = 0.9981 <<<
actions y_pred size (1132, 7) y_pred sum 1160.0
Actions: Macro_Pr = 0.9947 | Macro_Re = 0.7998 | Macro_F1  = 0.8223 | Micro_F1 = 0.9850
actions y_pred size (6125, 7) y_pred sum 4667.0
Actions: Macro_Pr = 0.7856 | Macro_Re = 0.7998 | Macro_F1  = 0.6568 | Micro_F1 = 0.3939
	DEV   / LOSS =     0.1705  Time 0h 00m 02s  Dispotion counts: 201/189/212/1010
Events : Macro_Pr = 0.9111 | Macro_Re = 0.8918 | Macro_F1  = 0.9011 | Micro_F1 = 0.9460 <<<
actions y_pred size (212, 7) y_pred sum 210.0
Actions: Macro_Pr = 0.8445 | Macro_Re = 0.6822 | Macro_F1  = 0.6845 | Micro_F1 = 0.7814
actions y_pred size (1010, 7) y_pred sum 881.0
Actions: Macro_Pr = 0.6747 | Macro_Re = 0.6822 | Macro_F1  = 0.5506 | Micro_F1 = 0.3052
Saving checkpoint
current best epoch: 6
-----Saving predictions for current epoch 6 -----
 -->

### Clinical Bert: entities marked with '@', epoch: 09 (Best action f1)
| Set | Task | Macro Pr | Macro Re | Macro f1 | Micro f1 |
|-------|--------|----------|----------|----------|----------|
| Train | Event  |  0.9997  |  0.9973  |  0.9985  |  0.9993  |
| Train | Action |  0.9957  |  0.8474  |  0.8501  |  0.9966  |
|  Dev  | Event  |  0.9182  |  0.8803  |  0.8984  |  0.9435  |
|  Dev  | Action |  0.8741  |  0.7368  |  0.7736  |  0.8047<<|

<!-- ---------- Epoch: 09 ----------
	TRAIN / LOSS =     0.0061  Time 0h 00m 51s  Dispotion counts: 1128/1125/1129/6125
Events : Macro_Pr = 0.9997 | Macro_Re = 0.9973 | Macro_F1  = 0.9985 | Micro_F1 = 0.9993 <<<
actions y_pred size (1129, 7) y_pred sum 1173.0
Actions: Macro_Pr = 0.9957 | Macro_Re = 0.8474 | Macro_F1  = 0.8501 | Micro_F1 = 0.9966
actions y_pred size (6125, 7) y_pred sum 3259.0
Actions: Macro_Pr = 0.7320 | Macro_Re = 0.8474 | Macro_F1  = 0.6602 | Micro_F1 = 0.5280
	DEV   / LOSS =     0.1986  Time 0h 00m 02s  Dispotion counts: 201/188/211/1010
Events : Macro_Pr = 0.9182 | Macro_Re = 0.8803 | Macro_F1  = 0.8984 | Micro_F1 = 0.9435 <<<
actions y_pred size (211, 7) y_pred sum 210.0
Actions: Macro_Pr = 0.8741 | Macro_Re = 0.7368 | Macro_F1  = 0.7736 | Micro_F1 = 0.8047
actions y_pred size (1010, 7) y_pred sum 395.0
Actions: Macro_Pr = 0.7266 | Macro_Re = 0.7368 | Macro_F1  = 0.6733 | Micro_F1 = 0.5626
current best epoch: 6 -->

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
