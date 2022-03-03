# Current Best Dev Results

Precision, recall, and F1 score are reported for the official dev set release.
Avg F1 refers to the average computed on the dev set of the 5 alternative data splits.

## Majority Baseline

Generally, average performance of the majority baseline across dev sets is lower than the test set
performance reported in CMED.pdf.

### Action

|      | prec  | rec   | f1    | Avg F1|
|------|-------|-------|-------|-------|
|MICRO | 0.439 | 0.439 | 0.439 | 0.427 |
|MACRO | 0.073 | 0.167 | 0.102 | 0.097 |

### Actor

|      | prec  | rec   | f1    | Avg F1|
|------|-------|-------|-------|-------|
|MICRO | 0.878 | 0.878 | 0.878 | 0.884 |
|MACRO | 0.293 | 0.333 | 0.312 | 0.313 |

### Certainty

|      | prec  | rec   | f1    | Avg F1|
|------|-------|-------|-------|-------|
|MICRO | 0.792 | 0.792 | 0.792 | 0.844 |
|MACRO | 0.264 | 0.333 | 0.295 | 0.275 |

### Negation

|      | prec  | rec   | f1    | Avg F1|
|------|-------|-------|-------|-------|
|MICRO | 0.982 | 0.982 | 0.982 | 0.976 |
|MACRO | 0.491 | 0.500 | 0.495 | 0.494 |

### Temporality
|      | prec  | rec   | f1    | Avg F1|
|------|-------|-------|-------|-------|
|MICRO | 0.593 | 0.593 | 0.593 | 0.533 |
|MACRO | 0.148 | 0.250 | 0.186 | 0.197 |

 
## SVM Baseline
|Action|prec   | rec   |f1     |Actor |prec   | rec   |f1     |Cert  |prec   | rec   |f1     |
|------|-------|-------|-------|------|-------|-------|-------|------|-------|-------|-------|
|MICRO | 0.552 | 0.552 | 0.552 |MICRO | 0.887 | 0.887 | 0.887 |MICRO | 0.792 | 0.792 | 0.792 |
|MACRO | 0.537 | 0.427 | 0.413 |MACRO | 0.621 | 0.507 | 0.546 |MACRO | 0.559 | 0.409 | 0.429 |
		
|Neg   |prec   | rec   |f1     |Temp  |prec   | rec   |f1     |
|------|-------|-------|-------|------|-------|-------|-------|
|MICRO | 0.986 | 0.986 | 0.986 |MICRO | 0.670 | 0.670 | 0.670 |
|MACRO | 0.993 | 0.625 | 0.697 |MACRO | 0.463 | 0.423 | 0.424 |


## BERT Baselines
See [bert\_baselines/Results.md](https://github.com/jvasilakes/n2c2-track1/blob/master/context/bert_baselines/Results.md).
