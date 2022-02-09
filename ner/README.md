# NER
`eval_scripts/ner.py` will compute the precision, recall, and F1 score between two brat ann files. E.g.,

```bash
python eval_scripts/ner.py biomedicus_baseline/output_brat/dev ../n2c2Track1TrainingData/data/dev/
```

The output is markdown formatted and can be copied directly into this file for keeping track of current progress.


# Results
All results are on the dev split unless otherwise specified

## Biomedicus Baseline
### Exact match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.380 | 0.839 | 0.523 |
|MACRO | 0.386 | 0.835 | 0.504 |

### Lenient match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.510 | 0.895 | 0.650 |
|MACRO | 0.487 | 0.878 | 0.601 |


## SciSpacy Baseline
### Exact match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.395 | 0.574 | 0.468 |
|MACRO | 0.404 | 0.586 | 0.456 |

### Lenient match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.469 | 0.684 | 0.556 |
|MACRO | 0.459 | 0.663 | 0.519 |

