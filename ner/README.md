# NER
`eval_scripts/ner.py` will compute the precision, recall, and F1 score between two brat ann files. E.g.,

```bash
python eval_scripts/ner.py biomedicus_baseline/output_brat/dev ../n2c2Track1TrainingData/data/dev/
```

The output is markdown formatted and can be copied directly into this file for keeping track of current progress.


# Results
All results are on the dev split unless otherwise specified.
Lenient match means that >=1 characters overlap between the predicted and gold span.

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


## Scispacy Baseline
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

## Medspacy Baseline (vector model)
### Exact match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.873 | 0.790 | 0.830 |
|MACRO | 0.831 | 0.773 | 0.790 |

### Lenient match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.934 | 0.842 | 0.886 |
|MACRO | 0.883 | 0.817 | 0.837 |

## Medspacy Baseline (transformer model)
### Exact match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.836 | 0.762 | 0.798 |
|MACRO | 0.826 | 0.790 | 0.798 |

### Lenient match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.950 | 0.916 | 0.933 |
|MACRO | 0.923 | 0.907 | 0.907 |
