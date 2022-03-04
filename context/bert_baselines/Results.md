# Best Overall Results
As of 2/3/2022

Precision, recall, and F1 score are reported for the official dev set release.
Avg F1 refers to the average computed on the dev set of the 5 alternative data splits.

There are large variations in performance on the Certainty and Negation tasks, suggesting 
that these training examples do not necessarily generalize well.

## Action
Bio\_ClinicalBERT
+/- 1 sentence window
Entity markers: use both
|         | P     | R     | F1    | Avg F1|
|---------|-------|-------|-------|-------|
|micro    | 0.814 | 0.814 | 0.814 | 0.857 |
|macro    | 0.789 | 0.727 | 0.753 | 0.776 |


## Actor
bert-base-uncased
0 sentence window
Entity markers: use both
|         | P     | R     | F1    | Avg F1|
|---------|-------|-------|-------|-------|
|micro    | 0.923 | 0.923 | 0.923 | 0.918 |
|macro    | 0.762 | 0.656 | 0.700 | 0.721 |

## Certainty
Bio\_ClinicalBERT
+/- 1 sentence window
Entity markers: use first only
|         | P     | R     | F1    | Avg F1|
|---------|-------|-------|-------|-------|
|micro    | 0.910 | 0.910 | 0.910 | 0.911 |
|macro    | 0.854 | 0.779 | 0.808 | 0.655 |

## Negation
bert-base-uncased
0 sentence window (same results with +/- 1 sentence window)
pooled output
|         | P     | R     | F1    | Avg F1|
|---------|-------|-------|-------|-------|
| micro   | 0.982 | 0.982 | 0.982 | 0.977 |
| macro   | 0.743 | 0.623 | 0.662 | 0.523 |

## Temporality
Bio\_ClinicalBERT
+/- 1 sentence window
Entity markers: use first only
|         | P     | R     | F1    | Avg F1|
|---------|-------|-------|-------|-------|
|micro    | 0.837 | 0.837 | 0.837 | 0.862 |
|macro    | 0.830 | 0.662 | 0.704 | 0.785 |


# Pooled Output Results
## Single Task
### bert-base-uncased
| Action  | P     | R     | F1    | Actor   | P     | R     | F1    | Cert    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.742 | 0.742 | 0.742 | micro   | 0.887 | 0.887 | 0.887 | micro   | 0.837 | 0.837 | 0.837 |
| macro   | 0.645 | 0.558 | 0.580 | macro   | 0.679 | 0.553 | **0.599** | macro   | 0.667 | 0.599 | 0.625 |

| Neg     | P     | R     | F1    | Temp    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.982 | 0.982 | 0.982 | micro   | 0.814 | 0.814 | 0.814 |
| macro   | 0.743 | 0.623 | **0.662** | macro   | 0.811 | 0.659 | **0.693** |

### Bio\_ClinicalBERT

| Action  | P     | R     | F1    | Actor   | P     | R     | F1    | Cert    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.729 | 0.729 | 0.729 | micro   | 0.896 | 0.896 | 0.896 | micro   | 0.891 | 0.891 | 0.891 |
| macro   | 0.759 | 0.645 | **0.669** | macro   | 0.725 | 0.520 | 0.573 | macro   | 0.804 | 0.718 | **0.746** |

| Neg     | P     | R     | F1    | Temp    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.968 | 0.968 | 0.968 | micro   | 0.787 | 0.787 | 0.787 |
| macro   | 0.593 | 0.616 | 0.603 | macro   | 0.555 | 0.574 | 0.558 |

## Multi-task
### bert-base-uncased

| Action  | P     | R     | F1    | Actor   | P     | R     | F1    | Cert    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.122 | 0.122 | 0.122 | micro   | 0.357 | 0.357 | 0.357 | micro   | 0.665 | 0.665 | 0.665 |
| macro   | 0.200 | 0.185 | 0.102 | macro   | 0.389 | 0.513 | 0.279 | macro   | 0.399 | 0.491 | 0.401 |

| Neg     | P     | R     | F1    | Temp    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.697 | 0.697 | 0.697 | micro   | 0.747 | 0.747 | 0.747 |
| macro   | 0.487 | 0.355 | 0.411 | macro   | 0.503 | 0.512 | 0.506 |

### Bio\_ClinicalBERT

| Action  | P     | R     | F1    | Actor   | P     | R     | F1    | Cert    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.086 | 0.086 | 0.086 | micro   | 0.235 | 0.235 | 0.235 | micro   | 0.353 | 0.353 | 0.353 |
| macro   | 0.050 | 0.205 | 0.066 | macro   | 0.345 | 0.405 | 0.154 | macro   | 0.161 | 0.111 | 0.132 |

| Neg     | P     | R     | F1    | Temp    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.172 | 0.172 | 0.172 | micro   | 0.787 | 0.787 | 0.787 |
| macro   | 0.511 | 0.578 | 0.156 | macro   | 0.555 | 0.557 | 0.546 |



# Entity Span Results
Surrounded the entity mention with '@' markers.
Using just first entity marker vs using both as input to the classification head.
bert-base-uncased (BERT) vs Bio\_ClinicalBERT (CBERT)

### Action
Winner: ClinicalBERT with +/- 1 sentence window, pooling both entity markers
|      |       |      |Win=0  |       |       | |Win=1  |       |       |
|------|-------|------|-------|-------|-------|-|-------|-------|-------|
| Model|Markers|      | prec  | rec   | f1    | | prec  | rec   | f1    |
| BERT | first |MICRO | 0.710 | 0.710 | 0.710 | | 0.769 | 0.769 | 0.769 |
|      |       |MACRO | 0.647 | 0.653 | 0.642 | | 0.732 | 0.706 | 0.713 |
|      | both  |MICRO | 0.719 | 0.719 | 0.719 | | 0.760 | 0.760 | 0.760 |
|      |       |MACRO | 0.669 | 0.654 | 0.650 | | 0.748 | 0.696 | 0.713 |
|CBERT | first |MICRO | 0.778 | 0.778 | 0.778 | | 0.787 | 0.787 | 0.787 |
|      |       |MACRO | 0.729 | 0.722 | 0.715 | | 0.751 | 0.708 | 0.713 |
|      | both  |MICRO | 0.774 | 0.774 | 0.774 | | 0.814 | 0.814 | **0.814** |
|      |       |MACRO | 0.762 | 0.727 | 0.734 | | 0.789 | 0.727 | **0.753** |


### Actor
Winner: BERT-base with 0 sentence window, pooling both entity markers
|      |       |      |Win=0  |       |       | |Win=1  |       |       |
|------|-------|------|-------|-------|-------|-|-------|-------|-------|
| Model|Markers|      | prec  | rec   | f1    | | prec  | rec   | f1    |
| BERT | first |MICRO | 0.914 | 0.914 | 0.914 | | 0.882 | 0.882 | 0.882 |
|      |       |MACRO | 0.775 | 0.617 | 0.675 | | 0.614 | 0.524 | 0.547 |
|      | both  |MICRO | 0.923 | 0.923 | **0.923** | | 0.887 | 0.887 | 0.887 |
|      |       |MACRO | 0.762 | 0.656 | **0.700** | | 0.567 | 0.499 | 0.515 |
|CBERT | first |MICRO | 0.891 | 0.891 | 0.891 | | 0.887 | 0.887 | 0.887 |
|      |       |MACRO | 0.585 | 0.477 | 0.506 | | 0.569 | 0.458 | 0.486 |
|      | both  |MICRO | 0.882 | 0.882 | 0.882 | | 0.891 | 0.891 | 0.891 |
|      |       |MACRO | 0.603 | 0.470 | 0.509 | | 0.622 | 0.537 | 0.555 |


### Certainty
Winner: BERT-base with +/- 1 sentence window, using only first entity marker
|      |       |      |Win=0  |       |       | |Win=1  |       |       |
|------|-------|------|-------|-------|-------|-|-------|-------|-------|
| Model|Markers|      | prec  | rec   | f1    | | prec  | rec   | f1    |
| BERT | first |MICRO | 0.900 | 0.900 | 0.900 | | 0.910 | 0.910 | **0.910** |
|      |       |MACRO | 0.807 | 0.749 | 0.772 | | 0.854 | 0.779 | **0.808** |
|      | both  |MICRO | 0.905 | 0.905 | 0.905 | | 0.873 | 0.873 | 0.873 |
|      |       |MACRO | 0.835 | 0.769 | 0.795 | | 0.809 | 0.643 | 0.703 |
|CBERT | first |MICRO | 0.900 | 0.900 | 0.900 | | 0.891 | 0.891 | 0.891 |
|      |       |MACRO | 0.852 | 0.701 | 0.759 | | 0.800 | 0.751 | 0.769 |
|      | both  |MICRO | 0.887 | 0.887 | 0.887 | | 0.891 | 0.891 | 0.891 |
|      |       |MACRO | 0.831 | 0.679 | 0.727 | | 0.783 | 0.751 | 0.765 |


### Negation
Winner: BERT-base with 0 sentence window, pooling both entity markers
|      |       |      |Win=0  |       |       | |Win=1  |       |       |
|------|-------|------|-------|-------|-------|-|-------|-------|-------|
| Model|Markers|      | prec  | rec   | f1    | | prec  | rec   | f1    |
| BERT | first |MICRO | 0.982 | 0.982 | **0.982** | | 0.982 | 0.982 | **0.982** |
|      |       |MACRO | 0.491 | 0.500 | 0.495 | | 0.491 | 0.500 | 0.495 |
|      | both  |MICRO | 0.973 | 0.973 | 0.973 | | 0.964 | 0.964 | 0.964 |
|      |       |MACRO | 0.618 | 0.618 | **0.618** | | 0.576 | 0.613 | 0.591 |
|CBERT | first |MICRO | 0.964 | 0.964 | 0.964 | | 0.982 | 0.982 | **0.982** |
|      |       |MACRO | 0.576 | 0.613 | 0.591 | | 0.491 | 0.500 | 0.495 |
|      | both  |MICRO | 0.982 | 0.982 | **0.982** | | 0.968 | 0.968 | 0.968 |
|      |       |MACRO | 0.491 | 0.500 | 0.495 | | 0.593 | 0.616 | 0.603 |


### Temporality
Winner: ClinicalBERT with +/- 1 sentence window, using only first entity marker
|      |       |      |Win=0  |       |       | |Win=1  |       |       |
|------|-------|------|-------|-------|-------|-|-------|-------|-------|
| Model|Markers|      | prec  | rec   | f1    | | prec  | rec   | f1    |
| BERT | first |MICRO | 0.792 | 0.792 | 0.792 | | 0.805 | 0.805 | 0.805 |
|      |       |MACRO | 0.542 | 0.536 | 0.535 | | 0.808 | 0.641 | 0.683 |
|      | both  |MICRO | 0.783 | 0.783 | 0.783 | | 0.810 | 0.810 | 0.810 |
|      |       |MACRO | 0.544 | 0.544 | 0.542 | | 0.820 | 0.636 | 0.675 |
|CBERT | first |MICRO | 0.805 | 0.805 | 0.805 | | 0.837 | 0.837 | **0.837** |
|      |       |MACRO | 0.804 | 0.629 | 0.674 | | 0.830 | 0.662 | **0.704** |
|      | both  |MICRO | 0.814 | 0.814 | 0.814 | | 0.801 | 0.801 | 0.801 |
|      |       |MACRO | 0.558 | 0.569 | 0.563 | | 0.803 | 0.624 | 0.668 |
