# Contents

* [Best Overall Results](#best-overall-results) 

* [Pooled Output Results](#pooled-output-results)

* [Entity Span Results](#entity-span-results)

* [DICE vs Cross Entropy Loss](#dice-vs-cross-entropy)


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
Bio\_ClinicalBERT  
Action + Negation multi-task setup  
+/- 1 sentence window  
Entity markers: use both  
|         | prec  | rec   | f1    | Avg F1|
|---------|-------|-------|-------|-------|
| micro   | 0.986 | 0.986 | 0.986 | 0.978 |
| macro   | 0.993 | 0.625 | 0.697 | 0.589 |


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



# DICE vs Cross Entropy

Below, I compare the performance of the current best models using self-adjusted DICE loss vs using standard cross entropy loss.

## Takeaways
DICE *can* improve recall on some less frequent classes, but losses in precision mean that it ultimately performs worse than cross entropy.

I still think that there just isn't enough variety in the training data to learn a generalizable representations of most infrequent classes.

### Action
|Loss  |      | prec  | rec   | f1    |
|------|------|-------|-------|-------|
| DICE |MICRO | 0.769 | 0.769 | 0.769 |
|      |MACRO | 0.734 | 0.719 | 0.722 |
| CE   |MICRO | 0.814 | 0.814 | 0.814 |
|      |MACRO | 0.789 | 0.727 | 0.753 |

| Loss |           | prec  | rec   | f1    | supp |
|------|-----------|-------|-------|-------|------|
| DICE |Decrease   | 0.615 | 0.615 | 0.615 | 13   |
|      |Increase   | 0.900 | 0.783 | 0.837 | 23   |
|      |Start      | 0.769 | 0.825 | 0.796 | 97   |
|      |Stop       | 0.804 | 0.683 | 0.739 | 60   |
|      |UniqueDose | 0.714 | 0.909 | 0.800 | 22   |
|      |Unknown    | 0.600 | 0.500 | 0.545 | 6    |
|  CE  |Decrease   | 0.667 | 0.615 | 0.640 | 13   |
|      |Increase   | 0.944 | 0.739 | 0.829 | 23   |
|      |Start      | 0.783 | 0.928 | 0.849 | 97   |
|      |Stop       | 0.878 | 0.717 | 0.789 | 60   |
|      |UniqueDose | 0.864 | 0.864 | 0.864 | 22   |
|      |Unknown    | 0.600 | 0.500 | 0.545 | 6    |


### Actor
|Loss  |      | prec  | rec   | f1    |
|------|------|-------|-------|-------|
| DICE |MICRO | 0.896 | 0.896 | 0.896 |
|      |MACRO | 0.722 | 0.529 | 0.578 |
| CE   |MICRO | 0.923 | 0.923 | 0.923 |
|      |MACRO | 0.762 | 0.656 | 0.700 |

| Loss |          | prec  | rec   | f1    | supp |
|------|----------|-------|-------|-------|------|
| DICE |Patient   | 0.583 | 0.412 | 0.483 | 17   |
|      |Physician | 0.917 | 0.974 | 0.945 | 194  |
|      |Unknown   | 0.667 | 0.200 | 0.308 | 10   |
| CE   |Patient   | 0.769 | 0.588 | 0.667 | 17   |
|      |Physician | 0.945 | 0.979 | 0.962 | 194  |
|      |Unknown   | 0.571 | 0.400 | 0.471 | 10   |



### Certainty
|Loss  |      | prec  | rec   | f1    |
|------|------|-------|-------|-------|
| DICE |MICRO | 0.882 | 0.882 | 0.882 |
|      |MACRO | 0.824 | 0.660 | 0.714 |
| CE   |MICRO | 0.891 | 0.891 | 0.891 |
|      |MACRO | 0.800 | 0.751 | 0.769 |


| Loss |             | prec  | rec   | f1    | supp |
|------|-------------|-------|-------|-------|------|
| DICE |Certain      | 0.895 | 0.971 | 0.932 | 175  |
|      |Conditional  | 0.750 | 0.353 | 0.480 | 17   |
|      |Hypothetical | 0.826 | 0.655 | 0.731 | 29   |
|  CE  |Certain      | 0.923 | 0.960 | 0.941 | 175  |
|      |Conditional  | 0.667 | 0.706 | 0.686 | 17   |
|      |Hypothetical | 0.810 | 0.586 | 0.680 | 29   |


### Negation
|Loss  |      | prec  | rec   | f1    |
|------|------|-------|-------|-------|
| DICE |MICRO | 0.869 | 0.869 | 0.869 |
|      |MACRO | 0.529 | 0.688 | 0.525 |
| CE   |MICRO | 0.982 | 0.982 | 0.982 |
|      |MACRO | 0.743 | 0.623 | 0.662 |


|Loss  |           | prec  | rec   | f1    | supp |
|------|-----------|-------|-------|-------|------|
| DICE |Negated    | 0.069 | 0.500 | 0.121 | 4    |
|      |NotNegated | 0.990 | 0.876 | 0.929 | 217  |
| CE   |Negated    | 0.500 | 0.250 | 0.333 | 4    |
|      |NotNegated | 0.986 | 0.995 | 0.991 | 217  |



### Temporality
|Loss  |      | prec  | rec   | f1    |
|------|------|-------|-------|-------|
| DICE |MICRO | 0.787 | 0.787 | 0.787 |
|      |MACRO | 0.869 | 0.559 | 0.579 |
| CE   |MICRO | 0.837 | 0.837 | 0.837 |
|      |MACRO | 0.830 | 0.662 | 0.704 |

| Loss |        | prec  | rec   | f1    | supp |
|------|--------|-------|-------|-------|------|
| DICE |Future  | 1.000 | 0.121 | 0.216 | 33   |
|      |Past    | 0.919 | 0.947 | 0.932 | 131  |
|      |Present | 0.556 | 0.833 | 0.667 | 54   |
|      |Unknown | 1.000 | 0.333 | 0.500 | 3    |
|  CE  |Future  | 0.700 | 0.636 | 0.667 | 33   |
|      |Past    | 0.932 | 0.939 | 0.935 | 131  |
|      |Present | 0.690 | 0.741 | 0.714 | 54   |
|      |Unknown | 1.000 | 0.333 | 0.500 | 3    |
