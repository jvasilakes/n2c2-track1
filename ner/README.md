# To train NER models you need to do the following steps
## Install required libraries
Please install the required libraries in the `requirements.txt` file

## Run the test set where we only have .txt files
- Firstly, we have to modify the script `run_test.sh` with the folder to the test data 
- Secondly, please run the following command from the root folder of the project
```bash
bash ner/run_test.sh
```

## Pre-process the data and download RoBERTa
You have to run the following command from the root folder of the project since I hard-code the path of the corpus
```bash
bash ner/train_step0.sh
```

## Train NER models
Please change to the `ner` folder and run this script
```bash
bash train_crossval.sh
```

It is noted that the `train_crossval.sh` script is to run three different BERT models (bert-base-uncased, clinicalBert, and BioRoberta) on the five folds. You can comment out them if you don't want to run that much.

The configurations for each experiment are stored in the folder `experiments/split_number/model_name`. This folder will also store all the output results.

The script also includes the prediction and evaluation steps on the development set. The final scores can be found at `result_org.txt` at the correponding output folders.

Currently, the default number of epochs is 10, you can change it by modifying the `train_crosseval.sh` at line 52 after the yaml option with this `--epoch 50`.

**The training script can be used in csf3 if you uncomment lines 11, 14, 17, and 23**


# NER results by 5-fold cross validation
As of 4th March
## Exact match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|BERT Base | 0.9018 | 0.8318 | 0.8651 |
|ClinicalBERT|0.9710 | 0.9520 | 0.9613 |
|BioRoBERTa	| 0.9682 | 0.9566 | 0.9624 |

## Lenient match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|BERT Base | 0.9269 | 0.8550 | 0.8892 |
|ClinicalBERT | 0.9809 | 0.9619 | 0.9713 |
|BioRoBERTa | 0.9759 | 0.9643 | 0.9701 |

# NER
`eval_scripts/ner.py` will compute the precision, recall, and F1 score between two brat ann files. E.g.,

```bash
python eval_scripts/ner.py biomedicus_baseline/output_brat/dev ../n2c2Track1TrainingData/data/dev/
```

The output is markdown formatted and can be copied directly into this file for keeping track of current progress.

`eval_scripts/n2c2.py` is the evaluation script used by the N2C2 Shared Task 2018. In this script we've added more corpora so that we can use it with them. Similarly to `ner.py`, this script also compares results between two brat ann folders. E.g.,

```bash
python eval_scripts/n2c2.py biomedicus_baseline/output_brat/dev ../n2c2Track1TrainingData/data/dev/ --ner-eval-corpus n2c2
```


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

## BERT base cased (Span-based classifier)
### Exact match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.900 | 0.771 | 0.830 |
|MACRO | 0.828 | 0.733 | 0.765 |

### Lenient match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.929 | 0.793 | 0.855 |
|MACRO | 0.851 | 0.750 | 0.785 |

## Clinical bert (Span-based classifier)
### Exact match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.963 | 0.943 | 0.952 |
|MACRO | 0.956 | 0.937 | 0.946 |

### Lenient match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.978 | 0.951 | 0.964 |
|MACRO | 0.965 | 0.942 | 0.952 |

## BioRoberta (Span-based classifier)
This is the results by RoBERTa-large-PM-M3-Voc, pre-trained on PubMed and PMC and MIMIC-III: https://github.com/facebookresearch/bio-lm  
### Exact match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.956 | 0.957 | 0.957 |
|MACRO | 0.934 | 0.932 | 0.931 |

### Lenient match
|      | prec  | rec   | f1    |
|------|-------|-------|-------|
|MICRO | 0.982 | 0.974 | 0.978 |
|MACRO | 0.967 | 0.955 | 0.960 |
