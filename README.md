# n2c2-track1


```
- README.md : This file
- brat_reader/ A small utility for reading brat annotations.
- n2c2Track1TrainingData/
   ├─ datastats.py  Script for computing dataset statistics from brat ann files.
   ├─ README  Original README file from the n2c2 data release.
   ├─ CMED.pdf  Preprint describing the dataset and baseline models.
   ├─ data/  Original data files released by n2c2.
   │   ├─ dev/  Development set. Contains txt and brat ann files.
   │   └─ train/  Training set. Contains txt and brat ann files.
   ├─ segmented/ Train and dev datasets with segmented sentences using biomedicus
   ├─ tokenised/ Train and dev datasets with tokenised sentences.
   └─ statistics/ Datasets statistics computed by datastats.py on train/dev/train+dev

- ner/   Directory for the named-entity recognition (NER) task
   ├─ eval_scripts/  Scripts for running NER evaluation
   │     └─ ner.py  Precision, recall, and F1 between two brat ann files.
   ├─ biomedicus_baseline/  NER baseline using the Biomedicus entity linker
   └─ scispacy_baseline/ NER baseline using scispacy en_ner_bc5cdr_md model.

- ee/   Directory for the event extraction (EE) task
```
