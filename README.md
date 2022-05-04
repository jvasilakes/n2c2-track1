# n2c2-track1


```
- README.md : This file
- brat_reader/ A small utility for reading brat annotations.
- n2c2Track1TrainingData{-v2,-v3}/
   ├─ README  Original README file from the n2c2 data release.
   ├─ data/  Original data files released by n2c2.
   │   ├─ dev/  Development set. Contains txt and brat ann files.
   │   └─ train/  Training set. Contains txt and brat ann files.
   ├─ segmented/ Train and dev datasets with segmented sentences using biomedicus
   ├─ tokenised/ Train and dev datasets with tokenised sentences.
   └─ statistics/ Datasets statistics computed by datastats.py on train/dev/train+dev
- n2c2TestData/
   ├─ release1/ The first data release containing only plain txt files for end-to-end evaluation.
   ├─ release2/ The first data release containing gold-standard NER for event+context evaluation.
   ├─ release3/ The first data release containing gold-standard event annotations for context evaluation.
   └─ segmented/ release1 txt files with sentences segmented using biomedicus
- evaluation/
   ├─ eval_script.py The official evaluation script.
   └─ segmented/ Official evaluation documentation.
- auxiliary_data/ Other datasets with similar tasks
- scripts/ Various utilities for preprocessing and summarizing data
- ner/   Directory for the named-entity recognition (NER) task
- ee/   Directory for the event extraction (EE) task
- context/ Directory for the context classification task
```
