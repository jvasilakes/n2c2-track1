# To run the model on the test set where we only have .txt files
- Firstly, we have to modify the script `run_test.sh` with the folder to the test data 
- Secondly, run the following command from the root folder of the project
```bash
bash ner/run_test.sh
```

# To train NER models you need to do the following steps
## Install required libraries
Please install the required libraries in the `requirements.txt` file

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

It is noted that the `train_crossval.sh` script is to run two different BERT models (clinicalBert and BioLM) on the five folds. You can comment out them if you don't want to run that much.

The configurations for each experiment are stored in the folder `experiments/split_number/model_name`. This folder will also store all the output results.

The script also includes the prediction and evaluation steps on the development set. The final scores can be found at `result_org.txt` at the correponding output folders.

Currently, the default number of epochs is 10, you can change it by modifying this parameter in the train-ner.yaml file.

# NER Evaluation
`eval_scripts/ner.py` will compute the precision, recall, and F1 score between two brat ann files. E.g.,

```bash
python eval_scripts/ner.py output_brat/dev ../n2c2Track1TrainingData/data/dev/
```

The output is markdown formatted and can be copied directly into this file for keeping track of current progress.

`eval_scripts/n2c2.py` is the evaluation script used by the N2C2 Shared Task 2022. Similarly to `ner.py`, this script also compares results between two brat ann folders. E.g.,

```bash
python eval_scripts/n2c2.py biomedicus_baseline/output_brat/dev ../n2c2Track1TrainingData/data/dev/ --ner-eval-corpus n2c2
```

