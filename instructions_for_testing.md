# Running the full pipeline

The pipeline contains three main sections:

 1. [NER](#ner)
 2. [Event Extraction](#event-extraction)
 3. [Context Classification](#context-classification)


## Preliminaries

`n2c2_track1_home` is set to the directory containing the `ner`, `ee`, and `context` subdirectories for each task.

The test data will be a set of plain text files.

First, do the following:

 1. A quick manual review of the text files to check for obvious problems, e.g. empty documents, documents in other languages, etc.
 2. Copy/move all text files to `${n2c2_track1_home}/n2c2TestData/test/0`.
 3. Count all text files to ensure we have the same number as the organizers released: `ls -1 ${n2c2_track1_home}/n2c2TestData/test/0/*.txt | wc -l`


## NER

**Input**: `${n2c2_track1_home}/n2c2TestData/test`
**Input Description**: plain text files.
**Model Location**:

### Step 1: set the prediction model

Set `ner_model` to **Model Location**.
For example, we'll predict using roberta trained on the first CV split.
```
ner_model=0/baseline_roberta
```
which corresponds to a subdirectory of `${n2c2_track1_home}/ner/experiments/`.


### Step 2: run prediction

 * Ensure that the `test_data` field in `${n2c2_track1_home}/ner/experiments/${ner_model}/predict-test.yaml` is set to `corpus/test/0/`. 
 * Ensure that a model checkpoint exists at `${n2c2_track1_home}/ner/experiments/${ner_model}/joint`
 * Run the pre-process, prediction, post-process pipeline with
```
cd ${n2c2_track1_home}
bash ner/run_test.sh
```

**Output**: `experiments/${ner_model}/predict-test-org`
**Output Description**: brat-formatted files, one per input file, containing detected medication spans.



## Event Extraction

**Input**: `${n2c2_track1_home}/ner/experiments/${model}/predict-test-org`

**Model Location**: `${n2c2_track1_home}/ee/saved_models/`

Set the following to specify the event extraction model in accordance with **Model Location** above.
For example, we'll evaluate BlueBERT trained on the ensemble split.
```
bert_model="blue"
train_split="ensemble"
```

```
cd ${n2c2_track1_home}/ee/src
sh xec_n2c2_predict.sh ${bert_model} ${train_split} ${n2c2_track1_home}/ner/experiments/${ner_model}/predict-test-org preprocess
```

**Output**: `${n2c2_track1_home}/ee/saved_models/${bert_model}_${train_split}/predictions/test`
**Output Description**: brat-formatted files, one per input file, containing detected Disposition and NoDisposition events.


## Context Classification

**Input**: `${n2c2_track1_home}/ee/saved_models/${bert_model}_${train_split}/predictions/test`

Context classification is further broken into 5 simultaneous tasks:

 * [Action](#action)
 * [Actor](#actor)
 * [Certainty](#certainty)
 * [Negation](#negation)
 * [Temporality](#temporality)


### Action

**Model Location Action**: See `${n2c2_track1_home}/context/bert_baselines/submissions/{1,2,3}/action.txt`

```
python run.py predict ${model_location_action}/config.yaml
```
**Output**: `${model_location_action}/predictions/n2c2ContextDataset/brat/test`, where `model_location` is defined below.


### Actor

**Model Location Actor**: See `${n2c2_track1_home}/context/bert_baselines/submissions/{1,2,3}/actor.txt`

```
python run.py predict ${model_location_actor}/config.yaml
```
**Output**: `${model_location_actor}/predictions/n2c2ContextDataset/brat/test`, where `model_location` is defined below.


### Certainty

**Model Location Certainty**: See `${n2c2_track1_home}/context/bert_baselines/submissions/{1,2,3}/certainty.txt`

```
python run.py predict ${model_location_certainty}/config.yaml
```
**Output**: `${model_location_certainty}/predictions/n2c2ContextDataset/brat/test`, where `model_location` is defined below.

### Negation

**Model Location Negation**: See `${n2c2_track1_home}/context/bert_baselines/submissions/{1,2,3}/negation.txt`

```
python run.py predict ${model_location_negation}/config.yaml
```
**Output**: `${model_location_negation}/predictions/n2c2ContextDataset/brat/test`, where `model_location` is defined below.

### Temporality

**Model Location Temporality**: See `${n2c2_track1_home}/context/bert_baselines/submissions/{1,2,3}/temporality.txt`

```
python run.py predict ${model_location_temporality}/config.yaml
```
**Output**: `${model_location_temporality}/predictions/n2c2ContextDataset/brat/test`, where `model_location` is defined below.


## Merging predictions

Once all models have been run, merge the brat files.

```
${n2c2_track1_home}/context/bert_baselines/utils/merge_brat_predictions.py n2c2ContextDataset test --model_dirs ${model_location_action} ${model_location_actor} ${model_location_certainty} ${model_location_negation} ${model_location_temporality} --outdir ${outdir}
```
**Output**: 
