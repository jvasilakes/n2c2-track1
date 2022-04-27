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

### Step 1: set the prediction model

For example, we'll predict using roberta trained on the first CV split.
```
ner_model=0/baseline_roberta
```
This corresponds to a subdirectory of `${n2c2_track1_home}/ner/experiments/`.


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

sh xec_n2c2_predict.sh blue experiments/${ner_model}/predict-test-org preprocess


**Output**: 
**Output Description**: brat-formatted files, one per input file, containing detected Disposition and NoDisposition events.


## Context Classification

**Input**: 

Context classification is further broken into 5 simultaneous tasks:

 * [Action](#action)
 * [Actor](#actor)
 * [Certainty](#certainty)
 * [Negation](#negation)
 * [Temporality](#temporality)

**Output**:
**Output Description**:

### Action

**Model Location**:


### Actor

**Model Location**:


### Certainty

**Model Location**:


### Negation

**Model Location**:


### Temporality

**Model Location**:
