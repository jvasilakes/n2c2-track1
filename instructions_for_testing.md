# Running the prediction pipeline on test data

The pipeline contains three main sections:

 1. [NER](#ner)
 2. [Event Extraction](#event-extraction)
 3. [Context Classification](#context-classification)

There are 3 data releases in the evaluation:

 1. End-to-end (NER, NER+Event, NER+Event+Context), given plain text files.
 2. Event, Event+Context given gold-standard NER.
 3. Context given gold-standard events.

For each release, we can submit up to 3 runs.


## Preliminaries

`n2c2_track1_home` is set to the directory containing the `ner`, `ee`, and `context` subdirectories for each task.

The test data will be a set of plain text files.

First, do the following:

 1. A quick manual review of the text files to check for obvious problems, e.g. empty documents, documents in other languages, etc.
 2. Copy/move all text files to `${n2c2_track1_home}/n2c2TestData/test/${release_number}`.
 3. Count all text files to ensure we have the same number as the organizers released: `ls -1 ${n2c2_track1_home}/n2c2TestData/test/0/*.txt | wc -l`
 4. Run and validate sentence splitting:

```
python ${n2c2_track1_home}/scripts/run_biomedicus_sentences.py \
		--biomedicus_data_dir ~/.biomedicus/data/sentences/ \
		--indir ${n2c2_track1_home}/n2c2TestData/test/0 \
                --outdir ${n2c2_track1_home}/n2c2TestData/test/segmented/ \

python ${n2c2_track1_home}/scripts/validate_segmentations.py \
		--segments_dir ${n2c2_track1_home}/n2c2TestData/test/segmented/ \
		--text_dir ${n2c2_track1_home}/n2c2TestData/test/0 \
```

# NER

**Input**: `${n2c2_track1_home}/n2c2TestData/test`
**Input Description**: plain text files.
**Model Location**:

## Step 1: set the prediction model

Set `ner_model` to **Model Location**.
For example, we'll predict using roberta trained on the first CV split.
```
ner_model=0/baseline_roberta
```
which corresponds to a subdirectory of `${n2c2_track1_home}/ner/experiments/`.


## Step 2: run prediction

 * Ensure that the `test_data` field in `${n2c2_track1_home}/ner/experiments/${ner_model}/predict-test.yaml` is set to `corpus/test/0/`. 
 * Ensure that a model checkpoint exists at `${n2c2_track1_home}/ner/experiments/${ner_model}/joint`
 * Run the pre-process, prediction, post-process pipeline with
```
cd ${n2c2_track1_home}
bash ner/run_test.sh
```
**Output**: `experiments/${ner_model}/predict-test-org`
**Output Description**: brat-formatted files, one per input file, containing detected medication spans.

### Step 3: run ensemble

 * Assuming that we have predictions from different models in a folder `${n2c2_track1_home}/ner/experiments/test_predictions`
 * Ensure that in that folder we have predictions produced by a different model in a different folder, such as `${n2c2_track1_home}/ner/experiments/test_predictions/${ner_model}/predict-test-org`. This folder is the output from Step 2.
 * Run the following command line

```
cd ${n2c2_track1_home}/ner
mkdir ${n2c2_track1_home}/ner/experiments/test_ensemble
python src/ensemble.py --indir ${n2c2_track1_home}/ner/experiments/test_predictions \
		       --outdir ${n2c2_track1_home}/ner/experiments/test_ensemble \
		       --type test
``` 
**Output**: `${n2c2_track1_home}/ner/experiments/test_ensemble`
**Output Description**: brat-formatted files, one per input file, containing detected medication spans.




# Event Extraction

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


# Context Classification

**Input**: `${n2c2_track1_home}/ee/saved_models/${bert_model}_${train_split}/predictions/test`

Context classification is further broken into 5 simultaneous tasks:

 * Action
 * Actor
 * Certainty
 * Negation
 * Temporality

Separate models have been trained for each subtask. We run prediction for each and then merge them.  
Prediction follows the same template for all tasks. Let `${task}` be one of `action, actor, certainty, negation, temporality`.


**Model Location**: See `/net/scratch2/mbassnt3/n2c2_2022/models_for_submission/` on csf3.

#### If using a single model (submissions 1 and 2)
```
python run.py predict ${model_location}/config.yaml \
                      --datasplit test \
                      --datadir ${n2c2_track1_home}/ee/saved_models/${bert_model}_${train_split}/predictions/test \
                      --sentences_dir ${n2c2_track1_home}/n2c2TestData/test/segmented/
```
**Output**: `${model_location}/predictions/n2c2ContextDataset/brat/test`



#### If using an ensemble of models (submission 3)
```
submissions_dir=/net/scratch2/mbassnt3/n2c2_2022/submissions/release_{1,2,3}/submission_{1,2,3}/

bash utils/predict_ensemble.sh --task ${task} --split test \
                               --modeldir ${submissions_dir}/context/models/${task}/ \
                               --anndir ${n2c2_track1_home}/ee/saved_models/${bert_model}_${train_split}/predictions/test \
                               --sentsdir ${n2c2_track1_home}/n2c2TestData/test/segmented/ \
                               --outdir ${submission_dir}/context/predictions/
```

**Output**: `${submission_dir}/context/predictions/test/${task}/${version}/ann`

`${version}` is automatically determined by `predict_ensemble.sh` and is printed to standard out when running the script.




## Merging predictions for submission

Once all models have been run, merge the brat files with the following command.

```
predictions_dir=/net/scratch2/mbassnt3/n2c2_2022/submissions/release_{1,2,3}/submission_{1,2,3}/context/predictions/test/
${n2c2_track1_home}/context/bert_baselines/utils/merge_brat_predictions.py \
                --pred_dirs ${predictions_dir}/{Action,Actor,Certainty,Negation,Temporality} \
                --outdir ${predictions_dir}/all
```
**Output**: `/net/scratch2/mbassnt3/n2c2_2022/submissions/release_{1,2,3}/submission_{1,2,3}/context/predictions/test/all/`
