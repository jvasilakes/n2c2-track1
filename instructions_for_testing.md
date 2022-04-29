# Running the prediction pipeline on test data

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


**Model Location**: See `${n2c2_track1_home}/context/bert_baselines/submissions/{1,2,3}/${task}.txt`

#### If using a single model (submissions 1 and 2)
```
python run.py predict --datasplit test ${model_location}/config.yaml
```
**Output**: `${model_location}/predictions/n2c2ContextDataset/brat/test`



#### If using an ensemble of models (submissions 3)
```
ensemble_outdir=/mnt/iusers01/nactem01/u14498jv/scratch/n2c2_track1/context/bert_baseline_logs/ensemble_predictions/
bash predict_ensemble.sh test \
                         ${n2c2_track1_home}/context/bert_baselines/submissions/3/${task}.txt \
                         ${ensemble_outdir}/test/${task}/${version}/
```
`${version}` is automatically determined by `predict_ensemble.sh` and is printed to standard out when running the script.

**Output**: `${ensemble_outdir}/test/${task}/${version}/`




## Merging predictions for submission

Once all models have been run, merge the brat files with the following command.
Let `sub_num` be one of `1, 2, 3`.

```
submission_outdir=/mnt/iusers01/nactem01/u14498jv/scratch/n2c2_track1/context/bert_baseline_logs/submission/${sub_num}
${n2c2_track1_home}/context/bert_baselines/utils/merge_brat_predictions.py \
                --pred_dirs ${outdir_action_sub_num} \
                            ${outdir_actor_sub_num} \
                            ${outdir_certainty_sub_num} \
                            ${outdir_negation_sub_num} \
                            ${outdir_temporality_sub_num} \
                --outdir ${submission_outdir}
```
Where `outdir_${task}_sub_num` corresponds to the test predictions for the given task and submission. For example, for Submission
2 on Action,

```
outdir_action_2=${model_location_action}/predictions/n2c2ContextDataset/brat/test/
```

For Submission 3 on Negation,
```
outdir_negation_3=${ensemble_outdir}/test/negation/${version}/
```

**Output**: `/mnt/iusers01/nactem01/u14498jv/scratch/n2c2_track1/context/bert_baseline_logs/submission/{1,2,3}/`
