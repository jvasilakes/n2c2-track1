import warnings
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from transformers import BertConfig, BertModel, AdamW
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import precision_recall_fscore_support

from src.models.losses import get_loss_function
from src.models.layers import TokenEmbeddingPooler

# Ignore warning that BertModel is not using some layer parameters.
from transformers import logging
logging.set_verbosity_error()


class BertMultiHeadedSequenceClassifier(pl.LightningModule):
    """
    bert_model_name_or_path: e.g., 'bert-base-uncased'
    label_spec: dict from task names to number of labels. Can be obtained
        from data.n2c2.n2c2SentencesDataModule.label_spec
    freeze_pretrained: bool. Default False. If True, freeze the BERT layers.
    use_entity_spans: bool. Default False. If True, use only the pooled
        entity embeddings as input to the classifier heads.
        If False, use the pooled embeddings of the entire input.
    entity_pool_fn: How to pool the token embeddings of the target
        entity_mention. Possible values are
        "max", "mean", "first", "last", "first-last".
    use_levitated_markers: bool. Default False. If True, uses packed levitated
        markers.  Only used if use_entity_spans=True.
    levitated_marker_pool_fn: How to pool the token embeddings of the
        levitated markers. Possible values are
        "max", "mean", "first", "last", "first-last".
    dropout_prob: Dropout probability for the classification layer.
    lr: learning rate of the optimizer
    weight_decay: weight decay rate of the optimizer
    """

    @classmethod
    def from_config(cls, config, datamodule, **override_kwargs):
        """
        :param config.ExperimentConfig config: config instance
        :param data.n2c2SentencesDataModule datamodule: data module instance
        """
        classifier_loss_kwargs = {}
        for task in datamodule.label_spec.keys():
            classifier_loss_kwargs[task] = {}
            for (key, val) in config.classifier_loss_kwargs.items():
                # TODO: I don't like how hard-coded this is.
                if key == "class_weights":
                    # Change to the much less clear name used by torch
                    key = "weight"
                    val = datamodule.class_weights[task]
                classifier_loss_kwargs[task][key] = val

        kwargs = {
            "bert_model_name_or_path": config.bert_model_name_or_path,
            "label_spec": datamodule.label_spec,
            "freeze_pretrained": config.freeze_pretrained,
            "use_entity_spans": config.use_entity_spans,
            "entity_pool_fn": config.entity_pool_fn,
            "use_levitated_markers": config.use_levitated_markers,
            "levitated_marker_pool_fn": config.levitated_marker_pool_fn,
            "dropout_prob": config.dropout_prob,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "classifier_loss_fn": config.classifier_loss_fn,
            "classifier_loss_kwargs": classifier_loss_kwargs,
        }
        for (key, val) in override_kwargs.items():
            if key == "classifier_loss_kwargs":
                warnings.warn("Overriding classifier_loss_kwargs not supported. Please change the config file instead.")  # noqa
                continue
            kwargs[key] = val
        return cls(**kwargs)

    def __init__(
            self,
            bert_model_name_or_path,
            label_spec,
            freeze_pretrained=False,
            use_entity_spans=False,
            entity_pool_fn="max",
            use_levitated_markers=False,
            levitated_marker_pool_fn="max",
            dropout_prob=0.1,
            lr=1e-3,
            weight_decay=0.0,
            classifier_loss_fn="cross-entropy",
            classifier_loss_kwargs=None,
            ):
        super().__init__()
        self.bert_model_name_or_path = bert_model_name_or_path
        self.label_spec = label_spec
        self.freeze_pretrained = freeze_pretrained
        self.use_entity_spans = use_entity_spans
        self.entity_pool_fn = entity_pool_fn
        self.use_levitated_markers = use_levitated_markers
        self.levitated_marker_pool_fn = levitated_marker_pool_fn
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.weight_decay = weight_decay
        # prepare loss function stuff
        self.classifier_loss_fn = classifier_loss_fn
        self.classifier_loss_kwargs = classifier_loss_kwargs

        # BERT
        self.bert_config = BertConfig.from_pretrained(
            self.bert_model_name_or_path)
        self.bert_config.hidden_dropout_prob = self.dropout_prob
        self.bert = BertModel.from_pretrained(
            self.bert_model_name_or_path, config=self.bert_config)
        if self.freeze_pretrained is True:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Entity and levitated marker embedding poolers
        if self.use_entity_spans is False and self.use_levitated_markers is True:  # noqa
            warnings.warn("levitated markers only supported when use_entity_spans=True. Not using levitated markers.")  # noqa
            self.use_levitated_markers = False
        if self.use_entity_spans is True:
            pooler_insize = pooler_outsize = self.bert_config.hidden_size
            self.entity_pooler = TokenEmbeddingPooler(
                pooler_insize, pooler_outsize, self.entity_pool_fn)
            if self.use_levitated_markers is True:
                self.levitated_marker_pooler = TokenEmbeddingPooler(
                    pooler_insize, pooler_outsize,
                    self.levitated_marker_pool_fn)

        # Classifiers, one per task
        self.classifier_heads = nn.ModuleDict()
        self.classifier_loss_fns = nn.ModuleDict()
        classifier_insize = self.bert_config.hidden_size
        if self.use_levitated_markers is True:
            # b/c we'll concat the pooled representations
            # from entities and markers
            classifier_insize = 2 * classifier_insize
        for (task, num_labels) in label_spec.items():
            self.classifier_heads[task] = nn.Sequential(
                    # TODO: try adding in a layernorm here.
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(classifier_insize, num_labels)
                    )
            clf_loss_fn = get_loss_function(self.classifier_loss_fn)
            kwargs = self.classifier_loss_kwargs[task]
            self.classifier_loss_fns[task] = clf_loss_fn(**kwargs)

        # save __init__ arguments to self.hparams, which is logged by pl
        self.save_hyperparameters()

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            entity_token_idxs=None,
            levitated_marker_idxs=None,
            labels=None,
            dataset=None,
            ):
        """
        input_ids, attention_mask, token_type_ids, position_ids,
            head_mask, inputs_embeds: Inputs to BertModel.
        entity_token_idxs: nested list of wordpiece token indices indicating
            the position of the entity in the input.
        levitated_marker_idxs: nested list of wordpiece token indices
            indicating the positions of the marked spans in the input.
        labels: dict from task names to torch.LongTensor labels of
                shape (batch_size,).

        Basically, the inputs are encoded and pooled via BERT (pooled_output).
        The use_entity_spans and use_levitated_markers options affect how
        pooled_output is computed.

         * use_entity_spans = use_levitated_markers = False:
                  X--(BERT)-->pooled_output
         * use_entity_spans = True, use_levitated_markers = False:
                  X--(BERT)-->last_hidden_state
                  last_hidden_state--(entity_pooler)-->pooled_output
         * use_entity_spans = use_levitated_markers = True:
                  X--(BERT)-->last_hidden_state
                  last_hidden_state--(entity_pooler)-->entity_pooled
                  last_hidden_state--(levitated_pooler)-->levitated_pooled
                  pooled_output <-- [entity_pooled;levitated_pooled]
        """
        bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
                )

        if self.use_entity_spans is True:
            if entity_token_idxs is not None:
                pooled_output = self.entity_pooler(
                    bert_outputs.last_hidden_state, entity_token_idxs)
            else:
                raise ValueError("use_entity_spans=True but no entity_token_idxs provided in forward()")  # noqa
            if self.use_levitated_markers is True:
                if levitated_marker_idxs is not None:
                    marker_output = self.levitated_marker_pooler(
                        bert_outputs.last_hidden_state, levitated_marker_idxs)
                    pooled_output = torch.cat(
                        (pooled_output, marker_output), dim=1)  # noqa
                else:
                    raise ValueError("use_levitated_markers=True but no levitated_marker_idxs provided in forward()")  # noqa
        else:
            pooled_output = bert_outputs.pooler_output

        clf_outputs = {}
        for (task, clf_head) in self.classifier_heads.items():
            if dataset is not None:
                # If doing multi-dataset learning, the task will
                # be formatted like {dataset}:{label},
                # e.g., "n2c2Context:Action"
                if dataset != task.split(':')[0]:
                    continue
            logits = clf_head(pooled_output)
            if labels is not None:
                task_labels = labels[task]
                clf_loss = self.classifier_loss_fns[task](
                        logits.view(-1, self.label_spec[task]),
                        task_labels.view(-1))
            else:
                clf_loss = None
            clf_outputs[task] = SequenceClassifierOutput(
                    loss=clf_loss,
                    logits=logits,
                    hidden_states=bert_outputs.hidden_states,
                    attentions=bert_outputs.attentions)
        return clf_outputs

    def get_model_outputs(self, batch):
        dataset = None
        if "dataset" in batch.keys():
            # We're doing multi-dataset learning
            dataset = batch["dataset"]
        levitated_marker_idxs = None
        if "levitated_marker_idxs" in batch.keys():
            levitated_marker_idxs = batch["levitated_marker_idxs"]
        bert_inputs = {k: v for (k, v) in batch["encodings"].items()
                       if k != "offset_mapping"}
        outputs_by_task = self(
                **bert_inputs,
                entity_token_idxs=batch["entity_token_idxs"],
                levitated_marker_idxs=levitated_marker_idxs,
                labels=batch["labels"],
                dataset=dataset)
        return outputs_by_task

    def training_step(self, batch, batch_idx):
        outputs_by_task = self.get_model_outputs(batch)
        total_loss = torch.tensor(0.).to(self.device)
        for (task, outputs) in outputs_by_task.items():
            total_loss += outputs.loss
            self.log(f"train_loss_{task}", outputs.loss)
        return total_loss

    def predict_step(self, batch, batch_idx):
        outputs_by_task = self.get_model_outputs(batch)
        tasks = list(outputs_by_task.keys())
        inputs_with_predictions = {
                "input_ids": batch["encodings"]["input_ids"],
                "texts": batch["texts"],
                "entity_token_idxs": batch["entity_token_idxs"],
                "entity_char_spans": batch["entity_char_spans"],
                "char_offsets": batch["char_offsets"],
                "docids": batch["docids"],
                "predictions": {task: [] for task in tasks}
                }
        for (task, outputs) in outputs_by_task.items():
            softed = nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(softed, dim=1)
            inputs_with_predictions["predictions"][task] = preds
        return inputs_with_predictions

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs_by_task = self.get_model_outputs(batch)
        task_metrics = {}
        for (task, outputs) in outputs_by_task.items():
            preds = torch.argmax(outputs.logits, axis=1)
            task_metrics[task] = {"loss": outputs.loss,
                                  "preds": preds,
                                  "labels": batch["labels"][task]}
        return task_metrics

    def validation_epoch_end(self, task_metrics):
        losses_by_task = defaultdict(list)
        preds_by_task = defaultdict(list)
        labels_by_task = defaultdict(list)

        for example in task_metrics:
            for (task, metrics) in example.items():
                losses_by_task[task].append(metrics["loss"].detach().cpu().numpy())  # noqa
                preds_by_task[task].extend(metrics["preds"].detach().cpu().numpy())  # noqa
                labels_by_task[task].extend(metrics["labels"].detach().cpu().numpy())  # noqa

        val_losses = []
        macro_f1s = []
        for task in losses_by_task.keys():
            losses_by_task[task] = np.array(losses_by_task[task]).mean()
            preds_by_task[task] = np.array(preds_by_task[task])
            labels_by_task[task] = np.array(labels_by_task[task])

            self.log(f"val_loss_{task}", losses_by_task[task], prog_bar=False)
            val_losses.append(losses_by_task[task])
            for avg_fn in ["micro", "macro"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p, r, f1, _ = precision_recall_fscore_support(
                            labels_by_task[task], preds_by_task[task],
                            average=avg_fn)
                    if avg_fn == "macro":
                        macro_f1s.append(f1)
                res = {f"{avg_fn}_{task}_precision": p,
                       f"{avg_fn}_{task}_recall": r,
                       f"{avg_fn}_{task}_F1": f1}
                self.log_dict(res, prog_bar=False)

        self.log_dict({"avg_val_loss": np.mean(val_losses),
                       "avg_macro_f1": np.mean(macro_f1s)}, prog_bar=True)

    def configure_optimizers(self):
        params = self.parameters()
        opt = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return opt

    def _validate_class_weights(self, class_weights, label_spec):
        if not isinstance(class_weights, (dict, type(None))):
            raise ValueError(f"class_weights must be None or dict(). Got {type(class_weights)}.")  # noqa 
        if class_weights is not None:
            for (task, weights) in class_weights.items():
                if not torch.is_tensor(weights):
                    raise TypeError(f"class weights must be torch.Tensor. Got {type(weights)}.")  # noqa
                num_classes = label_spec[task]
                if len(weights) != num_classes:
                    raise ValueError(f"Number of weights != number of classes for task {task}")  # noqa
        elif class_weights is None:
            class_weights = {task: None for task in label_spec.keys()}
        return class_weights
