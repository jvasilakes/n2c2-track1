import warnings
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel, AdamW
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import precision_recall_fscore_support

from .layers import EntityPooler

# Ignore warning that BertModel is not using some layer parameters.
from transformers import logging
logging.set_verbosity_error()


class BertMultiHeadedSequenceClassifier(pl.LightningModule):
    """
    bert_model_name_or_path: e.g., 'bert-base-uncased'
    label_spec: dict from task names to number of labels. Can be obtained
                from data.n2c2SentencesDataModule.label_spec
    freeze_pretrained: bool. Default False. If True, freeze the BERT layers.
    use_entity_spans: bool. Default False. If True, use only the pooled
                      entity embeddings as input to the classifier heads.
                      If False, use the pooled embeddings of the entire input.
    entity_pool_fn: How to pool the token embeddings of
                    the target entity_mention. Possible values are
                    "max", "mean", "first", "last", "first-last".
    dropout_prob: Dropout probability for the classification layer.
    lr: learning rate
    weight_decay: weight decay rate
    """

    @classmethod
    def from_config(cls, config, datamodule):
        """
        :param config.ExperimentConfig config: config instance
        :param data.n2c2SentencesDataModule datamodule: data module instance
        """
        return cls(
                config.bert_model_name_or_path,
                datamodule.label_spec,
                config.freeze_pretrained,
                config.use_entity_spans,
                config.entity_pool_fn,
                config.dropout_prob,
                config.lr,
                config.weight_decay,
                datamodule.class_weights,
                )

    def __init__(
            self,
            bert_model_name_or_path,
            label_spec,
            freeze_pretrained=False,
            use_entity_spans=False,
            entity_pool_fn="max",
            dropout_prob=0.1,
            lr=1e-3,
            weight_decay=0.0,
            class_weights=None,
            ):
        super().__init__()
        self.bert_model_name_or_path = bert_model_name_or_path
        self.label_spec = label_spec
        self.freeze_pretrained = freeze_pretrained
        self.use_entity_spans = use_entity_spans
        self.entity_pool_fn = entity_pool_fn
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = self._validate_class_weights(
            class_weights, self.label_spec)

        self.bert_config = BertConfig.from_pretrained(
            self.bert_model_name_or_path)
        self.bert_config.hidden_dropout_prob = self.dropout_prob
        self.bert = BertModel.from_pretrained(
            self.bert_model_name_or_path, config=self.bert_config)
        if self.freeze_pretrained is True:
            for param in self.bert.parameters():
                param.requires_grad = False

        if self.use_entity_spans is True:
            insize = outsize = self.bert_config.hidden_size
            if self.entity_pool_fn == "first-last":
                insize = 2 * insize
            self.entity_pooler = EntityPooler(
                insize, outsize, self.entity_pool_fn)

        self.classifier_heads = nn.ModuleDict()
        for (task, num_labels) in label_spec.items():
            self.classifier_heads[task] = nn.Sequential(
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(self.bert_config.hidden_size, num_labels)
                    )
        # save __init__ arguments to self.hparams, which is logged by pl
        self.save_hyperparameters()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            offset_mapping=None,
            entity_spans=None
            ):
        """
        labels: dict from task names to torch.LongTensor labels of
                shape (batch_size,).
        offset_mapping: output from a transformers.PreTrainedTokenizer{Fast}
                        with return_offsets_mapping=True
        entity_spans: torch.LongTensor of shape (batch_size, 2) indicating
                      the character offsets of the target entity in the input.
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

        if entity_spans is not None and self.use_entity_spans is True:
            pooled_output = self.entity_pooler(
                bert_outputs.last_hidden_state, offset_mapping, entity_spans)
        else:
            pooled_output = bert_outputs.pooler_output

        clf_outputs = {}
        for (task, clf_head) in self.classifier_heads.items():
            logits = clf_head(pooled_output)
            task_labels = labels[task]
            if self.class_weights[task] is not None:
                # Only copy the weights to the model device once.
                if self.class_weights[task].device != self.device:
                    dev = self.device
                    self.class_weights[task] = self.class_weights[task].to(dev)
            loss_fn = CrossEntropyLoss(weight=self.class_weights[task])
            loss = loss_fn(logits.view(-1, self.label_spec[task]),
                           task_labels.view(-1))
            clf_outputs[task] = SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=bert_outputs.hidden_states,
                    attentions=bert_outputs.attentions)
        return clf_outputs

    def training_step(self, batch, batch_idx):
        task_outputs = self(
                **batch["encodings"],
                labels=batch["labels"],
                entity_spans=batch["entity_spans"])
        for (task, outputs) in task_outputs.items():
            self.log(f"train_loss_{task}", outputs.loss)
        return outputs.loss

    def predict_step(self, batch, batch_idx):
        task_outputs = self(
                **batch["encodings"],
                labels=batch["labels"],
                entity_spans=batch["entity_spans"])

        tasks = list(task_outputs.keys())
        inputs_with_predictions = {
                "texts": batch["texts"],
                "labels": batch["labels"],
                "entity_spans": batch["entity_spans"],
                "char_offsets": batch["char_offsets"],
                "docids": batch["docids"],
                "predictions": {task: [] for task in tasks}
                }
        for (task, outputs) in task_outputs.items():
            softed = nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(softed, dim=1)
            inputs_with_predictions["predictions"][task] = preds
        return inputs_with_predictions

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        task_outputs = self(
                **batch["encodings"],
                labels=batch["labels"],
                entity_spans=batch["entity_spans"])

        task_metrics = {}
        for (task, outputs) in task_outputs.items():
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
