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


class BertMultiHeadedSequenceClassifier(pl.LightningModule):
    """
    bert_model_name_or_path: e.g., 'bert-base-uncased'
    label_spec: dict from task names to number of labels. Can be obtained
                from data.n2c2SentencesDataModule.label_spec
    freeze_pretrained: bool. Default False. If True, freeze the BERT layers.
    use_entity_spans: bool. Default False. If True, use only the pooled
                      entity embeddings as input to the classifier heads.
                      If False, use the pooled embeddings of the entire input.
    entity_pool_fn: "max" or "mean". How to pool the token embeddings of
                    the target entity_mention.
    lr: learning rate
    weight_decay: weight decay rate
    """

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
            class_weights=None):
        super().__init__()
        self.bert_model_name_or_path = bert_model_name_or_path
        self.label_spec = label_spec
        self.freeze_pretrained = freeze_pretrained
        self.use_entity_spans = use_entity_spans
        self.entity_pool_fn = entity_pool_fn
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.weight_decay = weight_decay
        self.entity_pool_fn = entity_pool_fn
        self._validate_class_weights(class_weights, self.label_spec)
        self.class_weights = class_weights

        self.bert_config = BertConfig.from_pretrained(
            self.bert_model_name_or_path)
        self.bert = BertModel.from_pretrained(
            self.bert_model_name_or_path, config=self.bert_config)
        if self.freeze_pretrained is True:
            for param in self.bert.parameters():
                param.requires_grad = False

        if self.use_entity_spans is True:
            self.entity_pooler = nn.Sequential(
                    nn.Linear(self.bert_config.hidden_size,
                              self.bert_config.hidden_size),
                    nn.Tanh())

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
            masked_hidden, token_mask = self.mask_hidden(
                bert_outputs.last_hidden_state, offset_mapping, entity_spans)
            pooled = self.pool_entity_embeddings(masked_hidden, token_mask)
            pooled_output = self.entity_pooler(pooled)
        else:
            pooled_output = bert_outputs.pooler_output

        clf_outputs = {}
        for (task, clf_head) in self.classifier_heads.items():
            logits = clf_head(pooled_output)
            task_labels = labels[task]
            # TODO: I don't like doing this op at every forward,
            # but I can't do it in __init__ since self.device is still cpu.
            weight = self.class_weights[task].to(self.device)
            loss_fn = CrossEntropyLoss(weight=weight)
            loss = loss_fn(logits.view(-1, self.label_spec[task]),
                           task_labels.view(-1))
            clf_outputs[task] = SequenceClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=bert_outputs.hidden_states,
                    attentions=bert_outputs.attentions)
        return clf_outputs

    def pool_entity_embeddings(self, masked_hidden, token_mask):
        if self.entity_pool_fn == "max":
            # Replace masked with -inf to avoid zeroing out
            # hidden dimensions if the non-masked values are all negative.
            masked_hidden[torch.logical_not(token_mask)] = -torch.inf
            pooled = torch.max(masked_hidden, axis=1)[0]
        elif self.entity_pool_fn == "mean":
            pooled = masked_hidden.sum(axis=1) / token_mask.sum(axis=1)
        else:
            raise ValueError(f"Unknown pool function {self.entity_pool_fn}")
        return pooled

    def mask_hidden(self, hidden_states, offset_mapping, entity_spans):
        # We need to replace the (0, 0) spans in the offset mapping
        # with (-1, -1) to avoid masking errors when the
        # start of the span is 0.
        offset_mask = offset_mapping == torch.tensor([0, 0], device=self.device)  # noqa
        offset_mask = offset_mask[:, :, 0] & offset_mask[:, :, 1]
        offset_mapping[offset_mask, :] = torch.tensor([-1, -1]).type_as(offset_mapping)  # noqa
        # Keep all tokens whose start char is >= the entity start and
        #   whose end char is <= the entity end.
        start_spans = entity_spans[:, 0].unsqueeze(-1).expand(
                -1, offset_mapping.size(1))
        end_spans = entity_spans[:, 1].unsqueeze(-1).expand(
                -1, offset_mapping.size(1))
        token_mask = (offset_mapping[:, :, 0] >= start_spans) & \
                     (offset_mapping[:, :, 1] <= end_spans)
        # Duplicate the mask across the hidden dimension
        token_mask_ = token_mask.unsqueeze(-1).expand(hidden_states.size())
        if len((token_mask.sum(axis=1) == torch.tensor(0.)).nonzero()) > 0:
            raise ValueError("Entity span not found! Try increasing max_seq_length.")  # noqa
        masked = hidden_states * token_mask_
        return masked, token_mask_

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
        for (task, weights) in class_weights.items():
            if not torch.is_tensor(weights):
                raise TypeError(f"class weights must be torch.Tensor. Got {type(weights)}.")  # noqa
            num_classes = label_spec[task]
            if len(weights) != num_classes:
                raise ValueError(f"Number of weights != number of classes for task {task}")  # noqa


MODEL_LOOKUP = {
        "bert-sequence-classifier": BertMultiHeadedSequenceClassifier,
        }
