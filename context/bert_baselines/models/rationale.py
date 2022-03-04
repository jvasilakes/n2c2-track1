import warnings
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertConfig, BertModel, AdamW
from sklearn.metrics import precision_recall_fscore_support

from .layers import EntityPooler, KumaMask, RecurrentEncoder
from .model_outputs import SequenceClassifierOutputWithTokenMask


def format_input_ids_and_masks(input_ids, masks, tokenizer):
    """
    Used with the output of BertRationaleClassifier.predict_step.
    Outputs a nested list of (token, z) for each input example. E.g.,

    dm = n2c2SentencesDataModule()
    m = BertRationaleClassifer()
    preds = m.predict_step(batch)
    masked_inputs = format_input_ids_and_masks(
                        preds["input_ids"],
                        preds["zmask"][task],
                        dm.tokenizer) -> List[List[Tuple]]
    """
    toks = tokenizer.convert_ids_to_tokens(input_ids)
    toks = [tok for (tok, tid) in zip(toks, input_ids) if tid != 0]
    mask = [z.item() for (z, tid)
            in zip(masks, input_ids) if tid != 0]
    assert len(toks) == len(mask)
    masked_tokens = list(zip(toks, mask))
    return masked_tokens


class BertRationaleClassifier(pl.LightningModule):

    @classmethod
    def from_config(cls, config, datamodule):
        """
        :param config.ExperimentConfig config: config instance
        :param data.n2c2SentencesDataModule datamodule: data module instance
        """
        if config.use_entity_spans is False:
            msg = """You specified use_entity_spans=False but
            BertRationaleClassifer always uses entity spans,
            so this has been ignored."""
            warnings.warn(msg)
        return cls(
                config.bert_model_name_or_path,
                datamodule.label_spec,
                config.freeze_pretrained,
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

        pooler_insize = pooler_outsize = self.bert_config.hidden_size
        if self.entity_pool_fn == "first-last":
            pooler_insize = 2 * pooler_insize
        self.entity_pooler = EntityPooler(
            pooler_insize, pooler_outsize, self.entity_pool_fn)

        self.kuma_masks = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        self.classifier_heads = nn.ModuleDict()
        for (task, num_labels) in label_spec.items():
            self.kuma_masks[task] = KumaMask(
                    self.bert_config.hidden_size + pooler_outsize)  # noqa

            self.encoders[task] = RecurrentEncoder(
                    insize=pooler_outsize,
                    hidden_size=200,
                    cell="lstm")

            self.classifier_heads[task] = nn.Sequential(
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(2 * self.encoders[task].hidden_size, num_labels)
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
            offset_mapping=None,
            labels=None,
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

        h = bert_outputs.last_hidden_state
        pooled_entity_output = self.entity_pooler(
                h, offset_mapping, entity_spans)

        clf_outputs = {}
        for (task, clf_head) in self.classifier_heads.items():
            # Compute HardKuma gates
            entity_expanded = pooled_entity_output.unsqueeze(1).expand(h.size())  # noqa
            h_with_entity = torch.cat([h, entity_expanded], dim=2)
            z = self.kuma_masks[task](h_with_entity)
            z = z * attention_mask.unsqueeze(-1)

            # Use the gates to mask the inputs and encode them.
            lengths = attention_mask.sum(dim=1)
            outputs, final = self.encoders[task](h * z, lengths)

            logits = clf_head(final)
            task_labels = labels[task]
            if self.class_weights[task] is not None:
                # Only copy the weights to the model device once.
                if self.class_weights[task].device != self.device:
                    dev = self.device
                    self.class_weights[task] = self.class_weights[task].to(dev)
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights[task])
            loss = loss_fn(logits.view(-1, self.label_spec[task]),
                           task_labels.view(-1))
            clf_outputs[task] = SequenceClassifierOutputWithTokenMask(
                    loss=loss,
                    logits=logits,
                    hidden_states=bert_outputs.hidden_states,
                    attentions=bert_outputs.attentions,
                    mask=z.squeeze(-1))
        return clf_outputs

    @staticmethod
    def compute_mask_ratio(z, token_mask):
        return z.sum(dim=1) / token_mask.sum(dim=1)

    def training_step(self, batch, batch_idx):
        task_outputs = self(
                **batch["encodings"],
                labels=batch["labels"],
                entity_spans=batch["entity_spans"])
        for (task, outputs) in task_outputs.items():
            self.log(f"train_loss_{task}", outputs.loss)
            mask_ratio = self.compute_mask_ratio(
                    outputs.mask, batch["encodings"]["attention_mask"])
            self.log(f"mask_ratio_{task}", mask_ratio.mean())
        return outputs.loss

    def predict_step(self, batch, batch_idx):
        task_outputs = self(
                **batch["encodings"],
                labels=batch["labels"],
                entity_spans=batch["entity_spans"])

        tasks = list(task_outputs.keys())
        inputs_with_predictions = {
                "texts": batch["texts"],
                "input_ids": batch["encodings"]["input_ids"],
                "labels": batch["labels"],
                "entity_spans": batch["entity_spans"],
                "char_offsets": batch["char_offsets"],
                "docids": batch["docids"],
                "predictions": {task: [] for task in tasks},
                "zmask": {task: [] for task in tasks},
                }
        for (task, outputs) in task_outputs.items():
            softed = nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(softed, dim=1)
            inputs_with_predictions["predictions"][task] = preds
            inputs_with_predictions["zmask"][task] = outputs.mask
        return inputs_with_predictions

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        task_outputs = self(
                **batch["encodings"],
                labels=batch["labels"],
                entity_spans=batch["entity_spans"])

        task_metrics = {}
        for (task, outputs) in task_outputs.items():
            preds = torch.argmax(outputs.logits, axis=1)
            mask_ratios = self.compute_mask_ratio(
                    outputs.mask, batch["encodings"]["attention_mask"])
            task_metrics[task] = {"loss": outputs.loss,
                                  "preds": preds,
                                  "labels": batch["labels"][task],
                                  "mask_ratios": mask_ratios}
        return task_metrics

    def validation_epoch_end(self, task_metrics):
        """
        Flatten batched metrics and summarize.
        """
        losses_by_task = defaultdict(list)
        preds_by_task = defaultdict(list)
        labels_by_task = defaultdict(list)
        mask_ratios_by_task = defaultdict(list)

        for batch in task_metrics:
            for (task, metrics) in batch.items():
                losses_by_task[task].append(metrics["loss"].detach().cpu().numpy())  # noqa
                preds_by_task[task].extend(metrics["preds"].detach().cpu().numpy())  # noqa
                labels_by_task[task].extend(metrics["labels"].detach().cpu().numpy())  # noqa
                mask_ratios_by_task[task].extend(metrics["mask_ratios"].detach().cpu().numpy())  # noqa

        val_losses = []
        macro_f1s = []
        for task in losses_by_task.keys():
            losses_by_task[task] = np.array(losses_by_task[task]).mean()
            preds_by_task[task] = np.array(preds_by_task[task])
            labels_by_task[task] = np.array(labels_by_task[task])
            mask_ratios_by_task[task] = np.array(labels_by_task[task]).mean()

            self.log(f"val_loss_{task}", losses_by_task[task], prog_bar=False)
            self.log(f"avg_mask_ratio_{task}",
                     mask_ratios_by_task[task], prog_bar=False)
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
