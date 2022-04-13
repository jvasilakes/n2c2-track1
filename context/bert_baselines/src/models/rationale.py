import warnings
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertConfig, BertModel, AdamW
from sklearn.metrics import precision_recall_fscore_support

from src.models.losses import get_loss_function
from src.models.layers import TokenEmbeddingPooler, KumaMask, RecurrentEncoder
from src.models.model_outputs import SequenceClassifierOutputWithTokenMask


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
    """
    :param dict classifier_loss_kwargs: {task: **kwargs}. That is,
            kwargs must be specified *per task*.
    """

    @classmethod
    def from_config(cls, config, datamodule, **override_kwargs):
        """
        :param config.ExperimentConfig config: config instance
        :param data.n2c2.n2c2SentencesDataModule datamodule: data module
        """
        if config.use_entity_spans is False:
            msg = """You specified use_entity_spans=False but BertRationaleClassifer always uses entity spans, so this has been ignored."""  # noqa
            warnings.warn(msg)

        # We'll always look up loss kwargs by task, so we duplicate/separate
        # them out here.
        classifier_loss_kwargs = {}
        mask_loss_kwargs = {}
        for task in datamodule.label_spec.keys():
            classifier_loss_kwargs[task] = {}
            for (key, val) in config.classifier_loss_kwargs.items():
                # TODO: I don't like how hard-coded this is.
                if key == "class_weights":
                    # Change to the much less clear name used by torch.
                    key = "weight"
                    val = datamodule.class_weights[task]
                classifier_loss_kwargs[task][key] = val
            mask_loss_kwargs[task] = {}
            for (key, val) in config.mask_loss_kwargs.items():
                mask_loss_kwargs[task][key] = val

        kwargs = {
            "bert_model_name_or_path": config.bert_model_name_or_path,
            "label_spec": datamodule.label_spec,
            "freeze_pretrained": config.freeze_pretrained,
            "entity_pool_fn": config.entity_pool_fn,
            "dropout_prob": config.dropout_prob,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "classifier_loss_fn": config.classifier_loss_fn,
            "classifier_loss_kwargs": classifier_loss_kwargs,
            "mask_loss_fn": config.mask_loss_fn,
            "mask_loss_kwargs": mask_loss_kwargs,
        }
        for (key, val) in override_kwargs.items():
            if key in ["classifier_loss_kwargs", "mask_loss_kwargs"]:
                warnings.warn(f"Overriding {key} not supported. Please change the config file instead.")  # noqa
                continue
            kwargs[key] = val
        return cls(**kwargs)

    def __init__(
            self,
            bert_model_name_or_path,
            label_spec,
            freeze_pretrained=False,
            entity_pool_fn="max",
            dropout_prob=0.1,
            lr=1e-3,
            weight_decay=0.0,
            # class_weights=None,  Pass these in classifier_loss_kwargs now
            classifier_loss_fn="cross-entropy",
            classifier_loss_kwargs=None,
            mask_loss_fn="ratio",
            mask_loss_kwargs=None,
            ):
        super().__init__()
        self.bert_model_name_or_path = bert_model_name_or_path
        self.label_spec = label_spec
        self.freeze_pretrained = freeze_pretrained
        self.entity_pool_fn = entity_pool_fn
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.weight_decay = weight_decay
        # prepare loss function stuff
        self.classifier_loss_fn = classifier_loss_fn
        self.classifier_loss_kwargs = classifier_loss_kwargs
        self.mask_loss_fn = mask_loss_fn
        self.mask_loss_kwargs = mask_loss_kwargs

        # save __init__ arguments to self.hparams, which is logged by pl
        self.save_hyperparameters()

        # build the BERT model.
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
        self.entity_pooler = TokenEmbeddingPooler(
            pooler_insize, pooler_outsize, self.entity_pool_fn)

        # Build the task-specific model components.
        self.kuma_masks = nn.ModuleDict()
        self.mask_loss_fns = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        self.classifier_heads = nn.ModuleDict()
        self.classifier_loss_fns = nn.ModuleDict()
        for (task, num_labels) in label_spec.items():
            self.kuma_masks[task] = KumaMask(
                    self.bert_config.hidden_size + pooler_outsize)
            # Mask loss function for this task
            msk_loss_fn = get_loss_function(self.mask_loss_fn)
            kwargs = self.mask_loss_kwargs[task]
            self.mask_loss_fns[task] = msk_loss_fn(**kwargs)

            self.encoders[task] = RecurrentEncoder(
                    insize=pooler_outsize,
                    hidden_size=200,
                    cell="lstm")

            self.classifier_heads[task] = nn.Sequential(
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(2 * self.encoders[task].hidden_size, num_labels)
                    )
            # Classifier loss function for this task
            clf_loss_fn = get_loss_function(self.classifier_loss_fn)
            kwargs = self.classifier_loss_kwargs[task]
            self.classifier_loss_fns[task] = clf_loss_fn(**kwargs)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            offset_mapping=None,
            entity_spans=None,
            labels=None,
            dataset=None,
            ):
        """
        labels: dict from task names to torch.LongTensor labels of
                shape (batch_size,).
        offset_mapping: output from a transformers.PreTrainedTokenizer(Fast)
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
            if dataset is not None:
                # If doing multi-dataset learning, the task will
                # be formatted like {dataset}:{label},
                # e.g., "n2c2Context:Action"
                if dataset != task.split(':')[0]:
                    continue
            # Compute HardKuma gates
            entity_expanded = pooled_entity_output.unsqueeze(1).expand(h.size())  # noqa
            h_with_entity = torch.cat([h, entity_expanded], dim=2)
            z, z_dists = self.kuma_masks[task](h_with_entity)
            # Mask out PAD tokens.
            z = z * attention_mask.unsqueeze(-1)
            # Mask out CLS and SEP too.
            z[:, 0] = torch.tensor(0.)  # CLS
            lengths = attention_mask.sum(dim=1)
            z[torch.arange(z.size(0)), lengths - 1] = torch.tensor(0.)  # SEP

            # Use the gates to mask the inputs and encode them.
            outputs, final = self.encoders[task](h * z, lengths)

            # Compute the classifier and mask losses.
            logits = clf_head(final)
            task_labels = labels[task]
            clf_loss = self.classifier_loss_fns[task](
                    logits.view(-1, self.label_spec[task]),
                    task_labels.view(-1))
            mask_loss = self.mask_loss_fns[task](
                z.squeeze(-1), z_dists, attention_mask)
            clf_outputs[task] = SequenceClassifierOutputWithTokenMask(
                    loss=clf_loss,
                    mask_loss=mask_loss,
                    logits=logits,
                    hidden_states=bert_outputs.hidden_states,
                    attentions=bert_outputs.attentions,
                    mask=z.squeeze(-1))
        return clf_outputs

    @staticmethod
    def compute_mask_ratio(z, token_mask):
        return z.sum(dim=1) / token_mask.sum(dim=1)

    @staticmethod
    def compute_transition_rate(z, token_mask):
        total = torch.zeros(z.size(0)).type_as(z)
        J = z.size(1)
        I = J - 1  # noqa E741: Ambiguous variable name I
        for (i, j) in zip(range(0, I), torch.arange(1, J)):
            zi = z[:, i]
            zj = z[:, j]
            total += torch.abs(zi - zj)
        return total / token_mask[:, :-1].sum(dim=1)

    def get_model_outputs(self, batch):
        dataset = None
        if "dataset" in batch.keys():
            dataset = batch["dataset"]
        outputs_by_task = self(
                **batch["encodings"],
                entity_spans=batch["entity_spans"],
                labels=batch["labels"],
                dataset=dataset)
        return outputs_by_task

    def training_step(self, batch, batch_idx):
        outputs_by_task = self.get_model_outputs(batch)
        batch_loss = None
        for (task, outputs) in outputs_by_task.items():
            if batch_loss is None:
                # So it's on the right device
                batch_loss = torch.tensor(0.).type_as(outputs.loss)
            batch_loss += outputs.loss + outputs.mask_loss
            self.log(f"train_loss_{task}", outputs.loss)
            self.log(f"mask_loss_{task}", outputs.mask_loss)
            mask_ratios = self.compute_mask_ratio(
                    outputs.mask, batch["encodings"]["attention_mask"])
            self.log(f"mask_ratio_{task}", mask_ratios.mean())
            trans_rates = self.compute_transition_rate(
                outputs.mask, batch["encodings"]["attention_mask"])
            self.log(f"transition_rate_{task}", trans_rates.mean())
        return batch_loss

    def predict_step(self, batch, batch_idx):
        outputs_by_task = self.get_model_outputs(batch)
        tasks = list(outputs_by_task.keys())
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
        for (task, outputs) in outputs_by_task.items():
            softed = nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(softed, dim=1)
            inputs_with_predictions["predictions"][task] = preds
            inputs_with_predictions["zmask"][task] = outputs.mask
        return inputs_with_predictions

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs_by_task = self.get_model_outputs(batch)
        task_metrics = {}
        for (task, outputs) in outputs_by_task.items():
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
            mask_ratios_by_task[task] = np.array(mask_ratios_by_task[task]).mean()  # noqa

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
