import os
import warnings
from glob import glob
from collections import defaultdict, Counter

import brat_reader as br


ENSEMBLE_LOOKUP = {}


def register_ensembler(ensembler_name):
    def add_to_lookup(cls):
        ENSEMBLE_LOOKUP[ensembler_name] = cls
        return cls
    return add_to_lookup


class BratEnsembler(object):

    def __init__(self, model_dirs, dataset="n2c2ContextDataset",
                 datasplit="dev", task="Action"):
        self.model_dirs = self.validate(model_dirs, dataset, datasplit)
        print(f"Ensembling {len(self.model_dirs)} models.")
        self.dataset = dataset
        self.datasplit = datasplit
        self.task = task

    def choose(self, predicted_values: list) -> str:
        raise NotImplementedError()

    def load_predictions(self) -> dict:
        # {model_name: {file_name: anns}}
        predictions_by_filename_model = defaultdict(dict)
        for mdir in self.model_dirs:
            preds_dir = os.path.join(
                    mdir, f"predictions/{self.dataset}/brat/{self.datasplit}")
            preds_glob = os.path.join(preds_dir, "*.ann")
            pred_files = glob(preds_glob)
            for pred_f in pred_files:
                ann_preds = br.BratAnnotations.from_file(pred_f)
                pred_f_bn = os.path.basename(pred_f)
                predictions_by_filename_model[pred_f_bn][mdir] = ann_preds
        return predictions_by_filename_model

    def run(self):
        preds_by_filename_model = self.load_predictions()
        ensembled = self.ensemble_brat_predictions(preds_by_filename_model)
        return ensembled

    def ensemble_brat_predictions(self, preds_by_filename_model):
        pred_anns_by_fname = defaultdict(list)
        for (fname, preds_by_model) in preds_by_filename_model.items():
            all_events = [anns.events for anns in preds_by_model.values()]

            num_events = len(all_events[0])
            for events in all_events:
                if len(events) != num_events:
                    raise ValueError(f"Got different number of predictions in file {fname}!")  # noqa

            out_events = []
            for i in range(num_events):
                pred_values = [events[i].attributes[self.task].value
                               for events in all_events]
                ensembled_pred = self.choose(pred_values)
                out_event = all_events[0][i].copy()
                out_event._source_file = fname
                out_event.attributes[self.task].value = ensembled_pred
                out_events.append(out_event)
            out_anns = br.BratAnnotations.from_events(out_events)
            pred_anns_by_fname[fname] = out_anns
        return pred_anns_by_fname

    def validate(self, model_dirs, dataset, datasplit):
        for mdir in model_dirs:
            ckpt_dir = os.path.join(mdir, "checkpoints")
            if not os.path.isdir(ckpt_dir):
                raise OSError(f"No checkpoint directory found at {ckpt_dir}")
            ckpt_glob = os.path.join(ckpt_dir, "*.ckpt")
            ckpt_files = glob(ckpt_glob)
            if len(ckpt_files) == 0:
                raise OSError(f"No checkpoints found at {ckpt_glob}")
            if len(ckpt_files) > 1:
                raise OSError(f"Multiple checkpoints found at {ckpt_glob}. I don't know which to use.")  # noqa

            pred_dir = os.path.join(
                    mdir, f"predictions/{dataset}/brat/{datasplit}")
            if not os.path.isdir(pred_dir):
                raise OSError(f"No {datasplit} predictions found at {pred_dir}")  # noqa
            preds_glob = os.path.join(pred_dir, "*.ann")
            pred_files = glob(preds_glob)
            if len(pred_files) == 0:
                raise OSError(f"No {datasplit} predictions found at {preds_glob}")  # noqa
        return model_dirs


@register_ensembler("max-voting")
class MaxVotingEnsembler(BratEnsembler):

    def __init__(self, model_dirs, dataset, datasplit="dev", task="Action"):
        super().__init__(model_dirs, dataset, datasplit, task)
        if len(self.model_dirs) <= 2:
            raise ValueError("Need at least 3 models for MaxVotingEnsembler.")

    def choose(self, predictions):
        name, count = Counter(predictions).most_common(1)[0]
        if count == 1:
            warnings.warn("None of the models agree!")
        return name
