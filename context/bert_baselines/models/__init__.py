# TODO: do the same register thing here that I did in data
from .sequence import BertMultiHeadedSequenceClassifier
from .rationale import BertRationaleClassifier

MODEL_LOOKUP = {
        "bert-sequence-classifier": BertMultiHeadedSequenceClassifier,
        "bert-rationale-classifier": BertRationaleClassifier,
        }
