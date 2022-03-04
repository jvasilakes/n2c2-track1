from .sequence import BertMultiHeadedSequenceClassifier
from .rationale import BertRationaleClassifier

MODEL_LOOKUP = {
        "bert-sequence-classifier": BertMultiHeadedSequenceClassifier,
        "bert-rationale-classifier": BertRationaleClassifier,
        }
