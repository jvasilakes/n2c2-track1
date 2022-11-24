# TODO: do the same register thing here that I did in data
from src.models.sequence import BertMultiHeadedSequenceClassifier
from src.models.sequence_attentions import BertSequenceClassifierWithAttentions
from src.models.rationale import BertRationaleClassifier
from src.models.stochastic import BertMultiHeadedStochasticClassifier

MODEL_LOOKUP = {
        "bert-sequence-classifier": BertMultiHeadedSequenceClassifier,
        "bert-sequence-attentions": BertSequenceClassifierWithAttentions,
        "bert-rationale-classifier": BertRationaleClassifier,
        "bert-stochastic-classifier": BertMultiHeadedStochasticClassifier,
        }
