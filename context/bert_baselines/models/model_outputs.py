from typing import Optional, Tuple

import torch
from dataclasses import dataclass
from transformers.file_utils import ModelOutput


@dataclass
class SequenceClassifierOutputWithTokenMask(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask: Optional[torch.FloatTensor] = None
