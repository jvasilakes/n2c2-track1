import torch


def register(name):
    def assign_name(func):
        func._tagged = name
        return func
    return assign_name


class TokenMask(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden, offset_mapping, entity_spans):
        """
        hidden: [batch_size, seq_len, hidden_dim]. The output of a
                bert or bert-like encoder.
        offset_mapping: [batch_size, seq_len, 2]. Character start, end
                        for each token in the input. Returned by
                        transformers.Tokenizer with return_offsets_mapping=True
        entity_spans: [batch_size, 2]. Character start, end for the target
                      entity mentions in this batch.
        """
        # We need to replace the (0, 0) spans in the token offsets
        # (corresponding to special tokens like <SOS>, <PAD>, etc.
        # with (-1, -1) to avoid masking errors when the
        # start of the span is 0.
        offset_mask = offset_mapping == torch.tensor([0, 0]).type_as(offset_mapping)
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
        token_mask_ = token_mask.unsqueeze(-1).expand(hidden.size())
        if len((token_mask.sum(axis=1) == torch.tensor(0.)).nonzero()) > 0:
            raise ValueError("Entity span not found! Try increasing max_seq_length.")  # noqa
        masked = hidden * token_mask_
        return masked, token_mask_

    def string(self):
        return "MaskLayer()"


class EntityPooler(torch.nn.Module):

    def __init__(self, insize, outsize, pool_fn):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        try:
            self.pooler = self.pooler_functions[pool_fn]
            self.pool_fn = pool_fn
        except KeyError:
            raise ValueError(f"Unknown pool function '{pool_fn}'")
        self.token_mask = TokenMask()
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.insize, self.outsize),
            torch.nn.Tanh())

    def forward(self, hidden, offset_mapping, entity_spans):
        """
        hidden: [batch_size, seq_len, hidden_dim]. The output of a
                bert or bert-like encoder.
        offset_mapping: [batch_size, seq_len, 2]. Character start, end
                        for each token in the input. Returned by
                        transformers.Tokenizer with return_offsets_mapping=True
        entity_spans: [batch_size, 2]. Character start, end for the target
                      entity mentions in this batch.
        """
        masked, token_mask = self.token_mask(
            hidden, offset_mapping, entity_spans)
        pooled = self.pooler(masked, token_mask)
        transformed = self.output_layer(pooled)
        return transformed

    @property
    def pooler_functions(self):
        if "_pooler_registry" in self.__dict__.keys():
            return self._pooler_registry
        else:
            self._pooler_registry = {}
            for name in dir(self):
                var = getattr(self, name)
                if hasattr(var, "_tagged"):
                    registry_name = var._tagged
                    self._pooler_registry[registry_name] = var
            return self._pooler_registry

    @register("max")
    def max_pooler(self, masked, token_mask):
        # Replace masked with -inf to avoid zeroing out
        # hidden dimensions if the non-masked values are all negative.
        masked[torch.logical_not(token_mask)] = -torch.inf
        pooled = torch.max(masked, axis=1)[0]
        return pooled

    @register("mean")
    def mean_pooler(self, masked, token_mask):
        pooled = masked.sum(axis=1) / token_mask.sum(axis=1)
        return pooled

    @register("first")
    def first_pooler(self, masked, token_mask):
        # token_mask is a boolean tensor, so max with give the
        # first occurrence of True.
        first_nonzero_idxs = token_mask.max(axis=1).indices[:, 0]
        batch_idxs = torch.arange(token_mask.size(0))
        pooled = masked[batch_idxs, first_nonzero_idxs, :]
        return pooled

    @register("last")
    def last_pooler(self, masked, token_mask):
        # Get the last True index in the mask by flipping it on the
        # token dimension computing the max index, and then subtracting
        # it from the total length in tokens.
        tmp_mask = token_mask.flip(1)
        last_nonzero_idxs = token_mask.size(1) - 1 - \
            tmp_mask.max(axis=1).indices[:, 0]
        batch_idxs = torch.arange(token_mask.size(0))
        pooled = masked[batch_idxs, last_nonzero_idxs, :]
        return pooled

    @register("first-last")
    def first_last_pooler(self, masked, token_mask):
        batch_idxs = torch.arange(token_mask.size(0))
        first_nonzero_idxs = token_mask.max(axis=1).indices[:, 0]
        tmp_mask = token_mask.flip(1)
        last_nonzero_idxs = token_mask.size(1) - 1 - \
            tmp_mask.max(axis=1).indices[:, 0]
        # [batch_size, 2, hidden_dim]
        # Where the 2 in dim 1 are the first and last token embeddings.
        pooled = torch.stack(
            [masked[batch_idxs, first_nonzero_idxs],
             masked[batch_idxs, last_nonzero_idxs]], dim=1)
        # Effectively concat the 2 vectors at dimension 1
        # [batch_size, 2 * hidden_dim]
        pooled = pooled.reshape(token_mask.size(0), -1)
        return pooled

    def string(self):
        return f"EntityPooler(insize={self.insize}, outsize={self.outsize}, pool_fn={self.pool_fn})"  # noqa