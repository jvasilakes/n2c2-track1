import warnings

import torch
import torch.nn as nn
import probabll.distributions as probd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from probabll.distributions.stretchrectify import StretchedAndRectifiedDistribution  # noqa


def register(name):
    def assign_name(func):
        func._tagged = name
        return func
    return assign_name


class TokenEmbeddingPooler(nn.Module):

    def __init__(self, insize, outsize, pool_fn):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        try:
            self.pooler = self.pooler_functions[pool_fn]
            self.pool_fn = pool_fn
        except KeyError:
            raise ValueError(f"Unknown pool function '{pool_fn}'")
        self.output_layer = nn.Sequential(
            nn.Linear(self.insize, self.outsize),
            nn.Tanh())

    def forward(self, hidden, token_idxs):
        """
        hidden: [batch_size, max_seq_length, hidden_dim]
        token_idxs: list of lists containing token_idxs to pool.
        """
        assert len(token_idxs) == hidden.size(0), "Number of token_idxs not equal to batch size!"  # noqa
        # Get a token-wise mask over hidden,
        token_mask = self.get_token_mask_from_indices(
                token_idxs, hidden.size())
        token_mask = token_mask.to(hidden.device)
        # apply the mask to keep only the specified tokens,
        masked_hidden = hidden * token_mask
        # and pool the embeddings.
        pooled = self.pooler(masked_hidden, token_mask)
        transformed = self.output_layer(pooled)
        return transformed

    def get_token_mask_from_indices(self, token_idxs, hidden_size):
        # Use the token_idxs to create a mask over the token dimension
        # in hidden, duplicated across the embedding dimension.
        token_mask = torch.zeros(hidden_size, dtype=torch.long)
        for (batch_i, idxs) in enumerate(token_idxs):
            token_mask[batch_i, idxs, :] = 1
        return token_mask

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
        return f"TokenEmbeddingPooler(insize={self.insize}, outsize={self.outsize}, pool_fn={self.pool_fn})"  # noqa


class KumaGate(nn.Module):
    """
    HardKumaraswamy gate layer based on
    https://github.com/bastings/interpretable_predictions/blob/master/latent_rationale/nn/kuma_gate.py  # noqa
    """
    def __init__(self, insize, outsize=1):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        self.anet = nn.Sequential(
                nn.Linear(self.insize, self.outsize),
                nn.Softplus()
                )
        self.bnet = nn.Sequential(
                nn.Linear(self.insize, self.outsize),
                nn.Softplus()
                )

    def forward(self, hidden):
        a = self.anet(hidden)
        b = self.bnet(hidden)
        kuma = probd.Kumaraswamy(a, b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Ignore torch.distributions warning about arg_constraints
            kuma = StretchedAndRectifiedDistribution(
                    kuma, lower=-0.1, upper=1.1)
        return kuma


class KumaMask(nn.Module):
    """
    Stochastic binary mask over tokens.
    """
    def __init__(self, insize):
        super().__init__()
        self.insize = insize
        self.gate = KumaGate(self.insize, 1)

    def forward(self, x):
        z = []
        z_dists = []
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            z_t_dist = self.gate(x_t)
            z_t = z_t_dist.rsample()
            z.append(z_t)
            z_dists.append(z_t_dist)
        z = torch.stack(z, dim=1)
        return z, z_dists


class RecurrentEncoder(nn.Module):

    def __init__(self, insize, hidden_size, cell="rnn"):
        super().__init__()
        self.insize = insize
        self.hidden_size = hidden_size
        try:
            self.recurrent = self.recurrent_layers[cell.lower()]()
            self.cell = cell.lower()
        except KeyError:
            raise ValueError(f"Unsupported cell type '{cell}'")

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        packed = pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.recurrent(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        final = torch.cat([direc for direc in hidden], dim=1)
        return outputs, final

    @property
    def recurrent_layers(self):
        if "_layer_registry" in self.__dict__.keys():
            return self._layer_registry
        else:
            self._layer_registry = {}
            for name in dir(self):
                var = getattr(self, name)
                if hasattr(var, "_tagged"):
                    registry_name = var._tagged
                    self._layer_registry[registry_name] = var
            return self._layer_registry

    @register("rnn")
    def rnn(self):
        return nn.RNN(
                input_size=self.insize,
                hidden_size=self.hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True)

    @register("lstm")
    def lstm(self):
        return nn.LSTM(
                input_size=self.insize,
                hidden_size=self.hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True)

    @register("gru")
    def gru(self):
        return nn.GRU(
                input_size=self.insize,
                hidden_size=self.hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True)
