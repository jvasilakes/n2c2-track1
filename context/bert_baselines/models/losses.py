import torch

from sadice import SelfAdjDiceLoss


def get_loss_function(function_name):
    lookup = LossLookup()
    try:
        return lookup[function_name]
    except KeyError:
        raise KeyError(f"Unsupported loss function '{function_name}'")


class LossLookup(object):
    """
    Register lookups for your desired loss functions here.
    """

    def loss_function(name):
        """
        Decorator for registering loss function getters.
        """
        def assign_tags(func):
            func._tagged = name
            return func
        return assign_tags

    @property
    def losses(self):
        if "_loss_registry" in self.__dict__.keys():
            return self._loss_registry
        else:
            self._loss_registry = {}
            for name in dir(self):
                var = getattr(self, name)
                if hasattr(var, "_tagged"):
                    name = var._tagged
                    self._loss_registry[name] = var
            return self._loss_registry

    def __getitem__(self, key):
        return self.losses[key]()

    # === CLASSIFICATION ===
    @staticmethod
    @loss_function("cross-entropy")
    def get_cross_entropy_loss():
        return torch.nn.CrossEntropyLoss

    @staticmethod
    @loss_function("self-adj-dice")
    def get_self_adj_dice_loss():
        """
        Dice Loss for Data-imbalanced NLP Tasks
        https://aclanthology.org/2020.acl-main.45/

        Parameters and their defaults:
          alpha: float = 1.0
          gamma: float = 1.0
        """
        return SelfAdjDiceLoss

    # === STOCHASTIC MASKS ===
    @staticmethod
    @loss_function("ratio")
    def get_ratio_loss():
        return RatioLoss


class RatioLoss(torch.nn.Module):
    """
    Constrain the sum of the token masks per input example to equal self.ratio.
    """
    def __init__(self, ratio=1.0, reduction="mean"):
        super().__init__()
        self.ratio = torch.tensor(ratio)

    def forward(self, _inputs, lengths):
        mask_ratio = _inputs.sum(dim=1) / lengths
        example_losses = torch.abs(mask_ratio - self.ratio)
        if self.reduction == "mean":
            return example_losses.mean()
        else:
            raise ValueError(f"Unsupported reduction '{self.reduction}'")

    def __repr__(self):
        return f"RatioLoss(ratio={self.ratio})"


class ControlledSparsityLoss(torch.nn.Module):
    """
    Stochasitic Mask Loss
    "Interpretable Neural Predictions with Differentiable Binary Variables"
    https://aclanthology.org/P19-1284/

    Default hyperparameters taken from accompanying code at
    https://github.com/bastings/interpretable_predictions/blob/master/latent_rationale/beer/models/latent.py#L17  # noqa
    """
    def __init__(
            self,
            selection_rate=1.0,  # target for L0
            transition_rate=0.0,  # target for fused lasso
            lagrange_alpha=0.5,
            lagrange_lr=0.05,
            lambda_init=0.0015,
            lambda_min=1e-12,
            lambda_max=5.0):
        super().__init__()
        self.selection_rate = selection_rate
        self.transition_rate = transition_rate
        self.lagrange_alpha = lagrange_alpha
        self.lagrange_lr = lagrange_lr
        self.lambda_init = lambda_init
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambdas = {
                "L0": torch.tensor(lambda_init),
                "fused_lasso": torch.tensor(lambda_init)}

    def forward(self, zs, z_dists):
        l0 = self.L0(zs, z_dists)
        constrained_l0 = self.constrain(l0, self.selection_rate, "L0")
        fused = self.fused_lasso(zs, z_dists)
        constrained_fused = self.constrain(
                fused, self.transition_rate, "fused_lasso")
        return constrained_l0 + constrained_fused

    def __repr__(self):
        return "ControlledSparsityLoss()"

    def L0(self, zs, z_dists):
        """
        Penalizes for the number of selected tokens.
        """
        raise NotImplementedError()

    def fused_lasso(self, zs, z_dists):
        """
        Penalizes for the number of transitions.
        """
        raise NotImplementedError()

    def constrain(self, value, target, constraint_name):
        """
        Compute constraint of value to target using lagrange multiplier
        at self.lambdas[constraint_name]
        """
        raise NotImplementedError()
        dissatisfaction = value - target
        # Moving average of the constraint
        c_ma = self.lagrange_alpha * self.c_mas[-1] + (1 - self.lagrange_alpha) * dissatisfaction  # noqa
        self.c_mas.append(c_ma)
        # Original paper does this but I don't know why as it just
        # equals c_ma...
        c0 = dissatisfaction + (c_ma - dissatisfaction)
        # Update the lagrangians
        tmp_lambda = self.lambdas[constraint_name]
        tmp_lambda = tmp_lambda * torch.exp(self.lagrange_lr * c0)
        tmp_lambda = tmp_lambda.clamp(self.lambda_min, self.lambda_max)
        self.lambdas[constraint_name] = tmp_lambda
        constrained_value = tmp_lambda * c0
        return constrained_value
