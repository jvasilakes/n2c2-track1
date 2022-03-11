import torch

from sadice import SelfAdjDiceLoss


def get_loss_function(function_name):
    lookup = LossLookup()
    try:
        return lookup[function_name]
    except KeyError:
        raise KeyError(f"Unsupported loss function '{function_name}'")


def get_supported_loss_functions():
    lookup = LossLookup()
    fns = lookup.losses.items()
    # Sort by loss type
    fns_by_type = {}
    for (name, fn) in fns:
        loss_type = fn._tagged[0]
        if loss_type not in fns_by_type:
            fns_by_type[loss_type] = []
        fns_by_type[loss_type].append((name, fn()))
    return fns_by_type


class LossLookup(object):
    """
    Register lookups for your desired loss functions here.
    """

    def loss_function(loss_type, name):
        """
        Decorator for registering loss function getters.
        """
        def assign_tags(func):
            func._tagged = (loss_type, name)
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
                    loss_type, name = var._tagged
                    self._loss_registry[name] = var
            return self._loss_registry

    def __getitem__(self, key):
        return self.losses[key]()

    # === CLASSIFICATION ===
    @staticmethod
    @loss_function("classification", "cross-entropy")
    def get_cross_entropy_loss():
        return torch.nn.CrossEntropyLoss

    @staticmethod
    @loss_function("classification", "self-adj-dice")
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
    @loss_function("mask", "ratio")
    def get_ratio_loss():
        return RatioLoss

    @staticmethod
    @loss_function("mask", "controlled-sparsity")
    def get_controlled_sparsity_loss():
        return ControlledSparsityLoss


class RatioLoss(torch.nn.Module):
    """
    Constrain the sum of the token masks per input example to equal self.ratio.
    """
    def __init__(self, ratio=1.0, reduction="mean"):
        super().__init__()
        self.ratio = torch.tensor(ratio)

    def forward(self, zs, z_dists, lengths):
        mask_ratio = zs.sum(dim=1) / lengths
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

    Default hyperparameters and compute code taken from accompanying code at
    https://github.com/bastings/interpretable_predictions/blob/master/latent_rationale/beer/models/latent.py#L17  # noqa
    
    Constrained optimization code based on Algorithm 1 of
    "Taming VAEs"
    https://arxiv.org/abs/1810.00597
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
                "lasso": torch.tensor(lambda_init)}
        self.c_mas = [torch.tensor(0.)]

    def forward(self, zs, z_dists, lengths):
        token_mask = torch.ones(lengths.size(0), lengths.max())
        for i in range(token_mask.size(0)):
            token_mask[i, lengths[i]:] = torch.tensor(0.)

        l0 = self.L0(z_dists, token_mask)
        constrained_l0 = self.constrain(l0, self.selection_rate, "L0")
        lasso = self.lasso(z_dists, token_mask)
        constrained_lasso = self.constrain(
                lasso, self.transition_rate, "lasso")
        return constrained_l0 + constrained_lasso

    def __repr__(self):
        return "ControlledSparsityLoss()"

    def L0(self, z_dists, token_mask):
        """
        Penalizes for the number of selected tokens.
        """
        pdf0 = []
        for z_dist in z_dists:
            # TODO: is log_prob.exp() stable?
            p0 = z_dist.log_prob(torch.tensor(0.)).exp()
            pdf0.append(p0)
        pdf0 = torch.stack(pdf0, dim=1).squeeze(-1)
        pdf_nonzero = 1. - pdf0
        pdf_nonzero *= token_mask
        l0 = pdf_nonzero.sum(dim=1) / token_mask.sum(dim=1)
        return l0.mean()

    def lasso(self, z_dists, token_mask):
        """
        Penalizes for the number of transitions.
        """
        pdf0 = []
        for z_dist in z_dists:
            logp0 = z_dist.log_prob(torch.tensor(0.)).exp()
            pdf0.append(logp0)
        pdf0 = torch.stack(pdf0, dim=1).squeeze(-1)
        p_zi_0 = pdf0[:, :-1]  # P(z_i = 0)
        p_zi1_0 = pdf0[:, 1:]  # P(z_i+1 = 0)
        p_zi_nonzero = 1. - p_zi_0  # 1 - P(z_i = 0)
        p_zi1_nonzero = 1. - p_zi1_0  # 1 - P(z_i+1 = 0)
        lasso = (p_zi_0 * p_zi1_nonzero) + (p_zi_nonzero * p_zi1_0)
        lasso *= token_mask[:, :-1]
        lasso = lasso.sum(dim=1) / token_mask.sum(dim=1)
        return lasso.mean()

    def constrain(self, value, target, constraint_name):
        """
        Compute constraint of value to target using lagrange multiplier
        at self.lambdas[constraint_name]
        """
        # ct_hat, using the notation from the paper, is the dissatisfaction
        # of the constraint.
        ct_hat = value - target
        # Compute the moving average of the constraint
        ct_ma = self.lagrange_alpha * self.c_mas[-1] + \
                (1 - self.lagrange_alpha) * ct_hat  # noqa
        self.c_mas.append(ct_ma)
        # Don't backprop the difference from the moving average
        ct = ct_hat + (ct_ma.detach() - ct_hat.detach())
        # Update the lagrangians
        new_lambda = self.lambdas[constraint_name]
        new_lambda = new_lambda * torch.exp(self.lagrange_lr * ct.detach())
        new_lambda = new_lambda.clamp(self.lambda_min, self.lambda_max)
        self.lambdas[constraint_name] = new_lambda
        constrained_value = new_lambda.detach() * ct
        return constrained_value


if __name__ == "__main__":
    fns_by_type = get_supported_loss_functions()
    for (loss_type, fn_names_and_classes) in fns_by_type.items():
        print(loss_type.title())
        for (name, cls) in fn_names_and_classes:
            print(f"  {name}: {cls}")
