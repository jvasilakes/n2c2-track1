import argparse

import torch
import torch.nn as nn
import torch.distributions as D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", type=float, default=0.)
    parser.add_argument("-t", type=float, default=1.)
    return parser.parse_args()


def main(args):
    """
    For testing.
    """
    x = torch.tensor([[0.9956, 0.9224, 5.5098, 0.6063, 0.8099],
                      [0.9724, 1.5499, -0.4157, 0.1914, -0.6092]],
                     requires_grad=True)
    print("X")
    print(x)
    print()

    print("Softmax")
    soft = torch.softmax(x, dim=1)
    print(soft)
    print(soft.sum(1))
    print(soft.argmax(1))
    print()

    print("Sparsegen-lin")
    sparsegen = SparsegenLin(dim=1, lam=args.l)
    rho = sparsegen(x)
    print(rho)
    print(rho.sum(1))
    print(rho.argmax(1))
    print()

    print("Gumbel Softmax Module")
    gum_fn = GumbelSoftmax(dim=1, tau=args.t)
    gum2 = gum_fn(x)
    print(gum2)
    print(gum2.sum(1))
    print(gum2.argmax(1))


class GumbelSoftmax(nn.Module):
    """
    The Gumbel Softmax function as described in

    Jang, E., Gu, S., & Poole, B. (2017). Categorical Reparameterization with
    Gumbel-Softmax. ArXiv:1611.01144 [Cs, Stat].
    http://arxiv.org/abs/1611.01144
    """

    def __init__(self, dim=None, tau=1.0):
        """
        Args:
            dim (int, optional): The dimension over which to apply
                                 the gumbel_softmax function.
            tau (float): The temperature parameter. Default 1.0.
        """
        super().__init__()
        self.dim = -1 if dim is None else dim
        assert tau > 0.0
        self.tau = tau
        self.softmax = torch.nn.Softmax(dim=self.dim)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): First dimension is batch size.
        Returns:
            torch.Tensor of the same size as inputs.
        """
        mean = torch.zeros_like(inputs)
        scale = torch.ones_like(inputs)
        # torch.distributions.Gumbel has rsample so we can rely
        # on backprop to compute gradients.
        gs = D.Gumbel(mean, scale).rsample()
        return self.softmax((inputs + gs) / self.tau)


class SparsegenLin(nn.Module):
    """
    Generic sparsegen-lin function as described in

    Laha, A., Chemmengath, S. A., Agrawal, P., Khapra, M.,
    Sankaranarayanan, K., & Ramaswamy, H. G. (2018).
    On Controllable Sparse Alternatives to Softmax.
    Advances in Neural Information Processing Systems, 31.
    https://proceedings.neurips.cc/paper/2018/hash/6a4d5952d4c018a1c1af9fa590a10dda-Abstract.html  # noqa

    Implementation modified from
    https://github.com/KrisKorrel/sparsemax-pytorch
    """

    def __init__(self, dim=None, lam=0.0):
        """
        Args:
            dim (int, optional): The dimension over which to apply
                                 the sparsegen function.
            lam (float): The lambda parameter. Default 0.0.
        """
        super().__init__()

        self.dim = -1 if dim is None else dim
        assert lam < 1
        self.lam = lam

    def forward(self, inputs):
        """Forward function.
        Args:
            inputs (torch.Tensor): Input tensor. First dimension is batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        inputs = inputs.transpose(0, self.dim)
        original_size = inputs.size()
        inputs = inputs.reshape(inputs.size(0), -1)
        inputs = inputs.transpose(0, 1)
        dim = 1

        number_of_logits = inputs.size(dim)

        # Translate inputs by max for numerical stability
        inputs = inputs - torch.max(inputs, dim=dim, keepdim=True)[0].expand_as(inputs)  # noqa

        # Sort inputs in descending order.
        # (NOTE: Can be replaced with linear time selection method:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html
        zs = torch.sort(input=inputs, dim=dim, descending=True)[0]
        ks = torch.arange(start=1, end=number_of_logits + 1, step=1,
                         dtype=inputs.dtype).view(1, -1)
        ks = ks.expand_as(zs).to(zs.device)

        # Determine sparsity of projection
        bound = 1 - self.lam + ks * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(inputs.type())
        k = torch.max(is_gt * ks, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1 + self.lam) / k
        taus = taus.expand_as(inputs)

        # Sparsemax
        ps = (inputs - taus) / (1 - self.lam)
        self.output = torch.max(torch.zeros_like(inputs), ps)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        grad_sum = torch.sum(grad_output * nonzeros, dim=dim)
        grad_sum /= torch.sum(nonzeros, dim=dim)
        self.grad_inputs = nonzeros * (grad_output - grad_sum.expand_as(grad_output))  # noqa

        return self.grad_inputs


if __name__ == "__main__":
    main(parse_args())
