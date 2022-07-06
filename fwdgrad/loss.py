from typing import Callable, Tuple

import torch
from torch.nn import functional as F


def xent(
    model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy loss. Given a pytorch model, it computes the cross-entropy loss.

    Args:
        model (torch.nn.Module): PyTorch model.
        x (torch.Tensor): Input tensor for the PyTorch model.
        t (torch.Tensor): Targets.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = model(x)
    return F.cross_entropy(y, t)


def functional_xent(
    params: Tuple[torch.nn.Parameter, ...],
    model: Callable[[Tuple[torch.nn.Parameter, ...], torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Functional cross-entropy loss. Given a functional version of a pytorch model, which can be obtained with
    `fmodel, params = functorch.make_functional(model)`, it computes the cross-entropy loss.

    Args:
        params (Tuple[torch.nn.Parameter, ...]): Model parameters obtained by `fmodel, params = fc.make_functional(model)`.
        model (Callable[[Tuple[torch.nn.Parameter, ...], torch.Tensor], torch.Tensor]): Functional version of a pytorch model,
            obtained by fmodel, `params = fc.make_functional(model)`
        x (torch.Tensor): Input tensor for the PyTorch model.
        t (torch.Tensor): Targets.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = model(params, x)
    return F.cross_entropy(y, t)
