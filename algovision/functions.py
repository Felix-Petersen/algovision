# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from algovision.core import *


class Min(torch.nn.Module):
    """Soft Minimum operator implemented via :class:`~torch.nn.Softmin`.
    """
    def __init__(self, beta):
        """

        :param beta: inverse temperature beta (required)
        """
        super(Min, self).__init__()
        self.beta = beta
        assert beta is not None, 'Beta is None but it has to be given explicitly for the Min module. (It is not ' \
                                 'possible to infer beta from the Algorithm it is used in.)'

    def __call__(self, *tensors: torch.Tensor) -> torch.Tensor:
        tensors = torch.stack(tensors, -1)
        return (torch.nn.Softmin(-1)(self.beta * tensors) * tensors).sum(-1)


class ArgMin(torch.nn.Module):
    """Soft Arg Minimum operator implemented via :class:`~torch.nn.Softmin`.
    """
    def __init__(self, beta):
        """

        :param beta: inverse temperature beta (required)
        """
        super(ArgMin, self).__init__()
        self.beta = beta
        assert beta is not None, 'Beta is None but it has to be given explicitly for the Min module. (It is not ' \
                                 'possible to infer beta from the Algorithm it is used in.)'

    def __call__(self, *tensors: torch.Tensor) -> torch.Tensor:
        tensors = torch.stack(tensors, -1)
        return torch.nn.Softmin(-1)(self.beta * tensors)


class Max(torch.nn.Module):
    """Soft Maximum operator implemented via :class:`~torch.nn.Softmax`.
    """
    def __init__(self, beta):
        """

        :param beta: inverse temperature beta (required)
        """
        super(Max, self).__init__()
        self.beta = beta
        assert beta is not None, 'Beta is None but it has to be given explicitly for the Min module. (It is not ' \
                                 'possible to infer beta from the Algorithm it is used in.)'

    def __call__(self, *tensors: torch.Tensor) -> torch.Tensor:
        tensors = torch.stack(tensors, -1)
        return (torch.nn.Softmax(-1)(self.beta * tensors) * tensors).sum(-1)


class ArgMax(torch.nn.Module):
    """Soft Arg Maximum operator implemented via :class:`~torch.nn.Softmax`.
    """
    def __init__(self, beta):
        """

        :param beta: inverse temperature beta (required)
        """
        super(ArgMax, self).__init__()
        self.beta = beta
        assert beta is not None, 'Beta is None but it has to be given explicitly for the Min module. (It is not ' \
                                 'possible to infer beta from the Algorithm it is used in.)'

    def __call__(self, *tensors: torch.Tensor) -> torch.Tensor:
        tensors = torch.stack(tensors, -1)
        return torch.nn.Softmax(-1)(self.beta * tensors)

