# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from algovision.core import *


class Min(AlgoModule):
    def __init__(self, beta=None):
        super(Min, self).__init__()
        self.beta = beta

    def __call__(self, *tensors: torch.Tensor) -> torch.Tensor:
        tensors = torch.stack(tensors, -1)
        return (torch.nn.Softmin(-1)(self.beta * tensors) * tensors).sum(-1)


class ArgMin(AlgoModule):
    def __init__(self, beta=None):
        super(ArgMin, self).__init__()
        self.beta = beta

    def __call__(self, *tensors: torch.Tensor) -> torch.Tensor:
        tensors = torch.stack(tensors, -1)
        return torch.nn.Softmin(-1)(self.beta * tensors)


class Max(AlgoModule):
    def __init__(self, beta=None):
        super(Max, self).__init__()
        self.beta = beta

    def __call__(self, *tensors: torch.Tensor) -> torch.Tensor:
        tensors = torch.stack(tensors, -1)
        return (torch.nn.Softmax(-1)(self.beta * tensors) * tensors).sum(-1)


class ArgMax(AlgoModule):
    def __init__(self, beta=None):
        super(ArgMax, self).__init__()
        self.beta = beta

    def __call__(self, *tensors: torch.Tensor) -> torch.Tensor:
        tensors = torch.stack(tensors, -1)
        return torch.nn.Softmax(-1)(self.beta * tensors)

