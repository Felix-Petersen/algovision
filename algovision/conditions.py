# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from algovision.core import *


class Eq(Condition):
    def __init__(self, left, right, **kwargs):
        super(Eq, self).__init__(left=left, right=right, **kwargs)

    def __call__(self, state: State) -> torch.Tensor:
        difference = self.get_left(state) - self.get_right(state)
        difference = difference * self.beta / 2.
        return 1. / torch.cosh(difference)**2


class NEq(Condition):
    def __init__(self, left, right, **kwargs):
        super(NEq, self).__init__(left=left, right=right, **kwargs)

    def __call__(self, state: State) -> torch.Tensor:
        difference = self.get_left(state) - self.get_right(state)
        difference = difference * self.beta / 2.
        return 1 - (1. / torch.cosh(difference)**2)


class LT(Condition):
    def __init__(self, left, right, **kwargs):
        super(LT, self).__init__(left=left, right=right, **kwargs)

    def __call__(self, state: State) -> torch.Tensor:
        difference = self.get_left(state) - self.get_right(state)
        difference = difference * self.beta
        return torch.sigmoid(- difference)


LEq = LT


class GT(Condition):
    def __init__(self, left, right, **kwargs):
        super(GT, self).__init__(left=left, right=right, **kwargs)

    def __call__(self, state: State) -> torch.Tensor:
        difference = self.get_left(state) - self.get_right(state)
        difference = difference * self.beta
        return torch.sigmoid(difference)


GEq = GT


class CatProbEq(Condition):
    def __init__(self, left, right, **kwargs):
        super(CatProbEq, self).__init__(left=left, right=right, **kwargs)

    def __call__(self, state: State) -> torch.Tensor:
        similarity = torch.nn.CosineSimilarity(-1)(self.get_left(state), self.get_right(state))
        difference = 1. - similarity
        difference = difference * self.beta
        return 1. / torch.cosh(difference)


class CosineSimilarity(Condition):
    def __init__(self, left, right, **kwargs):
        super(CosineSimilarity, self).__init__(left=left, right=right, **kwargs)

    def __call__(self, state: State) -> torch.Tensor:
        similarity = torch.nn.CosineSimilarity(-1)(self.get_left(state), self.get_right(state))
        return similarity


class IsTrue(Condition):
    def __init__(self, var):
        super(IsTrue, self).__init__(left=var, right=None)

    def __call__(self, state: State) -> torch.Tensor:
        value = self.get_left(state)
        assert value.min() >= 0 and value.max() <= 1, 'The probability cannot be outside the range [0, 1].'
        return value


class IsFalse(Condition):
    def __init__(self, var):
        super(IsFalse, self).__init__(left=var, right=None)

    def __call__(self, state: State) -> torch.Tensor:
        value = self.get_left(state)
        assert value.min() >= 0 and value.max() <= 1, 'The probability cannot be outside the range [0, 1].'
        return 1. - value

