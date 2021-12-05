# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from algovision import (
    Algorithm, Input, Output, Variable, Var, VariableInt, VarInt,               # core
    Eq, NEq, LT, LEq, GT, GEq, CatProbEq, CosineSimilarity, IsTrue, IsFalse,    # conditions
    If, While, For,                                                             # control_structures
    Let, LetInt, Print, Min, ArgMin, Max, ArgMax,                               # functions
)
import torch


if __name__ == '__main__':
    a = Algorithm(
        Input('values'),
        Variable('x', torch.tensor(0.)),
        While(
            LT('x', 5),
            Let('x', lambda x: x + 1)
        ),
        Print(lambda x: x),
        Output('x'),
        beta=1.25,
        # debug=True,
    )

    v = torch.randn(3)
    print(v)
    print(a(v))

