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


def get_bubble_sort(beta=5):
    return Algorithm(
        Input('array'),

        Var('a', torch.tensor(0.)),
        Var('b', torch.tensor(0.)),
        Var('swapped', torch.tensor(1.)),
        # Var('loss_prod', torch.tensor(0.)),
        Var('loss_sum', torch.tensor(0.)),
        VarInt('j', 0),
        VarInt('n', lambda array: array.shape[1] - 1),
        While(IsTrue('swapped'),
              Let('swapped', 0),
              For('i', 'n',
                  LetInt('j', lambda i: i + 1),
                  Let('a', 'array', ['i']),
                  Let('b', 'array', ['j']),
                  # Alternative notation for the two three lines above:
                  # Let('a', lambda array, i: array[:, i]),
                  # Let('b', lambda array, i: array[:, i+1]),
                  If(GT('a', 'b'),
                     if_true=[
                         Let('array', [lambda i: i + 1], 'a'),
                         Let('array', ['i'], 'b'),
                         Let('swapped', lambda swapped: 1.),
                         # Let('loss_prod', 1.),
                         Let('loss_sum', lambda loss_sum: loss_sum + 1.),
                     ]
                 ),
              ),
              LetInt('n', lambda n: n-1),
              ),
        Output('array'),
        # Output('loss_prod'),
        Output('loss_sum'),
        beta=beta,
        # debug=True,
    )


if __name__ == '__main__':
    a = get_bubble_sort(5)

    torch.manual_seed(0)
    v = torch.randn(5, 8)
    print(v)
    print(a(v))
