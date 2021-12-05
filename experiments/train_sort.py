# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from tqdm import tqdm
import random
import torch
from datasets.dataset import MultiDigitSplits
import models
import utils
from algovision import (
    Algorithm, Input, Output, Variable, Var, VariableInt, VarInt,               # core
    Eq, NEq, LT, LEq, GT, GEq, CatProbEq, CosineSimilarity, IsTrue, IsFalse,    # conditions
    If, While, For,                                                             # control_structures
    Let, LetInt, Print, Min, ArgMin, Max, ArgMax,                               # functions
)


def ranking_accuracy(data, targets):
    scores = model(data).squeeze(2)

    acc = torch.argsort(targets, dim=-1) == torch.argsort(scores, dim=-1)

    acc_em = acc.all(-1).float().mean()
    acc_ew = acc.float().mean()

    # EM5:
    scores = scores[:, :5]
    targets = targets[:, :5]
    acc = torch.argsort(targets, dim=-1) == torch.argsort(scores, dim=-1)
    acc_em5 = acc.all(-1).float().mean()

    return dict(
        acc_em=acc_em.type(torch.float32).mean().item(),
        acc_ew=acc_ew.type(torch.float32).mean().item(),
        acc_em5=acc_em5.type(torch.float32).mean().item(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST sorting benchmark')
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-n', '--num_compare', type=int, default=5)
    parser.add_argument('-i', '--num_steps', type=int, default=1_000_000, help='number of training steps')
    parser.add_argument('-e', '--eval_freq', type=int, default=20, help='the evaluation frequency')
    # parser.add_argument('-e', '--eval_freq', type=int, default=2_500, help='the evaluation frequency')
    parser.add_argument('-m', '--method', type=str, default='loss_sum', choices=['loss_sum', 'loss_prod'])
    parser.add_argument('--beta', type=float, default=8)
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'svhn'])
    parser.add_argument('-l', '--nloglr', type=float, default=3.5, help='Negative log learning rate')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    best_valid_acc = 0.

    # ---

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('----------------------------------------------------')
        print('--- WARNING: No GPU detected, running on CPU ... ---')
        print('----------------------------------------------------')
        args.device = 'cpu'

    splits = MultiDigitSplits(dataset=args.dataset, num_compare=args.num_compare, seed=args.seed)

    # drop_last needs to be true, otherwise error with testing for SVHN
    data_loader_train = splits.get_train_loader(
        args.batch_size, drop_last=True
    )
    data_loader_valid = splits.get_valid_loader(
        args.batch_size, drop_last=True
    )
    data_loader_test = splits.get_test_loader(
        args.batch_size, drop_last=True
    )

    if args.dataset == 'mnist':
        model = models.MultiDigitMNISTNet().to(args.device)
    elif args.dataset == 'svhn':
        model = models.SVHNConvNet().to(args.device)
    else:
        raise ValueError(args.dataset)

    optim = torch.optim.Adam(model.parameters(), lr=10**(-args.nloglr))

    bubble_sort = Algorithm(
        Input('array'),

        Var('a', torch.tensor(0.).to(args.device)),
        Var('b', torch.tensor(0.).to(args.device)),
        Var('swapped', torch.tensor(1.).to(args.device)),
        Var('loss_prod', torch.tensor(0.).to(args.device)),
        Var('loss_sum', torch.tensor(0.).to(args.device)),
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
                         Let('loss_prod', 1.),
                         Let('loss_sum', lambda loss_sum: loss_sum + 1.),
                     ]
                     ),
                  ),
              LetInt('n', lambda n: n - 1),
              ),
        Output('array'),
        Output('loss_prod'),
        Output('loss_sum'),
        beta=args.beta,
    )

    valid_accs = []
    test_acc = None

    for iter_idx, (data, targets) in tqdm(
        enumerate(utils.load_n(data_loader_train, args.num_steps)),
        desc="Training steps",
        total=args.num_steps,
    ):
        data = data.to(args.device)
        targets = targets.to(args.device)

        data_sorted = data[torch.arange(args.batch_size).to(args.device).unsqueeze(1), torch.argsort(targets, dim=-1)]

        outputs = model(data_sorted).squeeze(2)
        _, loss_prod, loss_sum = bubble_sort(outputs)

        if args.method == 'loss_sum':
            loss = loss_sum.mean()
        elif args.method == 'loss_prod':
            loss = loss_prod.mean()
        else:
            assert False, args.method

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (iter_idx + 1) % args.eval_freq == 0:

            current_valid_accs = []
            for data, targets in data_loader_valid:
                data, targets = data.to(args.device), targets.to(args.device)
                current_valid_accs.append(ranking_accuracy(data, targets))
            valid_accs.append(utils.avg_list_of_dicts(current_valid_accs))

            print(iter_idx, 'valid', valid_accs[-1])

            if valid_accs[-1]['acc_em5'] > best_valid_acc:
                best_valid_acc = valid_accs[-1]['acc_em5']

                current_test_accs = []
                for data, targets in data_loader_test:
                    data, targets = data.to(args.device), targets.to(args.device)
                    current_test_accs.append(ranking_accuracy(data, targets))
                test_acc = utils.avg_list_of_dicts(current_test_accs)

                print(iter_idx, 'test', test_acc)

    print('final test', test_acc)
