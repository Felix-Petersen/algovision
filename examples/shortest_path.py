from algovision import (
    Algorithm, Input, Output, Var, VarInt,                                          # core
    Eq, NEq, LT, LEq, GT, GEq, CatProbEq, CosineSimilarity, IsTrue, IsFalse,  # conditions
    If, While, For,                                                   # control_structures
    Let, LetInt, Print, Min, ArgMin, Max, ArgMax,                              # functions
)

import torch


INFTY = float('1e12')


def update_d_and_d_predecessor(d, cost, beta):
    unf = torch.nn.functional.unfold(d.unsqueeze(1), kernel_size=(3, 3)).transpose(-2, -1)
    unf = torch.cat([unf[..., :4], unf[..., 5:]], dim=-1)
    unf = unf.view(unf.shape[0], cost.shape[1], cost.shape[2], unf.shape[2])
    unf_sm = torch.nn.functional.softmin(unf * beta, dim=-1)
    new_d = (unf_sm * unf).sum(-1) + cost
    new_d = torch.nn.functional.pad(new_d, (1, 1, 1, 1), "constant", INFTY)
    return new_d, unf_sm


def get_next_location(trace_pos, d_predecessor):
    transport = (trace_pos.unsqueeze(-1) * d_predecessor)
    transport = torch.nn.functional.pad(transport, (0, 0, 1, 1, 1, 1), "constant", 0.)
    location = 0.
    slices = {
        -1: slice(2, None),
        0: slice(1, -1),
        1: slice(0, -2),
    }
    directions = [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ]
    for dir_idx in range(8):
        location = location + transport[:, slices[directions[dir_idx][0]], slices[directions[dir_idx][1]], dir_idx]
    return location


def get_shortest_path(beta, n_iter, t_conorm='probabilistic', p=None, loop_type='for', device='cpu'):

    if t_conorm == 'probabilistic':
        update_tracked_path_via_t_conorm = Let('tracked_path', lambda tracked_path, trace_pos:
            tracked_path + trace_pos - tracked_path * trace_pos)
    elif t_conorm == 'hamacher':
        update_tracked_path_via_t_conorm = Let('tracked_path', lambda tracked_path, trace_pos:
            (tracked_path + trace_pos + (p-2) * tracked_path * trace_pos) / (1 + (p-1) * tracked_path * trace_pos))
    elif t_conorm == 'yager_1':
        update_tracked_path_via_t_conorm = Let('tracked_path', lambda tracked_path, trace_pos:
            1 - torch.relu(1 - tracked_path - trace_pos))
    else:
        raise NotImplementedError(t_conorm)

    if loop_type == 'for':
        loop = lambda *x: For('i_traceback', n_iter, *x)
    elif loop_type == 'while':
        loop = lambda *x: While(LEq(lambda tracked_path: tracked_path[:, 0, 0], 1.), *x)
    else:
        raise NotImplementedError(loop_type)

    shortest_path = Algorithm(
        Input('cost'),

        Var('d',                lambda cost: torch.full((cost.shape[1]+2, cost.shape[2]+2), INFTY).to(device)),
        Var('d_predecessor',    lambda cost: torch.zeros(cost.shape[1], cost.shape[2], 8).to(device)),
        Var('trace_pos',        lambda cost: torch.zeros(cost.shape[1], cost.shape[2]).to(device)),
        Var('tracked_path',     lambda cost: torch.zeros(cost.shape[1], cost.shape[2]).to(device)),

        Let('d', [1, 1], 0.),
        Let('trace_pos', [-1, -1], 1.),

        # Forward
        For('i_iter', n_iter,
            Let(['d', 'd_predecessor'], lambda d, cost: update_d_and_d_predecessor(d, cost, beta)),
            Let('d', [1, 1], 0.),
        ),

        # Traceback
        loop(
            update_tracked_path_via_t_conorm,
            Let('trace_pos', lambda trace_pos, d_predecessor: get_next_location(trace_pos, d_predecessor)),
        ),

        Output('tracked_path'),
        beta=beta,
    )


    return shortest_path


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    torch.manual_seed(0)

    batch_size = 4
    dim_1 = 12
    dim_2 = 10

    beta = 10
    n_iter = 25
    p = None
    # t_conorm = 'probabilistic'
    t_conorm = 'hamacher'
    p = 5
    # t_conorm = 'yager_1'
    loop_type = 'for'
    # loop_type = 'while'

    cost = torch.sigmoid(
        torch.randn(batch_size, dim_1, dim_2) * 4
    ) + .2

    plt.figure(figsize=(8, 3.5))
    plt.suptitle('beta={}, n_iter={}, t_conorm={}, p={}, loop_type={}'.format(beta, n_iter, t_conorm, p, loop_type))

    plt.subplot(121)
    plt.imshow(cost[0].detach())
    plt.colorbar()

    tracked_path = get_shortest_path(beta, n_iter, t_conorm=t_conorm, p=p, loop_type=loop_type)(cost)
    plt.subplot(122)
    plt.imshow(tracked_path[0].detach())
    plt.colorbar()

    plt.show()