# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from algovision.core import *


class If(AlgoModule):
    def __init__(
            self,
            condition,
            if_true=None,
            if_false=None,
            epsilon=None,
            hard=None,
            debug=None,
    ):
        super(If, self).__init__()
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false
        self.epsilon = epsilon
        self.hard = hard
        self.debug = debug

    def __call__(self, state: State) -> State:
        p = self.condition(state)
        if self.debug:
            print(p)
        assert len(p.shape) >= 1, p.shape
        assert p.shape[0] == state.batch_size, (p.shape, state.batch_size)

        state_true = state
        state_false = state.clone()

        if self.if_true is not None and (p > self.epsilon).any():
            if isinstance(self.if_true, AlgoModule):
                state_true = self.if_true(state_true)
            elif isinstance(self.if_true, list):
                for module in self.if_true:
                    state_true = module(state_true)
            else:
                assert False, ('The true case has to be either None, an AlgoModule, or a list of AlgoModules; '
                               'however, it is {}: {}'.format(type(self.if_true), self.if_true))
        if self.if_false is not None and ((1 - p) > self.epsilon).any():
            if isinstance(self.if_false, AlgoModule):
                state_false = self.if_false(state_false)
            elif isinstance(self.if_false, list):
                for module in self.if_false:
                    state_false = module(state_false)
            else:
                assert False, ('The false case has to be either None, an AlgoModule, or a list of AlgoModules; '
                               'however, it is {}: {}'.format(type(self.if_false), self.if_false))

        if not self.hard:
            state_true.merge(state_false, 1 - p)
        else:
            # In case of hard, both cases are still executed as the condition might hold for some elements in the batch.
            state_true.merge(state_false, 1 - (p > .5).float())
        return state_true


class While(AlgoModule):
    def __init__(
            self,
            condition,
            *sequence,
            max_iter=None,
            epsilon=None,
            hard=None,
            debug=None,
    ):
        super(While, self).__init__()
        self.condition = condition
        self.sequence = sequence
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.hard = hard
        self.debug = debug

    def __call__(self, state: State) -> State:
        p_after = self.condition(state)
        p_before = torch.ones_like(p_after)
        i = 0
        accumulate_state = state.clone()
        accumulate_state.reset()

        accumulate_state.add(state, p_before - p_after)
        if self.debug:
            print('Before WHILE', p_before - p_after)
            print('state', state)
            print('accumulate_state', accumulate_state)
            print('-'*80)

        while p_after.max() > self.epsilon and i < self.max_iter:

            for elem in self.sequence:
                state = elem(state)

            p_before = p_after
            p_after = p_after * self.condition(state)

            accumulate_state.add(state, p_before - p_after)
            if self.debug:
                print('Inside WHILE', p_before - p_after, accumulate_state)
                print('state', state)
                print('accumulate_state', accumulate_state)
                print('-'*80)

            i += 1

        accumulate_state.add(state, p_after)
        if self.debug:
            print('After WHILE', p_before - p_after)
            print('state', state)
            print('accumulate_state', accumulate_state)
            print('-'*80)

        return accumulate_state


class For(AlgoModule):
    def __init__(
            self,
            var: str,
            range_or_list: Union[int, list, iter, str, LambdaType],
            *sequence: AlgoModule,
    ):
        super(For, self).__init__()
        self.var = var
        if isinstance(range_or_list, collections.abc.Iterable) and not isinstance(range_or_list, str):
            range_or_list = list(range_or_list)
        self.range_or_list = range_or_list
        self.sequence = sequence

    def __call__(self, state: State) -> State:

        if isinstance(self.range_or_list, int):
            range_or_list = range(self.range_or_list)
        elif isinstance(self.range_or_list, list):
            range_or_list = list(self.range_or_list)
        elif isinstance(self.range_or_list, LambdaType):
            input_args = inspect.getfullargspec(self.range_or_list)[0]
            args = [state[k] for k in input_args]
            range_or_list = self.range_or_list(*args)
            if isinstance(range_or_list, collections.abc.Iterable):
                range_or_list = list(range_or_list)
            assert type(range_or_list) in [int, list], (
                'The return value of the lambda expression has to be one of [int, list, iter] but was type '
                '{}. It was supposed to be used as range_or_list in a For loop'.format(type(range_or_list))
            )
        elif isinstance(self.range_or_list, str):
            range_or_list = state[self.range_or_list]
            assert isinstance(range_or_list, int) or isinstance(range_or_list, list), (
                'The variable {}, which was used for range_or_list should be an int or a list of ints but was neither. '
                'It was {} of type {}.'.format(self.range_or_list, range_or_list, type(range_or_list))
            )
        else:
            assert False, (
                'Invalid type {} for range_or_list. Supported is Union[int, list, iter, str, LambdaType]. '
                '({})'.format(type(self.range_or_list), self.range_or_list)
            )

        if isinstance(range_or_list, int):
            range_or_list = range(range_or_list)

        if self.var in state.state:
            var_existed = True
            assert False, 'The variable used in a For loop ({}) must not exist in the outer scope.'.format(self.var)
        else:
            var_existed = False
            # state.state[self.var] = torch.zeros(state.batch_size, device=state.get_device())
            state.state[self.var] = 0

        for i in range_or_list:

            state[self.var] = i

            for elem in self.sequence:
                state = elem(state)

        if not var_existed:
            del state.state[self.var]

        return state

