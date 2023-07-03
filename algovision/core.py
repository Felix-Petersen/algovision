# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Union, Tuple, List
from types import LambdaType
from abc import abstractmethod
import pprint
import copy
import inspect
import collections.abc
import re


VARIABLE_REGEX = r"^[_a-zA-Z][_a-zA-Z0-9]*$"


class Input(object):
    def __init__(self, name, shape=None, dtype=None):
        self.name = name
        assert re.match(VARIABLE_REGEX, name), 'Name {} is invalid, names have to match the following regex: {}'.format(
            name, VARIABLE_REGEX
        )
        self.shape = shape
        self.dtype = dtype

    def checks(self, x):
        if self.shape is not None:
            assert x.shape[1:] == self.shape, 'Shape of input {} does not match predefined shape {}. Note that the ' \
                                              'input is expected to have an additional batch dimension.' \
                                              ''.format(x.shape, self.shape)
        if self.dtype is not None:
            assert x.dtype == self.dtype, 'Data type of input {} does not match predefined {}.' \
                                          ''.format(x.dtype, self.dtype)


class Output(object):
    def __init__(self, name, shape=None, dtype=None):
        self.name = name
        assert re.match(VARIABLE_REGEX, name), 'Name {} is invalid, names have to match the following regex: {}'.format(
            name, VARIABLE_REGEX
        )
        self.shape = shape
        self.dtype = dtype

    def checks(self, x):
        if self.shape is not None:
            assert x.shape[1:] == self.shape, 'Shape of output {} does not match predefined shape {}. Note that the ' \
                                              'input is expected to have an additional batch dimension.' \
                                              ''.format(x.shape, self.shape)
        if self.dtype is not None:
            assert x.dtype == self.dtype, 'Data type of output {} does not match predefined {}.' \
                                          ''.format(x.dtype, self.dtype)


class Variable(object):
    def __init__(self, name, initial_value):
        self.name = name
        assert re.match(VARIABLE_REGEX, name), 'Name {} is invalid, names have to match the following regex: {}'.format(
            name, VARIABLE_REGEX
        )
        self.initial_value = initial_value

    def get_value(self):
        if isinstance(self.initial_value, LambdaType):
            return self.initial_value
        else:
            return self.initial_value.clone().detach()


class VariableInt(object):
    def __init__(self, name, initial_value):
        self.name = name
        assert re.match(VARIABLE_REGEX, name), 'Name {} is invalid, names have to match the following regex: {}'.format(
            name, VARIABLE_REGEX
        )
        self.initial_value = initial_value
        self.checks_and_cast()

    def checks_and_cast(self):
        if isinstance(self.initial_value, int):
            pass
        elif isinstance(self.initial_value, list):
            for val in self.initial_value:
                assert isinstance(val, int), (
                    'Integer variable is a hard variable and only supports `int`, `List[int]`, and Iter[int]. '
                    'For variable {} inserted {} of type {}. The problematic element is {}.'
                    ''.format(self.name, self.initial_value, type(self.initial_value), val)
                )
        elif isinstance(self.initial_value, collections.abc.Iterable):
            self.initial_value = list(self.initial_value)
            for val in self.initial_value:
                assert isinstance(val, int), (
                    'Integer variable is a hard variable and only supports `int`, `List[int]`, and Iter[int]. '
                    'For variable {} inserted {} of type {}. The problematic element is {}.'
                    ''.format(self.name, self.initial_value, type(self.initial_value), val)
                )
        elif isinstance(self.initial_value, LambdaType):
            pass
        else:
            assert False, (
                'Integer variable is a hard variable and only supports `int`, `List[int]`, `LambdaType`, '
                'and `Iter[int]`. '
                'For variable {} inserted {} of type {}.'
                ''.format(self.name, self.initial_value, type(self.initial_value))
            )

    def get_value(self):
        if isinstance(self.initial_value, int) or isinstance(self.initial_value, LambdaType):
            return self.initial_value
        else:
            return list(self.initial_value)


Var = Variable
VarInt = VariableInt


class State(object):
    def __init__(self, input_names, input_values, variables, variable_ints, batch_size):
        assert len(input_names) == len(input_values), 'The number of actual inputs {} does not match the ' \
                                                      'predefined number of expected inputs {}.'.format(
            len(input_values), len(input_names)
        )
        self.state = {}
        self.batch_size = batch_size

        for name, value in zip(input_names, input_values):
            assert isinstance(name, Input), name
            assert name.name not in self.state, 'Variable / Input with name `{}` is defined for the second time, ' \
                                                'which is not supported. The following variables are already ' \
                                                'defined: {}'.format(name.name, self.state.keys())
            name.checks(value)
            assert value.shape[0] == batch_size, (
                'The 0th dimension of Input `{}` is supposed to be the batch dimension (which was inferred from the '
                'first input to be of size {}); however, the shape is {}.'.format(name.name, batch_size, value.shape)
            )
            self.state[name.name] = value

        for variable in variables:
            assert isinstance(variable, Variable), variable
            assert variable.name not in self.state, (
                'Variable / Input with name `{}` is defined for the second time, '
                'which is not supported. The following variables are already '
                'defined: {}'.format(variable.name, self.state.keys()))

            self.state[variable.name] = variable.get_value()

            if isinstance(self.state[variable.name], LambdaType):
                input_args = inspect.getfullargspec(self.state[variable.name])[0]
                args = [self.state[k] for k in input_args]
                self.state[variable.name] = self.state[variable.name](*args)
                assert isinstance(self.state[variable.name], torch.Tensor), (
                    'The return value of the lambda expression has to be torch.Tensor but was type '
                    '{}. It was supposed to be written to variable {}.'.format(
                        type(self.state[variable.name]), variable.name)
                )

            self.state[variable.name] = self.state[variable.name].unsqueeze(0).repeat(
                batch_size,
                *[1]*len(self.state[variable.name].shape)
            )

        for variable_int in variable_ints:
            assert isinstance(variable_int, VariableInt), variable_int
            assert variable_int.name not in self.state, (
                'Variable / Input with name `{}` is defined for the second time, '
                'which is not supported. The following variables are already '
                'defined: {}'.format(variable_int.name, self.state.keys()))
            self.state[variable_int.name] = variable_int.get_value()

            if isinstance(self.state[variable_int.name], LambdaType):
                input_args = inspect.getfullargspec(self.state[variable_int.name])[0]
                args = [self.state[k] for k in input_args]
                self.state[variable_int.name] = self.state[variable_int.name](*args)
                assert type(self.state[variable_int.name]) in [int, list], (
                    'The return value of the lambda expression has to be one of [int, list] but was type '
                    '{}. It was supposed to be written to variable {}.'.format(
                        type(self.state[variable_int.name]), variable_int.name)
                )

    def return_outputs(self, outputs: List[Output]) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        return_values = []

        for output in outputs:
            assert output.name in self.state, 'Output with name `{}` is not defined. ' \
                                              'The following variables are already ' \
                                              'defined: {}'.format(output.name, self.state.keys())

            value = self.state[output.name]
            output.checks(value)
            return_values.append(value)

        if len(return_values) == 0:
            assert False, 'There has to be at least one return value.'
        elif len(return_values) == 1:
            return return_values[0]
        else:
            return tuple(return_values)

    def merge(self, to_merge, p):
        """(Internal)  Merges two states with where the new tensor is used to the extent of :math:`p` .

        For any tensors :math:`t_1, t_2` (`self`, `to_merge`) and probability :math:`p` , the new tensor and probability
        :math:`t^\prime` is defined as

        .. math:: t^\\prime = t_1 \\cdot (p-1) + t_2 \\cdot p
        """
        assert isinstance(to_merge, State), to_merge
        assert self.state.keys() == to_merge.state.keys(), (self.state.keys(), to_merge.state.keys())
        assert p.shape[0] == self.batch_size, (p.shape, self.batch_size)

        for key, value in to_merge.state.items():
            if isinstance(value, int) or (isinstance(value, list) and all([isinstance(v, int) for v in value])):
                assert value == self.state[key], 'You have probabilistically modified a hard Int discrete value ' \
                                                 '({}). {} {}'.format(key, value, self.state[key])
                continue
            if value.dtype == torch.long:
                assert (value == self.state[key]).all(), 'You have probabilistically modified a discrete value ' \
                                                            '(LongTensor) ({}).'.format(key)
                continue

            assert len(p.shape) <= len(value.shape), (
                'Error because the probability (which is produced by '
                'the Condition) is higher dimensional than the actual '
                'values to be interpolated. This is most likely because '
                'of too high dimensional inputs to the condition. '
                'Usually, the inputs to the condition should have a '
                'shape of (B, ) where B is the batch dimension, i.e., '
                'specifically not something like (B, 1). The shape of '
                'p is {} and the shape of value is {}.'
                ''.format(p.shape, value.shape)
            )

            p_0 = 1 - p
            p_1 = p
            while len(p_0.shape) < len(value.shape):
                p_0 = p_0.unsqueeze(-1)
                p_1 = p_1.unsqueeze(-1)

            assert self.state[key].shape == value.shape, (self.state[key].shape, value.shape)
            # If it crashes here, that is most likely because one of the elements in state does not have its batch
            # dimension:
            # print('merge', p_0.shape, self.state[key].shape, value.shape)
            self.state[key] = (self.state[key] * p_0 + value * p_1)

    def probabilistic_update(self, key, value, p):
        assert p.shape[0] == self.batch_size, (p.shape, self.batch_size)
        assert len(p.shape) == len(self.state[key].shape), (p.shape, self.state[key].shape)

        if not (isinstance(value, int) or isinstance(value, float)):
            while len(value.shape) < len(self.state[key].shape):
                value = value.unsqueeze(-1)

        # print('probabilistic_update', p.shape, self.state[key].shape, value.shape)
        self.state[key] = p * value + (1-p) * self.state[key]

    def reset(self):
        """(Internal)  Resets all tensors to zero.
        """
        for key in self.state.keys():
            if isinstance(self.state[key], torch.Tensor):
                self.state[key] = torch.zeros_like(self.state[key])
            else:
                assert isinstance(self.state[key], int) or isinstance(self.state[key], LambdaType) or \
                       (isinstance(self.state[key], list) and all([isinstance(v, int) for v in self.state[key]])), (
                    'Illegal data type {} found for variable {}.'.format(type(self.state[key]), key)
                )

    def add(self, to_add, p):
        """(Internal)  Merges two states by adding `to_add` weighted by `p`. Except of VariableInt types; there, the
        new value is used.
        """
        for key, value in to_add.state.items():
            p_0 = p
            if isinstance(value, int) or isinstance(value, LambdaType) or \
                    (isinstance(value, list) and all([isinstance(v, int) for v in value])):
                self.state[key] = value
            else:
                assert self.state[key].shape == value.shape, (self.state[key].shape, value.shape)

                assert len(p.shape) <= len(value.shape), (
                    'Error because the probability (which is produced by '
                    'the Condition) is higher dimensional than the actual '
                    'values to be interpolated. This is most likely because '
                    'of too high dimensional inputs to the condition. '
                    'Usually, the inputs to the condition should have a '
                    'shape of (B, ) where B is the batch dimension, i.e., '
                    'specifically not something like (B, 1). The shape of '
                    'p is {} and the shape of value is {}.'
                    ''.format(p.shape, value.shape)
                )

                while len(p_0.shape) < len(value.shape):
                    p_0 = p_0.unsqueeze(-1)
                # print('add', p_0.shape, self.state[key].shape, value.shape)
                self.state[key] = self.state[key] + p_0 * value

    def clone(self):
        """Duplicates a :class:`~State` .

        Does not duplicate the internal objects, i.e., ``copy.copy()`` instead of ``copy.deepcopy()``.
        """
        clone = copy.copy(self)
        clone.state = copy.copy(self.state)

        return clone

    def __str__(self):
        d = self.clone().__dict__
        return pprint.pformat(d, indent=4)

    def to(self, device):
        for obj_key in self.state:
            if isinstance(self.state[obj_key], torch.Tensor):
                self.state[obj_key] = self.state[obj_key].to(device)
            else:
                assert False, (obj_key, self.state[obj_key])

    def __setitem__(self, key, item):
        assert key in self.state, (
            'The variable {} is not declared but you attempted to write to it.'.format(key)
        )

        self.state[key] = item

    def __getitem__(self, key):
        assert key in self.state, (
            'The variable {} does not exist but you attempted to access it.'.format(key)
        )

        return self.state[key]

    def update(self, new_values: dict):
        # """(Internal)  Overrides the state with values from a ``dict``."""
        for key, value in new_values.items():
            if isinstance(self.state[key], torch.Tensor):
                if isinstance(value, torch.Tensor):
                    assert value.shape == self.state[key].shape, (
                        'A variable ({}) is being updated but the new shape ({}) does not match the original shape '
                        '({}), which is not legal. '
                        'This might be because the shape of a tensor is (B, 1) or something similar, i.e., where an '
                        'unnecessary dimension is in the end.'.format(key, value.shape, self.state[key].shape)
                    )
                self.state[key] = value * torch.ones_like(self.state[key])
            else:
                assert isinstance(value, type(self.state[key])), (
                    value, type(value), self.state[key], type(self.state[key])
                )
                self.state[key] = value

    def get_device(self):
        for key, value in self.state.items():
            return value.device


class AlgoModule(object):
    def __init__(self):
        self.beta = None

    @abstractmethod
    def __call__(self, state: State) -> State:
        pass

    def set_hyperparameters(
            self,
            beta,
            max_iter,
            epsilon,
            hard,
            debug,
    ):
        kwargs = dict(
            beta=beta,
            max_iter=max_iter,
            epsilon=epsilon,
            hard=hard,
            debug=debug,
        )
        for key, val in kwargs.items():
            if hasattr(self, key):
                if getattr(self, key) is None:
                    setattr(self, key, val)

        for attr_name in dir(self):
            if isinstance(getattr(self, attr_name), AlgoModule):
                if debug:
                    print('Setting hyperparameters for {}:'.format(attr_name))
                getattr(self, attr_name).set_hyperparameters(**kwargs)

            elif isinstance(getattr(self, attr_name), list) or isinstance(getattr(self, attr_name), tuple):
                for elem in getattr(self, attr_name):
                    if debug:
                        print(getattr(self, attr_name), isinstance(elem, AlgoModule))
                    if isinstance(elem, AlgoModule):
                        if debug:
                            print('Setting hyperparameters for {} {}:'.format(attr_name, elem))
                        elem.set_hyperparameters(**kwargs)

            elif isinstance(getattr(self, attr_name), Condition):
                if getattr(self, attr_name).beta is None:
                    if debug:
                        print('Setting beta for {} {}:'.format(attr_name, getattr(self, attr_name)))
                    getattr(self, attr_name).beta = beta


class Condition(object):
    def __init__(
            self,
            left,
            right,
            beta=None,
    ):
        self.left = left
        self.right = right
        self.beta = beta

    def get_left(self, state):
        if type(self.left) == str:
            return state[self.left]
        elif type(self.left) is LambdaType:
            kwargs = dict([(key, state[key]) for key in inspect.getfullargspec(self.left)[0]])
            return self.left(**kwargs)
        else:
            return self.left

    def get_right(self, state):
        if type(self.right) == str:
            return state[self.right]
        elif type(self.right) is LambdaType:
            kwargs = dict([(key, state[key]) for key in inspect.getfullargspec(self.right)[0]])
            return self.right(**kwargs)
        else:
            return self.right

    @abstractmethod
    def __call__(self, state: State) -> torch.Tensor:
        pass

    def __and__(self, other):
        if self.beta is None or other.beta is None:
            assert False, 'Warning: and / or currently do not support implicitly setting beta.'
        return lambda state: self(state) * other(state)

    def __or__(self, other):
        if self.beta is None or other.beta is None:
            assert False, 'Warning: and / or currently do not support implicitly setting beta.'

        def or_(state):
            a = self(state)
            b = other(state)
            return a + b - a * b

        return or_


class Algorithm(torch.nn.Module):
    def __init__(
            self,
            *sequence,
            beta=10.,
            max_iter=2**10,
            epsilon=1.e-5,
            hard=False,
            debug=False,
    ):
        super(Algorithm, self).__init__()

        self.inputs = []
        self.outputs = []
        self.variables = []
        self.variable_ints = []
        self.algo_modules = []

        self.beta = beta
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.hard = hard
        self.debug = debug

        for elem in sequence:
            if isinstance(elem, Input):
                self.inputs.append(elem)
            elif isinstance(elem, Output):
                self.outputs.append(elem)
            elif isinstance(elem, Variable):
                self.variables.append(elem)
            elif isinstance(elem, VariableInt):
                self.variable_ints.append(elem)
            elif isinstance(elem, AlgoModule):
                self.algo_modules.append(elem)
            else:
                raise SyntaxError('You inserted an object of value {} into Algorithm but this is not supported.'
                                  ''.format(elem))

        assert len(self.inputs) > 0, 'You need to have at least one Input.'
        assert len(self.outputs) > 0, 'You need to have at least one Output.'
        # assert len(self.algo_modules) > 0, 'You need to have at least one AlgoModule.'

        for algo_module in self.algo_modules:
            algo_module.set_hyperparameters(
                beta=beta,
                max_iter=max_iter,
                epsilon=epsilon,
                hard=hard,
                debug=debug,
            )

    def forward(self, *inputs: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        batch_size = inputs[0].shape[0]

        state = State(self.inputs, inputs, self.variables, self.variable_ints, batch_size)

        if self.debug:
            print('Before Algorithm')
            print(state)
            print('-'*80)

        for module in self.algo_modules:
            state = module(state)

        if self.debug:
            print('After Algorithm')
            print(state)
            print('-'*80)

        return state.return_outputs(self.outputs)


if __name__ == '__main__':

    a = Algorithm(
        Input('values'),
        Var('counter', torch.zeros(1)),
        VarInt('counter2', 1221),
        VarInt('counter3', range(10)),
        Output('values'),
    )

    v = torch.randn(3, 2)
    print(v)
    print(a(v))


