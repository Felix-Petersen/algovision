# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from algovision.core import *


class Let(AlgoModule):
    """

    .. list-table:: Use cases of :class:`Let`
       :widths: 38 24 37
       :header-rows: 1
       :class: tight-table

       * - AlgoVision
         - Python
         - Description
       * - ``Let('a', 'x')``
         - ``a = x``
         - Variable ``a`` is set to the value of variable ``x``.
       * - ``Let('a', lambda x: x**2)``
         - ``a = x**2``
         - As soon as we compute anything on the right hand side of the equation, we need to write it as a ``lambda`` expression.
       * - ``Let('a', 'array', ['i'])``
         - ``a = array[i]``
         - Indexing on the right hand requires an additional list parameter after the second argument.
       * - ``Let('a', lambda array, i: array[:, i])``
         - ``a = array[i]``
         - Equivalent to the row above: indexing can also be manually done inside of a ``lambda`` expression. Note that in this case, the batch dimension has to be written explicitly.
       * - ``Let('a', 'array', ['i', lambda j: j+1])``
         - ``a = array[i, j+1]``
         - Multiple indices and `lambda` expressions are also supported.
       * - ``Let('a', 'array', [None, slice(0, None, 2)])``
         - ``a = array[:, 0::2]``
         - ``None`` and ``slice`` s are also supported.
       * - ``Let('a', ['i'], 'x')``
         - ``a[i] = x``
         - Indexing can also be done on the left hand side of the equation.
       * - ``Let('a', ['i'], 'x', ['j'])``
         - ``a[i] = x['j']``
         - ...or on both sides.
       * - ``Let(['a', 'b'], lamba x, y: (x+y, x-y))``
         - ``a, b = x+y, x-y``
         - Multiple return values are supported.

    """
    def __init__(self, *args):
        """
        The :class:`Let` module executes a lambda or arbitrary other function and writes the return values back to
        the specified variable.

        """
        super(Let, self).__init__()

        if not isinstance(args[0], str):
            assert len(args) == 2 and isinstance(args[0], list), (
                'The first argument has to be a variable name string. An exception is if it is a list and there are '
                'only 2 arguments, the second of which is a lambda expression. Given was {}'.format(args))

        self.index_variable = None
        self.index_value = None

        if len(args) < 2:
            assert False, 'You need to provide at least two values for a Let assignment statement. {}'.format(args)
        elif len(args) == 2 and isinstance(args[0], str):
            # Let(str, lambda / str / int / float)
            self.variable = args[0]
            self.value = args[1]
        elif len(args) == 2 and isinstance(args[0], list):
            # Let([str], lambda)
            assert all([isinstance(x, str) for x in args[0]]), (
                'If the first input is a list, it requires that all inputs are strings and that the value it is '
                'set to is defined as a lambda expression. The notation for using a list is exclusively for collecting '
                'multiple return values of a lambda expression. Given was: {}'.format(args[0]))
            assert isinstance(args[1], LambdaType), (
                'If the first input is a list, it requires that all inputs are strings and that the value it is '
                'set to is defined as a lambda expression. The notation for using a list is exclusively for collecting '
                'multiple return values of a lambda expression. Instead of a lambda expression, '
                'given was: {} of type {}'.format(args[1], type(args[1])))
            self.variable = args[0]
            self.value = args[1]
        elif len(args) == 3 and isinstance(args[1], list):
            # Let(str, [], lambda / str / int / float)
            self.variable = args[0]
            self.index_variable = args[1]
            self.value = args[2]
        elif len(args) == 3 and isinstance(args[1], str) and isinstance(args[2], list):
            # Let(str, str, [])
            self.variable = args[0]
            self.value = args[1]
            self.index_value = args[2]
        elif len(args) == 4 and isinstance(args[1], list) and isinstance(args[2], str) and isinstance(args[3], list):
            # Let(str, [], str, [])
            self.variable = args[0]
            self.index_variable = args[1]
            self.value = args[2]
            self.index_value = args[3]
        else:
            assert False, 'Invalid set of arguments for Let: {}'.format(args)

    def __call__(self, state: State) -> State:

        assert (isinstance(self.value, LambdaType) or isinstance(self.value, str) or isinstance(self.value, int)
                or isinstance(self.value, float) or isinstance(self.value, torch.Tensor)), (
            'The value defined in the Let has to be one of "lambda / str / int / float / Tensor". However, {} '
            'is of type {}.'.format(self.value, type(self.value))
        )

        ################################################################################################################

        if isinstance(self.value, LambdaType):
            input_args = inspect.getfullargspec(self.value)[0]
            args = [state[k] for k in input_args]
            results = self.value(*args)

            if isinstance(results, tuple) or isinstance(results, list):
                assert self.index_variable is None and self.index_value is None, (
                    'When using a lambda expression to return a tuple or a list, indexing is not supported. For '
                    'indexing the value, you could also do this within the lambda expression. Anything else makes '
                    'hardly sense because indexing the variable on the left is ambiguous if there are more than '
                    'one variable on the left. If necessary, you could write the result to a temporary variable '
                    'and then use indexing in a separate Let statement. You fed the following to Let: {} {} {} {}'
                    ''.format(self.variable, self.index_variable, self.value, self.index_value)
                )
                assert len(self.variable) == len(results), (
                    'The number of variables to write to ({} {}) does not match the number of values returned by the '
                    'lambda expression ({}).'.format(len(self.variable), self.variable, len(results))
                )
                results = dict(zip(self.variable, results))
                state.update(results)
                return state

            assert type(results) in [torch.Tensor, int, float], (
                'The return value of the lambda expression has to be one of [torch.Tensor, int, float] but was type '
                '{}. It was supposed to be written to variable {}.'.format(type(results), self.variable)
            )

        elif isinstance(self.value, str):
            results = state[self.value]

        elif isinstance(self.value, torch.Tensor):
            results = self.value.unsqueeze(0).repeat(
                state.batch_size,
                *[1]*len(self.value.shape)
            )

        elif isinstance(self.value, int) or isinstance(self.value, float):
            results = self.value
        else:
            assert False, 'Internal error caused by an illegal value type fed to Let: {}'.format(type(self.value))

        ################################################################################################################

        if self.index_value is not None and len(self.index_value) > 0:
            assert isinstance(results, torch.Tensor), (
                'Indexing the value is only allowed if the value is a tensor, but the value for variable {} and '
                'which is supposed to by indexed by {} is of type {}.'
                ''.format(self.variable, self.index_value, type(self.value))
            )
            assert len(self.index_value) <= len(results.shape) - 1, (
                'To many indices were given ({}) for the value of shape {} to be set to variable {}. '
                'Note that the first dimension of the value is here the batch dimension and the first given '
                'index is not for the batch dimension. Instead, the batch dimension is handled implicitly.'
                ''.format(self.index_value, results.shape, self.variable)
            )

            indices = []
            for index_ in self.index_value:
                if isinstance(index_, LambdaType):
                    args = inspect.getfullargspec(index_)[0]
                    args = [state[k] for k in args]
                    index = index_(*args)
                    assert isinstance(index, int), (
                        'The index obtained with `{}` has to be of type int but is of type {}. '
                        'Indexing with tensors is not supported by the direct indexing function as it is an advanced '
                        'feature that requires the user to use it within a lambda expression due to the complexity of '
                        'potential relaxed indexings.'
                        ''.format(index_, type(index))
                    )
                elif isinstance(index_, int) or isinstance(index_, slice):
                    index = index_
                elif isinstance(index_, str):
                    index = state[index_]
                    assert isinstance(index, int), (
                        'The index obtained with `{}` has to be of type int but is of type {}. '
                        'Indexing with tensors is not supported by the direct indexing function as it is an advanced '
                        'feature that requires the user to use it within a lambda expression due to the complexity of '
                        'potential relaxed indexings.'
                        ''.format(index_, type(index))
                    )
                elif index_ is None:
                    index = slice(None)
                else:
                    assert False, (
                        'The index cannot be obtained as `{}` of type {} is not a legal index type. '
                        'Indexing with tensors is not supported by the direct indexing function as it is an advanced '
                        'feature that requires the user to use it within a lambda expression due to the complexity of '
                        'potential relaxed indexings.'
                        ''.format(index_, type(index_))
                    )

                indices.append(index)

            # print('Before Indexing', results.shape)
            results = results[[slice(None)] + indices]
            # print('After Indexing', results.shape)

        ################################################################################################################

        if self.index_variable is not None and len(self.index_variable) > 0:

            assert isinstance(state[self.variable], torch.Tensor), (
                'Indexing the value is only allowed if the variable is a tensor, but variable {} is of type {}. '
                'Most likely, you tried to index / and probabilistically modify a VariableInt, which is not possible.'
                ''.format(self.variable, type(self.variable))
            )
            assert len(self.index_variable) <= len(state[self.variable].shape) - 1, (
                'To many indices were given ({}) for the variable {} of shape {}. '
                'Note that the first dimension of the value is here the batch dimension and the first given '
                'index is not for the batch dimension. Instead, the batch dimension is handled implicitly.'
                ''.format(self.index_variable, self.variable, state[self.variable].shape)
            )

            indices = []

            for index_ in self.index_variable:
                if isinstance(index_, LambdaType):
                    args = inspect.getfullargspec(index_)[0]
                    args = [state[k] for k in args]
                    index = index_(*args)
                    assert isinstance(index, int) or isinstance(index, torch.Tensor), (
                        'The index obtained with `{}` has to be of type int or torch.Tensor but is of type {}. '
                        ''.format(index_, type(index))
                    )
                elif isinstance(index_, int) or index_ is slice:
                    index = index_
                elif index_ is None:
                    index = slice(None)
                elif isinstance(index_, str):
                    index = state[index_]
                    assert isinstance(index, int) or isinstance(index, torch.Tensor), (
                        'The index obtained with `{}` has to be of type int or torch.Tensor but is of type {}. '
                        ''.format(index_, type(index))
                    )
                elif isinstance(index_, torch.Tensor):
                    index = index_.unsqueeze(0).repeat(
                        state.batch_size,
                        *[1]*len(index_.shape)
                    )
                else:
                    assert False, (
                        'The index cannot be obtained as `{}` of type {} is not a legal index type. '
                        ''.format(index_, type(index_))
                    )

                indices.append(index)

            if any([isinstance(x, torch.Tensor) for x in indices]):
                assert len(indices) == 1, (
                    'Probabilistic tensor indexing for writing back to a variable supports only one index, '
                    'which is a probability distribution giving the probability / extent of writing back to each '
                    'element. The given index was {} for variable {}.'.format(self.index_variable, self.variable)
                )

                state.probabilistic_update(self.variable, results, indices[0])
                return state

            with torch.no_grad():
                p = torch.zeros_like(state[self.variable])
                p[[slice(None)] + indices] = 1

            state.probabilistic_update(self.variable, results, p)
            return state

        ################################################################################################################

        else:
            state.update({self.variable: results})
            return state


class LetInt(AlgoModule):
    def __init__(self, *args):
        """
        The :class:`LetInt` module executes a lambda or arbitrary other function and writes the return values back to
        the specified integer variable. Valid values are integers, lists of integers, and lambda expressions returning
        either of the first two, or strings corresponding to variables.

        """
        super(LetInt, self).__init__()

        self.index_variable = None
        self.index_value = None

        if len(args) != 2:
            assert False, 'You need to provide exactly two values for a LetInt assignment statement. {}'.format(args)
        else:
            # LetInt(str, lambda / str / int / list)
            self.variable = args[0]
            self.value = args[1]
            if isinstance(self.value, collections.abc.Iterable) and not isinstance(self.value, str):
                self.value = list(self.value)
                assert all([isinstance(x, int) for x in self.value]), 'There was a value that is not an int in {}.' \
                                                                      ''.format(self.value)

    def __call__(self, state: State) -> State:

        assert (isinstance(self.value, LambdaType) or isinstance(self.value, str) or isinstance(self.value, int)
                or isinstance(self.value, list)), (
            'The value defined in the Let has to be one of "lambda / str / int / List[int]". However, {} '
            'is of type {}.'.format(self.value, type(self.value))
        )

        if isinstance(self.value, LambdaType):
            input_args = inspect.getfullargspec(self.value)[0]
            args = [state[k] for k in input_args]
            results = self.value(*args)
            assert type(results) in [int, list, collections.abc.Iterable], (
                'The return value of the lambda expression has to be one of [int, list, iter] but was type '
                '{}. It was supposed to be written to variable {}.'.format(type(results), self.variable)
            )
            if type(results) in [list, collections.abc.Iterable]:
                results = list(results)
                assert all([isinstance(x, int) for x in results]), 'There was a value that is not an int in {}.' \
                                                                   ''.format(results)

        elif isinstance(self.value, str):
            results = state[self.value]
            assert type(results) in [int, list], (
                'The return value selected with key {} has to be one of [int, list, iter] but was type '
                '{}. It was supposed to be written to variable {}.'.format(self.value, type(results), self.variable)
            )

        elif isinstance(self.value, int):
            results = self.value

        elif isinstance(self.value, list):
            results = list(self.value)

        else:
            assert False, 'Internal error caused by an illegal value type fed to LetInt: {}'.format(type(self.value))

        state.update({self.variable: results})
        return state


class Print(AlgoModule):
    def __init__(
            self,
            function
    ):
        """
        Like :class:`Let` but does not write back and is only for debug purposes.

        Args:
            function: lambda function.
        """
        super(Print, self).__init__()
        self.function = function

    def __call__(self, state: State) -> State:
        """Identity, result of function is printed."""

        input_args = inspect.getfullargspec(self.function)[0]
        args = [state[k] for k in input_args]
        print(self.function(*args))

        return state


