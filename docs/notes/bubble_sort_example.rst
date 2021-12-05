Example: Bubble Sort
====================

Deriving a loss from a smooth algorithm can be as easy as

.. code-block:: python

    from examples import get_bubble_sort
    import torch

    torch.manual_seed(0)
    # Get an array (the first dimension is the batch dimension, which is always required)
    array = torch.randn(1, 8, requires_grad=True)

    bubble_sort = get_bubble_sort(beta=5)
    result, loss = bubble_sort(array)

    loss.backward()
    print(array)
    print(result)
    print(array.grad)

Here, the loss is a sorting loss corresponding to the number of swaps in the bubble sort algorithm.
But we can also define this algorithm from scratch:

.. code-block:: python

    from algovision import (
        Algorithm, Input, Output, Var, VarInt,  # core
        GT, IsTrue,                       # conditions
        If, While, For,           # control_structures
        Let, LetInt,                       # functions
    )
    import torch

    bubble_sort = Algorithm(
        # Define the variables the input corresponds to
        Input('array'),
        # Declare and initialize all differentiable variables
        Var('a',        torch.tensor(0.)),
        Var('b',        torch.tensor(0.)),
        Var('swapped',  torch.tensor(1.)),
        Var('loss',     torch.tensor(0.)),
        # Declare and initialize a hard integer variable (VarInt) for the control flow.
        # It can be defined in terms of a lambda expression. The required variables
        # are automatically inferred from the signature of the lambda expression.
        VarInt('n', lambda array: array.shape[1] - 1),
        # Start a relaxed While loop:
        While(IsTrue('swapped'),
            # Set `swapped` to 0 / False
            Let('swapped', 0),
            # Start an unrolled For loop. Corresponds to `for i in range(n):`
            For('i', 'n',
                # Set `a` to the `i`th element of `array`
                Let('a', 'array', ['i']),
                # Using an inplace lambda expression, we can include computations
                # based on variables to obtain the element at position i+1.
                Let('b', 'array', [lambda i: i+1]),
                # An If-Else statement with the condition a > b
                If(GT('a', 'b'),
                   if_true=[
                       # Set the i+1 th element of array to a
                       Let('array', [lambda i: i + 1], 'a'),
                       # Set the i th element of array to b
                       Let('array', ['i'], 'b'),
                       # Set swapped to 1 / True
                       Let('swapped', 1.),
                       # Increment the loss by 1 using a lambda expression
                       Let('loss', lambda loss: loss + 1.),
                   ]
               ),
            ),
            # Decrement the hard integer variable n by 1
            LetInt('n', lambda n: n-1),
        ),
        # Define what the algorithm should return
        Output('array'),
        Output('loss'),
        # Set the inverse temperature beta
        beta=5,
    )


