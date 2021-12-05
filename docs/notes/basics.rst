Basic Instructions
==================

At the core of our framework lies the ``Algorithm`` class, which is a ``torch.nn.Module``.
To implement an algorithm, follow the following structure:

.. code-block:: python

    algo = Algorithm(
        Input('x'),
        # ... more input arguments ...
        Var('y', torch.tensor(0.)),
        # ... more variables ...
        Let('y', 'x'),  # corresponds to y = x
        # ... more instructions ...
        Output('y'),
        # ... more return values ...
        # optionally define hyperparameters like beta
    )

and to execute simply run ``algo(x)`` where ``x`` is a respective PyTorch tensor of shape ``[B, ]`` where ``B`` is the batch size.

In the following, we go into detail for each of the module types.

``Input`` and ``Output``
------------------------

``Input`` and ``Output`` specify which variables are the inputs / outputs of the algorithm.
All returned variables (as specified via ``Output``), have to be either input variables or declared via ``Variable``.

``Var`` and ``Variable``
------------------------

``Variable`` declares and initialized a variable. ``Var`` is an alias for ``Variable``.
Note that variables that are inputs do not have to be / cannot additionally be initialized by ``Var``.
Also, note that if it is initialized with a tensor, a batch dimension will be added and the tensor will be repeated during execution of the algorithm.
For example, if the shape was ``[4, ]``, it will be changed to ``[B, 4, ]`` where ``B`` is the batch size.

λ Lambda Expressions
--------------------

Key to defining an algorithm are ``lambda`` expressions (see `here <https://www.w3schools.com/python/python_lambda.asp>`_ for a reference).
They allow defining anonymous functions and therefore allow expressing computations in-place.
In most cases in ``algovision``, it is possible to write a value in terms of a lambda expressions.
The name of the used variable will be inferred from the signature of the expression.
For example, ``lambda x: x**2`` will take the variable named ``x`` and return the square of it at the location where the expression is written.

``Let('z', lambda x, y: x**2 + y)`` corresponds to the regular line of code ``z = x**2 + y``.
This also allows inserting complex external functions including neural networks as part of the lambda expression.
Assuming ``net`` is a neural networks, one can write ``Let('y', lambda x: net(x))`` (corresponding to ``y = net(x)``).

``Let``
-------

``Let`` is one of the core instructions as it sets the variable given as the left argument to the value of the right expression.


``Let`` is a very flexible instruction.
In its most simple form ``Let`` obtains two arguments, a string naming the variable where the result is written, and the value that may be expressed via a ``lambda`` expression.

If the lambda expression returns multiple values, e.g., because a complex function is called and has two return values, the left argument can be a list of strings.
That is, ``Let(['a', 'b'], lamba x, y: (x+y, x-y))`` corresponds to ``a, b = x+y, x-y``.

``Let`` also supports indexing. This is denoted by an additional list argument after the left and/or the right argument.
For example, ``Let('a', 'array', ['i'])`` corresponds to ``a = array[i]``, while ``Let('array', ['i'], 'b')`` corresponds to ``array[i] = b``.
``Let('array', ['i'], 'array', ['j'])`` corresponding to ``array[i] = array[j]`` is also supported.

Note that indexing can also be expressed through ``lambda`` expressions.
For example, ``Let('a', 'array', ['i'])`` is equivalent to ``Let('a', lambda array, i: array[:, i])``. Note how in this case the batch dimension has to be explicitly taken into account (``[:, ]``).
Relaxed indexing on the right-hand side is only supported through ``lambda`` expressions due to its complexity.
Relaxed indexing on the left-hand side is supported if exactly one probability weight tensor is in the list (e.g., ``Let('array', [lambda x: get_weights(x)], 'a')``).

