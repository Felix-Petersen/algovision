For Loops and Hard Integers
===========================

``For``
-------

``For`` loops are bounded loops which are unrolled and with the condition that the number of iterations does not depend on the input value.
However, it may depend on the input shape.
This effectively acts as syntactic sugar for implementing more complex cases.
Note that loops are generally slower than a vectorized computation, which is why vectorization (if possible) is preferrable.

The concept of ``For`` loops also comes with hard integer variables, which can be initialized with ``VarInt`` and may also not depend on the input values but can depend on the shape.
``LetInt`` only supports setting the variable to an integer (Python ``int``) or list of integers (as well as the same type via lambda expressions).
Integers may be used for loops and for hard indexing.

An example is

.. code-block:: python

    Algorithm(
        # ...
        For('i', 'n',  # corresponds to: for i in range(n):
            # ... instructions and control structures ...
        ),
        # ...
    )

Instead of ``'n'``, one may also give a list or range (e.g., ``[0, 2, 4, 6]`` / ``range(0, 8, 2)``) or a ``lambda`` expression like ``lambda n: n-1`` or ``lambda array: array.shape[1]``.
