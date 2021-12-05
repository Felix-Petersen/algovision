If-Else and While Control Structures
====================================

Conditions
----------

To define an ``If`` or ``While`` statement, it is necessary to define the respective Condition.
This may be done via conditions such as ``Eq`` (equal), ``IsTrue``, ``LT`` (less than), and ``GT`` (greater than).
See the API for a list of all conditions: :ref:`conditions`.
In general, the tensors used for the condition, should be of shape ``[B, ]``, i.e., apart from batching be scalars, as the probability of execution should be a scalar value.

``If``
------

The ``If`` module implements If-Else statements and may be used in the following way:

.. code-block:: python

    Algorithm(
        # ...
        If(LT('a', lambda b: b + 1),  # corresponds to: if a < b + 1:
            if_true=[
                # ... instructions and control structures ...
            ],
            if_false=[
                # ... instructions and control structures ...
            ],
        ),
        # ...
    )

The ``if_true`` or ``if_false`` may also be omitted here.

``While``
---------

.. code-block:: python

    Algorithm(
        # ...
        While(LT('a', lambda b: b + 1),  # corresponds to: while a < b + 1:
            # ... instructions and control structures ...
        ),
        # ...
    )

While blocks consider two additional hyperparameters: ``max_iter`` and ``epsilon``, which corresponds to the maximum number of iterations and the minimum probability for continuing the loop.
These hyperparameters (similar to ``beta``) can be set locally for each instance or globally as an argument for the algorithm.
Note that if a global and a local value are given, the local value is preferred.

