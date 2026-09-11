"""A small linear causal world that can be observed or intervened on."""

from numbers import Integral
from numbers import Real

import numpy as np


class World:
    """Represent a test-world with hidden parameters.
    """

    def __init__(self, rng=None):
        """
        Initializes the world.

        Parameters
        ----------
        rng:
            Optional NumPy-compatible random generator. Pass
            ``numpy.random.default_rng(seed)`` to make experiments reproducible.
        """
        self._W = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.0, 0.0],
                            [0.0, 0.9, 0.0]])
        self._rng = np.random if rng is None else rng
        if not hasattr(self._rng, "normal"):
            raise TypeError("rng must provide a normal() method")

        # NumPy's normal distribution takes a standard deviation, so sqrt(5)
        # is used to obtain the requested variance of 5.00.
        self._noise = self._draw_noise()

    def _draw_noise(self):
        """Draw independent noise for A, B, and C."""
        return np.asarray(
            self._rng.normal(loc=0.0, scale=np.sqrt(5.0), size=3),
            dtype=float,
        )

    @staticmethod
    def _validate_value(name, value):
        if value is not None and (
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        ):
            raise TypeError(f"{name} must be a real number or None")

    def observe(self, n_experiments=1, A=None, B=None, C=None):
        """Generate, print, and return observations from the system.

        Passing a value for ``A``, ``B``, or ``C`` fixes that input for every
        experiment. Omission of a value for any variable results in an uncontrolled
        value.

        Also contains an n_experiments parameter, which defaults to 1 if not specified.

        Returns
        -------
        dict
            A dictionary keyed by ``"experiment: <index>"``. Each value is a
            dictionary containing the resulting ``A``, ``B``, and ``C`` measurements.
        """
        if (
            isinstance(n_experiments, (bool, np.bool_))
            or not isinstance(n_experiments, Integral)
        ):
            raise TypeError("n_experiments must be a positive integer")
        if n_experiments <= 0:
            raise ValueError("n_experiments must be a positive integer")

        self._validate_value("A", A)
        self._validate_value("B", B)
        self._validate_value("C", C)

        b_value = 0.0 if B is None else float(B)
        c_value = 0.0 if C is None else float(C)
        data = {}
        for experiment in range(n_experiments):
            if A is None:
                # NumPy takes standard deviation rather than variance.
                a_value = float(
                    self._rng.normal(loc=50.0, scale=np.sqrt(10.0))
                )
            else:
                a_value = float(A)

            x = np.array([a_value, b_value, c_value], dtype=float)
            y = (self._W @ x) + x + self._noise

            data[f"experiment: {experiment}"] = {
                "A": float(y[0]),
                "B": float(y[1]),
                "C": float(y[2]),
            }

            # Keep a fresh independent noise draw ready for the next
            # experiment (including one performed by a later observe call).
            self._noise = self._draw_noise()

        print(data)
        return data