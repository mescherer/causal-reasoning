from numbers import Integral


class Scientist:
    def __init__(self, budget):
        if isinstance(budget, bool) or not isinstance(budget, Integral):
            raise TypeError("budget must be a non-negative integer")
        if budget < 0:
            raise ValueError("budget must be a non-negative integer")
        self._budget = int(budget)

    @property
    def budget(self):
        return self._budget

    def experiment(self, world, n_experiments=1, A=None, B=None, C=None):
        """Generate, print, and return observations from the world in question.
                
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
        if isinstance(n_experiments, bool) or not isinstance(n_experiments, Integral):
            raise TypeError("n_experiments must be a positive integer")
        if n_experiments <= 0:
            raise ValueError("n_experiments must be a positive integer")
        if n_experiments > self._budget:
            return "Failed to run. Experiments exceed the remaining budget."

        observations = world.observe(n_experiments, A, B, C)
        self._budget -= n_experiments
        return observations
