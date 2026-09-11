import pytest

from scientist import Scientist


class StubWorld:
    def __init__(self, result=None, error=None):
        self.result = {} if result is None else result
        self.error = error
        self.calls = []

    def observe(self, n_experiments, A, B, C):
        self.calls.append((n_experiments, A, B, C))
        if self.error is not None:
            raise self.error
        return self.result


def test_experiment_observes_world_and_charges_each_experiment():
    expected = {"experiment: 0": {"A": 1.0, "B": 2.0, "C": 3.0}}
    world = StubWorld(result=expected)
    scientist = Scientist(budget=5)

    result = scientist.experiment(world, n_experiments=3, A=1, B=2, C=3)

    assert result is expected
    assert world.calls == [(3, 1, 2, 3)]
    assert scientist.budget == 2


def test_experiment_can_use_exact_remaining_budget():
    world = StubWorld()
    scientist = Scientist(budget=2)

    scientist.experiment(world, n_experiments=2)

    assert scientist.budget == 0
    assert world.calls == [(2, None, None, None)]


def test_experiment_cannot_exceed_remaining_budget():
    world = StubWorld()
    scientist = Scientist(budget=2)

    result = scientist.experiment(world, n_experiments=3)

    assert result == "Failed to run. Experiments exceed the remaining budget."
    assert scientist.budget == 2
    assert world.calls == []


def test_experiment_cannot_run_after_budget_is_exhausted():
    world = StubWorld()
    scientist = Scientist(budget=1)
    scientist.experiment(world)

    result = scientist.experiment(world)

    assert result == "Failed to run. Experiments exceed the remaining budget."
    assert scientist.budget == 0
    assert len(world.calls) == 1


def test_failed_world_observation_does_not_consume_budget():
    world = StubWorld(error=RuntimeError("observation failed"))
    scientist = Scientist(budget=2)

    with pytest.raises(RuntimeError, match="observation failed"):
        scientist.experiment(world, n_experiments=2)

    assert scientist.budget == 2


@pytest.mark.parametrize("budget", [-1, -10])
def test_scientist_rejects_negative_budget(budget):
    with pytest.raises(ValueError, match="non-negative integer"):
        Scientist(budget)


@pytest.mark.parametrize("budget", [True, 1.5, "2"])
def test_scientist_rejects_non_integer_budget(budget):
    with pytest.raises(TypeError, match="non-negative integer"):
        Scientist(budget)


@pytest.mark.parametrize("n_experiments", [0, -1])
def test_experiment_rejects_non_positive_counts(n_experiments):
    with pytest.raises(ValueError, match="positive integer"):
        Scientist(5).experiment(StubWorld(), n_experiments=n_experiments)


@pytest.mark.parametrize("n_experiments", [True, 1.5, "2"])
def test_experiment_rejects_non_integer_counts(n_experiments):
    with pytest.raises(TypeError, match="positive integer"):
        Scientist(5).experiment(StubWorld(), n_experiments=n_experiments)
