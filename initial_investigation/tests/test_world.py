import numpy as np
import pytest

from world import World


class StubRng:
    def __init__(self, noise_draws, a_draws=()):
        self.noise_draws = iter(noise_draws)
        self.a_draws = iter(a_draws)
        self.calls = []

    def normal(self, loc, scale, size=None):
        self.calls.append((loc, scale, size))
        if size == 3:
            return next(self.noise_draws)
        return next(self.a_draws)


def test_world_has_expected_causal_weights_and_noise_distribution():
    rng = StubRng(noise_draws=[[1.0, 2.0, 3.0]])

    world = World(rng=rng)

    np.testing.assert_array_equal(
        world._W,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.9, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(world._noise, [1.0, 2.0, 3.0])
    assert rng.calls == [(0.0, np.sqrt(5.0), 3)]


def test_observe_applies_weights_inputs_and_noise(capsys):
    rng = StubRng(noise_draws=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    world = World(rng=rng)

    data = world.observe(1, 10, 2, 3)

    assert data == {
        "experiment: 0": {"A": 11.0, "B": 9.0, "C": 7.8}
    }
    assert capsys.readouterr().out == f"{data}\n"


def test_observe_draws_a_with_mean_50_and_variance_10(capsys):
    rng = StubRng(
        noise_draws=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        a_draws=[10.0],
    )
    world = World(rng=rng)

    data = world.observe()

    assert data["experiment: 0"] == {"A": 11.0, "B": 7.0, "C": 3.0}
    assert rng.calls[1] == (50.0, np.sqrt(10.0), None)
    capsys.readouterr()


def test_observe_generates_requested_number_of_independent_experiments(capsys):
    rng = StubRng(
        noise_draws=[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )
    world = World(rng=rng)

    data = world.observe(2, A=2)

    assert list(data) == ["experiment: 0", "experiment: 1"]
    assert data["experiment: 0"] == {"A": 2.0, "B": 1.0, "C": 0.0}
    assert data["experiment: 1"] == {"A": 3.0, "B": 2.0, "C": 1.0}
    capsys.readouterr()


def test_zero_is_a_valid_explicit_value(capsys):
    rng = StubRng(noise_draws=[[0, 0, 0], [0, 0, 0]])
    world = World(rng=rng)

    data = world.observe(1, 0, 0, 0)

    assert data["experiment: 0"] == {"A": 0.0, "B": 0.0, "C": 0.0}
    capsys.readouterr()


@pytest.mark.parametrize("n_experiments", [0, -1])
def test_observe_rejects_non_positive_experiment_counts(n_experiments):
    world = World(rng=StubRng(noise_draws=[[0, 0, 0]]))

    with pytest.raises(ValueError, match="positive integer"):
        world.observe(n_experiments)


@pytest.mark.parametrize("n_experiments", [True, 1.5, "2"])
def test_observe_rejects_non_integer_experiment_counts(n_experiments):
    world = World(rng=StubRng(noise_draws=[[0, 0, 0]]))

    with pytest.raises(TypeError, match="positive integer"):
        world.observe(n_experiments)


@pytest.mark.parametrize(
    ("argument", "value"),
    [("A", "1"), ("B", object()), ("C", True)],
)
def test_observe_rejects_non_numeric_inputs(argument, value):
    world = World(rng=StubRng(noise_draws=[[0, 0, 0]]))

    with pytest.raises(TypeError, match=argument):
        world.observe(**{argument: value})
