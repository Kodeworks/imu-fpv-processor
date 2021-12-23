import pytest

from src.utils.damping import Damping


def test_init():
    initial_damping = 0.5
    min_damping = 0.05
    max_damping = 0.8
    damping_factor_damping = 0.05
    damping = Damping(initial_damping=initial_damping,
                      min_damping=min_damping,
                      max_damping=max_damping,
                      damping_factor_damping=damping_factor_damping)

    assert damping.damping == initial_damping
    assert damping.min_damping == min_damping
    assert damping.max_damping == max_damping
    assert damping.damping_factor_damping == damping_factor_damping


def test_value():
    initial_damping = 0.5
    min_damping = 0.05
    max_damping = 0.8
    damping_factor_damping = 0.05
    damping = Damping(initial_damping=initial_damping,
                      min_damping=min_damping,
                      max_damping=max_damping,
                      damping_factor_damping=damping_factor_damping)

    assert damping.value() == initial_damping


def test_update():
    initial_damping = 0.5
    min_damping = 0.05
    max_damping = 0.8
    damping_factor_damping = 0.05
    damping = Damping(initial_damping=initial_damping,
                      min_damping=min_damping,
                      max_damping=max_damping,
                      damping_factor_damping=damping_factor_damping)

    for i in range(100):
        previous_damping_value = damping.value()
        damping.update()
        current_damping_value = damping.value()
        assert previous_damping_value > current_damping_value


def test_boost():
    initial_damping = 0.5
    min_damping = 0.05
    max_damping = 0.8
    damping_factor_damping = 0.05
    damping = Damping(initial_damping=initial_damping,
                      min_damping=min_damping,
                      max_damping=max_damping,
                      damping_factor_damping=damping_factor_damping)

    damping.boost()

    assert damping.value() == max_damping

    new_damping = 0.9
    damping.boost(new_damping=new_damping)
    assert damping.value() == new_damping
