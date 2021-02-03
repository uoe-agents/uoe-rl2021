"""
These test that the utils.mdp works correctly
"""

import pytest
from rl2021.utils import MDP, Transition


def test_add_transition_1():
    mdp = MDP()
    t = Transition("s0", "a0", "s1", 0, 0)
    mdp.add_transition(t)
    assert "s0" in mdp.states
    assert "s1" in mdp.states
    assert "a0" in mdp.actions

    assert len(mdp.states) == 2
    assert len(mdp.actions) == 1


def test_add_transition_2():
    mdp = MDP()
    t1 = Transition("s0", "a0", "s1", 0, 0)
    t2 = Transition("s1", "a0", "s2", 0, 0)
    mdp.add_transition(t1)
    mdp.add_transition(t2)

    assert len(mdp.states) == 3
    assert len(mdp.actions) == 1


def test_add_transition_3():
    mdp = MDP()
    t1 = Transition("s0", "a0", "s0", 0, 0)
    t2 = Transition("s1", "a1", "s1", 0, 0)
    mdp.add_transition(t1)
    mdp.add_transition(t2)

    assert len(mdp.states) == 2
    assert len(mdp.actions) == 2


def test_add_transition_4():
    mdp = MDP()
    t1 = Transition("s0", "a0", "s1", 0, 0)
    t2 = Transition("s0", "a0", "s1", 0, 0)
    mdp.add_transition(t1)

    with pytest.raises(ValueError):
        mdp.add_transition(t2)

    assert len(mdp.states) == 2
    assert len(mdp.actions) == 1


def test_set_terminal():
    mdp = MDP()
    t1 = Transition("s0", "a0", "s1", 0, 0)
    mdp.add_transition(t1)

    mdp.add_terminal_state("s0")

    assert len(mdp.states) == 2
    assert len(mdp.actions) == 1
    assert len(mdp.terminal_states) == 1
    assert "s0" in mdp.terminal_states

    mdp.add_terminal_state("s4")
    assert len(mdp.states) == 3
    assert len(mdp.terminal_states)
    assert "s4" in mdp.terminal_states
    assert "s4" in mdp.states


def test_compile_1():
    mdp = MDP()
    t1 = Transition("s0", "a0", "s1", 1, 0)
    t2 = Transition("s1", "a0", "s0", 1, 0)
    mdp.add_transition(t1)
    mdp.add_transition(t2)

    mdp.ensure_compiled()
    assert mdp.P.shape == (2, 1, 2)
    assert mdp.R.shape == (2, 1, 2)


def test_compile_2():
    mdp = MDP()
    t1 = Transition("s0", "a0", "s1", 1, 5)
    t2 = Transition("s1", "a0", "s0", 1, 2)
    mdp.add_transition(t1)
    mdp.add_transition(t2)

    mdp.ensure_compiled()
    assert (
        mdp.R[mdp._state_dict["s0"], mdp._action_dict["a0"], mdp._state_dict["s1"]] == 5
    )
    assert (
        mdp.R[mdp._state_dict["s1"], mdp._action_dict["a0"], mdp._state_dict["s0"]] == 2
    )

    assert (
        mdp.P[mdp._state_dict["s0"], mdp._action_dict["a0"], mdp._state_dict["s1"]] == 1
    )
    assert (
        mdp.P[mdp._state_dict["s1"], mdp._action_dict["a0"], mdp._state_dict["s0"]] == 1
    )


def test_compile_3():
    mdp = MDP()
    t1 = Transition("s0", "a0", "s1", 1, 0)
    t2 = Transition("s1", "a0", "s0", 0.5, 0)
    mdp.add_transition(t1)
    mdp.add_transition(t2)

    with pytest.raises(ValueError):
        mdp.ensure_compiled()

    t3 = Transition("s1", "a0", "s2", 0.5, 0)
    t4 = Transition("s2", "a0", "s0", 1, 0)

    mdp.add_transition(t3)
    mdp.add_transition(t4)

    mdp.ensure_compiled()


def test_compile_4():
    mdp = MDP()
    t1 = Transition("s0", "a0", "s1", 1, 0)
    mdp.add_transition(t1)

    with pytest.raises(ValueError):
        mdp.ensure_compiled()

    mdp.add_terminal_state("s1")
    mdp.ensure_compiled()


def test_compile_5():
    mdp = MDP()
    t1 = Transition("s0", "a0", "s1", 1, 0)
    mdp.add_transition(t1)
    mdp.add_terminal_state("s1")

    mdp.ensure_compiled()
    assert mdp.compiled
    assert type(mdp.states) is tuple
    assert type(mdp.actions) is tuple
    assert type(mdp.terminal_states) is tuple
    mdp._decompile()
    assert not mdp.compiled
    assert type(mdp.states) is set
    assert type(mdp.actions) is set
    assert type(mdp.terminal_states) is set


def test_compile_6():
    mdp = MDP()
    t1 = Transition("s0", "a0", "s1", 1, 0)
    mdp.add_transition(t1)
    mdp.add_terminal_state("s1")
    mdp.ensure_compiled()
    assert mdp.terminal_mask[mdp._state_dict["s1"]] == True
    assert mdp.terminal_mask[mdp._state_dict["s0"]] == False
