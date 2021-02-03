import numpy as np
from collections import namedtuple
from typing import List, Hashable

Transition = namedtuple(
    "Transition", ["state", "action", "next_state", "prob", "reward"]
)

# introduce type aliases for state and action
State = Hashable
Action = Hashable


class MDP:
    """Class to represent a Markov Decision Process (MDP)

    Allows for easy creation and generation of numpy arrays for faster computation

    :attr transitions (List[Transition]): list of all transitions
    :attr states (Set[State]): set of all states
    :attr actions (Set[Action]): set of all actions
    :attr terminal_states (Set[State]): set of all terminal states (NOT USED)
    :attr init_state (State): initial state (NOT USED)
    :attr max_episode_length (int): maximum length of an episode (NOT USED)
    :attr _state_dict (Dict[State, int]): mapping from states to state indeces
    :attr _action_dict (Dict[Action, int]): mapping from actions to action indeces
    :attr P (np.ndarray of float with dim (num of states, num of actions, num of states)):
        3D NumPy array with transition probabilities.
        *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
        E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
        state 4 with action 2) can be accessed with `self.P[3, 2, 4]`
    :attr R (np.ndarray of float with dim (num of states, num of actions, num of states)):
        3D NumPy array with rewards for transitions.
        E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
        2) can be accessed with `self.R[3, 2, 4]`
    :attr terminal_mask (np.ndarray of bool with dim (num of state)) (NOT USED):
        1D NumPy array of bools indicating terminal states.
        E.g. `self.terminal_mask[3]` returns a boolean indicating whether state 3 is terminal
    :attr compiled (bool): flag indicating whether the MDP was already compiled

    Note:
        State and Action can be any hashable type!
    """

    def __init__(self):
        """Constructor of MDP

        Initialise an empty (!) MDP
        """
        self.transitions = []
        self.states = set()
        self.actions = set()
        self.terminal_states = set()

        self.init_state = None
        self.max_episode_length = None

        self._state_dict = {}
        self._action_dict = {}

        self.P = np.zeros([])
        self.R = np.zeros([])
        self.terminal_mask = np.zeros([])

        self.compiled = False

    def add_transition(self, *transitions: List[Transition]):
        """Adds transition tuples to the MDP

        Any states encountered will be added to the set of states. This will lead to a non-compiled
        MDP. Multiple transitions can be added using add_transition(t1, t2, ...)

        :param transitions (List[Transition]): list of transition tuples to add
        """
        for t in transitions:
            self._add_transition(t)

    def _add_transition(self, transition: Transition):
        """Adds a transition tuple to the MDP

        Any states encountered will be added to the set of states. This will lead to a non-compiled
        MDP.

        :param transition (Transition): transition tuple to add
        """
        if self.compiled:
            self._decompile()

        self.states.add(transition.state)
        self.states.add(transition.next_state)
        self.actions.add(transition.action)

        for t in self.transitions:
            if (
                t.state == transition.state
                and t.next_state == transition.next_state
                and t.action == transition.action
            ):
                raise ValueError("Transition with same {s,a, s'} exists")

        self.transitions.append(transition)

    def add_terminal_state(self, state: State):
        """Adds a terminal/ absorbing state to the MDP

        No outbound transitions are required for such states.

        :param state (State): the terminal state to add
        """
        if self.compiled:
            self._decompile()

        self.states.add(state)
        self.terminal_states.add(state)

    def set_init_state(self, state: State):
        """Sets the initial state of the MDP (optional)

        :param state (State): the initial state of the MDP
        """
        if state not in self.states:
            if self.compiled:
                self._decompile()
            self.states.add(state)

        self.init_state = state

    def ensure_compiled(self):
        """Compile MDP if not already compiled
        """
        if not self.compiled:
            self._compile()

    def _decompile(self):
        """Resets states and actions to modifiable sets and toggles the compiled flag off
        """
        self.states = set(self.states)
        self.actions = set(self.actions)
        self.terminal_states = set(self.terminal_states)
        self._state_dict = {}
        self._action_dict = {}

        self.P = np.zeros([])
        self.R = np.zeros([])
        self.terminal_mask = np.zeros([])
        self.compiled = False

    def _compile(self):
        """Calculates the transition and reward matrices (P and R)

        Calling this function is required to use these lookup matrices for transition outcomes
        """
        self.states = tuple(self.states)
        self.terminal_states = tuple(self.terminal_states)
        self.actions = tuple(self.actions)

        self.compiled = True

        self.terminal_mask = np.array(
            [s in self.terminal_states for s in self.states], dtype=bool
        )
        non_terminal_mask = np.invert(self.terminal_mask)

        for i, s in enumerate(self.states):
            self._state_dict[s] = i
        for i, a in enumerate(self.actions):
            self._action_dict[a] = i

        self.P = np.zeros([len(self.states), len(self.actions), len(self.states)])
        self.R = np.zeros([len(self.states), len(self.actions), len(self.states)])
        for t in self.transitions:
            self.P[
                self._state_dict[t.state],
                self._action_dict[t.action],
                self._state_dict[t.next_state],
            ] = t.prob
            self.R[
                self._state_dict[t.state],
                self._action_dict[t.action],
                self._state_dict[t.next_state],
            ] = t.reward

        if not np.allclose(self.P.sum(axis=2)[non_terminal_mask, :], 1.0):
            raise ValueError("Transition probabilities s0 -> a* must add to 1.")
        # ? Check and warn if terminal states have outbound transitions

    def render(self, filename: str):
        """Renders the MDP environment as a graph

        :param filename (str): name of file to write the grap to
        """
        import pygraphviz as pgv

        G = pgv.AGraph(strict=False, directed=True)

        for state in self.states:
            G.add_node(state)
        for t in self.transitions:
            G.add_edge(t.state, t.next_state, label=str(t.prob) + "/" + str(t.reward))
        G.layout()
        G.draw(filename)
