# To test install using:
```console
    semitable@machine:~$ cd rl2021
    semitable@machine:~$ pip install -e .
```
and run tests:
```console
    semitable@machine:~$ pytest
```



# Exercise 1
- Create and solve an MDP

e.g. the chain world mdp:

```python
from utils import MDP
mdp = MDP()
mdp.add_transition(
    #         start action end prob reward
    Transition("s0", "a", "s1", 1, 0),
    Transition("s0", "b", "s0", 1, 0),
    Transition("s1", "a", "s2", 1, 0),
    Transition("s1", "b", "s0", 1, 2),
    Transition("s2", "a", "s3", 1, 0),
    Transition("s2", "b", "s0", 1, 2),
    Transition("s3", "a", "s4", 1, 0),
    Transition("s3", "b", "s0", 1, 2),
    Transition("s4", "a", "s4", 1, 10),
    Transition("s4", "b", "s0", 1, 2),
)
mdp.max_episode_length = 10
mdp.set_init_state("s0")
```

solve mdp using:
exercise1/mdp_solver.py:PolicyIteration

```python
from exercise1.mdp_solver import PolicyIteration, ValueIteration

PolicyIteration(mdp, discount=0.99).solve()
# or
ValueIteration(mdp, discount=0.99).solve()
```
