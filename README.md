# uoe-rl2021
Coursework base code for Reinforcement Learning 2021 (UoE)

Find the coursework description here (might need Learn log-in): 

https://www.learn.ed.ac.uk/webapps/blackboard/content/listContentEditable.jsp?content_id=_5094964_1&course_id=_82627_1&mode=reset

## Installation Instructions tested in a new DICE account

You can also find instructions in the coursework description (Section 3)

Open the terminal and write:

```bash
mkvirtualenv --python=`which python3` rl2021
```
This will create a virtual environment called 'rl2021' which you should use to work.

It's a python3 environment since python3 will be used during tests.

Every time you open a terminal you will need to activate the environment using:

```bash
workon rl2021
```

You should see then a parenthesis on your terminal as such:
```bash
(rl2021) [vulcan]s1873000:
```
Now clone the environment:
```bash
git clone git@github.com:uoe-agents/uoe-rl2021.git
```

Navigate and install using:
```bash
cd uoe-rl2021
pip install -e .
```

## CAREFUL! Your Forks are PUBLIC!
Forks in github are public, and usually easy to find by clicking the number next to your forks.

If you want to mirror the repository you can do it this way: https://help.github.com/articles/duplicating-a-repository/ and don't forget to make it private afterwards.

You could then add the original remote and pull any updates (a bit less convenient but...)

using:
```bash
git remote add coursework https://github.com/uoe-agents/uoe-rl2021.git
```
and pulling as:
```bash
git pull coursework master
```
or just clone and work locally.

## PyTest

The test folder contains a few tests that should all pass for full marks. They include checks for usage of correct variable names and the existance of files.
To run them, navigate to the main folder `uoe-rl2021
and run:
```bash
pytest -v
```


