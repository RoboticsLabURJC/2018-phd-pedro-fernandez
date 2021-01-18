### Sources [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)




### Table of contents

## GridWorld
  - [Tabular Dyna-Q](#TABULAR-DYNA-Q)
  - [Dyna-Q +](#Dyna-Q-+)
  - [Prioritized Sweeping](#Prioritized-Sweeping)
## Short Corridor
  - [Policy Approximation](#Policy-approximation)
  - [REINFORCE Monte Carlo Policy Gradient](#REINFORCE-MC)
---

First implementation of a RL algoritmhs. It is a typical and beginner example for understanding basics ideas and concepts of RL.

We are going to follow the TextBook [Sutton]

The GridWorld and Maze are explained in chapter 8, and they are good examples of discrete and stochastic environments.


## Tabular Dyna-Q

The algorithm is
![tabular Dyna-Q](GridWorld&Maze/images_theory/Tabular-Dyna-Q.png)

Model and Planning verion that it is called Tabular Dyna-Q. 

If we take only until (d) step, it is called a **direct RL** or **Q-learning** algorithm. And steps (e) and (f) it is the Tabular Dyna-Q with Model and Planning version, which represents next figure

![model and planning](GridWorld&Maze/images_theory/model-and-planning.png)

![Dyna-Q architecture](GridWorld&Maze/images_theory/DynaQ-architecture.png)

<a href="https://www.youtube.com/watch?feature=player_embedded&v=K3DimDK5H6A
" target="_blank"><img src="https://img.youtube.com/vi/K3DimDK5H6A/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

[![Tabular Dyna Q](https://img.youtube.com/vi/K3DimDK5H6A/0.jpg)](https://www.youtube.com/watch?v=K3DimDK5H6A)



https://youtu.be/K3DimDK5H6A

- Tabular_Dyna-Q V2.ipynb and Tabular_Dyna-Q V2.py contains implementation of TABULAR DYNA-Q. 
Running the code, we obtain figures representing policies and values for a given value. They are examples of a given value.

![policies](GridWorld&Maze/images_theory/tabular-Dyna-POLICY_9_2_49.png)

![values](GridWorld&Maze/images_theory/tabular-Dyna-Q_VALUES_9_2_49.png)

---
## Dyna-Q +

The Dyna-Q+ agent that did solve the shortcut maze uses one such heuristic. This agent keeps track for each state–action pair of how many time steps have elapsed since the pair was last tried in a real interaction with the environment. The more time that has elapsed, the greater (we might presume) the chance that the dynamics of this pair has changed and that the model of it is incorrect. To encourage behavior that tests long-untried actions, a special “bonus reward” is given on simulated experiences involving these actions.

One possible example of a grid changing world is showed in next image

![changing grid world](GridWorld&Maze/images_theory/DynaQ+.png)


The algorithm is in "DynaQ+ V1.py" and .pynb

---
## Prioritized Sweeping

The algorithm is shown below

![prioritized sweeping](GridWorld&Maze/images_theory/prioritized-sweeping.png)

The idea behind this algorithm is that we can work only in states that have changed recently and update states closed to him, that is, backward focusing. We have a queu of states and theirs predecessors, so if one state is changed, all its predecessors too. Thus we only update a few states each time.

The algorithm is in "prioritized sweeping V1.py" file




---
## Policy Approximation 

1. Short corridor with switched actions.

![short corridor](GridWorld&Maze/images_theory/short-corridor.png)

Algorithm can find in "short_corridor(chapter_13_policy_approximation).py"

## REINFORCE Monte Carlo Policy Gradient

Generic algorithm can see below

![Reinforce](GridWorld&Maze/images_theory/reinforce.png)

and code in "short_corridor(chapter_13_REINFORCE_MonteCarlo_Policy_Gradient_Control).py"




