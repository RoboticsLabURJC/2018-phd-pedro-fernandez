### Sources - [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/RoboticsLabURJC/2018-phd-pedro-fernandez)




### Table of contents


  - [Example 3.5 GridWorld](#3-5-GRID)
  - [Example 3.8 GridWorld with optimal Policy and Value Function](#3-8-GRID)
  - [Example 4.1 GridWorld Iterative Policy Evaluation (prediction)](#4-1-iteration-policy)
  - [Example 5.4 off policy ordinary importance sampling](#5-4-off-policy-MonteCarlo)
  - [Example 6.2 Temporal Differences (0) Random Walk](#TD0)
  - [Example 6.3 Windy GridWorld SARSA](#SARSA)


  - [Tabular Dyna-Q](#TABULAR-DYNA-Q)
  - [Dyna-Q +](#Dyna-Q-+)
  - [Prioritized Sweeping](#Prioritized-Sweeping)
  - [Policy Approximation](#Policy-approximation)
  - [REINFORCE Monte Carlo Policy Gradient](#REINFORCE-MC)
---

Ideas and code taking from: [2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com) and 2016 Kenta Shimada(hyperkentakun@gmail.com)](https://github.com/LyWangPX/Reinforcement-Learning-2nd-Edition-by-Sutton-Exercise-Solutions)
 


First implementation of a RL algoritmhs. It is a typical and beginner example for understanding basics ideas and concepts of RL.

We are going to follow the TextBook [Sutton]


# Part One: simple didactic algorithms 
## Example 3.5 GridWorld

Finite episodic MDP (Markov Decision Processes) with a grid of 5x5 states, and 4 equiprobables actions : left, right, up, down. Every action gives us 0 reward, except in those states which actions move the agent out, which results in -1, or in A state with reward +10, and B state, with reward +5. Discount rate of 0.9

[![3.5](https://img.youtube.com/vi/yR71faR0FMI/0.jpg)](https://www.youtube.com/watch?v=yR71faR0FMI)



The final Value Function generated is showing below:

![valueFunction](images/3_5_GridWorld_value_function.png)

Code is here: [3_5_GridWorld_(simple_finite_MDP_Example_3_5).ipynb](3_5_GridWorld_(simple_finite_MDP_Example_3_5).ipynb)



## Example 3.8 GridWorld with optimal Policy and Value Function

Following exercise 3.5 to find optimal policy and Value function in a discrete space with four discrete equiprobables actions. The framework is the same seen before, but now, we use optimal Bellman equations

[![3.5](https://img.youtube.com/vi/YZA0osDwUPM/0.jpg)](https://www.youtube.com/watch?v=YZA0osDwUPM)

The optimal Value Function generated is showing below:

![optimalvalueFunction](images/3_8_bestValueFunction.png)

and optimal policy


![optimalpolicy](images/3_8_optimalPolicy.png)


Code is in: [3_8_GridWorld_(simple_finite_MDP_with_Bellma_optimality_equations_Example_3_8).ipynb](3_8_GridWorld_(simple_finite_MDP_with_Bellma_optimality_equations_Example_3_8).ipynb)




## Example 4.1 GridWorld Iterative Policy Evaluation (prediction)

Next example follows algorithm to find Optimal Policiy in a deterministic and discrete state and actions world. There are 4 equiprobables actions with no discount rate in the problem. Algorithm is shown below

![policyevaluation](images_theory/4_1_iterative_policy_evaluation.png)

Result with best Value Function obtained:

![bestvaluefuntion](images/figure_4_1.png)


[![4.1](https://img.youtube.com/vi/BvdCEA3C5us/0.jpg)](https://www.youtube.com/watch?v=BvdCEA3C5us)


Code is in: [4_1_GridWorld_(optimal_policy_figure_4.1).ipynb](4_1_GridWorld_(optimal_policy_figure_4.1).ipynb)



## Example 5.4 off policy ordinary importance sampling

First off-policy algorithm in a simple world with a single state and two actions, one of then with different probability to be taken. 
Off-policy means we have to deal with 2 policies, Target policy which is the policy to be learned, and Behavior policy which generates behavior. 
As now we have 2 differents policies, we have to deal with different probability distributions, so importance sampling is the technique to estimating expected values under one distribution given samples form another. 
In this exercise, we work with ordinary importance sampling, where importance sampling is done as a simple average.

The algorithm is useful as a intro to off policy methods, and to understand their concepts.
In this case, the target policy and the behavior policy have to taken the same states at least one, being behavior policy more exploratory. 
The target policy becomes a deterministic optimal policy and the behavior remains stochastic, such as a greedy policy. 


[![4.1](https://img.youtube.com/vi/YliO0xXp8Z0/0.jpg)](https://www.youtube.com/watch?v=YliO0xXp8Z0)


Code is in: [5_4_off_policy_one_state_MDP.ipynb](5_4_off_policy_one_state_MDP.ipynb)


# Part Two: more complex algorithms, beginning with Temporal Differences (TD)

## Example 6.2 Temporal Differences (0) Random Walk

This is the first example of Temporal Difference algorithm. Such as algorithms have several important characteristics:
- no need models 
- off-policy or on-policy
- online interactions with envs
- bootstrap, means they get estimations over estimations, or guess from a guess.
- they depend only on next state to have info and rewards, unlike MC methods where they come into the end of the episode.





[![6.2](https://img.youtube.com/vi/A1MUJYEOj5w/0.jpg)](https://www.youtube.com/watch?v=A1MUJYEOj5w)


Code is in: [6_2_Random_walk(comparing_alphaMC_vs_TD(0).ipynb](6_2_Random_walk(comparing_alphaMC_vs_TD(0).ipynb))

## Example 6.3 Windy GridWorld SARSA

SARSA means State Action Reward State Action where we can try to learn the values of state action-pairs. The algorithm is show below
![SARSA](images_theory/6_3_SARSA.png)


The main equation is in

>Q(S,A) = Q(S,A) + alfa (R + gammaQ(S',A') - Q(S,A)) 



[![6.3](https://img.youtube.com/vi/GQb24xN3hyI/0.jpg)](https://www.youtube.com/watch?v=GQb24xN3hyI)


Code is in: [6_3_windy_gridworld(e-Greedy_SARSA).ipynb](6_3_windy_gridworld(e-Greedy_SARSA).ipynb)




## (UNDER CONSTRUCTION)
## Tabular Dyna-Q

The algorithm is
![tabular Dyna-Q](GridWorld&Maze/images_theory/Tabular-Dyna-Q.png)

Model and Planning verion that it is called Tabular Dyna-Q. 

If we take only until (d) step, it is called a **direct RL** or **Q-learning** algorithm. And steps (e) and (f) it is the Tabular Dyna-Q with Model and Planning version, which represents next figure

![model and planning](GridWorld&Maze/images_theory/model-and-planning.png)

![Dyna-Q architecture](GridWorld&Maze/images_theory/DynaQ-architecture.png)





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

---
## REINFORCE Monte Carlo Policy Gradient

Generic algorithm can see below

![Reinforce](GridWorld&Maze/images_theory/reinforce.png)

and code in "short_corridor(chapter_13_REINFORCE_MonteCarlo_Policy_Gradient_Control).py"




