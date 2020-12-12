### PhD Pedro [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)



### Table of contents

  - [Weblog](#weblog)
    - [Noviembre](#noviembre)
  - [Resources](#resources)
    - [Useful links JdRobot](#useful-links-jdrobot)
    - [Courses](#courses)
    - [Lectures](#lectures)
    - [TextBooks](#textbooks)
    - [Tutorials/Tasks](#tutorialstasks)
    - [Links](#links)
    - [Repos](#repos)
    - [Papers](#papers)
    - [Open Source Reinforcement Learning Platforms](#open-source-reinforcement-learning-platforms)
    - [Applications](#Applications)

 
---
## Weblog
---
### November

**Week from 2 to 7**


I am going to read TFM's Ignacio Arranz in 📂   **/Users/user/PhD**
This TFM is my foundation in this part of research. I'll take all code and ideas to begin with.

Next week Proposal: Install and execute all parts, frameworks, libraries and "play" with them.


**Week up to 30 November**

Goal: play with OpenAI Gym Pendulum.

The explanations of environments are here:
- https://github.com/openai/gym/wiki/Pendulum-v0
- https://mspries.github.io/jimmy_pendulum.html
- https://github.com/ZhengXinyue/Model-Predictive-Control/blob/master/Naive_MPC/Pure_MPC_Pendulum.py

- https://gym.openai.com/docs/
- https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
- https://leonardoaraujosantos.gitbook.io/artificial-inteligence/artificial_intelligence/markov_decision_process


I finally implemented Pendulum and cartPole, even with NN in Pendulum.

---
### December

**Week from 1 to 16 december**

- DQN. I am going to work wiht DQN. First steps only learn and undestand its implementation. I will follow DeepMind paper: [Human-level control through deep reinforcement learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning)

- [Installing Gazebo and ROS in MacOS 10.15.7 (Catalina)](http://gazebosim.org/tutorials?cat=install&tut=install_on_mac)

- Install Ubuntu20.04, Python 3.7, Gazebo 11, BehaviorStudio, Ros Noetic (lastest Version). See details in Install.md


- Experience Replay

### Lectures and URLs of the week

- [Reinforcement Learning with ROS and Gazebo](https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial7/README.md)

- [Erle Robotics: Accelerated Robot Training through Simulation
with ROS and Gazebo](https://roscon.ros.org/2018/presentations/ROSCon2018_AcceleratedRobotTraining.pdf)

- [gym-gazebo2](https://github.com/AcutronicRobotics/gym-gazebo2)

- [Tensor Flow and Neural Nets (in Spanish)](https://relopezbriega.github.io/blog/2016/06/05/tensorflow-y-redes-neuronales/)
- [RL with Bayesian approach (just for curiosity)](https://towardsdatascience.com/coordinating-multiple-rl-agents-on-overcooked-ee1b4ce11054) and [here](https://arxiv.org/pdf/2003.11778.pdf)

- [Deep Reinforcement Learning - an overview](https://arxiv.org/pdf/1701.07274.pdf)
- [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/pdf/1708.05866.pdf)

- https://github.com/vmayoral/basic_reinforcement_learning
- https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner
- https://github.com/keras-rl/keras-rl
- https://github.com/Lazydok/RL-Pytorch-cartpole
- https://github.com/erlerobot/gym-gazebo && https://github.com/AcutronicRobotics/gym-gazebo2
- [Extending the OpenAI Gym for robotics: a toolkit
for reinforcement learning using ROS and Gazebo](https://arxiv.org/pdf/1608.05742.pdf)

- [Toward Self-Driving Bicycles Using State-of-the-Art Deep Reinforcement Learning Algorithms](https://www.mdpi.com/2073-8994/11/2/290/htm)

- [A Beginner's Guide to Deep Reinforcement Learning](https://wiki.pathmind.com/deep-reinforcement-learning)

- [World Models](https://arxiv.org/abs/1803.10122) && [here](https://worldmodels.github.io/)

- [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)

- [Google X’s Deep Reinforcement Learning in Robotics using Vision](https://hackernoon.com/google-xs-deep-reinforcement-learning-in-robotics-using-vision-7a78e87ab171) && [QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation](https://arxiv.org/abs/1806.10293) && [here](https://ai.googleblog.com/2018/06/scalable-deep-reinforcement-learning.html)

- [Aprendizaje por Refuerzo: Procesos de Decisión de Markov — Parte 1](https://medium.com/aprendizaje-por-refuerzo-introducci%C3%B3n-al-mundo-del/aprendizaje-por-refuerzo-procesos-de-decisi%C3%B3n-de-markov-parte-1-8a0aed1e6c59) y [Aprendizaje por Refuerzo: Procesos de Decisión de Markov — Parte 2](https://medium.com/aprendizaje-por-refuerzo-introducci%C3%B3n-al-mundo-del/aprendizaje-por-refuerzo-procesos-de-decisi%C3%B3n-de-markov-parte-2-d219358ecd76) y [Aprendizaje por Refuerzo: Planificando con Programación Dinámica](https://medium.com/aprendizaje-por-refuerzo-introducci%C3%B3n-al-mundo-del/aprendizaje-por-refuerzo-planificando-con-programaci%C3%B3n-din%C3%A1mica-200ebd2af48f) y [Aprendizaje por Refuerzo: Predicción Libre de Modelo](https://medium.com/aprendizaje-por-refuerzo-introducci%C3%B3n-al-mundo-del/aprendizaje-por-refuerzo-predicciones-sin-modelo-45e66528aa98) y [Aprendizaje por Refuerzo: Control Libre de Modelo](https://medium.com/aprendizaje-por-refuerzo-introducci%C3%B3n-al-mundo-del/aprendizaje-por-refuerzo-control-libre-de-modelo-e175f50217a) y [Aprendizaje por Refuerzo: Aproximación de Función de Valor](https://medium.com/aprendizaje-por-refuerzo-introducci%C3%B3n-al-mundo-del/aprendizaje-por-refuerzo-aproximaci%C3%B3n-de-funci%C3%B3n-de-valor-61b8f5e22e21)

- [Reinforcement learning tutorial with TensorFlow](https://adventuresinmachinelearning.com/reinforcement-learning-tensorflow/) && [Policy Gradient Reinforcement Learning in TensorFlow 2](https://adventuresinmachinelearning.com/policy-gradient-tensorflow-2/) && [Prioritised Experience Replay in Deep Q Learning](https://adventuresinmachinelearning.com/prioritised-experience-replay/)

- [Vanilla Deep Q Networks](https://towardsdatascience.com/dqn-part-1-vanilla-deep-q-networks-6eb4a00febfb) && [Double Deep Q Networks](https://towardsdatascience.com/double-deep-q-networks-905dd8325412)

- [Simple reinforcement learning methods to learn CartPole](http://kvfrans.com/simple-algoritms-for-solving-cartpole/)

- [Q-learning: Aprendizaje automático por refuerzo](https://rubenlopezg.wordpress.com/2015/05/12/q-learning-aprendizaje-automatico-por-refuerzo/)

- [Aprendizaje por refuerzo: algoritmo Q Learning](http://www.cs.us.es/~fsancho/?e=109)

- [Entrenamiento de Redes Neuronales: mejorando el Gradiente Descendiente](http://www.cs.us.es/~fsancho/?e=165)

- [DQN from Scratch with TensorFlow 2](https://levelup.gitconnected.com/dqn-from-scratch-with-tensorflow-2-eb0541151049)

---
### General Resources

### Useful links JdRobot

- Ignacio Arranz Master Thesis Work in: https://roboticslaburjc.github.io/2019-tfm-ignacio-arranz/
- BehaviorStudio project from JdRobot: https://github.com/JdeRobot/BehaviorStudio
- Gym-Gazebo 2 project from JdRobot: https://github.com/JdeRobot/gym-gazebo-2


### Courses

- [Coursera] [RL Specialization University of Alberta](https://www.coursera.org/specializations/reinforcement-learning)
- [Coursera] [Practical Reinforcement Learning by National Research University Higher School of Economics](https://www.coursera.org/learn/practical-rl?specialization=aml)
- [Coursera] [Reinforcement Learning in Finance New York University](https://www.coursera.org/learn/reinforcement-learning-in-finance)
- [Udacity][ Reinforcement Learning](https://www.udacity.com/course/reinforcement-learning--ud600): by Georgia Tech, free available
- [Stanford] [CS234: Reinforcement Learning - Winter 2020](http://web.stanford.edu/class/cs234/index.html)
- [David Silver's Reinforcement Learning Course - UCL(University College London) course on RL](https://www.davidsilver.uk/teaching/)


### Lectures
- [Berkeley] [Lectures for [UC Berkeley] CS 285 Fall 2020: Deep Reinforcement Learning](https://www.youtube.com/playlist?list=PL_iWQOsE6TfURIIhCrlt-wj9ByIVpbfGc)

### TextBooks

- [Reinforcement Learning: An Introduction, second edition](http://incompleteideas.net/book/RLbook2018.pdf). Richard S. Sutton and Andrew G. Barto. The MIT Press (https://mitpress.mit.edu/books/reinforcement-learning-second-edition) [[Code]](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)



### Tutorials/Tasks




### Links
- [Applications of Reinforcement Learning in Real World](https://towardsdatascience.com/applications-of-reinforcement-learning-in-real-world-1a94955bcd12): an intro to RL world with very very useful links
- [An Outsider's Tour of Reinforcement Learning](http://www.argmin.net/2018/06/25/outsider-rl/)


- [Awesome Reinforcement Learning:](https://github.com/aikorea/awesome-rl) is a entry door to entire RL world
- [Implementation of RL algorithms:](https://github.com/dennybritz/reinforcement-learning) with many notebooks in main subjects such as TD, MDP, Q

### Repos
- [Udacity RL](https://github.com/udacity/deep-reinforcement-learning)


### Papers



### Open Source Reinforcement Learning Platforms
- [OpenAI gym](https://github.com/openai/gym) - A toolkit for developing and comparing reinforcement learning algorithms
- [OpenAI universe](https://github.com/openai/universe) - A software platform for measuring and training an AI's general intelligence across the world's supply of games, websites and other applications
- [DeepMind Lab](https://github.com/deepmind/lab) - A customisable 3D platform for agent-based AI research
- [Project Malmo](https://github.com/Microsoft/malmo) - A platform for Artificial Intelligence experimentation and research built on top of Minecraft by Microsoft
- [ViZDoom](https://github.com/Marqt/ViZDoom) - Doom-based AI research platform for reinforcement learning from raw visual information
- [Retro Learning Environment](https://github.com/nadavbh12/Retro-Learning-Environment) - An AI platform for reinforcement learning based on video game emulators. Currently supports SNES and Sega Genesis. Compatible with OpenAI gym.
- [torch-twrl](https://github.com/twitter/torch-twrl) - A package that enables reinforcement learning in Torch by Twitter
- [UETorch](https://github.com/facebook/UETorch) - A Torch plugin for Unreal Engine 4 by Facebook
- [TorchCraft](https://github.com/TorchCraft/TorchCraft) - Connecting Torch to StarCraft
- [garage](https://github.com/rlworkgroup/garage) - A framework for reproducible reinformcement learning research, fully compatible with OpenAI Gym and DeepMind Control Suite (successor to rllab)
- [TensorForce](https://github.com/reinforceio/tensorforce) - Practical deep reinforcement learning on TensorFlow with Gitter support and OpenAI Gym/Universe/DeepMind Lab integration.
- [tf-TRFL](https://github.com/deepmind/trfl/) - A library built on top of TensorFlow that exposes several useful building blocks for implementing Reinforcement Learning agents.
- [OpenAI lab](https://github.com/kengz/openai_lab) - An experimentation system for Reinforcement Learning using OpenAI Gym, Tensorflow, and Keras.
- [keras-rl](https://github.com/matthiasplappert/keras-rl) - State-of-the art deep reinforcement learning algorithms in Keras designed for compatibility with OpenAI.
- [BURLAP](http://burlap.cs.brown.edu) - Brown-UMBC Reinforcement Learning and Planning, a library written in Java
- [MAgent](https://github.com/geek-ai/MAgent) - A Platform for Many-agent Reinforcement Learning. 
- [Ray RLlib](http://ray.readthedocs.io/en/latest/rllib.html) - Ray RLlib is a reinforcement learning library that aims to provide both performance and composability.
- [SLM Lab](https://github.com/kengz/SLM-Lab) - A research framework for Deep Reinforcement Learning using Unity, OpenAI Gym, PyTorch, Tensorflow.
- [Unity ML Agents](https://github.com/Unity-Technologies/ml-agents) - Create reinforcement learning environments using the Unity Editor
- [Intel Coach](https://github.com/NervanaSystems/coach) - Coach is a python reinforcement learning research framework containing implementation of many state-of-the-art algorithms.
- [Microsoft AirSim](https://microsoft.github.io/AirSim/docs/reinforcement_learning/) - Open source simulator based on Unreal Engine for autonomous vehicles from Microsoft AI & Research.



## Applications

- [Resource Management with Deep Reinforcement Learning](https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf) for computer clusters
- [Reinforcement learning-based multi-agent system for network traffic signal control](http://web.eecs.utk.edu/~ielhanan/Papers/IET_ITS_2010.pdf) for Traffic Control

- [End-to-End training of deep visuomotor policies](https://www.youtube.com/watch?v=Q4bMcUk6pcw&feature=emb_logo) in Robotics

- [Web system Configuration](http://ranger.uta.edu/~jrao/papers/ICDCS09.pdf)

- [Chemistry in optimizing chemical reactions with Deep Reinforcement Learning](https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00492) and its [Github](https://github.com/lightingghost/chemopt)

- [News Personalized Recommendations](http://www.personal.psu.edu/~gjz5038/paper/www2018_reinforceRec/www2018_reinforceRec.pdf)
- [Real-Time Bidding with Multi-Agent Reinforcement Learningin Display Advertising](https://arxiv.org/pdf/1802.09756.pdf)
- Games such as [AlphaGo](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) , [AlphaGo Zero](https://deepmind.com/blog/article/alphago-zero-starting-scratch), [playing Atari of DeepMind](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and [LSTM DL (DRQN)](https://arxiv.org/pdf/1507.06527.pdf)
- [GANs and RL](https://arxiv.org/pdf/1804.01118.pdf) from [DeepMind](https://www.youtube.com/watch?v=N5oZIO8pE40)
- [Pros and cons of RL state of art (2018)][Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)










