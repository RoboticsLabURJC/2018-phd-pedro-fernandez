### PhD Pedro - [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/RoboticsLabURJC/2018-phd-pedro-fernandez)

### Table of contents

- [Weblog](#weblog)

  - [June 2023](#June-2023)
  - [May 2023](#May-2023)
  - [October 2022](#October)
  - [February 2022](#February)
  - [January 2022](#January)
  - [December 2021](#December)
  - [November 2021](#November)
  - [October 2021](#October)
  - [September 2021](#September)
  - [May 2021](#May)
  - [April 2021](#April)
  - [March 2021](#March)
  - [February 2021](#February)
  - [January 2021](#January)
  - [December 2020](#December)
  - [November 2020](#november)

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

## 

### June-2023

$$Training \ 5 - target \ speed \ 20km/h$$ 

Working with the height of the point of perception

Features

- Qlearning
- states: 8
- actions: 3
- Tonw07 circuit in the mountains, start position fixed
- Epochs: 1000
- 4 point of perception, at height x= [30, 60, 100, 120]

The speed at which we try to work is 20km/h, although the autopilot does it at a maximum of 30km/h. Due to Carla's accelerations (linear and angular) it is an average speed, reaching up to 26km/h during the circuit.

![Metrics](./plots/four_points.png)

Since there are more states, we make her train for 1000 epochs.
It seems that the fact of having 4 states makes it learn in a more solid way, consolidating what it has learned as it progresses in the epochs. Although, it only finish 1 lap of the circuit but it does not consolidate the result.

![Metrics](./plots/20230614-150612_ie_metrics.jpg)
![Metrics](./plots/20230614-150821_histogram_actions.jpg)


$$Training \ 4 - target \ speed \ 20km/h$$

Working with the height in the image of the point of perception

Features

- Qlearning
- states: 6
- actions: 3
- Tonw07 circuit in the mountains, start position fixed
- Epochs: 500
- 1 point of perception, at height x=20 pixels

We try to work at 20km/h, although the autopilot does it at a maximum of 30km/h. Due to Carla's accelerations (linear and angular) it is an average speed, reaching up to 26km/h during the circuit.

By having the point of perception very high in the image, the car learns to follow it, but with the problem that it changes lane more than we want. With more epochs we could have better results.
Now, it turns out that when you reach the top and start the downhill of the circuit, the point of x=20 loses the line and is reset, preventing you from continuing the training.

Trainings with different x give us different results
- X = 30 : also loses the line
- X = 40 : training does not get good results and never reaches the top
- X = 36 : the training is not as good as with x = 30 and does not finish reaching the top despite the small difference with x=30


In the image you can see the point of perception taken at x = 100
![Metrics](./plots/one_point.png)




### May-2023

- From 23 to 31

We are trying to increment the effective velocity to reach maximun allowed in the circuit, aprox. 30km/h



$$Training 3$$

- Task: Follow Lane
- City: Town07 mountain circuit
- Q-Learning algorithm
- States: simplified perception with 1 control point and **8** states (we reduce the number of states)
- Position of the control point on the x-axis: 100
- Actions: 3
- Target speed: **10** km/h (the range is taken from 8-14km/h)
- Reward function: distance to lane center
- Epochs: 500 or 12 hours
- Steps per epoch: 300
- Target Steps in this circuit with above target velocity: approx. 130 steps


![Metrics](./plots/20230531-105929_ie_metrics.jpg)

The training ended after 500 epochs 
In the graph on the top left, we find the reward and the steps in each iteration. We can observe that almost at the end of training, the agent achieved its best effort. 


Now, at higher speed, the car gets out of its lane much more time.

![Metrics](./plots/20230531-105935_lane_changed.jpg)



States and actions taken are very similar.

![Metrics](./plots/20230531-110051_histogram_actions.jpg)
![Metrics](./plots/20230531-111044_histogram_states.jpg)

Histogram of states shows eight states. Each one is 80 pixels long.

Finally, Q-table values are shown below where state - action (6,0), (7,2) and (8,2) are the most valuables. 
![Metrics](./plots/20230531-111034_qtable.jpg)



- From 16 - 22


$$Training 2$$
All features are the same as Training 1 except for velocity, which we set to maximum 10km/h. This velocity is taken going down the hill, while going up usually the car goes to 2-4 km/h. But now the margin is greater.


- Task: Follow Lane
- City: Town07 mountain circuit
- Q-Learning algorithm
- States: simplified perception with 1 control point and 16 states
- Position of the control point on the x-axis: 100
- Actions: 3
- Target speed: 10 km/h (but the range oscilates from 2-11 km/h)
- Reward function: distance to lane center
- Epochs: 500 or 12 hours
- Steps per epoch: 400
- Target Steps in this circuit with above target velocity: approx. 250 steps


![Metrics](./plots/20230522-155327_ie_metrics.jpg)

The training ended after 12 hours. 
In the graph on the top left, we find the reward and the steps in each iteration. There are some oscilation in results, getting the best rewards at the end of epochs. That is very visble in left below where we can reach the target almost in the last episodes. The time per epoch gives the same information.


Now, at higher speed, the car gets out of its lane much more time.

![Metrics](./plots/20230522-155336_lane_changed.jpg)



States and actions taken are very similar.

![Metrics](./plots/20230522-160243_histogram_actions.jpg)
![Metrics](./plots/20230522-160241_histogram_states.jpg)

Histogram of states shows state 8 (in the center of the lane) is taken the most, along with state 7 which is soft left.

Finally, Q-table values are shown below where state - action (6,0), (7,2) and (8,2) are the most valuables. 
![Metrics](./plots/20230522-160754_qtable.jpg)








$$Training 1$$
The below plots show the results of the training carried out with the following characteristics:

- Task: Follow Lane
- City: Town07 mountain circuit
- Q-Learning algorithm
- States: simplified perception with 1 control point, and 16 states
- Position of the control point on the x-axis: 100
- Actions: 3
- Target speed: 4km/h
- Reward function: distance to lane center
- Epochs: 500 or 12 hours
- Steps per epoch: 400
- Target Steps in this circuit with above target velocity: approx. 300 steps

![Metrics](./plots/20230522-115345_ie_metrics.jpg)

The training ended after 12 hours. 
In the graph on the top left, we find the reward and the steps in each iteration. If we look at the steps the agent takes, we see that there are oscillations until about training 150, and from there, the number of steps in each episode stays constant at the maximum. Despite that, there are a few oscillations around episode 300.
Likewise, the reward behaves similarly, and from episode 300 the reward is maximum. It coincides with the low values of the epsilon parameter, which can be seen in the graph on the upper right.

In the graph below on the left, we can see the distance it would take to reach our goal in each episode. Although there are oscillations up to episode 280, it is from there that we can see that in each episode the agent achieves the result

In the graph below to the right, the time the car takes in each episode is shown, and from iteration 280 the time is maximum in all of them, indicating that the objective has been achieved.




Within the task of staying in the lane, it is important to find out if the car has left it and how many times, which indicates good behavior. In the following graphs we can analyze the results. In some training sessions it comes out up to 32 times, but after 280 and it doesn't, behaving perfectly.

![Metrics](./plots/20230522-115352_lane_changed.jpg)



States and actions taken could be seen in next two graphics. In the first one, we can observe that action 0 (turn left) is taken the most, and one possible explanation is due to the circuit has one more left curve than right curves (3 vs. 2) and the agent learnt this behaviour

![Metrics](./plots/20230522-112953_histogram_actions.jpg)
![Metrics](./plots/20230522-112952_histogram_states.jpg)

Histogram of states shows state 8 (in the center of the lane) is taken the most, along with state 7 which is soft left.

Finally, Q-table values are shown below where state - action (6,0), (7,2) and (8,2) are the most valuables. 
![Metrics](./plots/20230522-111534_qtable.jpg)

- From 1 - 15

In this long period of time, we have been working in the Carla 0.9.13 environment, and taking everything implemented in Gazebo, this is the follow line and follow lane tasks.

At this moment, we have trained an agent in a complex circuit for the lane follow task, so that the vehicle manages to complete a target distance without leaving its lane.

The features of what we have implemented are:

- Task: follow lane
- Algorithm: Q-Learning
- States: a point of perception, obtained from a mask applied to the RGB camera sensor. In this way we obtain the center of the lane for a certain point of the x-axis of the image. In the examples shown below, x=100 pixels.
The center of the image is obtained from the right and center lines of the image, which in this part of the circuit are both continuous.
- Actions: 3 actions, 0=[0.4, -0.1], 1=[0.6,0.0], 2=[0.4, 0.1] where the values correspond to [throttle, steering]
- Reward function: depends only on the distance to the center of the lane

The chosen circuit corresponds to the city 7 of Carla, being a winding circuit with up and down slopes, which makes training very difficult.
The trainings have been executed during 1000 iterations with 500 steps in each of them.

Once the algorithm was able to converge and finish the circuit correctly, we tried to make an inference with the result of the Q table trained. The inference has been made in 3 different routes, within the same city due to the fact that the central line of the road is continuous, which ensures that we get the center correctly. There is no similar circuit in other cities of Carla. Therefore, what we have done has been to infer in 3 ways.

1. Infer in the same section of the training.

<div align="center">
  <a href="https://www.youtube.com/watch?v=nyCWHW-5rsI"><img src="https://img.youtube.com/vi/nyCWHW-5rsI/0.jpg"></a>
</div>

[![Youtube](https://img.youtube.com/vi/nyCWHW-5rsI/0.jpg)](https://www.youtube.com/watch?v=nyCWHW-5rsI "Inference in the same training lane circuit")




2. Infer on a different section of the training, with different curves and slopes

<div align="center">
  <a href="https://www.youtube.com/watch?v=7SAIw8kB0rI"><img src="https://img.youtube.com/vi/7SAIw8kB0rI/0.jpg"></a>
</div>

[![Youtube](https://img.youtube.com/vi/7SAIw8kB0rI/0.jpg)](https://www.youtube.com/watch?v=7SAIw8kB0rI "Inference in the same training lane circuit")



3. Infer in the same section of the training but in the opposite direction.

<div align="center">
  <a href="https://www.youtube.com/watch?v=HAXq4K7tWOU"><img src="https://img.youtube.com/vi/HAXq4K7tWOU/0.jpg"></a>
</div>


[![Youtube](https://img.youtube.com/vi/HAXq4K7tWOU/0.jpg)](https://www.youtube.com/watch?v=HAXq4K7tWOU "Inference in the same training lane circuit")





### October

During this time we have been working on integrating all the code in RL-Studio.
Now we have two complete tasks, follow lane and follow line, with different configurations.

### Task Follow Lane

The agent has to run over right lane, trying to avoid line center, and no surpass it.
Different configurations has been trained:

| Algorithm |             q-learn             |                        $DDPG^1$ |       $DDPG^2$ |                         $DDPG^3$ |
| --------- | :-----------------------------: | ------------------------------: | -------------: | -------------------------------: |
| State     | simplified perception (1 point) | simplified perception (1 point) | image as input |                   image as input |
| Actions   |        discrete (3 set)         |                      continuous |     continuous |                       continuous |
| Rewards   |            f(center)            |                       f(center) |      f(center) | f(center, linear velocity, time) |

You can watch training process for $DDPG^1$ with simplified perception


<div align="center">
  <a href="https://www.youtube.com/watch?v=5pq6hzku4w8"><img src="https://img.youtube.com/vi/5pq6hzku4w8/0.jpg"></a>
</div>

[![Youtube](https://img.youtube.com/vi/5pq6hzku4w8/0.jpg)](https://www.youtube.com/watch?v=5pq6hzku4w8)

and for $DDPG^2$ with image as a neural net input


<div align="center">
  <a href="https://www.youtube.com/watch?v=-GuPUjT2sy0"><img src="https://img.youtube.com/vi/-GuPUjT2sy0/0.jpg"></a>
</div>

[![Youtube](https://img.youtube.com/vi/-GuPUjT2sy0/0.jpg)](https://www.youtube.com/watch?v=-GuPUjT2sy0)

### Rewards

We created a set of different reward functions for Follow Lane and Follow Line tasks.

For Follow Line task:

- **In function of the center line**. The robot gets the most reward by being positioned in the center of the center line. As the error with respect to the center of the line increases, the rewards decrease, 2 and 1, respectively. In case of leaving the road, there is a strong penalty of -100 to avoid misbehavior.

$$
reward =
  \begin{cases}
    10       & \quad \text{distancetocenter } <= | 0.2 | \\
    2        & \quad |0.4| >= \text{distancetocenter } > |0.2|  \\
    1        & \quad |0.9| >= \text{distancetocenter } > |0.4|  \\
    -100     & \text{distancetocenter } > |0.9|  \\
  \end{cases}
$$

- **In function of linear velocity, angular velocity and distance of the center line**.

We can assume that linear velocity and angular velocity are linearly related: when one increases, the other decreases. On long straight lines, the linear velocity **v** will be high and the angular velocity **w** must be 0, while on curves, as w increases,v decreases. The formula that governs this assumption is given by:

$$w_{target} = \beta_0 - \beta_1v$$

Assuming that the smallest radius in meters of a curve in our circuits is 5 meters, then we move in ranges of $v=[2, 30]$ and $w=[-6, 6]$

So, we can easily compute the error between actual and desirable angular velocity
$$error = w_{target} - w_{goal}$$

and the distance to center, so the reward function comes from:

$$reward = \frac{1}{\textit{e}^{(error + distancetocenter)}} $$

For Follow Lane task:

- **In function of the center line**. The robot gets the most reward by being positioned in the center of the right lane. As the error with respect to the center of the line increases, the rewards decrease, 2 and 1, respectively. In case of leaving the road, there is a strong penalty of -100 to avoid misbehavior.

$$
reward =
  \begin{cases}
    10       & \quad 0.65 >= \text{distancetocenter } > 0.25 \\
    2        & \quad 0.9 > \text{distancetocenter } > 0.65  \\
    2        & \quad 0.25 >= \text{distancetocenter } > 0  \\
    1        & \quad 0 >= \text{distancetocenter } > -0.9  \\
    -100     & \text{distancetocenter } > |0.9|  \\
  \end{cases}
$$

- **In function of the center line, linear velocity and time to complete**. The idea is to force the agent going fast and try to finish the task as soon as posible. So, we rewarded fast velocities in center of lane positions, and penalize long steps. It has been added scalars, 10, 2 and 1, to maximize near the center.

$$
reward =
  \begin{cases}
    10 + velocity - ln(step)      & \quad 0.65 >= \text{distancetocenter } > 0.25 \\
    2 + velocity - ln(step)      & \quad 0.9 > \text{distancetocenter } > 0.65  \\
    2 + velocity - ln(step)        & \quad 0.25 >= \text{distancetocenter } > 0  \\
    1 + velocity - ln(step)        & \quad 0 >= \text{distancetocenter } > -0.9  \\
    -100     & \text{distancetocenter } > |0.9|  \\
  \end{cases}
$$

### February

Partial code is in the provisional repo until I can integrate it into RL Studio
https://github.com/RoboticsLabURJC/2018-phd-pedro-fernandez/rl-studio

- From 15 - 28

Training DDPG algorithm with different parameters and configurations. Always taking image as a state, playing with different segmentations, color reduction, neural nets configuration, or image size to get the best results. Actions always are continuous and new linear reward funtion to isolate from the environment, and being a function of linear velocity, angular velocity and center of image. Thus, the agent gets rewards optimizing the three params

- From 1 - 15

Integrating my RL Studio branch with main repo RL-Studio v1.1. Adding new deep deterministic policy gradient (DDPG) algorithm which allows working with continuous actions and multidimensional states space such as images

### January

- From 15 - 31

Training qlearn, dqn and ddpg algorithms with differents features set: images and simplified perception as states and discrete or contiuous actions.
Those parameters give us dozens of trainings, which serve us as learning

- From 1 - 15

Up to this point, for our F1 agent and the follow the line problem, we have different algorithms (Qlearning, Deep QLearning and DDPG), different input states (simplified perception and the raw image) and 2 types of actions for linear velocity and the angular (discrete and continuous).
In this period we are integrating all the parameters in a single yaml file to carry out the training in the simplest way.
The trainings that we are going to run are:

| States                                    | Discrete Actions     | Continuous Actions |
| ----------------------------------------- | -------------------- | ------------------ |
| SP (simplified perception (1 to n points) | qlearning, DQN, DDPG | DDPG               |
| image                                     | DQN, DDPG            | DDPG               |

Within the simplified perception (SP) we are going to train with 1, 3, 5 and N points.
Therefore we will have 19 training models. But we must bear in mind that within each of them there may be different parameters. For example, for SP1, we will have to play with different positions of the point within the height of the image. The same for sp3 and sp5. Nor for SPN since we take all points of the height in the image.

For discrete actions we will play with a small preset range. But for continuous actions we will have to try many other ranges. In the trainings that we have carried out with continuous actions with the DDPG algorithm and the raw image as the input state, we have verified that the agent does not achieve good results in the first 5000 episodes

### December

We finish implementing algorithm, with two continuous actions. Now we are training with differents hyperparams and configurations.

Good new is algorithm and architecture is working fine: we have 5 neural nets with more than 8 millions params each one, image as state, reward function is working and Actor-critic architecure is having good behavior. But we are not get our goals yet. There are a large number of hyperparams and params and in next days, we are going to tunind and tweaking to get the best configuration and try to finish simple circiut

### November

- From 1 to 15

We are integrating the developments made on my local machine, with the framework of the RL-Studio workgroup. In this way we can advance in a single and unified tool to create RL algorithms

https://github.com/keras-rl/keras-rl

https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

### October

- From 16 to 27th

In this period we are implementing the DDPG algorithm with continuous actions. So:

1. We follow the Keras API and I had to upgrade to TensorFlow 2.6.

2. we work with constant linear velocity and continuous angular velocity to analyze how the algorithm behaves

3. The environment provides us with the dimensions of the space, through the camera sensor, whereas before we provided it through the configuration file. This is an important advance in the future so that we do not depend on the parameter file and it is provided by the sensor.

4. The input images are reduced to 20% of the size of the original image so that the neural networks can process the data well. I had a memory allocation error that could be fixed in this way.
   Also, the images are normalized so that the color goes from 0 to 1. This is typical in NN image processing.
   For the moment we leave it like that to move forward, without going into analyzing how it could affect the recognition of the input status and the actions to take. It seems that the memory problems in Tamino are due to the training of other companions simultaneously.

5. At the moment, the rewards are being obtained from the pre-processing of the image, obtaining the lines and centers of the image to obtain the position of the car and be able to give it the reward values. Due to the previous point, where we change the size of the image, some constants have had to be varied by hand, within the corresponding functions. In next steps we will try to eliminate the preprocessing of the image so that the reward is obtained depending on the physics of the environment.

- From 1 to 15th

**Goal**:

In this work period, we are going to focus on starting to work with spaces for continuous actions.

So far, the actions that we have developed in our simulator of the circuit with a car, consists of a discrete set of actions designed ad hoc.

An example is in the following table, where we have 2 actions that allow us to handle our agent, the linear velocity v in m/s, and the angular velocity w in rad/s:

| Actions               |  0  |   1 |   2 |
| --------------------- | :-: | --: | --: |
| linear vel v (m/s)    |  3  |   2 |   2 |
| angular vel w (rad/s) |  0  |   1 |  -1 |

As you can see, the values ​​of each action are established by hand and based on the developer's experience with the simulator, which makes our algorithm behave appropriately for training and tests.
As we already know, the discretization of variables is the first step to understand the algorithms and see how they work, but it does not represent the real world that we are trying to simulate.

From now on, we are going to work with continuous actions. In this period we leave the constant linear velocity v, and we take the angular velocity w to be continuous.
Analyzing the behavior of a real Formula1 on circuits like Montmelo, we can see that the minimum race speed is 90km / h, which corresponds to the tightest curves on the circuit. With a curve radius of 10 meters (it is only an assumption because I have not been able to verify it), we have an angular velocity w = 2.5 rad / sec (approx) while the minimum obtained at maximum speeds with a very large curve radius, can be close to 0.
This gives us a continuous interval of w = [-2.5, +2.5]

Therefore we are going to work with 2 actions:

- v = 10 m/sec which will be constant in training. We will do tests with different values ​​of v to understand the behavior of the agent
- w = [-2.5, +2.5]

**Coninuous actions**

To work with continuous actions, there are 2 options widely used in the research community:

- use noisy perturbations, specifically an Ornstein-Uhlenbeck process for generating noise, as described in the paper. It samples noise from a correlated normal distribution. (https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process)

- Like in the TD Advantage Actor-Critic method, where the output are two scalar functions, μ (s) and σ (s), which are used as the mean and standard deviation of a Gaussian (normal) distribution. We will choose our actions by sampling from this distribution. The figure below shows the process with one neural net with two outputs, one for mean and other one for standard deviation.

![alt text](https://miro.medium.com/max/700/1*GlQCfOGm0slsuL-7qB_vYg.jpeg)

**Neural Networks**

Currently our neural network consists of a number of outputs that are delimited by our construction of the discrete set of actions. And with a softmax function we choose the one with the best probability for the network.
In the previous example, we have a set of 3 actions: 0, 1 and 2, where action 0 corresponds to v = 3 and w = 0 and so on.

Now we have to redesign the neural network so that it gives us 2 outputs, one for v and one for w. The neural network we build will support multi-label classification
Although, in this first moment the v will be constant, the new architecture of the neural network will allow us later to be able to use a continuous space for the v, and even to be able to add more actions to our agents easily.

One example is shown below with 2 categories, one for object color, and the other one for category. We will substitute for velocity and angular velocity:
![alt text](https://pyimagesearch.com/wp-content/uploads/2018/05/keras_multi_output_fashionnet_bottom.png?_ga=2.236320897.1058554134.1633706853-1748351799.1633706853)

**Reward and Image Preprocessing**

So far we have worked obtaining the reward through the image captured by the camera sensor placed on the agent. Once the information is received, the central point of the image gives us the rewards: the closer we are to the center of the image, the greater the reward.
Additionally, that rewards are established by the developer based on his experience, similar to how it has been done with actions above.

From now on, we will try to isolate any sensor from the RL algorithm to avoid image processing and to be able to scale our algorithms to other agents and to other environments.
Likewise, we are going to obtain the reward by isolating it from the image processing, by means of a formula applied to the physics of our experiments using the two variables that are moving our agent in this environment, the linear velocity and the angular velocity.
The reward to be obtained by the agent is given by the formula:

![formula](<https://render.githubusercontent.com/render/math?math=abs(|v|-v^(\frac{1}{\mathrm{-e}^{w}}))>)

where | v | represents linear velocity normalized.

So that mean, our state is represented by the input image obtained through the camera, and the reward is obtained from the linear velocity and the angular velocity, which are the 2 magnitudes that define the movement of our agent. That means that we do not preprocess the image and allow us to easily escalate to other problems.

**References in this period**:

- [Continuous control with Deep RL](https://arxiv.org/pdf/1509.02971.pdf)
- https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
- https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
- https://keras.io/examples/rl/ddpg_pendulum/
- https://towardsdatascience.com/building-a-multi-output-convolutional-neural-network-with-keras-ed24c7bc1178
- https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/

---

### September

The Deep QLearning (DQN) algorithm is already implemented in the RL-Studio framework following the algorithm developed by DeepMind for the Go game.

The most relevant features are:

- Two convolutional neural networks, one main and one secondary, where the weights of the inner network are updated in the latter every defined time to decrease the variance in the training data.
- Replay Memory where we store the information every certain number of training steps
- The neural network receives the raw image as input and generates the corresponding actions
- In order to train faster, there is a image processing with the OpenCv libraries that helps us to calculate the reward from the position of the car with respect to the center of the red line that is painted on the circuit
- Different positions and alternative displacement can be generated so that the car sees more situations and accelerates training.

Below several short videos of training in a simple circuit with an F1

Training 1

[![Alt text](https://img.youtube.com/vi/oTYWpTp0Lk8/0.jpg)](http://www.youtube.com/watch?v=oTYWpTp0Lk8)

Training 2

[![Alt text](https://img.youtube.com/vi/Z3VSA8sig0I/0.jpg)](http://www.youtube.com/watch?v=Z3VSA8sig0I)

Training 3

[![Alt text](https://img.youtube.com/vi/cRuxlHOrmJI/0.jpg)](http://www.youtube.com/watch?v=cRuxlHOrmJI)

TRaining 4

[![Alt text](https://img.youtube.com/vi/TmdUy6gfmoI/0.jpg)](http://www.youtube.com/watch?v=TmdUy6gfmoI)

The full code is in the next repo until I can integrate it into RL Studio
https://github.com/pjfernandecabo/rl-studio-arm64

---

### June

**Weeks from 1 to 15**

- Working on RL Studio: implementing in Mac AMD64 and Mac ARM64
- Running new algorithms: DQN, AC, PPO...

---

### May

**Weeks from 15 to 31**

- ROS Noetic tutorials
- Gazebo tutorials

**Weeks from 1 to 15**

Ending tabular TD methods: Qlearning, SARSA and expected SARSA in Mountain Car Open AI environment. Getting conclusions in different runnings playing with different hyperparams to fully understand each one behavior.

Going on with Ruben environment to Gazebo

---

### April

**Weeks from April 16 to 30 April**

Working on new RL environment, trying to connet it to Gazebo

**Weeks from April 1 to 15**

Trying to replicate my workmate's Ruben Lucas dedicated RL environment, both in Ubuntu and Mac. Ruben has differents projects which have implemented in his own RL environment: https://roboticslaburjc.github.io/2020-phd-ruben-lucas/install/

---

### March

**Weeks from 16 to 31 March**

Installing main architecture in Ubuntu and Mac with Gazebo 11, Gym Gazebo and ROS 2

**Weeks from 1 to 15 March**

Installing working platform and simulators to implement inside RL algorithms

- Ubuntu 20.04, ROS Noetic, Gym Gazebo 2, Gazebo 11
- Mac Big sur, ROS 2, Gazebo 11

Learning stochastic and deterministic policy gradient algorithms to solve continuos actions and states:

- https://spinningup.openai.com/en/latest/user/algorithms.html
- https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
- https://danieltakeshi.github.io/new-start-here.html

Following [lilianweng](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html), then we show the main characteristics of some of the SOTA policy based algorithms:

- REINFORCE (Monte-Carlo policy gradient) using episode samples to update the policy parameter θ. It relies on a full trajectory (MC).

A widely used variation of REINFORCE is to **subtract a baseline value** from the return Gt to reduce the variance of gradient estimation while keeping the bias unchanged. It is a **on-policy** method

- Actor-Critic, **on-policy** method as MC before: : training samples are collected according to the target policy.

- Off-policy Policy gradient:

1. The off-policy approach does not require full trajectories and can reuse any past episodes (**“experience replay”**) for much better sample efficiency.
2. The sample collection follows a behavior policy different from the target policy, bringing better **exploration**.

This algorithm works with a behavior policy and a target policy. Importance weight is used to calculate the gradient.

- A3C Asynchronous Advantage Actor-Critic (Mnih et al., 2016), short for A3C, is a classic policy gradient method with a special focus on parallel training: the critics learn the value function while multiple actors are trained in parallel and get synced with global parameters from time to time.

- A2C is a synchronous, deterministic? version of A3C. A coordinator in A2C waits for all the parallel actors to finish their work before updating the global parameters and then in the next iteration parallel actors starts from the same policy.

- DPG, Deterministic policy gradient (DPG), models the policy as a deterministic decision: a=μ(s).

- DDPG (Deep Deterministic Policy Gradient) is a **model free** and **off-policy** actor-critic. Combine DPG with DQN:

experience replay to stabilize the learning

frozen target network

continuous space

learns a deterministic policy

while learns, it builds an exploratory policy

soft updates on the parameters actor and critic algorithms

batch normalization in every dimension across samples in one minibatch

- D4PG (Distributed Distributional DDPG):

Distributional critic

N-step returns

Multiple distributed parallels actors

Prioritized Experience Replay

- MADDPG (Multi Agent DDPG) s an actor-critic model redesigned particularly for handling such a changing environment and interactions between agents. For one agent, the environment is non-stationary as policies of other agents are quickly upgraded and remain unknown.
  To mitigate the high variance triggered by the interaction between competing or collaborating agents in the environment, MADDPG proposed one more element - policy ensembles:

Train K policies for one agent;
Pick a random policy for episode rollouts;
Take an ensemble of these K policies to do gradient update.
In summary, MADDPG added three additional ingredients on top of DDPG to make it adapt to the multi-agent environment:

Centralized critic + decentralized actors;
Actors are able to use estimated policies of other agents for learning;
Policy ensembling is good for reducing variance.

---

Next algorithms go in sequence:

- TRPO (TRust region policy optimization) avoid parameter updates that change the policy too much at one step. It carries out this idea by enforcing a KL divergence constraint on the size of policy update at each iteration. TRPO can guarantee a monotonic improvement over policy iteration.

- PPO (proximal policy optimization) acting in a similar way as TRPO, but imposes teh constraint by forcing to stay in a small interval around 1.

- PPG (Phasic policy gradient) modifies the traditional on-policy actor-critic policy gradient algorithm. precisely PPO, to have separate training phases for policy and value functions.

---

- ACER (actor critic with Experience Replay) off-policy technique. It is counterpart of A3C on-policy method. It uses Retrace Q-value estimation, importance weigthts truncation and efficient TRPO

- ACTKR (actor-critic using Kronecker-factored trust region) proposed to use Kronecker-factored approximation curvature (K-FAC which uses natural gradient) to do the gradient update for both the critic and actor.

- SAC (soft actor-critic) is a off-policy, where incorporates the entropy measure of the policy into the reward to encourage exploration. It has:

An Actor-critic arquitecture

An off-policy formulation

Entropy maximization leads to policies that can (1) explore more and (2) capture multiple modes of near-optimal strategies (i.e., if there exist multiple options that seem to be equally good, the policy should assign each with an equal probability to be chosen).

- SAC with Automatically Adjusted Temperature, as SAC is brittle with respect to the temperature parameter. Unfortunately it is difficult to adjust temperature, because the entropy can vary unpredictably both across tasks and during training as the policy becomes better. An improvement on SAC formulates a constrained optimization problem: while maximizing the expected return, the policy should satisfy a minimum entropy constraint.

- TD3 (Twin Delayed Deep Deterministic) applies improves in DDPG to prevent the overestimation of the value function:

Clipped Double Q-Learning

Delayed update of Target and Policy Networks

Target Policy Smoothing

- SVPG (Stein Variational Policy Gradient) applies the Stein variational gradient descent (SVGD) algorithm to update the policy parameter

- IMPALA (importance Weighted Actor-Learner Architecture) Multiple actors generate experience in parallel, while the learner optimizes both policy and value function parameters using all the generated experience.

---

### February

**Weeks from 16 to 28 February**

1. Install platform in Mac Big Sur:

- [Gazebo 11.3.0](http://gazebosim.org/tutorials?cat=install&tut=install_on_mac&ver=11.0)
- [ROS2](https://index.ros.org/doc/ros2/Installation/Crystal/macOS-Install-Binary/)
- Gym Gazebo
- [Ignition Gazebo](https://ignitionrobotics.org/docs/all/getstarted)

2. Documentation in:

- [Gazebo](http://gazebosim.org/tutorials?cat=connect_ros)
- [ROS2](https://index.ros.org/doc/ros2/Tutorials/Configuring-ROS2-Environment/)

3. Really good tutorial in:

- [Open Source Robotics: Getting Started with Gazebo and ROS 2](https://www.infoq.com/articles/ros-2-gazebo-tutorial/)

References in this period:

- [Reward Engineering for Classic Control Problems on OpenAI Gym |DQN |RL](https://towardsdatascience.com/open-ai-gym-classic-control-problems-rl-dqn-reward-functions-16a1bc2b007), very beginning article but with a seminal idea as how **changing the rewards** we can get better results, optimizing convergence.

- [Actor Critic implementation with TensorFlow](https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic) and with [Keras](https://keras.io/examples/rl/)
- [DQN with Pythorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

**Weeks from 1 to 15 February**

We introduce in continuous actions and states. Unlike discrete actions, now continuous open ne procedures, algorithms and differents ways to do the analysis and almost all research drives to deep learning as a way to find features and parameters in search policy function.

References in this period:

- TFM Alex cabañeros UPC: Autonomous vehicle navigation with deep reinforcement
- Reinforcement Learning, Sutton 2018, chapter 13
- [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1509.02971.pdf). Timothy P. Lillicrap, Jonathan J. Hunt,Alexander Pritzel, Nicolas Heess,Tom Erez, Yuval Tassa, David Silver & Daan Wierstra. Google Deepmind
- [Deep Reinforcement Learning in Continuous Action Spaces: a Case Study in the Game of Simulated Curling](http://proceedings.mlr.press/v80/lee18b/lee18b.pdf). Kyowoon Lee, Sol-A Kim, Jaesik Choi, Seong-Whan Lee ; Proceedings of the 35th International Conference on Machine Learning, PMLR 80:2937-2946, 2018
- [Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution](https://www.ri.cmu.edu/publications/improving-stochastic-policy-gradients-in-continuous-control-with-deep-reinforcement-learning-using-the-beta-distribution/), Po-Wei Chou, Daniel Maturana and Sebastian Scherer.
- [Reinforcement Learning with Exploration by Random Network Distillation](https://towardsdatascience.com/reinforcement-learning-with-exploration-by-random-network-distillation-a3e412004402), Or Rivlin - Medium
- [Natural actor–critic algorithms](https://www.sciencedirect.com/science/article/pii/S0005109809003549), Shalabh Bhatnagar, Richard S. Sutton, Mohammad Ghavamzadeh, Mark Lee

- [Deep Reinforcement Learning That Matters](https://ojs.aaai.org/index.php/AAAI/article/view/11694), Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep Reinforcement Learning That Matters. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1)
- [Actor-critic algorithms. In Advances in neural information processing systems (pp. 1008-1014).](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.821.1075&rep=rep1&type=pdf) Konda, V. R., & Tsitsiklis, J. N. (2000).
- [Policy gradient methods for reinforcement learning with function approximation. In NIPs (Vol. 99, pp. 1057-1063).](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.6.696&rep=rep1&type=pdf)Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (1999, November).
- [Reinforcement learning in continuous action spaces through sequential monte carlo methods. Advances in neural information processing systems, 20, 833-840.](http://chercheurs.lille.inria.fr/~lazaric/Webpage/Publications_files/lazaric2008reinforcement.pdf) Lazaric, A., Restelli, M., & Bonarini, A. (2007).
- [Deterministic policy gradient algorithms. In International conference on machine learning (pp. 387-395). PMLR.](http://proceedings.mlr.press/v32/silver14.html) Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014, January).

- [Optimizing expectations: From deep reinforcement learning to stochastic computation graphs (Doctoral dissertation, UC Berkeley).](https://escholarship.org/uc/item/9z908523) Schulman, J. (2016).
- [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)

---

### January

**Weeks from 18 to 31 January**
Continuing developing GridWorlds, trying to reach textbook chapter 13. This period we focus on discrete worls, understanding value functions, policy search and models worlds, the three main approaches to every RL agent can have.

**Week from 4 to 18 January**

We are going to work with discrete and continuous examples as before.

- GridWorld & maze. Show images of agent with different configurations of state space, obstacles and Start and end points. Create graphics and results of implementation. Algorithms are in directory /GridWorld&Maze

- MountainCar the same as above point.

---

### December

**Week from 17 to 31 december**

We are going to work in two simple projects:

- Grid World with Robot

  - https://github.com/qqiang00/Reinforce/tree/master/reinforce
  - https://github.com/imraviagrawal/Reinforcement-Learning-Implementation
  - https://github.com/adik993/reinforcement-learning-sutton
  - https://github.com/adityajain07/ReinforcementLearning-Gridworld
  - https://github.com/LyWangPX/Reinforcement-Learning-2nd-Edition-by-Sutton-Exercise-Solutions

- Mountain car

- Some other interesting gyms:
  - https://github.com/MattChanTK/gym-maze
  - https://github.com/maximecb/gym-minigrid
  - https://github.com/maximecb/gym-miniworld

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

- [World Models](https://arxiv.org/abs/1803.10122) && [here](https://worldmodels.github.io/) && [Hallucinogenic Deep Reinforcement Learning Using Python and Keras](https://medium.com/applied-data-science/how-to-build-your-own-world-model-using-python-and-keras-64fb388ba459)

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

### November

**Week from 2 to 7**

I am going to read TFM's Ignacio Arranz in 📂 **/Users/user/PhD**
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

### General Resources

### Useful links JdRobot

- Ignacio Arranz Master Thesis Work in: https://roboticslaburjc.github.io/2019-tfm-ignacio-arranz/
- BehaviorStudio project from JdRobot: https://github.com/JdeRobot/BehaviorStudio
- Gym-Gazebo 2 project from JdRobot: https://github.com/JdeRobot/gym-gazebo-2

### Courses

- [Coursera] [RL Specialization University of Alberta](https://www.coursera.org/specializations/reinforcement-learning)
- [Coursera] [Practical Reinforcement Learning by National Research University Higher School of Economics](https://www.coursera.org/learn/practical-rl?specialization=aml)
- [Coursera] [Reinforcement Learning in Finance New York University](https://www.coursera.org/learn/reinforcement-learning-in-finance)
- [Udacity][ reinforcement learning](https://www.udacity.com/course/reinforcement-learning--ud600): by Georgia Tech, free available
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
- [Pros and cons of RL state of art (2018)][deep reinforcement learning doesn't work yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)
