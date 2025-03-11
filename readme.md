# Reinforcement Learning of Variable Admittance Control for Human-Robot Co-Manipulation
Authors: Mason Smith, Neeraj Chandratre, Vikramaditya Nanhai, Yousef Soltanian





## Abstract
In this project, a Fuzzy Q-Learning (FQL) algorithm
is used to dynamically change the damping values in a variable
admittance controller for human robot co-manipulation tasks.
The controller is used on a real UR5 robot that is physically
coupled with a human participant during a co-manipulation
target tracking task. The end effector is moved by the human
from the start to the goal position, while the reinforcement
learning (RL) algorithm learns to adapt the damping value
based on the force applied by the human. Four subjects each
perform 50 iterations of the experiment. The goal of the robotic
FQL-agent is to reduce jerk and achieve a smooth trajectory
while moving towards a target with an unknown position. The
UR5 robot arm is unaware of the goal position during training
but observes higher order dynamics and interaction forces.
The results are promising, with a significant reduction in the
amount of effort required by humans. 

A video of the training procedure using the proposed RL algorithm implemented on a
UR5 robot can be found here: [https://youtu.be/Tp1SyK3Iteo](https://youtu.be/Tp1SyK3Iteo).

For a full technical writeup of this project see [Reinforcement_in_robotics_project.pdf](https://github.com/user-attachments/files/19172852/Reinforcement_in_robotics_project.pdf)

![Fig_ExperimentalSetup](https://github.com/user-attachments/assets/8f899f5d-80fa-4c23-940d-7726737fc59f)

**Fig 1:** Experimental setup with a UR5 robot and human
partner attempting to jointly move the end effector to the goal
position (shaded red).


## Introduction
Human-Robot co-manipulation has been a major contributor
over the past few years. Earlier, during the industrial revolu-
tion, man and machine interacted without much intelligence
aspects, and humans mainly relied on operating the machines
and trying to get the best from them. Since the beginning of
development in robotics and intelligent machines, the concept
of human-robot interaction has evolved, and today it is an
essential part of our lives. It involves the use of robots for
doing complex tasks by humans, and also it includes doing
tasks with humans. The former term is helpful in rehabilitation,
critical, harsh, mentally and physically challenging, and in
complex environments. The latter term includes all of these,
along with robots interacting with humans for entertainment
and doing normal daily life tasks which are doable by humans.
In this project, the robot can help humans to do complex tasks
like lifting heavy objects, doing precise co-manipulation in
assembly, or robot surgery. The project includes making the
robot follow a 1 Dimensional path, which has start and end
points known to the human but unknown to the robot. The
robot has to learn to traverse this path with minimum jerks
and by providing damping so that the motion is smooth.

## Method

### Admittance Control
For the task we have considered, the end effector is the point of contact between the human and robot arm, where the human is the leader, and the robot is a follower. The motion of the end effector is restricted to a single dimension to perform the task of moving it from one point to the other. In this task, no information about the target position is available to the robot. The controller's goal is to adjust the  velocity of the end effector by varying the damping value. This task is divided into two sections:
1) High velocity and low precision when the target is far, requiring a low damping value.
2) Low velocity and high precision when the target is near, requiring a high damping value.

The admittance controller can be represented in the Cartesian frame of the end effector as:
$\mathbf{M}_d \dot{\mathbf{V}}_{ref} + \mathbf{C}_d \mathbf{V}_{ref}  = \mathbf{F}_H$, where $\mathbf{M}_d$ and $\mathbf{C}_d$ are admittance controller gains, positive diagonal matrices and represent the desired inertia and the damping.$\mathbf{V}_{ref}$ is the reference Cartesian velocity of the end effector and $\mathbf{F}_h$ is the measured external force by the operator. Due to the effectiveness of the damping parameter on human-robot co-operation, it is considered for further calculations and training.

### Fuzzy Q-Learning
Instead of using discrete sets that would yield
the problem intractable, FQL realizes state representation
using fuzzy states, constituted by the fuzzy set $S$. The agent
can visit a state partially, in the sense that the real-valued
input variables $X = [V, F_h, \dot{V} , \dot{F_h}]$ may belong up to a
degree to the membership functions of the fuzzy sets.
All combinations of the input membership functions form the rule base, but unlike the standard fuzzy systems, the actions are selected and not combined. In FQL, the conclusion of each rule $R_i, i = 1, ..., n$, where $n$ is the number of rules, is a crisp action $a_i'$ selected from the set of discrete actions $A$, according to the policy $\pi$. The action set $A$ consists of a
discrete number of crisp damping values. The selected action by each rule $R_i$ contributes to a continuous global action $U_t$, which is the damping value $C_d$ provided to the admittance controller, according to the premise strength $\phi_i$ of that rule. The Fuzzy Inference System (FIS) is implicitly used as a function approximation of the Q value functions.

Crisp actions $a_i'$ are selected from the following hybrid directed-undirected exploration strategy in [1,2].

The selected damping is the global action $U_t$ given by the following equation:

$U_t(X_t)=\sum\limits_{i=1}^n a_i'\phi_i$

A Q-function quantifies the quality of a given action with
respect to the current state and is given by:

$Q_t(X_t,U_t)=\sum\limits_{i=1}^n q_t(S_i,a_j)'\phi_i$

The optimal action for a rule is given by the Q*-function:

$Q_t(X_t)=\sum\limits_{i=1}^n max(q_t(S_i,a_j)'\phi_i)$

The q-values are updated at each iteration of the algorithm
according to:

$q_{t+1}=q_t(S_i,a_j)+\beta \epsilon_{t+1} e_t(S_i,a_j)$

where $\beta$ is the learning rate, $e_t$ the eligibility trace of the past
visited rules and $\epsilon_{t+1}$ is the Temporal Difference (TD) error given by:

$\epsilon_{t+1}=r_{t+1}+\gamma Q_t^*(X_t)-Q_t(X_t,U_t)$

The term $r_{t+1}$ is the reward received at time $t + 1$ and $\gamma$ is a discount factor that weights the effect of the future rewards. 
The goal of the FQL agent is to regulate the damping accordingly, so as to maximise the reward $r_{tot}$, which is the opposite of the non-negative jerk $J$ throughout the movement:

$r_{tot}=\sum\limits_{i=1}^{t_f} r_i=-\sum\limits_{i=1}^{t_f} \dddot{x}^2=-J^2$

The reward to the agent is provided at a frequency 10 times slower than the frequency that the admittance control is being done. This frequency is experimentally found to allow an accumulated reward signal that is more robust to the noise generated by the numeric differentiation for $\dddot{x}$.
## Results

![1951cc321c6af1028176c54a4f8fed9215836130](https://github.com/user-attachments/assets/a3dfb720-6847-4000-90ad-b9f9adeab196)

**Fig 2:** Results of the learning algorithm converging for each subject over 50 episodes. The top plot shows the cumulative jerk for each episode while the bottom plot shows decreasing episode duration in seconds.  The line in each plot shows the mean value while the shaded region shows $\pm$ one standard deviation.

![Fig_Dynamics](https://github.com/user-attachments/assets/5530618b-4b35-4404-8392-0615598bf26c)

**Fig 3:** Results from physical experiments. The top plot shows the observed end effector velocity $\dot{x}$ and the optimal MJT velocity $\dot{x}_{MJT}$ profile for comparison. The middle plot the continuous robot damping action applied over the course of the episode. The bottom plot shows the sensed human force $F_H$, the robot damping force $F_R$ and the net force $F_{net}$ applied to the end effector.
    The solid line for each plot represents the mean value over the final three episodes of all four participants (12 total episodes) while the shaded region shows $\pm$ one standard deviation. Here, all data were compared on a normalized time index to account for trials with different durations. 

## Conclusion
The project gives us insight that the cumulative reward obtained is highest when the human subject gives consistent force, followed by maintaining proper stopping positions (by not applying force when under goal position.) We find that the damping values are high during the start or end of the trajectory and are smaller during the times of high velocity movements.

The final experiments for 4 trials with 2 subjects resulted in a much better minimum jerk trajectory. The policy converged to the optimal between 35-40 iterations. The last 5 iterations required minimum effort from the human, and the time required to complete the episode also decreased. However, no trend in energy exerted was observed over the 50 episodes.

This project provided promising results for practical human-robot cooperative tasks with minimum efforts for day-to-day tasks. The algorithm was considered successful when the damping was regulated to get the minimum jerk, and cooperation became smooth. Fig. 2 shows an improvement to the human's trajectory in terms of the defined jerk-based cost function.
## References
[1] M. J. Er and C. Deng, “Online tuning of fuzzy inference systems using
dynamic fuzzy q-learning,” IEEE Transactions on Systems, Man, and
Cybernetics, Part B (Cybernetics), vol. 34, no. 3, pp. 1478–1489, 2004.

[2] L. Jouffe, “Fuzzy inference system learning by reinforcement methods,”
IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applica-
tions and Reviews), vol. 28, no. 3, pp. 338–355, 1998.
## Running the Code
To view simulation run the "Simulation/run.py" script

The controller and online learning algorithm can be found in the UR5_ROS_workspace directory. Custom implmentation of the learning algorithm can be found in "UR5_ROS_workspace/src/Compliant-Control-and-Application/control_algorithms/admittance/src/Admittance/"

Contact:mosmith3@asu.edu

