### Collaboration and Competition

## Introduction

In this project the task was to solve the environment where 2 agents play table tennis against each other. 
To solve this task I've used [Multi Agents Deep Deterministic Policy Gradient](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). In this paper Critic is augmented with extra information about other agent's policies whereas Actor has information  only about current agent. Agents also share Experience Replay Buffer.

## Learning Algorithm

Multi Agent DDPG is an adaptation of DDPG algorithm for the multiple agents. Here, all agents have access to each others' state observations and actions during Critic training but use only their own state observation to predict their actions. This ensures that the environment is stationary from the agent's point of view. Since the Critic is centralised, all agents' states and actions are concatenated and sent to Critic network as an input. The network outputs Q-value for that state. Then Q-value is used to train Actor for each agent which outputs predicted action values. This [link](https://medium.com/brillio-data-science/improving-openai-multi-agent-actor-critic-rl-algorithm-27719f3cafd4) provides more detailed summary of the algorithm.

## Model Architectures and Hyperparameters Used

In terms of Actor and Critic network architectures, I've started by using the same ones as in the previous assignments. Namely, two hidden layers, first layer with 400 and second layer with 300 neurons. But the models did not train well. After that I've reduced number of neurons in both layers to 256 and tried introducing Batch Normalization. It turned out that Normalization improved training only when used in the first hidden layer. With regards to training, I've started by using different samples of experience (from the same Experience Replay buffer) for each agent, but that did not work very well so I've adjusted the code to use exactly the same experiences for both agents and it improved training. The environment was solved in 4261 episodes. 

In terms of parameters, here are the list I've used in `multi_agent_ddpg.py` file:</br>

`BUFFER_SIZE = int(1e6)  # replay buffer size`</br>
`BATCH_SIZE = 256        # minibatch size`</br>
`GAMMA = 0.99            # discount factor`</br>
`TAU = 1e-3              # for soft update of target parameters`</br>
`LR_ACTOR = 1e-4         # learning rate of the actor`</br>
`LR_CRITIC = 1e-3        # learning rate of the critic`</br>
`WEIGHT_DECAY = 0        # L2 weight decay`</br>
`FC1_UNITS = 256         # Number of Neurons in the 1st fully connected layer`</br>
`FC2_UNITS = 256         # Number of Neurons in the 2nd fully connected layer`</br>

In addition, adding extra layers to either Actor or Critic networks did not improve learning. I've also tried using batch normalization on more than 1 layer but it did not help either. The more interesting aspect was the role of noise in the training process. To begin with, I've left the noise for the whole training process but agents were not learning well. Then, I've decided to _turn off_ the noise after some time and it turned out to improve training as well. Its probably because after some training and making _noisy_ decisions agents have explored enough and removing noise improves training further.</br>

![Training Scores](https://github.com/Sarunas-Girdenas/drlnd_tennis/blob/master/train.png)

As we can see, models did not train well for nearly 4000 episodes and then suddenly the scores jumped (at about 4000th episode). I think one possible explanation could be that the noise might have been switched off at about that time so agents predictions have became less noisy and training improved.


## Future Work

To improve the the training further, we could try the following:</br>
1. Prioritized Experience Replay: sampling experiences that resulted in larger errors more frequently would speed up the training.</br>
2. Different architectures for Actor and Critic Networks.</br>
3. Use gradient clipping to stabilize training.</br>