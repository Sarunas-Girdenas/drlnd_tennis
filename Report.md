### Collaboration and Competition

In this project the task was to solve the environment where 2 agents play table tennis against each other. 
To solve this task I've used [Multi Agents Deep Deterministic Policy Gradient](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). In this paper Critic is augmented with extra information about other agent's policies whereas Actor has information  only about current agent. Agents also share Experience Replay Buffer.

To begin with, I've started by using different samples of experience (from the same Experience Replay buffer) for each agent, but that did not work very well so I've adjusted the code to use exactly the same experiences for both agents and it improved training.

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

To improve the the training further, we could try the following:</br>
1. Prioritized Experience Replay: sampling experiences that resulted in larger errors more frequently would speed up the training.</br>
2. Different architectures for Actor and Critic Networks.</br>
3. Use gradient clipping to stabilize training.</br>