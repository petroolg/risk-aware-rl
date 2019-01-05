# Risk-aware reinforcement learning

The repository contains scripts for safe reinforcement learning in autonomous driving environment. 
It includes the implementation of an MDP with an autonomous driving environment, modifiable reward function specification and the possibility of empirical model learning.


# Policy initialization techniques
The module contains implementations of two initialization algorithms:
- Behavioral Cloning (BC)
- Generative Adversarial Imitation Learning (GAIL) [1]

# Risk-aware reinforcement learning
The module contains implementations of two reinforcement learning algorithms:
- Q-learning with risk-directed exploration [2]
- Policy Gradient with variance constraint [3]

# Get started
The code is compatible with python 3.6. Install the requirements with 

`pip install -r requirements.txt`

## Policy initialization 
Policy initialization algorithms need expert demonstrations to run. The repository consists of 350 trajectories sampled manually. They are stored in the folder `policy_initialization/trajectories`.  
</br>
To run BC script from the folder `policy_initialization/`:</br>
`python BC_agent.py --tp trajectories`</br>
</br>
To run GAIL script from the folder `policy_initialization/`:</br>
`python GAIL_agent.py --tp trajectories`</br>
</br>
Scripts output performance metrics, such as graphs and/or text.
</br>
## Risk-aware reinforcement learning
To run Q-learning in model-free mode* from the folder `risk_aware_rl/`:</br>
`python q_learning_agent.py`.</br>
</br>
To run Policy Gradient in risk-neutral mode* from the folder `risk_aware_rl/`: </br>
`python policy_gradient_agent.py`

</br>
Implementations are based on: </br>
[1] Ho, Jonathan and Ermon, Stefano. “Generative Adversarial Imitation Learning”. In: (June 2016). eprint: 1606.03476. url: https://arxiv.org/pdf/1606.03476. </br>
[2] L.M. Law, Edith. “Risk-directed Exploration in Reinforcement Learn- ing”. MA thesis. Montreal, Quebec: McGill University, Feb. 2005. </br>
[3] Castro, Dotan Di, Tamar, Aviv, and Mannor, Shie. “Policy Gradients with Variance Related Risk Criteria”. In: (June 2012). eprint: 1206. 6404. url: https://arxiv.org/pdf/1206.6404. </br>
</br>
</br>
\* for more modes and parameters access script's help:</br>
`pyhton <script_name> -h`
