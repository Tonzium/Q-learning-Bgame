This project implements a simple grid-based reinforcement learning environment where agents, represented as "blobs", navigate to achieve goals such as reaching food while avoiding enemies.
The core of the project is built around three main components: the Blob class, the QLearningAgent class, and the GameEnvironment class. The system utilizes Q-learning, a model-free reinforcement learning algorithm, to enable the agent to learn from interactions with the environment.

Features

1 ) Blob Entities: Autonomous entities that can move within a grid environment. Each blob can perform actions such as moving up, down, left, or right.

2 ) Game Environment: A grid-based world where blobs interact. The environment supports spawning blobs at random locations, including food and enemies.

3 ) QLearningAgent: Implements the Q-learning algorithm, allowing the blob designated as the player to learn optimal actions based on rewards received for reaching food and penalties for encountering enemies.

4 ) Reward System: Configurable rewards and penalties to shape the learning behavior of the agent, including rewards for reaching food and penalties for hitting enemies or performing unnecessary moves.

5 ) Visual Representation: Basic visual output using OpenCV to display the state of the environment after each action, showing the positions of the player, food, and enemies.
