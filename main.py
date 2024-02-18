from classes import Blob
from classes import QLearningAgent
from classes import GameEnvironment
import numpy as np
from PIL import Image
import cv2
import time
import pickle

# Pickle q table
start_q_table = "qtable-size7-pretrained.pickle"    # None for fresh or load filename (q_table)

# Settings
BOARD_SIZE = 7              # BOARD_SIZE x BOARD_SIZE
FULL_EPISODES = 1000        # training loops
MAX_STEPS_EPISODE = 200     # maximum amount of steps per episode
MOVE_PENALTY = 1            # -reward
ENEMY_PENALTY = 300         # -reward
FOOD_REWARD = 300           # reward
epsilon = 0                 # epsilon is not constant
EPS_DECAY = 0.999998        # epsilon * EPS_DECAY
LEARNING_RATE = 0.1         # learning rate
DISCOUNT = 0.99             # discount factor
SHOW_EVERY = 1              # visual presentation every n episode
TIME_STEP_MS = 100          # how long each step is presented n milliseconds

# Color settings
blob_colors = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
PLAYER_COLOR = 1  # player
FOOD_COLOR = 2  # food
ENEMY_COLOR = 3  # enemy

# Initialize objects
agent = QLearningAgent(board_size=BOARD_SIZE, start_q_table=start_q_table, learning_rate=LEARNING_RATE, discount=DISCOUNT, epsilon=epsilon, eps_decay=EPS_DECAY)
env = GameEnvironment(board_size=BOARD_SIZE, episodes=FULL_EPISODES, move_penalty=MOVE_PENALTY, enemy_penalty=ENEMY_PENALTY, food_reward=FOOD_REWARD, show_every=SHOW_EVERY)

# Start Iteration
episode_rewards = []

for episode in range(FULL_EPISODES):
    player_positions = set()    #Dont let food and enemy spawn on player
    player = Blob(board_size=BOARD_SIZE)
    player_positions.add((player.x, player.y))

    # Ensure food does not spawn at the player's position
    food = env.spawn_blob(player_positions, blob_colors[2])
    player_positions.add((food.x, food.y))

    # Ensure enemy does not spawn at the player's or food's position
    enemy = env.spawn_blob(player_positions, blob_colors[3]) # Enemy

    episode_reward = 0
    for i in range(MAX_STEPS_EPISODE):
        obs = (player-food, player-enemy)
        
        # Epsilon-greedy action
        if np.random.random() > epsilon:
            action = np.argmax(agent.q_table[obs])
        else:
            action = np.random.randint(0, 4)
        # Execute chosen action
        player.action(action)

        # Food and enemy random actions (uncomment to enable)
        #enemy.move()
        #food.move()

        # rewards
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        
        # Calculate relative state and max q value
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(agent.q_table[new_obs])
        current_q = agent.q_table[obs][action]

        # FOOD_REWARD is positive terminal state
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        # CALC q value for non terminal step
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        # Update new_q to q_table
        agent.q_table[obs][action] = new_q

        # Update reward and epsilon decay
        episode_reward += reward
        episode_rewards.append(episode_reward)
        epsilon *= EPS_DECAY

        # Terminal condition
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            print(f"Episode reward: {episode_rewards[-1]}")
            break

        # For visualization purposes
        if episode % SHOW_EVERY == 0:
            show = True
        else:
            show = False

        # Just for Visualization
        if show:
            visualization_env = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)
            visualization_env[food.x][food.y] = blob_colors[FOOD_COLOR]
            visualization_env[player.x][player.y] = blob_colors[PLAYER_COLOR]
            visualization_env[enemy.x][enemy.y] = blob_colors[ENEMY_COLOR]
            img = Image.fromarray(visualization_env, 'RGB')
            img = img.resize((1000, 1000), Image.NEAREST)
            cv2.imshow("QAgent", np.array(img))
            # Show game
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):     # terminal wait key 500 ms hardcoded
                    break
            else:
                if cv2.waitKey(TIME_STEP_MS) & 0xFF == ord('q'):     # adjust visualization time for each step
                    break

# Save q_table results
with open(f"qtable-size{BOARD_SIZE}-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(agent.q_table, f)