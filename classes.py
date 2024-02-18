import numpy as np
import pickle


# Blob class definition
class Blob:
    """
    Randomly place the blob within the environment's boundaries and enable actions.
    """
    def __init__(self, board_size, color=(255, 255, 255)):
        self.board_size = board_size
        self.x = np.random.randint(0, board_size)
        self.y = np.random.randint(0, board_size)
        self.color = color

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        if choice == 0:  # right
            self.move(x=1)
        elif choice == 1:  # left
            self.move(x=-1)
        elif choice == 2:  # up
            self.move(y=1)
        elif choice == 3:  # down
            self.move(y=-1)

    def move(self, x=None, y=None):
        # Allow only 1 step movement for Food and Enemy
        if x is None and y is None:
            direction = np.random.choice(['x', 'y'])  # Choose an axis to move along
            step = np.random.choice([-1, 1])  # Choose the direction of movement on that axis
            if direction == 'x':
                self.x += step
            else:  # direction == 'y'
                self.y += step

        # Allow only 1 step movement for Player (action)
        else:
            if x is not None:
                self.x += x
            if y is not None:
                self.y += y

        # Boundary conditions
        self.x = max(0, min(self.x, self.board_size - 1))
        self.y = max(0, min(self.y, self.board_size - 1))

# QLearningAgent class definition
class QLearningAgent:
    """
    Learning agent creating and updating q_table.
    """
    def __init__(self, board_size, start_q_table, learning_rate, discount, epsilon, eps_decay):
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.eps_decay = eps_decay

        if start_q_table is None:
            self.q_table = self.init_q_table()
        else:
            with open(start_q_table, "rb") as f:
                self.q_table = pickle.load(f)

    def init_q_table(self):
        q_table = {}
        for x1 in range(-self.board_size + 1, self.board_size):
            for y1 in range(-self.board_size + 1, self.board_size):
                for x2 in range(-self.board_size + 1, self.board_size):
                    for y2 in range(-self.board_size + 1, self.board_size):
                        q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for _ in range(4)]  #x1, y1 relative pos food to player #x2,y2 relative pos enemy to player
        return q_table

    def select_action(self, state):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.randint(0, 4)
        return action

    def update_q_values(self, state, new_state, action, reward):
        max_future_q = np.max(self.q_table[new_state])
        current_q = self.q_table[state][action]
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount * max_future_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        self.epsilon *= self.eps_decay


class GameEnvironment:
    """
    Board and colours
    """
    def __init__(self, board_size, episodes, move_penalty, enemy_penalty, food_reward, show_every):
        self.board_size = board_size
        self.episodes = episodes
        self.move_penalty = move_penalty
        self.enemy_penalty = enemy_penalty
        self.food_reward = food_reward
        self.show_every = show_every
        self.colors = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}  # 1:Player, 2:Food, 3:Enemy

    # To make sure there is no overlapping spawns
    def spawn_blob(self, exclude_positions, color):
        while True:
            blob = Blob(self.board_size, color=color)
            if (blob.x, blob.y) not in exclude_positions:
                return blob
