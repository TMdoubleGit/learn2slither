import torch
import random
import numpy as np
from collections import deque
from src.displayer import Displayer
from src.game import SnakeGameAI, Direction, Point
from src.model import Linear_QNet, QTrainer
from src.helper import plot
from src.constants import BLOCK_SIZE
from src.interpreter import Interpreter
import os
import argparse


MAX_MEMORY = 100_000
BATCH_SIZE = 256
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # controls the randomness
        self.gamma = 0.95 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if we exceed memory
        self.model = Linear_QNet(16, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.interpreter = None

    def get_state(self, game):
        """
        Uses the Interpreter to calculate the state of the environment based on the snake's vision.

        Args:
            game (SnakeGameAI): The current game instance.

        Returns:
            np.ndarray: The state of the environment as a NumPy array.
        """
        if self.interpreter is None:
            self.interpreter = Interpreter(game)
        
        return self.interpreter.get_state()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))   

    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(10, 80 - self.n_games)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move=random.randint(0,2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move

def train(training_sessions=None):
    plot_score = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    displayer = Displayer(game, agent)

    while training_sessions is None or agent.n_games < training_sessions:
        state_old = agent.get_state(game)

        displayer.display(None, agent, state_old, None, False)

        final_move = agent.get_action(state_old)

        action_dir = Direction.RIGHT
        if np.array_equal(final_move, [1, 0, 0]):
            action_dir = game.direction
        elif np.array_equal(final_move, [0, 1, 0]):
            clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            action_dir = clockwise[(clockwise.index(game.direction) + 1) % 4]
        elif np.array_equal(final_move, [0, 0, 1]):
            clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            action_dir = clockwise[(clockwise.index(game.direction) - 1) % 4]


        reward, done, score, new_direction = game.play_step(final_move)
        state_new = agent.get_state(game)

        displayer.display(action_dir, agent, state_new, reward, True)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
            
            if agent.n_games % 1000 == 0:
                model_folder_path = './models'
                if not os.path.exists(model_folder_path):
                    os.makedirs(model_folder_path)
                new_dir_path = "./models/game_" + str(agent.n_games)
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
                agent.model.save(f"./models/game_{agent.n_games}/model_{agent.n_games}.pth")
                plot_path = f"{new_dir_path}/plot_{agent.n_games}.png"
                plot(plot_score, plot_mean_score, save_path=plot_path)


            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_score.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_score, plot_mean_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Snake AI agent.')
    parser.add_argument('--sessions', type=int, default=None, help='Number of training sessions (default: None for infinite training)')
    args = parser.parse_args()
    train(training_sessions=args.sessions)