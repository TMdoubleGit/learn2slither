import torch
import random
import numpy as np
from collections import deque
from src.displayer import Displayer
from src.game import SnakeGameAI, Direction, Point
from src.model import Linear_QNet, QTrainer
from src.helper import plot
from src.constants import BLOCK_SIZE
import os
import argparse


MAX_MEMORY = 500_000
BATCH_SIZE = 1_024
LR = 0.0001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.record = 0
        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.001
        self.gamma = 0.99
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(16, 256, 4)
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
        return game.interpreter.get_state()


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
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.n_games)
        final_move = [0, 0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        if random.uniform(0, 1) < self.epsilon:
            top_actions = torch.argsort(prediction, descending=True)[:3]
            move = random.choice(top_actions).item()
        else:
            move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move

def train(training_sessions=None, model_path=None):
    plot_score = []
    plot_mean_score = []
    total_score = 0
    agent = Agent()

    if model_path:
        if os.path.exists(model_path):
            metadata = agent.model.load(model_path)
            agent.n_games = metadata.get('n_games', 0)
            agent.record = metadata.get('record', 0)
            agent.epsilon = metadata.get('epsilon', 1.0)
            agent.max_epsilon = metadata.get('max_epsilon', 1.0)
            agent.min_epsilon = metadata.get('min_epsilon', 0.01)
            agent.decay_rate = metadata.get('decay_rate', 0.001)
            agent.gamma = metadata.get('gamma', 0.99)
            agent.model.eval()

    game = SnakeGameAI()
    displayer = Displayer(game, agent)

    while training_sessions is None or agent.n_games < training_sessions:
        state_old = agent.get_state(game)

        env_old = displayer.display(None, state_old, None, None, False)

        final_move = agent.get_action(state_old)

        # action_dir = Direction.RIGHT
        if np.array_equal(final_move, [1, 0, 0, 0]):
            action_dir = game.direction
        elif np.array_equal(final_move, [0, 1, 0, 0]):
            clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            action_dir = clockwise[(clockwise.index(game.direction) + 1) % 4]
        elif np.array_equal(final_move, [0, 0, 1, 0]):
            clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            action_dir = clockwise[(clockwise.index(game.direction) - 1) % 4]
        elif np.array_equal(final_move, [0, 0, 0, 1]):
            clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            action_dir = clockwise[(clockwise.index(game.direction) + 2) % 4]

        reward, done, score, new_direction = game.play_step(final_move)
        state_new = agent.get_state(game)

        displayer.display(action_dir, state_new, reward, env_old, True)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > agent.record:
                agent.record = score
            
            if agent.n_games % 1000 == 0:
                model_folder_path = './models'
                if not os.path.exists(model_folder_path):
                    os.makedirs(model_folder_path)
                new_dir_path = "./models/game_" + str(agent.n_games)
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
                metadata = {
                    'n_games': agent.n_games,
                    'record': agent.record,
                    'epsilon': agent.epsilon,
                    'max_epsilon': agent.max_epsilon,
                    'min_epsilon': agent.min_epsilon,
                    'decay_rate': agent.decay_rate,
                    'gamma': agent.gamma
                }
                agent.model.save(f"./models/game_{agent.n_games}/model_{agent.n_games}.pth", metadata=metadata)
                plot_path = f"{new_dir_path}/plot_{agent.n_games}.png"
                plot(plot_score, plot_mean_score, save_path=plot_path)


            plot_score.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_score, plot_mean_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Snake AI agent.')
    parser.add_argument('--sessions', type=int, default=None, help='Number of training sessions (default: None for infinite training)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a previously saved model (default: None)')
    args = parser.parse_args()
    train(training_sessions=args.sessions, model_path=args.model_path)