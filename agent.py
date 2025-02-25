import torch
import random
import numpy as np
from collections import deque
from src.game import SnakeGameAI, Direction, Point
from src.model import Linear_QNet, QTrainer
from src.helper import plot
import os


MAX_MEMORY = 100_000
BATCH_SIZE = 256
BLOCK_SIZE = 20
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # controls the randomness
        self.gamma = 0.95 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if we exceed memory
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        if game.green_apples:
            closest_apple = min(game.green_apples, key=lambda apple: abs(apple[0] - head.x // BLOCK_SIZE) + abs(apple[1] - head.y // BLOCK_SIZE))
        else:
            closest_apple = (head.x // BLOCK_SIZE, head.y // BLOCK_SIZE)
        
        food_left = closest_apple[0] < head.x // BLOCK_SIZE
        food_right = closest_apple[0] > head.x // BLOCK_SIZE
        food_up = closest_apple[1] < head.y // BLOCK_SIZE
        food_down = closest_apple[1] > head.y // BLOCK_SIZE

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
        
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            food_left,
            food_right,
            food_up,
            food_down
            ]
        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))   

    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
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

def train():
    plot_score = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
            
            if agent.n_games % 10 == 0:
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
    train()
    plt.show()