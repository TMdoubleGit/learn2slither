import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from src.constants import RED_APPLE, GREEN_APPLE, WALL, EMPTY, NEGATIVE_REWARD, POSITIVE_REWARD, SMALLER_NEGATIVE_REWARD, BIGGER_NEGATIVE_REWARD

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 150, 0)
GREEN2 = (0, 255, 0)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 10


class SnakeGameAI:
    
    nb_red_apples = 1
    nb_green_apples = 2

    def __init__(self, w=BLOCK_SIZE * 12, h=BLOCK_SIZE * 12):
        self.w = w
        self.h = h

        self.board = [
            [WALL if (x == 0 or x == 11 or y == 0 or y == 11) else EMPTY
             for x in range(self.w // BLOCK_SIZE)]
            for y in range(self.h // BLOCK_SIZE)
            ]

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        

    def reset(self):
        self.direction = Direction.RIGHT

        self.board = [
            [WALL if (x == 0 or x == 11 or y == 0 or y == 11) else EMPTY
             for x in range(self.w // BLOCK_SIZE)]
            for y in range(self.h // BLOCK_SIZE)
            ]

        self.init_snake()
        if (
            self.nb_red_apples +
            self.nb_green_apples +
            len(self.snake)
            >
            (self.h - 2) * (self.w - 2)
        ):
            raise ValueError("Too many apples or snake is too long")
        
        self.score = 0
        self.green_apples = []
        self.current_green_apples = 0
        for _ in range(self.nb_green_apples):
            self.place_new_apple(GREEN_APPLE)

        self.red_apples = []
        self.current_red_apples = 0
        for _ in range(self.nb_red_apples):
            self.place_new_apple(RED_APPLE)

        self.is_game_over = False
        self.game_over_message = ""
        self.frame_iteration = 0

    def get_random_empty_cell(self):
        empty_cells = {
            (x, y)
            for x in range(1, self.w // BLOCK_SIZE - 1)
            for y in range(1, self.h // BLOCK_SIZE - 1)
            if self.board[y][x] == EMPTY
        }
        if not empty_cells:
            return None, None
        x, y = random.choice(list(empty_cells))
        return x, y

    def init_snake(self):
        initialized_snake = False
        while not initialized_snake:
            x, y = self.get_random_empty_cell()
            if x is not None and y is not None and x > 2:
                self.head = Point(x, y)
                self.snake = [self.head, 
                              Point(self.head.x-1, self.head.y),
                              Point(self.head.x-(2), self.head.y)]
                initialized_snake = True
                break
        print("Snake have been initialized at: ", self.head)


    def place_new_apple (self, apple):
        x, y = self.get_random_empty_cell()
        if x is not None and y is not None:
            self.board[y][x] = apple
            if apple == RED_APPLE:
                self.red_apples.append((x, y))
                self.current_red_apples += 1
            elif apple  == GREEN_APPLE:
                self.green_apples.append((x, y))
                self.current_green_apples += 1
        else:
            if apple == RED_APPLE:
                self.current_red_apples -= 1
                self.game_over("Can't place a new red apple")
            elif apple  == GREEN_APPLE:
                self.current_green_apples -= 1
                if self.current_green_apples == 0:
                    self.game_over("No more green apples, you win!")

    def game_over(self, game_over_message):
        self.is_game_over = True
        self.game_over_message = game_over_message

        
    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        new_direction = self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0

        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            self.game_over("Another brick in the wall.... ")
            reward = BIGGER_NEGATIVE_REWARD
            return reward, self.is_game_over, self.score, new_direction
            
        if (self.head.x, self.head.y) in self.green_apples:
            self.score += 1
            reward = POSITIVE_REWARD
            self.green_apples.remove((self.head.x, self.head.y))
            self.place_new_apple(GREEN_APPLE)

        elif (self.head.x, self.head.y) in self.red_apples:
            reward = NEGATIVE_REWARD
            self.snake.pop()
            self.snake.pop()
            if len(self.snake) == 0:
                self.game_over("You ate a red apple and died")
                reward = BIGGER_NEGATIVE_REWARD
                return reward, self.is_game_over, self.score, new_direction
            self.red_apples.remove((self.head.x, self.head.y))
            self.place_new_apple(RED_APPLE)
        
        else:
            reward = SMALLER_NEGATIVE_REWARD
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, self.is_game_over, self.score, new_direction
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > 10 or pt.x < 0 or pt.y > 10 or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            
        for apple in self.green_apples:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(apple[0] * BLOCK_SIZE, apple[1]* BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(apple[0]* BLOCK_SIZE+ 4 , apple[1]* BLOCK_SIZE + 4, 12, 12))

        for apple in self.red_apples:
            pygame.draw.rect(self.display, RED, pygame.Rect(apple[0]* BLOCK_SIZE, apple[1]* BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[index]
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (index + 1) % 4
            new_dir = clock_wise[next_index]
        else:
            next_index = (index - 1) % 4
            new_dir = clock_wise[next_index]
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1
            
        self.head = Point(x, y)
        return self.direction
