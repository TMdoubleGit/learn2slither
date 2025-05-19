import pygame
import random
from enum import Enum
from collections import namedtuple, deque, Counter
import numpy as np
import src.constants as c
from src.interpreter import Interpreter

pygame.init()
font = pygame.font.Font('arial.ttf', 24)


class Direction(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


Point = namedtuple('Point', 'x, y')


class SnakeGameAI:

    nb_red_apples = 1
    nb_green_apples = 2

    def __init__(self, w, h):
        self.w = (w + 2) * c.BLOCK_SIZE
        self.h = (h + 2) * c.BLOCK_SIZE

        self.human_mode = False
        self.step_by_step = False
        self.cycle_memory = deque(maxlen=30)
        self.cycle_repeat_threshold = 4

        self.board = [
            [c.WALL if (x == 0 or x == self.w // c.BLOCK_SIZE - 1
                        or y == 0 or y == self.h // c.BLOCK_SIZE - 1)
                else c.EMPTY
             for x in range(self.w // c.BLOCK_SIZE - 1)]
            for y in range(self.h // c.BLOCK_SIZE - 1)
            ]

        if c.GRAPHIC_MODE:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
        else:
            self.display = None
            self.clock = None
        self.reset()

        self.interpreter = Interpreter(self)

    def reset(self):
        self.direction = Direction.RIGHT

        self.board = [
            [c.WALL if (x == 0 or x == self.w // c.BLOCK_SIZE - 1
                        or y == 0 or y == self.h // c.BLOCK_SIZE - 1)
                else c.EMPTY
             for x in range(self.w // c.BLOCK_SIZE)]
            for y in range(self.h // c.BLOCK_SIZE)
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
            self.place_new_apple(c.GREEN_APPLE)

        self.red_apples = []
        self.current_red_apples = 0
        for _ in range(self.nb_red_apples):
            self.place_new_apple(c.RED_APPLE)

        self.is_game_over = False
        self.game_over_message = ""
        self.frame_iteration = 0

    def get_random_empty_cell(self):
        empty_cells = {
            (x, y)
            for x in range(1, self.w // c.BLOCK_SIZE - 1)
            for y in range(1, self.h // c.BLOCK_SIZE - 1)
            if self.board[y][x] == c.EMPTY
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
                if self.board[y][x - 1] == c.EMPTY and\
                        self.board[y][x - 2] == c.EMPTY:
                    self.snake = [self.head,
                                  Point(self.head.x-1, self.head.y),
                                  Point(self.head.x-(2), self.head.y)]
                    self.board[y][x] = c.SNAKE_HEAD
                    self.board[y][x - 1] = c.SNAKE_BODY
                    self.board[y][x - 2] = c.SNAKE_BODY
                    initialized_snake = True
                break

    def place_new_apple(self, apple):
        x, y = self.get_random_empty_cell()
        if x is not None and y is not None:
            self.board[y][x] = apple
            if apple == c.RED_APPLE:
                self.red_apples.append((x, y))
                self.current_red_apples += 1
            elif apple == c.GREEN_APPLE:
                self.green_apples.append((x, y))
                self.current_green_apples += 1
        else:
            if apple == c.RED_APPLE:
                self.current_red_apples -= 1
                self.game_over("Can't place a new red apple")
            elif apple == c.GREEN_APPLE:
                self.current_green_apples -= 1
                if self.current_green_apples == 0:
                    self.game_over("No more green apples, you win!")

    def game_over(self, game_over_message):
        self.is_game_over = True
        self.game_over_message = game_over_message

    def get_action_from_input(self, current_direction, key):
        if current_direction == Direction.UP:
            if key == pygame.K_UP:
                return [1, 0, 0, 0]
            elif key == pygame.K_RIGHT:
                return [0, 1, 0, 0]
            elif key == pygame.K_LEFT:
                return [0, 0, 1, 0]
            elif key == pygame.K_DOWN:
                return [0, 0, 0, 1]
        elif current_direction == Direction.DOWN:
            if key == pygame.K_DOWN:
                return [1, 0, 0, 0]
            elif key == pygame.K_RIGHT:
                return [0, 0, 1, 0]
            elif key == pygame.K_LEFT:
                return [0, 1, 0, 0]
            elif key == pygame.K_UP:
                return [0, 0, 0, 1]
        elif current_direction == Direction.LEFT:
            if key == pygame.K_LEFT:
                return [1, 0, 0, 0]
            elif key == pygame.K_UP:
                return [0, 1, 0, 0]
            elif key == pygame.K_DOWN:
                return [0, 0, 1, 0]
            elif key == pygame.K_RIGHT:
                return [0, 0, 0, 1]
        elif current_direction == Direction.RIGHT:
            if key == pygame.K_RIGHT:
                return [1, 0, 0, 0]
            elif key == pygame.K_UP:
                return [0, 0, 1, 0]
            elif key == pygame.K_DOWN:
                return [0, 1, 0, 0]
            elif key == pygame.K_LEFT:
                return [0, 0, 0, 1]
        return None

    def play_step(self, action):
        new_direction = self.direction
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    self.human_mode = not self.human_mode
                    print("Human mode:", self.human_mode)

                if event.key == pygame.K_p:
                    self.step_by_step = not self.step_by_step
                    print("Step-by-step mode:", self.step_by_step)

                if self.step_by_step and event.key == pygame.K_RETURN:
                    self.step_triggered = True

        if self.step_by_step:
            while not getattr(self, 'step_triggered', False):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    elif event.type == pygame.KEYDOWN \
                            and event.key == pygame.K_RETURN:
                        self.step_triggered = True
                    elif event.type == pygame.KEYDOWN \
                            and event.key == pygame.K_p:
                        self.step_by_step = not self.step_by_step
                        print("Step-by-step mode:", self.step_by_step)
                        return 0, self.is_game_over, self.score, new_direction
            self.step_triggered = False

        if self.human_mode:
            while not getattr(self, 'step_triggered', False):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    elif event.type == pygame.KEYDOWN \
                            and event.key == pygame.K_UP:
                        action = self.get_action_from_input(
                            self.direction, event.key)
                        self.step_triggered = True
                    elif event.type == pygame.KEYDOWN \
                            and event.key == pygame.K_DOWN:
                        action = self.get_action_from_input(
                            self.direction, event.key)
                        self.step_triggered = True
                    elif event.type == pygame.KEYDOWN \
                            and event.key == pygame.K_LEFT:
                        action = self.get_action_from_input(
                            self.direction, event.key)
                        self.step_triggered = True
                    elif event.type == pygame.KEYDOWN \
                            and event.key == pygame.K_RIGHT:
                        action = self.get_action_from_input(
                            self.direction, event.key)
                        self.step_triggered = True
                    elif event.type == pygame.KEYDOWN \
                            and event.key == pygame.K_h:
                        self.human_mode = not self.human_mode
                        print("Human mode:", self.human_mode)
                        return 0, self.is_game_over, self.score, new_direction
            self.step_triggered = False

        self.frame_iteration += 1
        new_direction = self._move(action)
        self.snake.insert(0, self.head)
        self.cycle_memory.append(self.head)

        reward = 0
        if self.is_collision() or self.frame_iteration > 20*len(self.snake):
            self.game_over("Another brick in the wall.... ")
            reward += c.BIGGER_NEGATIVE_REWARD
            return reward, self.is_game_over, self.score, new_direction

        if len(self.cycle_memory) == self.cycle_memory.maxlen:
            counts = Counter(self.cycle_memory)
            if counts.most_common(1)[0][1] >= self.cycle_repeat_threshold:
                self.game_over("Cycle detected")
                reward += c.BIGGER_NEGATIVE_REWARD
                return reward, self.is_game_over, self.score, new_direction

        green_apple_distance = self.w // c.BLOCK_SIZE - 2
        wall_distance = 0

        for i in range(1, self.w // c.BLOCK_SIZE):
            next_x = self.head.x + i * (
                1 if self.direction == Direction.RIGHT
                else -1 if self.direction == Direction.LEFT
                else 0)
            next_y = self.head.y + i * (
                1 if self.direction == Direction.DOWN
                else -1 if self.direction == Direction.UP
                else 0)

            if next_x < 0 or next_x >= self.w // c.BLOCK_SIZE\
                    or next_y < 0 or next_y >= self.h // c.BLOCK_SIZE:
                wall_distance = i - 1
                break

            next_cell = self.board[next_y][next_x]
            if next_cell == c.WALL:
                wall_distance = i - 1
                break
            elif next_cell == c.GREEN_APPLE:
                green_apple_distance = i - 1

        if green_apple_distance < wall_distance:
            reward -= 2 * c.SMALLER_NEGATIVE_REWARD
        else:
            reward += c.SMALLER_NEGATIVE_REWARD

        if (self.head.x, self.head.y) in self.green_apples:
            self.score += 1
            reward += c.POSITIVE_REWARD
            self.green_apples.remove((self.head.x, self.head.y))
            self.place_new_apple(c.GREEN_APPLE)

        elif (self.head.x, self.head.y) in self.red_apples:
            reward += c.NEGATIVE_REWARD
            self.snake.pop()
            self.snake.pop()
            if len(self.snake) == 0:
                self.game_over("You ate a red apple and died")
                reward += c.BIGGER_NEGATIVE_REWARD
                return reward, self.is_game_over, self.score, new_direction
            self.red_apples.remove((self.head.x, self.head.y))
            self.place_new_apple(c.RED_APPLE)

        else:
            self.snake.pop()
        self._update_board()
        self.interpreter.get_state()
        self._update_ui()
        if c.GRAPHIC_MODE and self.clock:
            self.clock.tick(c.SPEED)
        return reward, self.is_game_over, self.score, new_direction

    def _update_board(self):
        """
        Updates the grid given the state we are in.
        """
        self.board = [
            [c.WALL if (x == 0 or x == self.w // c.BLOCK_SIZE - 1
                        or y == 0 or y == self.h // c.BLOCK_SIZE - 1)
                else c.EMPTY
             for x in range(self.w // c.BLOCK_SIZE)]
            for y in range(self.h // c.BLOCK_SIZE)
            ]

        for segment in self.snake:
            self.board[segment.y][segment.x] = c.SNAKE_BODY

        for apple in self.green_apples:
            self.board[apple[1]][apple[0]] = c.GREEN_APPLE

        for apple in self.red_apples:
            self.board[apple[1]][apple[0]] = c.RED_APPLE

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x >= self.w // c.BLOCK_SIZE - 1 or pt.x < 1\
                or pt.y >= self.h // c.BLOCK_SIZE - 1 or pt.y < 1:
            return True
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        if not c.GRAPHIC_MODE:
            return
        self.display.fill(c.BLACK)

        pygame.draw.rect(
            self.display,
            c.GRAY1,
            pygame.Rect(0, 0, self.w, c.BLOCK_SIZE))
        pygame.draw.rect(
            self.display,
            c.GRAY1,
            pygame.Rect(0, self.h - c.BLOCK_SIZE, self.w, c.BLOCK_SIZE))
        pygame.draw.rect(
            self.display,
            c.GRAY1,
            pygame.Rect(0, 0, c.BLOCK_SIZE, self.h))
        pygame.draw.rect(
            self.display,
            c.GRAY1,
            pygame.Rect(self.w - c.BLOCK_SIZE, 0, c.BLOCK_SIZE, self.h))

        for x in range(0, self.w, c.BLOCK_SIZE):
            pygame.draw.line(self.display, c.WHITE, (x, 0), (x, self.h))

        for y in range(0, self.h, c.BLOCK_SIZE):
            pygame.draw.line(self.display, c.WHITE, (0, y), (self.w, y))

        for i, pt in enumerate(self.snake):
            color = c.LIGHT_BLUE if i == 0 else c.BLUE1
            pygame.draw.rect(
                self.display,
                color,
                pygame.Rect(
                    pt.x * c.BLOCK_SIZE,
                    pt.y * c.BLOCK_SIZE,
                    c.BLOCK_SIZE,
                    c.BLOCK_SIZE))

        for apple in self.green_apples:
            pygame.draw.rect(
                self.display,
                c.GREEN1,
                pygame.Rect(
                    apple[0] * c.BLOCK_SIZE,
                    apple[1] * c.BLOCK_SIZE,
                    c.BLOCK_SIZE,
                    c.BLOCK_SIZE))
            pygame.draw.rect(
                self.display,
                c.GREEN2,
                pygame.Rect(
                    apple[0] * c.BLOCK_SIZE + 4,
                    apple[1] * c.BLOCK_SIZE + 4, 16, 16))

        for apple in self.red_apples:
            pygame.draw.rect(
                self.display,
                c.RED1,
                pygame.Rect(
                    apple[0] * c.BLOCK_SIZE,
                    apple[1] * c.BLOCK_SIZE,
                    c.BLOCK_SIZE,
                    c.BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, c.WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [
            Direction.UP,
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT]
        index = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0, 0]):
            new_dir = clock_wise[index]
        elif np.array_equal(action, [0, 1, 0, 0]):
            next_index = (index + 1) % 4
            new_dir = clock_wise[next_index]
        elif np.array_equal(action, [0, 0, 1, 0]):
            next_index = (index - 1) % 4
            new_dir = clock_wise[next_index]
        elif np.array_equal(action, [0, 0, 0, 1]):
            next_index = (index + 2) % 4
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
