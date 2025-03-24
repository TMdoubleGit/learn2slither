import pyfiglet
from src.constants import WALL, BLOCK_SIZE, EMPTY
from src.constants import GREEN_APPLE, RED_APPLE, SNAKE_HEAD, SNAKE_BODY, RESET
import time
import os

class Displayer:
    """
    The Displayer class is responsible for rendering a text-based representation
    of the Learn2Slither game. It shows the snake's perception in four directions,
    the action taken, and the resulting perception.
    """

    def __init__(self, game):
        self.game = game
        self.display_intro()

    def display_intro(self):
        title = pyfiglet.figlet_format("Learn2Slither")
        print(title)
        print("Reinforcement Learning Snake Agent")
        print("The snake makes decisions based only on its 4-directional vision.\n")

    def get_vision_grid(self):
        """
        Returns a 2D grid showing the whole board, but only what the snake sees in the 4 directions.
        Everything else is blank (" ").
        """
        grid = [[" " for _ in range(self.game.w // BLOCK_SIZE)] for _ in range(self.game.h // BLOCK_SIZE)]

        head_x, head_y = self.game.head.x // BLOCK_SIZE, self.game.head.y // BLOCK_SIZE

        # Directions
        directions = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }

        # Place visible cells
        for dx, dy in directions.values():
            x, y = head_x, head_y
            while 0 <= x < self.game.w // BLOCK_SIZE and 0 <= y < self.game.h // BLOCK_SIZE:
                if (x, y) in self.game.green_apples:
                    grid[int(y)][int(x)] = GREEN_APPLE
                elif (x, y) in self.game.red_apples:
                    grid[int(y)][int(x)] = RED_APPLE
                elif (x, y) in [(seg.x // BLOCK_SIZE, seg.y // BLOCK_SIZE) for seg in self.game.snake]:
                    if (x, y) == (head_x, head_y):
                        grid[int(y)][int(x)] = SNAKE_HEAD
                    else:
                        grid[int(y)][int(x)] = SNAKE_BODY
                elif self.game.board[int(y)][int(x)] == WALL:
                    grid[int(y)][int(x)] = WALL
                    break
                else:
                    grid[int(y)][int(x)] = EMPTY

                x += dx
                y += dy

        return grid


    def display(self, action):
        """
        Displays vision before and after taking the action.
        """
        os.system("clear")  # or 'cls' on Windows
        print(pyfiglet.figlet_format("Learn2Slither"))
        print("Project: Reinforcement Learning Snake AI\n")

        vision_before = self.get_vision_grid()

        print("====== Snake Vision at t ======")
        for row in vision_before:
            print("".join(row))

        print("\nAction Taken:", action.name)

        # Advance the game state
        reward, done, score = self.game.play_step(action)

        vision_after = self.get_vision_grid()
        print("\n====== Snake Vision at t+1 ======")
        for row in vision_after:
            print("".join(row))

        print("\n===============================")
        return reward, done, score
