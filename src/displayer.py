import pyfiglet
from constant import CLEAR, BLUE, RESET, RED
import time
import os

class Displayer:
    """
    This class is responsible for displaying the snake's vision and actions
    in a text-based format.
    """

    SYMBOLS = {
        'WALL': "W",
        'EMPTY': "0",
        'RED_APPLE': "R",
        'GREEN_APPLE': "G",
        'SNAKE_BODY': "S",
        'SNAKE_HEAD': "H"
    }

    def __init__(self, game):
        self.game = game

    def get_vision_grid(self):
        """
        Returns a text-based representation of the snake's vision.
        The vision is in four directions: UP, DOWN, LEFT, RIGHT.
        """
        head_x, head_y = self.game.head.x // BLOCK_SIZE, self.game.head.y // BLOCK_SIZE
        vision_grid = []

        # Define directions
        directions = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }

        for direction, (dx, dy) in directions.items():
            vision_line = []
            x, y = head_x, head_y


            while 0 <= x < (self.game.w // BLOCK_SIZE) and 0 <= y < (self.game.h // BLOCK_SIZE):
                if (x, y) in self.game.green_apples:
                    vision_line.append(self.SYMBOLS['GREEN_APPLE'])
                elif (x, y) in self.game.red_apples:
                    vision_line.append(self.SYMBOLS['RED_APPLE'])
                elif (x, y) in [(seg.x // BLOCK_SIZE, seg.y // BLOCK_SIZE) for seg in self.game.snake]:
                    vision_line.append(self.SYMBOLS['SNAKE_BODY'])
                elif self.game.board[y][x] == WALL:
                    vision_line.append(self.SYMBOLS['WALL'])
                    break
                else:
                    vision_line.append(self.SYMBOLS['EMPTY'])

                x += dx
                y += dy

            vision_grid.append((direction, vision_line))

        return vision_grid

    def display(self, action):
        """
        Displays the snake's vision, its last action, and the updated vision.
        """
        vision_before = self.get_vision_grid()

        print("\n====== Snake Vision at t ======")
        for direction, vision in vision_before:
            print(f"{direction}: {''.join(vision)}")

        print("\nAction Taken:", action.name)

        # Update game state (simulate the move)
        self.game.play_step(action)

        vision_after = self.get_vision_grid()
        print("\n====== Snake Vision at t+1 ======")
        for direction, vision in vision_after:
            print(f"{direction}: {''.join(vision)}")

        print("\n===============================")
