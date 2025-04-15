import pyfiglet
import time

from src.constants import WALL, BLOCK_SIZE, EMPTY
from src.constants import GREEN_APPLE, RED_APPLE, SNAKE_HEAD, SNAKE_BODY, RESET, CLEAR, BLUE

class Displayer:
    def __init__(self, game, fps=10):
        self.game = game
        self.fps = fps
        print(CLEAR + BLUE + pyfiglet.figlet_format("Learn2Slither") + RESET)
        print("Reinforcement Learning Snake Agent")
        print("The snake makes decisions based only on its 4-directional vision.\n")

    def get_vision_grid(self):
        """
        Returns a 2D grid showing only what the snake sees in the 4 directions.
        """
        width = len(self.game.board[0])
        height = len(self.game.board)
        grid = [[" " for _ in range(width)] for _ in range(height)]

        head_x, head_y = self.game.head.x, self.game.head.y

        directions = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }

        for dx, dy in directions.values():
            x, y = head_x, head_y
            while 0 <= x < width and 0 <= y < height:
                if (x, y) in self.game.green_apples:
                    grid[y][x] = GREEN_APPLE
                elif (x, y) in self.game.red_apples:
                    grid[y][x] = RED_APPLE
                elif (x, y) in [(seg.x, seg.y) for seg in self.game.snake]:
                    grid[y][x] = SNAKE_HEAD if (x, y) == (head_x, head_y) else SNAKE_BODY
                elif self.game.board[y][x] == WALL:
                    grid[y][x] = WALL
                    break
                else:
                    grid[y][x] = EMPTY
                x += dx
                y += dy

        return grid

    def display(self, action):
        output = []

        output.append(BLUE + pyfiglet.figlet_format("Learn2Slither") + RESET)
        output.append("Project: Reinforcement Learning Snake AI\n")

        width = len(self.game.board[0])
        height = len(self.game.board)

        full_board = []
        for y in range(height):
            row = []
            for x in range(width):
                coord = (x, y)
                if coord in self.game.green_apples:
                    row.append(GREEN_APPLE)
                elif coord in self.game.red_apples:
                    row.append(RED_APPLE)
                elif coord == (self.game.head.x, self.game.head.y):
                    row.append(SNAKE_HEAD)
                elif coord in [(seg.x, seg.y) for seg in self.game.snake]:
                    row.append(SNAKE_BODY)
                elif self.game.board[y][x] == WALL:
                    row.append(WALL)
                else:
                    row.append(EMPTY)
            full_board.append(" ".join(row))

        # Vision at t
        vision_before = self.get_vision_grid()

        # Play step
        reward, done, score, direction = self.game.play_step(action)

        # Vision at t+1
        vision_after = self.get_vision_grid()

        # Header
        output.append("FULL BOARD".center(23) + "||" + "VISION t".center(23) + "||" + "VISION t+1".center(23))
        output.append("-" * 75)

        # Merge 3 columns
        for fb_row, vis_t, vis_tp1 in zip(full_board, vision_before, vision_after):
            row_t = " ".join(vis_t)
            row_tp1 = " ".join(vis_tp1)
            output.append(fb_row.ljust(10) + "||" + row_t.ljust(10) + "||" + row_tp1)

        # Action summary
        output.append(f"\nAction Taken: {direction.name}")
        output.append("\n" + "=" * 80)

        print("\n".join(output))
        time.sleep(1 / self.fps)

        return reward, done, score
