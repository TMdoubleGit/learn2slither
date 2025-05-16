import pyfiglet
import time

from src.constants import WALL, BLOCK_SIZE, EMPTY, TRAINING_MODE
from src.constants import GREEN_APPLE, RED_APPLE, SNAKE_HEAD, SNAKE_BODY, RESET, CLEAR, BLUE

class Displayer:
    def __init__(self, game, agent, fps=10):
        self.game = game
        self.agent = agent
        self.fps = fps
        
        # if TRAINING_MODE:
        #     print(CLEAR + BLUE + pyfiglet.figlet_format("Learn2Slither") + RESET)
        #     print("Reinforcement Learning Snake Agent")
        #     print("The snake makes decisions based only on its 4-directional vision.\n")

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


    def display(self, action, state, reward, prev_state = [], is_new_state = False):
        """
        Displays the full board, the snake's vision, and the state information.

        Args:
            action (str): The action taken by the snake.
            state (np.ndarray): The state of the environment as a NumPy array.
        """
        if TRAINING_MODE:
            if is_new_state == True:
                print(f"Action Taken: {action.name} (epsilon = {self.agent.epsilon:.2f}, {self.agent.model_type}, gamma = {self.agent.gamma:.2f}) - Reward: {reward} - Record: {self.agent.record}")
            return
        output = []
        state_tempo = "(T + 1)" if is_new_state else "(T)"

        # if is_new_state == True:
        #     output.append(CLEAR + BLUE + pyfiglet.figlet_format("Learn2Slither") + RESET)
        #     output.append("Project: Reinforcement Learning Snake AI\n")
        #     output.append(f"Game #{self.agent.n_games} - Turn #{self.game.frame_iteration} - Snake length: {len(self.game.snake)} - Score: {self.game.score}\n")
        #     output.append(prev_state)

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

        vision = self.get_vision_grid()

        state_labels = [
            "danger_up", "danger_down", "danger_left", "danger_right",
            "green_apple_up", "green_apple_down", "green_apple_left", "green_apple_right",
            "red_apple_up", "red_apple_down", "red_apple_left", "red_apple_right","wall_up",
            "wall_down", "wall_left", "wall_right"
        ]

        if is_new_state == True:
            output.append(f"\nAction Taken: {action.name} (epsilon = {self.agent.epsilon:.2f}, {self.agent.model_type}, gamma = {self.agent.gamma:.2f}) - Reward: {reward} - Record: {self.agent.record}")
            output.append("\n" + "=" * 95)
            
        output.append(f"FULL BOARD {state_tempo}".center(23) + "  ||  " + f"VISION {state_tempo}".center(23) + "  ||  " + f"STATE {state_tempo}".center(23))
        output.append("-" * 95)

        for i in range(height):
            fb_row = full_board[i]
            vis_row = " ".join(vision[i])
            state_info = f"{state_labels[i]}: {state[i]:.2f}" if i < len(state) else ""
            extra_state_info = f"{state_labels[12 + i]}: {state[12 + i]:.2f}" if i < 4 else ""
            output.append(fb_row.ljust(23) + "     " + vis_row.ljust(23) + "     " + state_info.ljust(23) + extra_state_info.ljust(23))


        if is_new_state == True:
            print("\n".join(output))
            time.sleep(1/self.fps)
        
        return "\n".join(output)
