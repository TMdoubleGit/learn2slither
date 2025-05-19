import pyfiglet
import time
import src.constants as c


class Displayer:
    def __init__(self, game, agent, fps=10):
        self.game = game
        self.agent = agent
        self.fps = fps

        if not c.GRAPHIC_MODE:
            print(c.CLEAR + c.BLUE +
                  pyfiglet.figlet_format("Learn2Slither") + c.RESET)
            print("Reinforcement Learning Snake Agent")
            print("The snake makes decisions based\
                only on its 4-directional vision.\n")

    def get_vision_grid(self):
        """
        Returns a 2D grid showing only what the snake sees in the 4 directions.
        """
        width = len(self.game.board[0])
        height = len(self.game.board)
        grid = [["-" for _ in range(width)] for _ in range(height)]

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
                    grid[y][x] = c.GREEN_APPLE
                elif (x, y) in self.game.red_apples:
                    grid[y][x] = c.RED_APPLE
                elif (x, y) in [(seg.x, seg.y) for seg in self.game.snake]:
                    grid[y][x] = c.SNAKE_HEAD \
                        if (x, y) == (head_x, head_y) else c.SNAKE_BODY
                elif self.game.board[y][x] == c.WALL:
                    grid[y][x] = c.WALL
                    break
                else:
                    grid[y][x] = c.EMPTY
                x += dx
                y += dy

        return grid

    def display(
            self,
            action,
            state,
            reward,
            prev_state=[],
            is_new_state=False
            ):
        """
        Displays the full board, the snake's vision, and the state information.

        Args:
            action (str): The action taken by the snake.
            state (np.ndarray): The state of the environment as a NumPy array.
        """
        if not c.GRAPHIC_MODE:
            return
        output = []
        extra_state_info = ""
        extra_state_info2 = ""
        extra_state_info3 = ""
        state_tempo = "(T + 1)" if is_new_state else "(T)"

        if is_new_state:
            output.append(c.CLEAR + c.BLUE +
                          pyfiglet.figlet_format("Learn2Slither") + c.RESET)
            output.append("Project: Reinforcement Learning Snake AI\n")
            output.append(f"Game #{self.agent.n_games} -" +
                          f"Turn #{self.game.frame_iteration} -" +
                          f" Snake length: {len(self.game.snake)} -" +
                          f" Score: {self.game.score}\n")
            output.append(prev_state)

        width = len(self.game.board[0])
        height = len(self.game.board)

        full_board = []
        for y in range(height):
            row = []
            for x in range(width):
                coord = (x, y)
                if coord in self.game.green_apples:
                    row.append(c.GREEN_APPLE)
                elif coord in self.game.red_apples:
                    row.append(c.RED_APPLE)
                elif coord == (self.game.head.x, self.game.head.y):
                    row.append(c.SNAKE_HEAD)
                elif coord in [(seg.x, seg.y) for seg in self.game.snake]:
                    row.append(c.SNAKE_BODY)
                elif self.game.board[y][x] == c.WALL:
                    row.append(c.WALL)
                else:
                    row.append(c.EMPTY)
            full_board.append(" ".join(row))

        vision = self.get_vision_grid()

        state_labels = [
            "danger_up",
            "danger_down",
            "danger_left",
            "danger_right",
            "green_apple_up",
            "green_apple_down",
            "green_apple_left",
            "green_apple_right",
            "red_apple_up",
            "red_apple_down",
            "red_apple_left",
            "red_apple_right",
            "wall_up",
            "wall_down",
            "wall_left",
            "wall_right"
        ]

        if is_new_state:
            output.append(f"\nAction Taken: {action.name} " +
                          f"(epsilon = {self.agent.epsilon:.2f}, " +
                          f"{self.agent.model_type}, " +
                          f"gamma = {self.agent.gamma:.2f}) "
                          f"- Reward: {reward} - Record: {self.agent.record}")
            output.append("\n" + "=" * 120)

        output.append("FULL BOARD, " +
                      " SNAKE VISION AND" +
                      f" STATE AT {state_tempo}\n")
        output.append("-" * 120)

        state_data = []
        for i in range(len(state)):
            state_data.append(f"{state_labels[i]}: {state[i]:.2f}")

        for i in range(height):
            fb_row = full_board[i]
            vis_row = " ".join(vision[i])
            state_info = state_data[i] if i < len(state) else ""
            if len(state) > height:
                extra_state_info =\
                    state_data[height + i]\
                    if i < len(state) - height else ""
            if len(state) > 2 * height:
                extra_state_info2 =\
                    state_data[2 * height + i]\
                    if i < len(state) - 2 * height else ""
            if len(state) > 3 * height:
                extra_state_info3 =\
                    state_data[3 * height + i]\
                    if i < len(state) - 3 * height else ""
            output.append(fb_row.ljust(23) +
                          "     " + vis_row + "     " +
                          state_info.ljust(23) + "     " +
                          extra_state_info.ljust(23) + "     " +
                          extra_state_info2.ljust(23) + "     " +
                          extra_state_info3.ljust(23))

        if is_new_state:
            print("\n".join(output))
            time.sleep(1/self.fps)

        return "\n".join(output)
