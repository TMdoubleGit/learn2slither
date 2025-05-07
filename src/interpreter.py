from src.constants import WALL
import numpy as np

class Interpreter:
    """
    Class responsible for interpreting the game environment and providing
    the state of the environment as a NumPy array.
    """

    def __init__(self, game):
        """
        Initializes the Interpreter with the game instance.

        Args:
            game (SnakeGameAI): An instance of the SnakeGameAI class.
        """
        self.game = game
        self.snake = game.snake
        self.grid = game.board
        self.snake_head = self.game.head

    def interpret(self, action):
        """
        Executes the given action and updates the game state.

        Args:
            action (int): The action to perform (0: LEFT, 1: RIGHT, 2: UP, 3: DOWN).
        """
        if action == 0:
            self.game.direction = Direction.LEFT
        elif action == 1:
            self.game.direction = Direction.RIGHT
        elif action == 2:
            self.game.direction = Direction.UP
        elif action == 3:
            self.game.direction = Direction.DOWN

        self.game.play_step()
        self.snake_head = self.game.head
        self.grid = self.game.board
        self.snake = self.game.snake
    
    def get_state(self):
        """
        Returns the current state of the environment as a NumPy array.

        The state includes:
        - Distances to green apples in all four directions.
        - Distances to red apples in all four directions.
        - Distances to dangers in all four directions.

        Returns:
            np.ndarray: A NumPy array representing the state of the environment.
        """
        directions = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0)
        }

        state = np.array([
            self._distance_danger(directions["up"]),
            self._distance_danger(directions["down"]),
            self._distance_danger(directions["left"]),
            self._distance_danger(directions["right"]),

            self._distance_apple(directions["up"], "green"),
            self._distance_apple(directions["down"], "green"),
            self._distance_apple(directions["left"], "green"),
            self._distance_apple(directions["right"], "green"),

            self._distance_apple(directions["up"], "red"),
            self._distance_apple(directions["down"], "red"),
            self._distance_apple(directions["left"], "red"),
            self._distance_apple(directions["right"], "red"),

            self._wall_distance(directions["up"]),
            self._wall_distance(directions["down"]),
            self._wall_distance(directions["left"]),
            self._wall_distance(directions["right"]),
        ], dtype=np.float32)

        return state.copy()


    def _wall_distance(self, direction):
        """
        Calculates the normalized distance from the snake's head to the nearest wall
        in the specified direction.

        Args:
            direction (tuple): A tuple representing the direction to check (dx, dy).

        Returns:
            float: The normalized distance to the nearest wall (between 0 and 1).
        """
        x, y = self.snake_head.x, self.snake_head.y
        distance = 0
        grid_height = len(self.grid)
        grid_width = len(self.grid[0])
        while (
            0 <= x < grid_width and
            0 <= y < grid_height and
            self.grid[y][x] != WALL
        ):
            distance += 1
            x += direction[0]
            y += direction[1]
        
        return distance / (grid_height - 2)

    def _distance_apple(self, direction, apple_type):
        """
        Calculates the normalized distance from the snake's head to the nearest apple
        of the specified type (green or red) in the given direction.

        Args:
            direction (tuple): A tuple representing the direction to check (dx, dy).
            apple_type (str): The type of apple to check ("green" or "red").

        Returns:
            float: The normalized distance to the nearest apple (between 0 and 1),
                or 0 if no apple is found in the given direction.
        """
        x, y = self.snake_head.x, self.snake_head.y
        distance = 0
        grid_height = len(self.grid)
        grid_width = len(self.grid[0])

        apples = self.game.green_apples if apple_type == "green" else self.game.red_apples

        while (0 <= x < grid_width and 0 <= y < grid_height):
            if (x, y) in apples:
                return distance / (grid_height - 2)
            distance += 1
            x += direction[0]
            y += direction[1]
        return 0

    def _distance_danger(self, direction):
        """
        Calculates the normalized distance from the snake's head to the nearest danger
        (wall, body, or red apple) in the specified direction.

        Args:
            direction (tuple): A tuple representing the direction to check (dx, dy).

        Returns:
            float: The normalized distance to the nearest danger (between 0 and 1).
        """
        x, y = self.snake_head.x, self.snake_head.y
        distance = 0
        grid_height = len(self.grid)
        grid_width = len(self.grid[0])

        while (0 <= x < grid_width and 0 <= y < grid_height):
            if self.grid[y][x] == WALL:
                return distance / (grid_height - 2)

            if (x, y) in [(segment.x, segment.y) for segment in self.snake[1:]]:
                return distance / (grid_height - 2)

            if (x, y) in self.game.red_apples and len(self.snake) <= 1:
                return distance / (grid_height - 2)

            x += direction[0]
            y += direction[1]
            distance += 1

        return 0

