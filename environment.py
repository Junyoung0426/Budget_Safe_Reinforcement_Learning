import numpy as np
import gym
from gym import spaces


class TicTacToe(gym.Env):
    """
    Gomoku Environment in OpenAI Gym style for a 3x3 board.
    """
    def __init__(self, board_size=3):
        super(TicTacToe, self).__init__()
        self.board_size = board_size
        self.action_space = spaces.Discrete(self.board_size * self.board_size)

        # Redefine observation space to accommodate hashable states
        self.observation_space = spaces.Tuple(
            [spaces.Tuple([spaces.Discrete(3) for _ in range(self.board_size)]) for _ in range(self.board_size)]
        )

        self.reset()

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_observation(), True

    def step(self, action):
        """
        Executes a move.
        Args:
            action: An integer representing the position (row * board_size + col).
        Returns:
            observation: The current board state as a hashable tuple of tuples.
            reward (float): Reward for the move.
            done (bool): Whether the game has ended.
            truncated (bool): Whether the game was truncated (not applicable here, set to False).
            info (dict): Additional information.
        """
        if self.done:
            raise RuntimeError("Game is over. Please reset the environment.")

        row, col = divmod(action, self.board_size)

        # Check if the move is valid
        if self.board[row, col] != 0:
            return self._get_observation(), -1, True, False, {"info": "Invalid move"}  # Penalty for invalid move

        # Make the move
        self.board[row, col] = self.current_player

        # Check for win or draw
        reward, self.done = self._check_game_status()

        # Switch players
        if not self.done:
            self.current_player = 3 - self.current_player

        return self._get_observation(), reward, self.done, False, {}

    def render(self, mode="human"):
        """
        Displays the current state of the board.
        """
        symbols = {0: ".", 1: "X", 2: "O"}
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))
        print("\n")


    def _get_observation(self):
        """
        Converts the board state into a hashable format (tuple of tuples).
        """
        return tuple(map(tuple, self.board))

    def _check_game_status(self):
        """
        Checks the game state for a win or draw.
        Returns:
            reward (float): Reward for the current player.
            done (bool): Whether the game has ended.
        """
        # Check rows and columns
        for i in range(self.board_size):
            if np.all(self.board[i, :] == self.current_player) or np.all(self.board[:, i] == self.current_player):
                return 1, True  # Current player wins

        # Check diagonals
        if np.all(np.diag(self.board) == self.current_player) or np.all(np.diag(np.fliplr(self.board)) == self.current_player):
            return 1, True  # Current player wins

        # Check for a draw
        if not np.any(self.board == 0):
            return -1, True  # Draw

        return 0, False  # Continue game