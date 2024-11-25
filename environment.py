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

import gym
from gym import spaces
import numpy as np

class TwentyOneGameEnv(gym.Env):
    """
    A simple 21-game environment.
    """
    def __init__(self):
        super(TwentyOneGameEnv, self).__init__()
        
        # Action space: Choose to add 1, 2, or 3
        self.action_space = spaces.Discrete(3)  # Actions: 0 -> +1, 1 -> +2, 2 -> +3
        
        # Observation space: Current count (0 to 21)
        self.observation_space = spaces.Discrete(22)  # Observations: integers 0 to 21
        
        # Initialize the state
        self.reset()

    def reset(self):
        """
        Reset the game to the initial state.
        """
        self.current_count = 0  # Start the count at 0
        self.done = False
        return self.current_count, {}

    def step(self, action):
        """
        Take a step in the environment.
        Args:
            action (int): Action to take (0 -> +1, 1 -> +2, 2 -> +3)
        Returns:
            obs (int): Current count
            reward (float): Reward for the action
            done (bool): Whether the game is over
            info (dict): Additional debug info
        """
        if self.done:
            raise RuntimeError("Game is over. Please reset the environment.")
        
        # Translate action to number (1, 2, or 3)
        action = action + 1
        
        # Update the current count
        self.current_count += action
        
        # Check if the game is over
        if self.current_count >= 21:
            self.done = True
            reward = -1  # Player loses if they reach 21
        else:
            reward = 0  # No reward yet; game continues
            
        # Opponent's turn (simple AI to play optimally)
        if not self.done:
            optimal_move = np.random.choice([1,2,3])
            self.current_count += optimal_move
            
            # Check if opponent causes the player to win
            if self.current_count >= 21:
                self.done = True
                reward = +1  # Player wins if opponent says 21
        
        return self.current_count, reward, self.done, {}, {}

    def render(self, mode='human'):
        """
        Render the current state of the game.
        """
        print(f"Current count: {self.current_count}")

# Register the environment (optional, for integration with Gym)
from gym.envs.registration import register

register(
    id='TwentyOneGame-v0',
    entry_point=__name__ + ':TwentyOneGameEnv',
)
