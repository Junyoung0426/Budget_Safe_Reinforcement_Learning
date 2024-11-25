import gym
from gym import spaces
import numpy as np
from collections import Counter

class CardCountingBlackjackEnv(gym.Env):
    def __init__(self):
        # Define action space: Stick (0) or Hit (1)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: Player hand sum, Dealer's face-up card, and remaining deck composition
        # Deck composition is a vector indicating the counts of each card rank (1-10)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # Player's hand sum (0-31)
            spaces.Discrete(11),  # Dealer's face-up card (1-10)
            spaces.Box(low=0, high=4, shape=(10,), dtype=np.int32)  # Deck composition
        ))
        
        # Initialize deck, player hand, and dealer hand
        self.reset()

    def reset(self):
        self.deck = self._initialize_deck()
        self.player_hand = []
        self.dealer_hand = []

        # Deal initial cards
        self.player_hand.append(self._draw_card())
        self.player_hand.append(self._draw_card())
        self.dealer_hand.append(self._draw_card())
        self.dealer_hand.append(self._draw_card())

        # Return the initial observation and additional info
        return self._get_observation(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        if action == 1:  # Hit
            self.player_hand.append(self._draw_card())
            if self._hand_value(self.player_hand) > 21:
                return self._get_observation(), -1, True, {}  # Player busts
        
        elif action == 0:  # Stick
            # Dealer's turn
            while self._hand_value(self.dealer_hand) < 17:
                self.dealer_hand.append(self._draw_card())
            
            # Compare hands
            player_value = self._hand_value(self.player_hand)
            dealer_value = self._hand_value(self.dealer_hand)
            
            if dealer_value > 21 or player_value > dealer_value:
                reward = 1  # Player wins
            elif player_value == dealer_value:
                reward = 0  # Draw
            else:
                reward = -1  # Dealer wins
            return self._get_observation(), reward, True, {}
        
        return self._get_observation(), 0, False, {}

    def render(self, mode='human'):
        print(f"Player hand: {self.player_hand} (value: {self._hand_value(self.player_hand)})")
        print(f"Dealer hand: {self.dealer_hand[:1]} + [hidden]")

    def _initialize_deck(self):
        # Create a standarda deck with 4 of each rank (1-10)
        deck = Counter({i: 4 for i in range(1, 11)})
        deck[10] *= 4  # 10 counts for face cards (J, Q, K)
        return deck

    def _draw_card(self):
        ranks = list(self.deck.elements())
        card = np.random.choice(ranks)
        self.deck[card] -= 1
        if self.deck[card] == 0:
            del self.deck[card]
        return card

    def _hand_value(self, hand):
        value = sum(hand)
        # Adjust for Aces (1 can also count as 11 if it doesn't bust the hand)
        aces = hand.count(1)
        while aces > 0 and value + 10 <= 21:
            value += 10
            aces -= 1
        return value

    def _get_observation(self):
        player_sum = self._hand_value(self.player_hand)
        dealer_face_up = self.dealer_hand[0]
        deck_composition = np.array([self.deck[i] for i in range(1, 11)])
        return (player_sum, dealer_face_up, deck_composition)

# Register the environment (optional for Gym compatibility)
# gym.envs.registration.register(
#     id='CardCountingBlackjack-v0',
#     entry_point=CardCountingBlackjackEnv,
#     max_episode_steps=100
# )

# Example usage
if __name__ == "__main__":
    env = CardCountingBlackjackEnv()
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action = int(input("Enter action (0=Stick, 1=Hit): "))
        obs, reward, done, _ = env.step(action)

    print(f"Game over! Reward: {reward}")