from pathlib import Path
import sys
import dash
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output, State
from flask import Flask
import random
import gym
import numpy as np
from fastapi.responses import RedirectResponse

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "models"))
from dqn.dqn_agent import DQNAgent

app = FastAPI()


external_stylesheets = ["./assets/style.css"]

dash_app = Dash(
    __name__,
    requests_pathname_prefix="/game/",  
    external_stylesheets=external_stylesheets
)

def get_card_image(value, suit):
    card_value = str(value).upper()
    card_suit = suit.upper().replace(" ", "_")
    card_filename = f"{card_value}_of_{card_suit}.png"
    return f"assets/card_images/{card_filename}"

suits = ["HEARTS", "DIAMONDS", "CLUBS", "SPADES"]
values = [2, 3, 4, 5, 6, 7, 8, 9, 10, "JACK", "QUEEN", "KING", "ACE"]

def draw_random_card():
    value = random.choice(values)
    suit = random.choice(suits)
    return (value, suit)

def calculate_hand_value(cards):
    value = 0
    aces = 0
    for card in cards:
        if card[0] in ["JACK", "QUEEN", "KING"]:
            value += 10
        elif card[0] == "ACE":
            value += 11
            aces += 1
        else:
            value += int(card[0])
    while value > 21 and aces:
        value -= 10
        aces -= 1
    return value

score = {"win": 0, "lose": 0, "draw": 0, "total_reward": 0, "Current_Money": 0}
initial_player_cards = []
initial_dealer_cards = []
game_over = False


env = gym.make('Blackjack-v1')
agent = DQNAgent(env)
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models/action_file/dqn_model.pth"
agent.load_model(str(model_path))



dash_app.layout = html.Div([
    html.H1("Blackjack Game", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Initial Balance:", style={'fontSize': '16px', 'marginRight': '10px'}),
        dcc.Input(
            id="initial-balance", 
            type="number", 
            value=1000000, 
            min=0, 
            disabled=False,  
            style={'width': '100px', 'textAlign': 'center'}
        ),
        html.Button(
            "▲", id="initial-balance-up", 
            n_clicks=0, disabled=False,
            style={'marginLeft': '5px', 'padding': '0 10px', 'background-color': '#007BFF', 'color': 'white'}
        ),
        html.Button(
            "▼", id="initial-balance-down", 
            n_clicks=0, disabled=False,
            style={'marginLeft': '5px', 'padding': '0 10px', 'background-color': '#007BFF', 'color': 'white'}
        ),
        html.Label("Bet Amount:", style={'marginLeft': '20px'}),
        dcc.Input(
            id="bet-amount", 
            type="number", 
            value=100, 
            min=1, 
            style={'width': '100px', 'textAlign': 'center'}
        ),
        html.Button(
            "▲", id="bet-amount-up", 
            n_clicks=0, 
            style={'marginLeft': '5px', 'padding': '0 10px', 'background-color': '#007BFF', 'color': 'white'}
        ),
        html.Button(
            "▼", id="bet-amount-down", 
            n_clicks=0, 
            style={'marginLeft': '5px', 'padding': '0 10px', 'background-color': '#007BFF', 'color': 'white'}
        ),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div([
        html.H3("Dealer's Cards", style={'textAlign': 'center'}),
        html.Div(id="dealer-cards", className="card-container", style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
        html.Div(id="dealer-sum", style={'textAlign': 'center', 'color': 'white', 'fontSize': '20px'})  
    ]),
    html.Div([
        html.H3("Your Cards", style={'textAlign': 'center'}),
        html.Div(id="player-cards", className="card-container", style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
        html.Div(id="player-sum", style={'textAlign': 'center', 'color': 'white', 'fontSize': '20px'})  
    ]),
    html.Div([
        html.H3("Current Money", style={'textAlign': 'center'}),
        html.Div(id="current-money-display", style={
            'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px','marginBottom': '50px',
            'border': '2px solid #007BFF', 'padding': '10px', 'borderRadius': '10px', 'color': 'white'
        })
    ], style={'position': 'absolute', 'top': '50%', 'right': '10%', 'transform': 'translateY(-50%)'}),
    html.Div([
        html.Button("Play", id="play-btn", n_clicks=0, disabled=False),
        html.Button("Hit", id="hit-btn", n_clicks=0, disabled=True),
        html.Button("Stand", id="stand-btn", n_clicks=0, style={'marginLeft': '10px'}, disabled=True),
        html.Button("New Round", id="new-round-btn", n_clicks=0, style={'marginLeft': '10px'}, disabled=True),
        html.Button("AI Play", id="ai-play-btn", n_clicks=0, style={'marginLeft': '10px'}, disabled=True)
    ], style={'textAlign': 'center', 'marginTop': '20px'}),
    html.Div(id="game-message", style={'textAlign': 'center', 'marginTop': '20px'}),
    html.Div([
        html.H3("Scoreboard", style={'textAlign': 'center'}),
        html.Div(id="scoreboard", className="scoreboard-container")
    ])
])

def render_card_images(cards):
    return [html.Img(src=get_card_image(value, suit), className="card") for (value, suit) in cards]

def update_scoreboard():
    total_games = score['win'] + score['lose'] + score['draw']
    winning_rate = (score['win'] / total_games) * 100 if total_games > 0 else 0 
    return f"Wins: {score['win']} | Losses: {score['lose']} | Draws: {score['draw']} | Total Reward: {score['total_reward']} | Winning Rate: {winning_rate:.3f}%"


@dash_app.callback(
    [
        Output("initial-balance", "value"),
        Output("initial-balance", "disabled"),  
        Output("initial-balance-up", "disabled"),
        Output("initial-balance-down", "disabled"),
    ],
    [
        Input("initial-balance-up", "n_clicks"), 
        Input("initial-balance-down", "n_clicks"),
        Input("play-btn", "n_clicks"),  
    ],
    State("initial-balance", "value")
)
def update_initial_balance(up_clicks, down_clicks, play_clicks, current_balance):
    ctx = callback_context.triggered[0]["prop_id"]
    if ctx == "play-btn.n_clicks":
        return current_balance, True, True, True  
    elif ctx == "initial-balance-up.n_clicks":
        return current_balance + 10000, False, False, False
    elif ctx == "initial-balance-down.n_clicks" and current_balance >= 10000:
        return current_balance - 10000, False, False, False
    return current_balance, False, False, False

@dash_app.callback(
    Output("bet-amount", "value"),
    [Input("bet-amount-up", "n_clicks"), Input("bet-amount-down", "n_clicks")],
    State("bet-amount", "value")
)
def update_bet_amount(up_clicks, down_clicks, current_bet):
    ctx = callback_context.triggered[0]["prop_id"]
    if ctx == "bet-amount-up.n_clicks":
        return current_bet + 100
    elif ctx == "bet-amount-down.n_clicks" and current_bet > 100:
        return current_bet - 100
    return current_bet

@dash_app.callback(
    [
        Output("dealer-cards", "children"),
        Output("player-cards", "children"),
        Output("game-message", "children"),
        Output("scoreboard", "children"),
        Output("dealer-sum", "children"),
        Output("player-sum", "children"),
        Output("new-round-btn", "disabled"),
        Output("hit-btn", "disabled"),
        Output("stand-btn", "disabled"),
        Output("play-btn", "disabled"),
        Output("ai-play-btn", "disabled"),
        Output("current-money-display", "children")  
    ],
    [Input("play-btn", "n_clicks"), Input("hit-btn", "n_clicks"), Input("stand-btn", "n_clicks"), Input("new-round-btn", "n_clicks"), Input("ai-play-btn", "n_clicks")],
    [State("initial-balance", "value"), State("bet-amount", "value")]
)
def update_game(play_clicks, hit_clicks, stand_clicks, new_round_clicks, ai_play_clicks, initial_balance, bet_amount):
    global initial_player_cards, initial_dealer_cards, score, game_over

    dealer_value = calculate_hand_value(initial_dealer_cards) if initial_dealer_cards else 0
    player_value = calculate_hand_value(initial_player_cards) if initial_player_cards else 0

    ctx = callback_context
    triggered_button = ctx.triggered[0]['prop_id'].split('.')[0]
    if play_clicks == 0:
        return (
            render_card_images([]),
            render_card_images([]),
            "Click 'Play' to Start", update_scoreboard(),
            "", "",
            True, True, True, False, True,  
            f"Current Money: {score['Current_Money']}"
    )
    if triggered_button == "play-btn":
        score = {"win": 0, "lose": 0, "draw": 0, "total_reward": 0, 'Current_Money': initial_balance}
        initial_player_cards = [draw_random_card(), draw_random_card()]
        initial_dealer_cards = [draw_random_card()]
        game_over = False
        dealer_value = calculate_hand_value(initial_dealer_cards)
        player_value = calculate_hand_value(initial_player_cards)
        message = f"Game Started with Initial Balance: {initial_balance}, Bet Amount: {bet_amount}"
        return (
            render_card_images([initial_dealer_cards[0]] if initial_dealer_cards else []),
            render_card_images(initial_player_cards if initial_player_cards else []),
            message, update_scoreboard(),
            f"Dealer Sum: {dealer_value}", f"Your Sum: {player_value}",
            True, False, False, True, False,
            f"Current Money: {score['Current_Money']}"  
        )

    
    elif triggered_button == "new-round-btn":
        initial_player_cards = [draw_random_card(), draw_random_card()]
        initial_dealer_cards = [draw_random_card()]
        game_over = False
        dealer_value = calculate_hand_value(initial_dealer_cards)
        player_value = calculate_hand_value(initial_player_cards)
        message = "New Round Started!"
        return (
            render_card_images([initial_dealer_cards[0]] if initial_dealer_cards else []),  
            render_card_images(initial_player_cards if initial_player_cards else []),       
            message, update_scoreboard(),
            f"Dealer Sum: {dealer_value}", f"Your Sum: {player_value}",
            True, False, False, True, False  ,
            f"Current Money: {score['Current_Money']}"
        )


    elif triggered_button == "ai-play-btn" and not game_over:
        player_value = calculate_hand_value(initial_player_cards)
        dealer_open_card = initial_dealer_cards[0][0] if isinstance(initial_dealer_cards[0][0], int) else 10
        is_soft = any(card[0] == "ACE" for card in initial_player_cards) and player_value + 10 <= 21
        state = (player_value, dealer_open_card, is_soft)
        try:
            action = agent.act(state, testing=True)
        except Exception as e:
            print(f"Exception in agent.act: {e}")
            action = np.random.choice([0, 1])  
        if action == 1:  
            initial_player_cards.append(draw_random_card())
            player_value = calculate_hand_value(initial_player_cards)
            if player_value > 21:
                message = "Bust! You exceeded 21. You Lose."
                score["lose"] += 1
                score["total_reward"] -= bet_amount
                score["Current_Money"] -= bet_amount
                game_over = True
                dealer_value = calculate_hand_value(initial_dealer_cards)
                return (
                    render_card_images(initial_dealer_cards),
                    render_card_images(initial_player_cards),
                    message, update_scoreboard(),
                    f"Dealer Sum: {dealer_value}", f"Your Sum: {player_value}",
                    False, True, True, True, True,
                    f"Current Money: {score['Current_Money']}"
                )
            else:
                message = "AI chose Hit. Continue..."
                return (
                    render_card_images([initial_dealer_cards[0]] if initial_dealer_cards else []),  
                    render_card_images(initial_player_cards if initial_player_cards else []),       
                    message, update_scoreboard(),
                    f"Dealer Sum: {dealer_value}", f"Your Sum: {player_value}",
                    True, False, False, True, False,
                    f"Current Money: {score['Current_Money']}"
                )
        elif action == 0:  
            dealer_value = calculate_hand_value(initial_dealer_cards)
            while dealer_value < 17:
                initial_dealer_cards.append(draw_random_card())
                dealer_value = calculate_hand_value(initial_dealer_cards)
            if dealer_value > 21 or player_value > dealer_value:
                message = "You Win!"
                score["win"] += 1
                score["total_reward"] += bet_amount
                score["Current_Money"] += bet_amount
            elif player_value < dealer_value:
                message = "You Lose!"
                score["lose"] += 1
                score["total_reward"] -= bet_amount
                score["Current_Money"] -= bet_amount
            else:
                message = "Draw!"
                score["draw"] += 1
            game_over = True
            return (
                render_card_images(initial_dealer_cards),
                render_card_images(initial_player_cards),
                message, update_scoreboard(),
                f"Dealer Sum: {dealer_value}", f"Your Sum: {player_value}",
                False, True, True, True, True,
                f"Current Money: {score['Current_Money']}"
            )

    elif triggered_button == "hit-btn" and not game_over:
        initial_player_cards.append(draw_random_card())
        player_value = calculate_hand_value(initial_player_cards)
        if player_value > 21:
            message = "Bust! You exceeded 21. You Lose."
            score["lose"] += 1
            score["total_reward"] -= bet_amount
            score["Current_Money"] -= bet_amount

            game_over = True
            dealer_value = calculate_hand_value(initial_dealer_cards)
            return (
                render_card_images(initial_dealer_cards),
                render_card_images(initial_player_cards),
                message, update_scoreboard(),
                f"Dealer Sum: {dealer_value}", f"Your Sum: {player_value}",
                False, True, True, True, True,
                f"Current Money: {score['Current_Money']}"
            )
        return (
            render_card_images([initial_dealer_cards[0]] if initial_dealer_cards else []),  
            render_card_images(initial_player_cards if initial_player_cards else []),       
            "Hit or Stand?", update_scoreboard(),
            f"Dealer Sum: {dealer_value}", f"Your Sum: {player_value}",
            True, False, False, True, False,
            f"Current Money: {score['Current_Money']}"
        )
    

    elif triggered_button == "stand-btn" and not game_over:
        dealer_value = calculate_hand_value(initial_dealer_cards)
        while dealer_value < 17:
            initial_dealer_cards.append(draw_random_card())
            dealer_value = calculate_hand_value(initial_dealer_cards)

        if dealer_value > 21 or player_value > dealer_value:
            message = "You Win!"
            score["win"] += 1
            score["total_reward"] += bet_amount
            score["Current_Money"] += bet_amount

        elif player_value < dealer_value:
            message = "You Lose!"
            score["lose"] += 1
            score["total_reward"] -= bet_amount
            score["Current_Money"] -= bet_amount

        else:
            message = "Draw!"
            score["draw"] += 1

        game_over = True
        return render_card_images(initial_dealer_cards), render_card_images(initial_player_cards), message, update_scoreboard(), f"Dealer Sum: {dealer_value}", f"Your Sum: {player_value}", False, True, True, True, True,f"Current Money: {score['Current_Money']}"

    return (
        render_card_images([initial_dealer_cards[0]] if initial_dealer_cards else []),      
        render_card_images(initial_player_cards if initial_player_cards else []),           
        "Game in Progress", update_scoreboard(),
        f"Dealer Sum: {dealer_value}", f"Your Sum: {player_value}",
        not game_over, game_over, game_over, game_over, game_over,f"Current Money: {score['Current_Money']}"
    )

app.mount("/", WSGIMiddleware(dash_app.server))

