from pathlib import Path
import sys
import dash
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import gym
import random
import numpy as np
import pickle
from flask import Flask

from fastapi.responses import RedirectResponse

app = FastAPI()
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "models"))
from ddqn.ddqn_agent import DDQNAgent
from dqn.dqn_agent import DQNAgent


def get_card_image(value, suit):
    card_value = str(value).upper()
    card_suit = suit.upper().replace(" ", "_")
    return f"assets/card_images/{card_value}_of_{card_suit}.png"

def render_card_images(cards):
    return [html.Img(src=get_card_image(value, suit), style={"height": "150px", "margin": "5px"}) for value, suit in cards]

with open("./compare/action_file/q_learning_model.pkl", "rb") as f:
    q_table = pickle.load(f)

with open("./compare/action_file/ucb_qlearning_model.pkl", "rb") as f:
    ucb_q_table = pickle.load(f)


env = gym.make("Blackjack-v1")
dqn_agent = DQNAgent(env)
dqn_agent.load_model("./compare/action_file/dqn_model.pth")

ddqn_agent = DDQNAgent(env)
ddqn_agent.load_model("./compare/action_file/ddqn_model.pth")
external_stylesheets = ["./assets/style.css"]

agents = ["q_learning", "ucb_q_learning", "dqn", "ddqn"]
current_balance = {agent: 1000000 for agent in agents}
bet_amount = 100
score = {agent: {"win": 0, "lose": 0, "draw": 0} for agent in agents}
winning_rate_list = {agent: [] for agent in agents}
balance_list = {agent: [1000000] for agent in agents}
player_cards = {agent: [] for agent in agents}
dealer_cards = []

def draw_random_card():
    values = [2, 3, 4, 5, 6, 7, 8, 9, 10, "JACK", "QUEEN", "KING", "ACE"]
    suits = ["HEARTS", "DIAMONDS", "CLUBS", "SPADES"]
    return random.choice(values), random.choice(suits)

def calculate_hand_value(cards):
    value, aces = 0, 0
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
external_stylesheets = ["./assets/style.css"]
def navbar():
    return html.Div(
        [
            html.A(
                "Game",
                href="/game",
                style={
                    "marginRight": "20px",
                    "padding": "10px 20px",
                    "textDecoration": "none",
                    "color": "#4CAF50", 
                    "backgroundColor": "#FFFFFF",  
                    "borderRadius": "5px",
                    "fontSize": "16px",
                    "fontWeight": "bold",
                    "boxShadow": "0px 2px 4px rgba(0, 0, 0, 0.2)", 
                    "transition": "background-color 0.3s ease",
                },
            ),
            html.A(
                "Compare Models",
                href="/compare",
                style={
                    "marginRight": "20px",
                    "padding": "10px 20px",
                    "textDecoration": "none",
                    "color": "#4CAF50",  
                    "backgroundColor": "#FFFFFF",  
                    "borderRadius": "5px",
                    "fontSize": "16px",
                    "fontWeight": "bold",
                    "boxShadow": "0px 2px 4px rgba(0, 0, 0, 0.2)",  
                    "transition": "background-color 0.3s ease",
                },
            ),
        ],
        style={
            "padding": "15px",
            "textAlign": "center",
            "backgroundColor": "rgba(255, 255, 255, 0.8)",  
            "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
            "borderRadius": "10px", 
            "margin": "10px 20px",  
        },
    )


dash_app = Dash(__name__, requests_pathname_prefix="/compare/",external_stylesheets=external_stylesheets)

dash_app.layout = html.Div([
    html.H1("AI Blackjack Simulation", style={"textAlign": "center"}),
    navbar(),
    

    html.Div([
        html.Label("Initial Balance:"),
        dcc.Input(id="initial-balance", type="number", value=1000000, min=0, style={"width": "100px", "marginRight": "20px"}),

        html.Label("Number of Games:"),
        dcc.Input(id="num-of-games", type="number", value=500, min=1, style={"width": "80px", "marginRight": "10px"}),

        html.Label("Betting Amount:"),
        dcc.Input(id="bet-amount", type="number", value=100, min=1, style={"width": "80px", "marginRight": "10px"}),

        html.Button("AI Play", id="ai-play-btn", style={"backgroundColor": "#28a745", "color": "white", "marginRight": "10px"}),
        html.Button("Stop", id="stop-btn", style={"backgroundColor": "#ffc107", "color": "black", "marginRight": "10px"}),
        html.Button("Reset", id="reset-btn", style={"backgroundColor": "#dc3545", "color": "white"})
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    html.Div([
        html.H3("Dealer's Cards", style={"textAlign": "center"}),
        html.Div(id="dealer-cards", style={"display": "flex", "justifyContent": "center", "marginBottom": "20px"}),
    ]),
    html.Div(
    [
        dcc.Graph(id="win-rate-graph", style={"marginBottom": "20px"}),  
        dcc.Graph(id="balance-trend-graph"),
    ],
    style={
        "display": "block",  
        "width": "100%", 
        "textAlign": "center",  
    },
),


    html.Div([
        html.Div([
            html.H4(f"{agent.upper()} Player's Cards", style={"textAlign": "center"}),
            html.Div(id=f"{agent}-player-cards", style={"display": "flex", "justifyContent": "center", "marginBottom": "20px"}),
        ]) for agent in agents
    ]),

    dcc.Interval(id="ai-interval", interval=10, n_intervals=0, disabled=True)
])

def generate_result_figures():
    win_rate_fig = {
        "data": [{"x": list(range(1, len(winning_rate_list[agent]) + 1)), "y": winning_rate_list[agent], "type": "line", "name": agent} for agent in agents],
        "layout": {"title": "Win Rate Over Games", "xaxis": {"title": "Games"}, "yaxis": {"title": "Win Rate (%)"}}
    }
    balance_fig = {
        "data": [{"x": list(range(1, len(balance_list[agent]) + 1)), "y": balance_list[agent], "type": "line", "name": agent} for agent in agents],
        "layout": {"title": "Balance Over Games", "xaxis": {"title": "Games"}, "yaxis": {"title": "Balance"}}
    }
    return win_rate_fig, balance_fig

def simulate_game(agent, initial_player_cards=None, initial_dealer_cards=None, shared_draws=None):
    global current_balance, score, player_cards, dealer_cards

    if initial_dealer_cards is not None:
        dealer_cards[:] = initial_dealer_cards
    else:
        if not dealer_cards:
            dealer_cards.extend([draw_random_card(), draw_random_card()])

    player_cards[agent] = [draw_random_card(), draw_random_card()]

    player_value = calculate_hand_value(player_cards[agent])
    dealer_value = calculate_hand_value(dealer_cards)

    shared_draw_index = 0
    
    while player_value < 17:
        if agent == "q_learning":
            if player_value not in q_table:
                action = np.random.choice([0, 1])  
            else:
                action = np.argmax(q_table[player_value])

        elif agent == "ucb_q_learning":
            if player_value not in ucb_q_table:
                action = np.random.choice([0, 1])  
            else:
                action = np.argmax(ucb_q_table[player_value])

        elif agent == "dqn":
            try:
                q_values = dqn_agent.model.predict(player_value)  
                action = np.argmax(q_values)
            except Exception as e:
                action = np.random.choice([0, 1])  

        elif agent == "ddqn":
            try:
                q_values = ddqn_agent.model.predict(player_value)  
                action = np.argmax(q_values)
            except Exception as e:
                action = np.random.choice([0, 1])  

        else:
            action = 0  

        if action == 1:
            if shared_draws and shared_draw_index < len(shared_draws):
                card = shared_draws[shared_draw_index]
                shared_draw_index += 1
            else:
                card = draw_random_card()
                if shared_draws is not None:
                    shared_draws.append(card)

            player_cards[agent].append(card)
            player_value = calculate_hand_value(player_cards[agent])
        else:
            break

    if player_value > 21 or (dealer_value <= 21 and dealer_value >= player_value):
        score[agent]["lose"] += 1
        current_balance[agent] -= bet_amount
    elif dealer_value > 21 or player_value > dealer_value:
        score[agent]["win"] += 1
        current_balance[agent] += bet_amount
    else:
        score[agent]["draw"] += 1

    total_games = sum(score[agent].values())
    winning_rate = (score[agent]["win"] / total_games) * 100 if total_games > 0 else 0
    winning_rate_list[agent].append(winning_rate)
    balance_list[agent].append(current_balance[agent])


@dash_app.callback(
    [
        Output("dealer-cards", "children"),
        *[Output(f"{agent}-player-cards", "children") for agent in agents],
        Output("win-rate-graph", "figure"),
        Output("balance-trend-graph", "figure"),
        Output("ai-interval", "disabled"),
        Output("ai-interval", "n_intervals"),
        Output("initial-balance", "value"),  
        Output("num-of-games", "value"),    
        Output("bet-amount", "value"),      
    ],
    [
        Input("ai-play-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        Input("ai-interval", "n_intervals"),
    ],
    [
        State("num-of-games", "value"),
        State("initial-balance", "value"),
        State("bet-amount", "value"),
    ],
)
def ai_play(ai_clicks, stop_clicks, reset_clicks, n_intervals, num_of_games, initial_balance, bet_amount_input):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [], *[[] for _ in agents], {}, {}, True, 0, initial_balance, num_of_games, bet_amount_input

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if triggered_id == "reset-btn":
        default_balance = 1000000
        default_games = 100
        default_bet = 100

        # 상태 초기화
        current_balance.update({agent: default_balance for agent in agents})
        dealer_cards.clear()
        player_cards.update({agent: [] for agent in agents})
        winning_rate_list.update({agent: [] for agent in agents})
        balance_list.update({agent: [default_balance] for agent in agents})

        return (
            [],  
            *[[] for _ in agents],  
            {},  
            {},  
            True,  
            0,  
            default_balance, 
            default_games,  
            default_bet,  
        )

    if triggered_id == "ai-play-btn":
        initial_dealer_cards = [draw_random_card(), draw_random_card()]
        initial_player_cards = [draw_random_card(), draw_random_card()]
        shared_draws = []  

        for agent in agents:
            simulate_game(agent, initial_player_cards, initial_dealer_cards, shared_draws)

        win_rate_fig, balance_fig = generate_result_figures()
        return render_card_images(dealer_cards), *[render_card_images(player_cards[agent]) for agent in agents], win_rate_fig, balance_fig, False, 0, initial_balance, num_of_games, bet_amount_input

    if triggered_id == "stop-btn":
        win_rate_fig, balance_fig = generate_result_figures()
        return (
            render_card_images(dealer_cards),
            *[render_card_images(player_cards[agent]) for agent in agents],
            win_rate_fig,
            balance_fig,
            True,   
            n_intervals,  
            initial_balance,
            num_of_games,
            bet_amount_input,
        )


    if triggered_id == "ai-interval" and n_intervals < num_of_games:
        initial_dealer_cards = [draw_random_card(), draw_random_card()]
        initial_player_cards = [draw_random_card(), draw_random_card()]
        for agent in agents:
            simulate_game(agent, initial_player_cards, initial_dealer_cards)
        win_rate_fig, balance_fig = generate_result_figures()
        return render_card_images(dealer_cards), *[render_card_images(player_cards[agent]) for agent in agents], win_rate_fig, balance_fig, False, n_intervals + 1, initial_balance, num_of_games, bet_amount_input

    if n_intervals >= num_of_games:
        win_rate_fig, balance_fig = generate_result_figures()
        return render_card_images(dealer_cards), *[render_card_images(player_cards[agent]) for agent in agents], win_rate_fig, balance_fig, True, n_intervals, initial_balance, num_of_games, bet_amount_input

    return [], *[[] for _ in agents], {}, {}, True, 0, initial_balance, num_of_games, bet_amount_input


app.mount("/", WSGIMiddleware(dash_app.server))