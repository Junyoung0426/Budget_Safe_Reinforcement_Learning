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
import torch
import uvicorn
from fastapi.responses import RedirectResponse

# FastAPI 애플리케이션 초기화
app = FastAPI()

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "models"))


# Flask 서버 초기화

# 카드 이미지 경로 함수
def get_card_image(value, suit):
    card_value = str(value).upper()
    card_suit = suit.upper().replace(" ", "_")
    return f"assets/card_images/{card_value}_of_{card_suit}.png"

def render_card_images(cards):
    return [html.Img(src=get_card_image(value, suit), style={"height": "150px", "margin": "5px"}) for value, suit in cards]

# 모델 로드
with open("./compare/action_file/q_learning_model.pkl", "rb") as f:
    q_table = pickle.load(f)

with open("./compare/action_file/ucb_qlearning_model.pkl", "rb") as f:
    ucb_q_table = pickle.load(f)

# DQN/DDQN 에이전트 초기화 (PyTorch 모델 로드)
from ddqn.ddqn_agent import DDQNAgent
from dqn.dqn_agent import DQNAgent
env = gym.make("Blackjack-v1")
dqn_agent = DQNAgent(env)
dqn_agent.load_model("./compare/action_file/dqn_model.pth")

ddqn_agent = DDQNAgent(env)
ddqn_agent.load_model("./compare/action_file/ddqn_model.pth")
external_stylesheets = ["./assets/style.css"]

# 전역 변수
agents = ["q_learning", "ucb_q_learning", "dqn", "ddqn"]
current_balance = {agent: 1000000 for agent in agents}
bet_amount = 100
score = {agent: {"win": 0, "lose": 0, "draw": 0} for agent in agents}
winning_rate_list = {agent: [] for agent in agents}
balance_list = {agent: [1000000] for agent in agents}
player_cards = {agent: [] for agent in agents}
dealer_cards = []

# 카드 계산 및 생성 함수
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
                    "color": "#4CAF50",  # 초록색 텍스트
                    "backgroundColor": "#FFFFFF",  # 흰색 버튼
                    "borderRadius": "5px",
                    "fontSize": "16px",
                    "fontWeight": "bold",
                    "boxShadow": "0px 2px 4px rgba(0, 0, 0, 0.2)",  # 버튼 그림자
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
                    "color": "#4CAF50",  # 초록색 텍스트
                    "backgroundColor": "#FFFFFF",  # 흰색 버튼
                    "borderRadius": "5px",
                    "fontSize": "16px",
                    "fontWeight": "bold",
                    "boxShadow": "0px 2px 4px rgba(0, 0, 0, 0.2)",  # 버튼 그림자
                    "transition": "background-color 0.3s ease",
                },
            ),
        ],
        style={
            "padding": "15px",
            "textAlign": "center",
            "backgroundColor": "rgba(255, 255, 255, 0.8)",  # 약간 투명한 흰색 배경
            "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
            "borderRadius": "10px",  # 살짝 둥근 네비게이션 바
            "margin": "10px 20px",  # 화면에 여백 추가
        },
    )

# Dash 앱 초기화
dash_app = Dash(__name__, requests_pathname_prefix="/compare/",external_stylesheets=external_stylesheets)

# Dash 레이아웃
dash_app.layout = html.Div([
    html.H1("AI Blackjack Simulation", style={"textAlign": "center"}),
    navbar(),
    

    html.Div([
        html.Label("Initial Balance:"),
        dcc.Input(id="initial-balance", type="number", value=1000000, min=0, style={"width": "100px", "marginRight": "20px"}),

        html.Label("Number of Games:"),
        dcc.Input(id="num-of-games", type="number", value=10, min=1, style={"width": "80px", "marginRight": "10px"}),

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
        dcc.Graph(id="win-rate-graph", style={"marginBottom": "20px"}),  # 그래프 간 간격 추가
        dcc.Graph(id="balance-trend-graph"),
    ],
    style={
        "display": "block",  # 위아래 배치
        "width": "100%",  # 전체 화면 너비 사용
        "textAlign": "center",  # 가운데 정렬 (옵션)
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

# 결과 생성 함수
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

# 게임 시뮬레이션
def simulate_game(agent, initial_player_cards=None, initial_dealer_cards=None, shared_draws=None):
    global current_balance, score, player_cards, dealer_cards

    # 초기 딜러 카드 설정
    if initial_dealer_cards is not None:
        dealer_cards[:] = initial_dealer_cards
    else:
        if not dealer_cards:
            dealer_cards.extend([draw_random_card(), draw_random_card()])

    # 초기 플레이어 카드 설정
    player_cards[agent] = [draw_random_card(), draw_random_card()]

    player_value = calculate_hand_value(player_cards[agent])
    dealer_value = calculate_hand_value(dealer_cards)

    # 공유된 드로우 인덱스 초기화
    shared_draw_index = 0
    
    while player_value < 17:
        if agent == "q_learning":
            # Q-learning: 상태가 Q-table에 없으면 랜덤 선택
            if player_value not in q_table:
                action = np.random.choice([0, 1])  # 랜덤 행동
            else:
                action = np.argmax(q_table[player_value])

        elif agent == "ucb_q_learning":
            # UCB: 상태가 UCB Q-table에 없으면 랜덤 선택
            if player_value not in ucb_q_table:
                action = np.random.choice([0, 1])  # 랜덤 행동
            else:
                action = np.argmax(ucb_q_table[player_value])

        elif agent == "dqn":
            # DQN: 상태가 새로운 경우 (예외 처리)
            try:
                q_values = dqn_agent.model.predict(player_value)  # Q-value 예측
                action = np.argmax(q_values)
            except Exception as e:
                action = np.random.choice([0, 1])  # 랜덤 행동

        elif agent == "ddqn":
            # DDQN: 상태가 새로운 경우 (예외 처리)
            try:
                q_values = ddqn_agent.model.predict(player_value)  # Q-value 예측
                action = np.argmax(q_values)
            except Exception as e:
                action = np.random.choice([0, 1])  # 랜덤 행동

        else:
            action = 0  # 기본값
    # while player_value < 17:
    #     if agent == "q_learning":
    #         action = q_table.get(player_value)
    #     elif agent == "ucb_q_learning":
    #         action =  ucb_q_table.get(player_value)
    #     elif agent == "dqn":
    #         action = dqn_agent.act(player_value, testing=True)  # 상태 벡터 전달
    #     elif agent == "ddqn":
    #         action = ddqn_agent.act(player_value, testing=True)  # 상태 벡터 전달
    #     else:
    #         action = 0

        if action == 1:
            if shared_draws and shared_draw_index < len(shared_draws):
                # 이미 공유된 카드를 사용
                card = shared_draws[shared_draw_index]
                shared_draw_index += 1
            else:
                # 새 카드를 생성하여 공유
                card = draw_random_card()
                if shared_draws is not None:
                    shared_draws.append(card)

            player_cards[agent].append(card)
            player_value = calculate_hand_value(player_cards[agent])
        else:
            break

    # 게임 결과 계산
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
        Output("initial-balance", "value"),  # 초기 잔고 초기화
        Output("num-of-games", "value"),    # 게임 수 초기화
        Output("bet-amount", "value"),      # 베팅 금액 초기화
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
        # 초기화된 값 설정
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
            [],  # 딜러 카드 초기화
            *[[] for _ in agents],  # 플레이어 카드 초기화
            {},  # 승률 그래프 초기화
            {},  # 잔고 그래프 초기화
            True,  # Interval 비활성화
            0,  # Interval 카운트 리셋
            default_balance,  # 초기 잔고 값
            default_games,  # 초기 게임 수
            default_bet,  # 초기 베팅 금액
        )

    if triggered_id == "ai-play-btn":
        initial_dealer_cards = [draw_random_card(), draw_random_card()]
        initial_player_cards = [draw_random_card(), draw_random_card()]
        shared_draws = []  # 모든 에이전트가 공유할 드로우 리스트

        for agent in agents:
            simulate_game(agent, initial_player_cards, initial_dealer_cards, shared_draws)

        win_rate_fig, balance_fig = generate_result_figures()
        return render_card_images(dealer_cards), *[render_card_images(player_cards[agent]) for agent in agents], win_rate_fig, balance_fig, False, 0, initial_balance, num_of_games, bet_amount_input

    if triggered_id == "stop-btn":
        # 현재 상태 유지
        win_rate_fig, balance_fig = generate_result_figures()
        return render_card_images(dealer_cards), *[render_card_images(player_cards[agent]) for agent in agents], win_rate_fig, balance_fig, True, n_intervals, initial_balance, num_of_games, bet_amount_input

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