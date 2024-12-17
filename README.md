# Budget-Safe Reinforcement Learning for Gambling
![image](https://github.com/user-attachments/assets/96de336e-53cd-4298-a744-d429dc4c2c02)

## Table of Contents
- [Introduction](#introduction)
  - [Motivation](#motivation)
  - [Goal](#goal)
  - [Environment](#environment)
  - [Directory Structure](#directory-structure)
  
- [Proposed Method](#proposed-method)
  - [Budget-Safe Q-learning](#budget-safe-q-learning)
  - [Situation-Aware Betting Strategy (SABS)](#situation-aware-betting-strategy-sabs)
- [Experiments](#experiments)
  - [Tic-Tac-Toe](#tic-tac-toe)
  - [Blackjack with Dynamic Betting](#blackjack-with-dynamic-betting)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)

---

## Introduction

### Motivation
Reinforcement Learning (RL) has achieved remarkable results in games such as Backgammon, Atari, Chess, and Go. However, applying RL to gambling games introduces a unique challenge: **budget constraints**. Without considering these constraints, an agent risks depleting its budget before learning an optimal strategy. 

This project aims to address this limitation by designing a **budget-safe RL algorithm** for gambling games with real-time budget constraints.

---

### Goal
Our objective is to develop an RL algorithm that ensures:
- **Low risk of bust**: Avoid running out of budget during learning.
- **High cumulative reward**: Maximize budget growth over time.
Our approach combines the advantages of both to achieve the goal.
---

### Environment

The following libraries and tools were used( Check requirenment.txt file)
- **Python**: 3.10.1
- **PyTorch**: 2.3.0+cu118
- **NumPy**: 1.26.4
- **FastAPI**: 0.111.0
- **Dash**: 2.17.1
- **Matplotlib**: 3.9.0
- **Gym Environment**: `Blackjack-v1`
---

## Directory Structure
The project is organized as follows:

- betting_agent       # Safe RL execution logic and betting strategy
- compare             # Compare different RL models for Blackjack
-  game                # Gameplay with the best-performing RL model
-  models              # Pretrained Blackjack RL models
-  main.py              # Main file to run the FastAPI application


## Proposed Method

### Budget-Safe Q-learning
- Incorporates Upper Confidence Bound (UCB) and Lower Confidence Bound (LCB) into Q-learning.
- Key features:
  - **QUCB**: Optimistically estimates the action-value function.
  - **QLCB**: Pessimistically estimates the action-value function to ensure safety.
- **Update Rule**:
  - Combines Temporal Difference (TD) error with UCB/LCB terms for exploration and safety.

### Situation-Aware Betting Strategy (SABS)
- Dynamically adjusts bets based on the current state and safety constraints.
- Ensures that bets remain within a safe range by leveraging QLCB for pessimistic estimation.
- The safety level is controlled by a parameter η.

---

## Experiments

### Tic-Tac-Toe
- Objective: Evaluate the algorithm in a simple turn-based game.
- Results:
  - Budget remains stable during learning, avoiding bust.
  - Betting strategy adapts to the level of learning.

### Blackjack with Dynamic Betting
- Introduces dynamic betting, where players can adjust bets mid-game.
- Results:
  - SABS prevents budget depletion and ensures long-term growth.
  - Adaptive betting strategies outperform fixed betting strategies.

---

## Conclusion
This project proposed a **Budget-Safe RL framework** that separates the learning and betting tasks:
- **Budget-Safe Q-learning** ensures efficient learning with safety constraints.
- **Situation-Aware Betting Strategy** dynamically adjusts bets to maximize rewards while minimizing risk.

Experiments demonstrated that our approach effectively balances risk and reward, making it a promising solution for budget-constrained RL applications.

## How to Run 

Follow the steps below to set up and run the RL project.

### 1. Clone the Repository
Clone the repository from GitHub:
```bash
git clone https://github.com/Junyoung0426/RL_Project.git
cd RL_Project
```

### 2. Create a Virtual Environment
Set up a Python virtual environment to manage dependencies:

#### On Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI Application
Start the FastAPI server:
```bash
python main.py
```

### 5. Access the Application
Once the server is running, open your browser and navigate to:

**URL:** [http://127.0.0.1:8080/game/](http://127.0.0.1:8080/game/)

#### Features:
- **Compare RL Models**: Evaluate the performance of different RL models for Blackjack.
- **Play Blackjack**: Interact and play Blackjack using trained RL models directly from the interface.

---

## Notes
- Ensure all dependencies are installed before running the application.
- The application uses FastAPI for the backend. If you encounter any issues, check the logs or verify the `main.py` script.

For further assistance or issues, please reach out via the [GitHub repository](https://github.com/Junyoung0426/RL_Project/issues).

