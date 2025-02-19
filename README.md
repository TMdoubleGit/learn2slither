# 🐍 Learn2Slither - Reinforcement Learning Snake Game

**Learn2Slither** is an AI-powered **Snake game** where a reinforcement learning agent learns to play using **Q-learning**. The goal is to train the agent to maximize its survival and reach a length of at least 10 cells.

## 🚀 Features
- **Reinforcement Learning (Q-learning)**
- **Custom 10x10 Board Environment**
- **State-based decision-making**
- **Reward system for learning optimization**
- **Graphical Display using Pygame**
- **Train & Save models using Joblib**

## 🎮 Installation & Setup

### 1️⃣ Clone the Repository

```sh 
git clone <your-repo-url>
cd Learn2Slither
```

### 2️⃣ Create & Activate Virtual Environment
``` sh
make install
. .venv/bin/activate
```
it will automatically setup your .venv and install all the dependencies
## 🏗️ Running the Project

### 1️⃣ Train the AI
Train the agent with a specified number of sessions:
```py
python src/train.py --sessions 500
```

This will generate trained models inside the /models/ folder.

### 2️⃣ Play with a Trained Model
Run the game with a pre-trained model:
python src/main.py --load models/100sess.pkl --sessions 10 --dontlearn --step-by-step

### 3️⃣ Run Tests
To ensure everything works correctly:
pytest tests/

## 🏆 Q-Learning Algorithm Overview
The agent follows the Q-learning approach to update its Q-table based on rewards.

### 🔹 State Representation
The snake's vision consists of 4 directions from its head:
W = Wall, H = Snake Head, S = Snake Body, G = Green Apple, R = Red Apple, 0 = Empty Space

### 🔹 Actions
The agent can move in 4 directions:

- UP
- DOWN
- LEFT
- RIGHT

### 🔹 Reward System
- Eating Green Apple → +1
- Eating Red Apple → -1
- Hitting Wall/Self → -10
- Doing Nothing → -0.1
### 🔹 Q-learning Update Rule
The Q-values are updated using the Bellman equation:
```sh 
Q(s, a) = Q(s, a) + α * (reward + γ * max(Q(s', a')) - Q(s, a))
```

Where:

- α (alpha) = Learning rate
- γ (gamma) = Discount factor
- max(Q(s', a')) = Maximum future reward estimate

### 🔧 Automate with Makefile
Run commands easily using make:
- make train      # Train the model
- make play       # Play using trained model
- make test       # Run unit tests
- make clean      # Remove cached files

