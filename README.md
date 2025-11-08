# ❌ AlphaX – Machine Learning Tic-Tac-Toe Bot in Python

### 🔹 Overview
**AlphaX** is a **Machine Learning–based Tic-Tac-Toe game** implemented in **Python**, where the bot learns to play optimally through **Q-Learning** and optionally **Deep Q-Networks (DQN)**.  

The bot can train itself via **self-play**, improve over time, and challenge human players using an interactive **Tkinter GUI**.  
This project demonstrates reinforcement learning concepts applied to a classic board game, bridging rule-based and learning-based AI.

---

### 🧠 Core Features
- 🧩 **Reinforcement Learning Agents**  
  - **Q-Learning** agent for fast learning using a Q-table.  
  - Optional **DQN** agent leveraging neural networks for better generalization.  
- 🤖 **Self-Play Training** – Automatically generates game data to improve the bot.  
- 🎮 **Human vs Bot Play** – Play interactively through a GUI.  
- 🔁 **Evaluation Mode** – Test the trained agent against a rule-based baseline.  
- 💾 **Save & Load Models** – Persist trained Q-tables or DQN weights for reuse.  
- 📊 **Explainable Q-values** – Visualize the Q-values for each move (Q-agent only).  

---

### 🧰 Tech Stack
| Category | Tools / Frameworks |
|-----------|--------------------|
| **Programming Language** | Python |
| **Libraries** | `numpy`, `tkinter`, `pickle`, `torch` (optional for DQN), `matplotlib` (optional) |
| **Concepts Used** | Q-Learning, Deep Q-Networks, Self-Play, Reinforcement Learning |
| **Version Control** | Git & GitHub |

---

### 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/alphax-tictactoe.git
   cd alphax-tictactoe
