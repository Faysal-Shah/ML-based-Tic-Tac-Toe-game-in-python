import os
import random
import threading
import time
import pickle
from collections import defaultdict, deque
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np

# Optional imeports
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Try to import torch for DQN support
USE_DQN = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    E_DQN = True
except Exception:
    USE_DQN = False

# ------------------------
# Environment (same as before)
# ------------------------
class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0]*9
        self.current_player = 1
        self.done = False
        self.winner = None
        return tuple(self.board), self.current_player

    def available_actions(self):
        return [i for i,v in enumerate(self.board) if v==0]

    def step(self, action):
        if self.done:
            raise ValueError("Game finished")
        if self.board[action] != 0:
            self.done = True
            self.winner = -self.current_player
            return tuple(self.board), -10, True, {"illegal": True}
        self.board[action] = self.current_player
        self.winner = self.check_winner()
        if self.winner is not None:
            self.done = True
            if self.winner == 0:
                return tuple(self.board), 0, True, {"draw": True}
            else:
                return tuple(self.board), 1 if self.winner == self.current_player else -1, True, {"winner": self.winner}
        else:
            self.current_player *= -1
            return tuple(self.board), 0, False, {}

    def check_winner(self):
        b = self.board
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for (i,j,k) in wins:
            s = b[i]+b[j]+b[k]
            if s==3: return 1
            if s==-3: return -1
        if all(x!=0 for x in b):
            return 0
        return None

    def render_text(self):
        chars = {1:'X', -1:'O', 0:' '}
        out = ''
        for r in range(3):
            out += '|'.join(chars[self.board[3*r+c]] for c in range(3)) + '\n'
            if r<2: out += '-----\n'
        return out

# ------------------------
# Q-learning Agent
# ------------------------
class QLearningAgent:
    def __init__(self, alpha=0.8, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995):
        self.Q = defaultdict(lambda: np.zeros(9, dtype=float))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def get_canonical_state(self, state, player):
        return tuple([v*player for v in state])

    def choose_action(self, state, player, available_actions, explore=True):
        s = self.get_canonical_state(state, player)
        q = self.Q[s]
        if explore and random.random() < self.epsilon:
            return random.choice(available_actions)
        av_q = [(q[a], a) for a in available_actions]
        max_q = max(av_q, key=lambda x: x[0])[0]
        best = [a for qv,a in av_q if qv==max_q]
        return random.choice(best)

    def update(self, state, player, action, reward, next_state, next_player, done, next_avail):
        s = self.get_canonical_state(state, player)
        s_next = self.get_canonical_state(next_state, next_player) if next_state is not None else None
        q_sa = self.Q[s][action]
        if done:
            target = reward
        else:
            next_qs = self.Q[s_next]
            max_next = max(next_qs[a] for a in next_avail) if next_avail else 0.0
            target = reward + self.gamma * max_next
        self.Q[s][action] = q_sa + self.alpha*(target - q_sa)

    def decay_epsilon(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon<self.epsilon_min:
                self.epsilon = self.epsilon_min

    def save(self, filename='alphax_qtable.pkl'):
        with open(filename,'wb') as f:
            pickle.dump(dict(self.Q), f)
        print('Saved Q-table to', filename)

    def load(self, filename='alphax_qtable.pkl'):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        with open(filename,'rb') as f:
            qdict = pickle.load(f)
            self.Q = defaultdict(lambda: np.zeros(9, dtype=float), qdict)
        print('Loaded Q-table from', filename)

# ------------------------
# DQN Agent (optional)
# ------------------------
class DQNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,9)
        )
    def forward(self,x):
        return self.net(x)

class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995, device='cpu'):
        self.device = device
        self.net = DQNNet().to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay = deque(maxlen=10000)
        self.batch_size = 64

    def state_to_tensor(self, state, player):
        s = np.array(state, dtype=np.float32) * player
        return torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)

    def choose_action(self, state, player, available_actions, explore=True):
        if explore and random.random() < self.epsilon:
            return random.choice(available_actions)
        with torch.no_grad():
            t = self.state_to_tensor(state, player)
            out = self.net(t).cpu().numpy().flatten()
            # mask unavailable
            mask = np.full(9, -1e9)
            mask[available_actions] = out[available_actions]
            return int(np.argmax(mask))

    def push(self, transition):
        self.replay.append(transition)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        batch = random.sample(self.replay, self.batch_size)
        states = torch.tensor([np.array(b[0])*b[1] for b in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[2] for b in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([np.array(b[4])*b[5] if b[4] is not None else np.zeros(9) for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([1.0 if b[6] else 0.0 for b in batch], dtype=torch.float32, device=self.device)

        qvals = self.net(states)
        q_sa = qvals.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.net(next_states)
            q_next_max = q_next.max(1)[0]
            target = rewards + self.gamma * q_next_max * (1 - dones)
        loss = nn.functional.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def save(self, fname='alphax_dqn.pth'):
        torch.save({'net': self.net.state_dict(), 'opt': self.opt.state_dict()}, fname)
        print('Saved DQN to', fname)

    def load(self, fname='alphax_dqn.pth'):
        if not os.path.exists(fname):
            raise FileNotFoundError(fname)
        data = torch.load(fname, map_location=self.device)
        self.net.load_state_dict(data['net'])
        self.opt.load_state_dict(data['opt'])
        print('Loaded DQN from', fname)

# ------------------------
# Rule-based opponent
# ------------------------
def rule_based_policy(state, player, avail):
    b = list(state)
    for a in avail:
        b2 = b.copy(); b2[a]=player
        if check_winner_static(b2) == player:
            return a
    opp = -player
    for a in avail:
        b2 = b.copy(); b2[a]=opp
        if check_winner_static(b2) == opp:
            return a
    if 4 in avail: return 4
    corners=[0,2,6,8]
    ca=[c for c in corners if c in avail]
    if ca: return random.choice(ca)
    sides=[1,3,5,7]
    sa=[s for s in sides if s in avail]
    if sa: return random.choice(sa)
    return random.choice(avail)

def check_winner_static(b):
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for (i,j,k) in wins:
        s = b[i]+b[j]+b[k]
        if s==3: return 1
        if s==-3: return -1
    if all(x!=0 for x in b): return 0
    return None

# ------------------------
# GUI Application
# ------------------------
class AlphaXGUI:
    def __init__(self, root):
        self.root = root
        root.title('AlphaX - Tic Tac Toe')
        self.env = TicTacToeEnv()
        self.agent_type = 'q'  # 'q' or 'dqn'
        self.qagent = QLearningAgent()
        self.dqn_agent = DQNAgent() if USE_DQN else None
        self.human_is = None  # 1 for X, -1 for O
        self.human_starts = True
        self.game_speed = 0.2
        self.selfplay_speed = 0.01
        self.running = False
        self.create_widgets()
        self.update_status('Ready')

    def create_widgets(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, pady=8)
        self.info_label = tk.Label(top, text='AlphaX', font=('Arial',14))
        self.info_label.pack()

        board_frame = tk.Frame(self.root)
        board_frame.pack()
        self.cells = []
        for i in range(9):
            b = tk.Button(board_frame, text=' ', font=('Arial',30), width=4, height=2,
                          command=lambda i=i: self.cell_click(i))
            b.grid(row=i//3, column=i%3)
            self.cells.append(b)

        side = tk.Frame(self.root)
        side.pack(pady=8)

        # Controls
        self.play_btn = tk.Button(side, text='New Game', command=self.new_game)
        self.play_btn.grid(row=0, column=0, padx=4)
        self.random_assign_btn = tk.Button(side, text='Random X/O', command=self.random_assign)
        self.random_assign_btn.grid(row=0, column=1, padx=4)
        self.explain_var = tk.IntVar()
        tk.Checkbutton(side, text='Explain (show Q)', variable=self.explain_var).grid(row=0, column=2)

        tk.Label(side, text='Agent:').grid(row=1, column=0)
        self.agent_menu = tk.StringVar(value='q')
        tk.OptionMenu(side, self.agent_menu, 'q', 'dqn' if USE_DQN else 'q').grid(row=1, column=1)

        tk.Button(side, text='Train (self-play)', command=self.start_train_thread).grid(row=1, column=2)
        tk.Button(side, text='Self-Play Viz', command=self.start_selfplay_viz).grid(row=2, column=0)
        tk.Button(side, text='Evaluate', command=self.evaluate_agent).grid(row=2, column=1)

        tk.Button(side, text='Save Model', command=self.save_model).grid(row=3, column=0)
        tk.Button(side, text='Load Model', command=self.load_model).grid(row=3, column=1)

        tk.Label(side, text='Speed (play / self-play ms):').grid(row=4, column=0, columnspan=2)
        self.speed_scale = tk.Scale(side, from_=10, to=1000, orient=tk.HORIZONTAL)
        self.speed_scale.set(int(self.game_speed*1000))
        self.speed_scale.grid(row=4, column=2)

        self.status = tk.Label(self.root, text='', font=('Arial',12))
        self.status.pack(pady=6)

    def update_status(self, txt):
        self.status.config(text=txt)

    def random_assign(self):
        if random.random() < 0.5:
            self.human_is = 1
            self.human_starts = True
            messagebox.showinfo('Assigned', 'You are X and will start')
        else:
            self.human_is = -1
            self.human_starts = False
            messagebox.showinfo('Assigned', 'You are O and agent will start')
        self.new_game()

    def new_game(self):
        self.env.reset()
        self.refresh_board()
        # choose who starts
        if self.human_is is None:
            # default human is O (second)
            self.human_is = -1
            self.human_starts = False
        # if agent starts and is X
        if self.human_starts is False and self.human_is == -1:
            # agent plays first
            self.root.after(100, self.agent_move)
        self.update_status('New game')

    def refresh_board(self):
        chars={1:'X',-1:'O',0:' '}
        for i,v in enumerate(self.env.board):
            self.cells[i].config(text=chars[v])
        self.root.update()

    def cell_click(self, idx):
        if self.env.done:
            messagebox.showinfo('Game over','Start a new game')
            return
        # if human's turn
        if self.env.current_player == self.human_is:
            if self.env.board[idx] != 0:
                return
            self.env.step(idx)
            self.refresh_board()
            if self.env.done:
                self.on_game_end()
                return
            # agent move
            self.root.after(100, self.agent_move)

    def agent_move(self):
        if self.env.done: return
        avail = self.env.available_actions()
        agent_kind = self.agent_menu.get()
        if agent_kind == 'q':
            a = self.qagent.choose_action(tuple(self.env.board), self.env.current_player, avail, explore=False)
        else:
            if self.dqn_agent is not None:
                a = self.dqn_agent.choose_action(tuple(self.env.board), self.env.current_player, avail, explore=False)
            else:
                a = random.choice(avail)
        # optionally show explainable Q-values
        if self.explain_var.get() and agent_kind == 'q':
            s = self.qagent.get_canonical_state(tuple(self.env.board), self.env.current_player)
            qvals = self.qagent.Q[s]
            txts = [f'{i}:{qvals[i]:.2f}' if i in avail else '' for i in range(9)]
            messagebox.showinfo('Q-values','\n'.join(txts))
        self.env.step(a)
        self.refresh_board()
        if self.env.done:
            self.on_game_end()

    def on_game_end(self):
        w = self.env.winner
        if w == 1:
            msg = 'X wins'
        elif w == -1:
            msg = 'O wins'
        else:
            msg = 'Draw'
        self.update_status('Game over: '+msg)
        messagebox.showinfo('Result', msg)

    # ------------------------
    # Training (self-play) manager
    # ------------------------
    def start_train_thread(self):
        if self.running:
            messagebox.showwarning('Busy','Already running')
            return
        t = threading.Thread(target=self.train_self_play, daemon=True)
        t.start()

    def train_self_play(self, episodes=20000):
        self.running = True
        self.update_status('Training...')
        env = TicTacToeEnv()
        # choose backend
        backend = self.agent_menu.get()
        if backend == 'q':
            agent = self.qagent
        else:
            agent = self.dqn_agent if self.dqn_agent is not None else self.qagent
        window = deque(maxlen=1000)
        log_every = max(1, episodes//10)
        for ep in range(1, episodes+1):
            state, player = env.reset()
            done = False
            while not done:
                avail = env.available_actions()
                a = agent.choose_action(state, player, avail, explore=True)
                next_state, reward, done, info = env.step(a)
                next_avail = env.available_actions() if not done else []
                next_player = env.current_player if not done else None
                # update
                if backend == 'q':
                    agent.update(state, player, a, reward, next_state if not done else None, next_player if not done else None, done, next_avail)
                else:
                    # DQN transitions
                    agent.push((state, player, a, reward, next_state if not done else None, next_player if not done else None, done))
                    agent.learn()
                state = next_state
                player = env.current_player if not done else player
            # record result relative to player 1
            if env.winner == 1:
                window.append(1)
            elif env.winner == -1:
                window.append(-1)
            else:
                window.append(0)
            agent.decay_epsilon()
            if ep % log_every == 0 or ep==1:
                wins = sum(1 for x in window if x==1)
                losses = sum(1 for x in window if x==-1)
                draws = sum(1 for x in window if x==0)
                print(f'Ep {ep}/{episodes} | last {len(window)}: w{wins} l{losses} d{draws} eps={agent.epsilon:.4f}')
        # save after training
        if backend == 'q':
            agent.save()
        else:
            if self.dqn_agent is not None:
                agent.save()
        self.update_status('Training finished')
        if plt:
            # optionally show plots
            pass
        self.running = False

    # ------------------------
    # Self-play visualizer: play games and animate on the board
    # ------------------------
    def start_selfplay_viz(self, games=200):
        if self.running:
            messagebox.showwarning('Busy','Already running')
            return
        t = threading.Thread(target=self.selfplay_viz_thread, args=(games,), daemon=True)
        t.start()

    def selfplay_viz_thread(self, games=200):
        self.running = True
        self.update_status('Self-play visualization...')
        env = TicTacToeEnv()
        backend = self.agent_menu.get()
        agent = self.qagent if backend=='q' else (self.dqn_agent if self.dqn_agent is not None else self.qagent)
        speed_ms = max(10, self.speed_scale.get())
        for g in range(games):
            env.reset()
            self.refresh_board()
            while not env.done:
                avail = env.available_actions()
                a = agent.choose_action(tuple(env.board), env.current_player, avail, explore=False)
                env.step(a)
                self.refresh_board()
                time.sleep(speed_ms/1000.0)
            # small pause between games
            time.sleep(0.05)
        self.update_status('Self-play done')
        self.running = False

    # ------------------------
    # Evaluate agent vs baseline
    # ------------------------
    def evaluate_agent(self, episodes=1000):
        if self.running:
            messagebox.showwarning('Busy','Already running')
            return
        t = threading.Thread(target=self._evaluate_thread, args=(episodes,), daemon=True)
        t.start()

    def _evaluate_thread(self, episodes):
        self.running = True
        self.update_status('Evaluating...')
        env = TicTacToeEnv()
        backend = self.agent_menu.get()
        agent = self.qagent if backend=='q' else (self.dqn_agent if self.dqn_agent is not None else self.qagent)
        res = {'win':0,'loss':0,'draw':0}
        for _ in range(episodes):
            env.reset()
            while not env.done:
                if env.current_player == 1:
                    a = agent.choose_action(tuple(env.board), 1, env.available_actions(), explore=False)
                else:
                    a = rule_based_policy(tuple(env.board), -1, env.available_actions())
                env.step(a)
            if env.winner == 1: res['win'] += 1
            elif env.winner == -1: res['loss'] += 1
            else: res['draw'] += 1
        self.update_status(f"Eval done: {res}")
        messagebox.showinfo('Evaluation', str(res))
        self.running = False

    # ------------------------
    # Save / Load
    # ------------------------
    def save_model(self):
        backend = self.agent_menu.get()
        if backend == 'q':
            f = filedialog.asksaveasfilename(defaultextension='.pkl', filetypes=[('Pickle','*.pkl')], title='Save Q-table')
            if f:
                with open(f,'wb') as fh:
                    pickle.dump(dict(self.qagent.Q), fh)
                messagebox.showinfo('Saved', f'Saved Q-table to {f}')
        else:
            if self.dqn_agent is None:
                messagebox.showwarning('No DQN', 'DQN backend not available')
                return
            f = filedialog.asksaveasfilename(defaultextension='.pth', filetypes=[('PyTorch','*.pth')], title='Save DQN')
            if f:
                torch.save({'net': self.dqn_agent.net.state_dict(), 'opt': self.dqn_agent.opt.state_dict()}, f)
                messagebox.showinfo('Saved', f'Saved DQN to {f}')

    def load_model(self):
        backend = self.agent_menu.get()
        if backend == 'q':
            f = filedialog.askopenfilename(filetypes=[('Pickle','*.pkl')], title='Load Q-table')
            if f:
                with open(f,'rb') as fh:
                    qdict = pickle.load(fh)
                    self.qagent.Q = defaultdict(lambda: np.zeros(9, dtype=float), qdict)
                messagebox.showinfo('Loaded', f'Loaded Q-table from {f}')
        else:
            if self.dqn_agent is None:
                messagebox.showwarning('No DQN', 'DQN backend not available')
                return
            f = filedialog.askopenfilename(filetypes=[('PyTorch','*.pth')], title='Load DQN')
            if f:
                data = torch.load(f, map_location='cpu')
                self.dqn_agent.net.load_state_dict(data['net'])
                self.dqn_agent.opt.load_state_dict(data['opt'])
                messagebox.showinfo('Loaded', f'Loaded DQN from {f}')

# ------------------------
# Entry point
# ------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = AlphaXGUI(root)
    root.mainloop()
