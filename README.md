# AI630 Project: Intelligent Autonomous Robot Navigation
**Course:** Robotics and Planning (AI630)  
**Domain:** 8x8 Grid-World Navigation using MDP Planning & Reinforcement Learning

## Requirements
- Python 3.8+
- `numpy`
- `matplotlib`
- `pygame` (for the interactive live simulator)

Install the required libraries:
```bash
pip install numpy matplotlib pygame
```

## Project Structure
```text
.
├── grid_env.py          # Task 1 & 2: 8x8 Grid World environment and MDP model
├── planning.py          # Task 3: Value Iteration & Policy Iteration
├── rl_agents.py         # Task 4: Q-Learning & SARSA agents
├── experiments.py       # Task 5 & 6: Experiments A, B, and C (multi-seed)
├── interactive_sim.py   # Interactive Pygame live simulator (Demo)
└── README.md            # Project documentation and run instructions
```

## How to Run
Run the scripts in the following order from the project directory:

1. **Task 1 & 2 — Environment & MDP Model (`grid_env.py`)**  
   Creates the 8x8 grid world, transition probabilities, rewards, and saves the environment map (`task1_grid_environment.png`).
   ```bash
   python grid_env.py
   ```

2. **Task 3 — Model-Based Planning (`planning.py`)**  
   Runs Value Iteration and Policy Iteration, saves value/policy maps at selected iterations (1, 5, and convergence), and plots the optimal path.
   ```bash
   python planning.py
   ```

3. **Task 4 — Model-Free Reinforcement Learning (`rl_agents.py`)**  
   Trains Q-Learning and SARSA agents, saves full Q-tables (`*.txt`), learned policy maps, and learning curves.
   ```bash
   python rl_agents.py
   ```

4. **Task 5 & 6 — Experiments A, B, and C (`experiments.py`)**  
   Runs multi-seed experiments comparing Planning vs. Learning, Exploration strategies (epsilon-greedy, alpha, gamma), and Environment Complexity (simple, complex, and 20% stochastic slip). Saves all plots and `experiments_summary.txt`.
   ```bash
   python experiments.py
   ```

5. **Interactive Live Simulation (`interactive_sim.py`)**  
   Launches a real-time Pygame visualization to watch the robot train and navigate live.
   ```bash
   python interactive_sim.py
   ```
   **Simulator Controls:**
   - `1` / `2` / `3` / `4`: Switch between Q-Learning, SARSA, Value Iteration, and Policy Iteration
   - `F`: Fast-forward training (+50 episodes)
   - `UP` / `DOWN`: Increase or decrease simulation speed (FPS)
   - `SPACE`: Pause / Resume
   - `M`: Toggle Simple vs. Complex maze
   - `S`: Toggle Deterministic vs. Stochastic (20% slip) dynamics
   - `A`: Show / Hide policy arrows
   - `R`: Reset training