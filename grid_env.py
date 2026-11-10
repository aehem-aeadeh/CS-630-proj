import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridWorldEnv:
    def __init__(self, size=8, is_stochastic=False, custom_obstacles=None):
        self.size = size
        self.n_states = size * size  # 64 states (from 0 to 63)
        self.n_actions = 4           # 0: Up, 1: Down, 2: Left, 3: Right
        self.is_stochastic = is_stochastic
        
        # Start near upper-left, Goal near lower-right
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        
        # Default obstacles for 8x8 simple maze
        if custom_obstacles is not None:
            self.obstacles = custom_obstacles
        else:
            self.obstacles = [
                (1, 2), (1, 3), (2, 2),
                (3, 5), (4, 5), (5, 5),
                (5, 1), (5, 2), (6, 2),
                (2, 6), (6, 4)
            ]
            
        # Rewards specification
        self.step_reward = -1
        self.collision_penalty = -10
        self.goal_reward = 100
        self.gamma = 0.95  # Discount factor
        
        self.action_names = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
        self.action_deltas = {
            0: (-1, 0), # Up
            1: (1, 0),  # Down
            2: (0, -1), # Left
            3: (0, 1)   # Right
        }
        
        self.agent_pos = self.start_pos

    def pos_to_state(self, pos):
        return pos[0] * self.size + pos[1]

    def state_to_pos(self, state):
        return (state // self.size, state % self.size)

    def reset(self):
        self.agent_pos = self.start_pos
        return self.pos_to_state(self.agent_pos)

    def get_transitions(self, state, action):

        pos = self.state_to_pos(state)
        
        if pos == self.goal_pos or pos in self.obstacles:
            return [(1.0, state, 0, True)]

        transitions = []
        if not self.is_stochastic:
            intended_actions = [(1.0, action)]
        else:
            if action in [0, 1]: # Up or Down -> slip Left or Right
                slip_actions = [2, 3]
            else:                # Left or Right -> slip Up or Down
                slip_actions = [0, 1]
            intended_actions = [
                (0.8, action),
                (0.1, slip_actions[0]),
                (0.1, slip_actions[1])
            ]

        for prob, act in intended_actions:
            dr, dc = self.action_deltas[act]
            next_r, next_c = pos[0] + dr, pos[1] + dc
            next_pos = (next_r, next_c)

            if (next_r < 0 or next_r >= self.size or 
                next_c < 0 or next_c >= self.size or 
                next_pos in self.obstacles):
                next_state = state
                reward = self.collision_penalty
                done = False
            elif next_pos == self.goal_pos:
                next_state = self.pos_to_state(next_pos)
                reward = self.goal_reward
                done = True
            else:
                next_state = self.pos_to_state(next_pos)
                reward = self.step_reward
                done = False

            transitions.append((prob, next_state, reward, done))

        return transitions

    def step(self, action):
        current_state = self.pos_to_state(self.agent_pos)
        transitions = self.get_transitions(current_state, action)
        
        probs = [t[0] for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        _, next_state, reward, done = transitions[idx]
        
        self.agent_pos = self.state_to_pos(next_state)
        return next_state, reward, done

    def render(self, path=None, save_path=None, title="8x8 Grid World Environment"):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.invert_yaxis() 
        ax.set_xticks(np.arange(0, self.size + 1, 1))
        ax.set_yticks(np.arange(0, self.size + 1, 1))
        ax.grid(True, color='black', linewidth=1.2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for r in range(self.size):
            for c in range(self.size):
                if (r, c) == self.start_pos:
                    ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor='#4CAF50', alpha=0.8))
                    ax.text(c + 0.5, r + 0.5, 'S\n(Start)', ha='center', va='center', color='white', fontweight='bold')
                elif (r, c) == self.goal_pos:
                    ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor='#2196F3', alpha=0.8))
                    ax.text(c + 0.5, r + 0.5, 'G\n(Goal)', ha='center', va='center', color='white', fontweight='bold')
                elif (r, c) in self.obstacles:
                    ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor='#424242'))
                    ax.text(c + 0.5, r + 0.5, 'X', ha='center', va='center', color='white', fontweight='bold')
                else:
                    # كتابة رقم الحالة بخط صغير بالزاوية
                    state_num = self.pos_to_state((r, c))
                    ax.text(c + 0.15, r + 0.2, str(state_num), fontsize=8, color='gray')

        if path:
            path_coords = [self.state_to_pos(s) for s in path]
            ys = [p[0] + 0.5 for p in path_coords]
            xs = [p[1] + 0.5 for p in path_coords]
            ax.plot(xs, ys, color='red', marker='o', linewidth=2.5, markersize=6, label='Robot Path')
            ax.legend(loc='upper right')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✅ تم حفظ الرسمة بنجاح في: {save_path}")
        plt.show()

if __name__ == "__main__":
    env = GridWorldEnv(size=8, is_stochastic=False)
    env.render(save_path="task1_grid_environment.png")