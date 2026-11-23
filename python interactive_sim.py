import sys
import pygame
import numpy as np
from grid_env import GridWorldEnv
from planning import value_iteration, policy_iteration
from rl_agents import choose_action

# --- إعدادات الشاشة والألوان ---
GRID_SIZE = 8
CELL_SIZE = 80
GRID_PX = GRID_SIZE * CELL_SIZE  # 640x640
PANEL_WIDTH = 300
WIDTH = GRID_PX + PANEL_WIDTH    # 940
HEIGHT = GRID_PX                 # 640

BG_COLOR = (245, 247, 250)
PANEL_BG = (30, 34, 42)
GRID_LINE = (200, 206, 214)
START_COLOR = (76, 175, 80)
GOAL_COLOR = (33, 150, 243)
OBSTACLE_COLOR = (55, 62, 71)
ROBOT_COLOR = (231, 76, 60)
TRAIL_COLOR = (255, 193, 7)
ARROW_COLOR = (211, 47, 47)
TEXT_WHITE = (240, 244, 248)
TEXT_MUTED = (160, 170, 185)
ACCENT_COLOR = (79, 195, 247)

# نفس المتاهة المعقدة من Experiment C
COMPLEX_OBSTACLES = [
    (0, 4),
    (1, 2), (1, 3), (1, 6),
    (2, 2), (2, 4), (2, 6),
    (3, 1), (3, 2), (3, 5),
    (4, 5),
    (5, 0), (5, 1), (5, 2), (5, 5),
    (6, 2), (6, 4), (6, 6)
]

SPEED_LEVELS = [4, 10, 25, 60, 120]  # مستويات السرعة (FPS)


class InteractiveSimulator:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI630 - Autonomous Robot Navigation Simulator")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_text = pygame.font.SysFont("segoeui", 16)
        self.font_small = pygame.font.SysFont("segoeui", 13)
        self.font_cell = pygame.font.SysFont("segoeui", 18, bold=True)

        # إعدادات البيئة والوضع الحالي
        self.is_stochastic = False
        self.is_complex = False
        self.mode = "Q-Learning"  # Q-Learning, SARSA, Value Iteration, Policy Iteration
        self.paused = False
        self.show_arrows = True
        self.speed_idx = 1  # يبدأ بسرعة 10 FPS (واضحة ومناسبة للمشاهدة)

        self.build_env_and_reset()

    def build_env_and_reset(self):
        obs = COMPLEX_OBSTACLES if self.is_complex else None
        self.env = GridWorldEnv(size=GRID_SIZE, is_stochastic=self.is_stochastic, custom_obstacles=obs)

        # متغيرات التعلم والتخطيط
        self.Q = np.zeros((self.env.n_states, self.env.n_actions))
        self.policy = np.zeros(self.env.n_states, dtype=int)
        self.alpha = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = 0.992
        self.min_epsilon = 0.05

        if self.mode == "Value Iteration":
            _, self.policy, _, _ = value_iteration(self.env)
        elif self.mode == "Policy Iteration":
            _, self.policy, _ = policy_iteration(self.env)

        # إحصائيات الجولات
        self.episode = 1
        self.last_reward = 0.0
        self.last_steps = 0
        self.collision_flash = 0

        self.reset_episode()

    def reset_episode(self):
        self.state = self.env.reset()
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.ep_collisions = 0
        self.trail = [self.env.state_to_pos(self.state)]

        if self.mode == "SARSA":
            self.curr_action = choose_action(self.Q, self.state, self.epsilon, self.env.n_actions)

    def step_once(self):
        """تنفيذ خطوة واحدة للروبوت حسب الخوارزمية المختارة"""
        if self.mode == "Q-Learning":
            action = choose_action(self.Q, self.state, self.epsilon, self.env.n_actions)
            next_state, reward, done = self.env.step(action)
            best_next_q = 0.0 if done else np.max(self.Q[next_state])
            self.Q[self.state, action] += self.alpha * (reward + self.env.gamma * best_next_q - self.Q[self.state, action])
            self.state = next_state

        elif self.mode == "SARSA":
            next_state, reward, done = self.env.step(self.curr_action)
            next_action = choose_action(self.Q, next_state, self.epsilon, self.env.n_actions)
            next_q = 0.0 if done else self.Q[next_state, next_action]
            self.Q[self.state, self.curr_action] += self.alpha * (reward + self.env.gamma * next_q - self.Q[self.state, self.curr_action])
            self.state = next_state
            self.curr_action = next_action

        else:  # Value Iteration or Policy Iteration
            action = self.policy[self.state]
            next_state, reward, done = self.env.step(action)
            self.state = next_state

        self.ep_reward += reward
        self.ep_steps += 1
        if reward == self.env.collision_penalty:
            self.ep_collisions += 1
            self.collision_flash = 4  # وميض أحمر خفيف عند الاصطدام

        pos = self.env.state_to_pos(self.state)
        if not self.trail or self.trail[-1] != pos:
            self.trail.append(pos)

        # إنهاء الجولة عند الوصول للهدف أو تجاوز 150 خطوة
        if done or self.ep_steps >= 150:
            self.last_reward = self.ep_reward
            self.last_steps = self.ep_steps
            self.episode += 1
            if self.mode in ["Q-Learning", "SARSA"]:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.reset_episode()

    def fast_forward_training(self, n_episodes=50):
        """تقديم سريع 50 جولة تدريب بلمح البصر (مثالي لعرض التطور قدام الدكتور)"""
        if self.mode not in ["Q-Learning", "SARSA"]:
            return
        for _ in range(n_episodes):
            state = self.env.reset()
            ep_reward = 0.0
            action = choose_action(self.Q, state, self.epsilon, self.env.n_actions) if self.mode == "SARSA" else None
            for step in range(150):
                if self.mode == "Q-Learning":
                    action = choose_action(self.Q, state, self.epsilon, self.env.n_actions)
                    next_state, reward, done = self.env.step(action)
                    best_next_q = 0.0 if done else np.max(self.Q[next_state])
                    self.Q[state, action] += self.alpha * (reward + self.env.gamma * best_next_q - self.Q[state, action])
                    state = next_state
                else:
                    next_state, reward, done = self.env.step(action)
                    next_action = choose_action(self.Q, next_state, self.epsilon, self.env.n_actions)
                    next_q = 0.0 if done else self.Q[next_state, next_action]
                    self.Q[state, action] += self.alpha * (reward + self.env.gamma * next_q - self.Q[state, action])
                    state = next_state
                    action = next_action
                ep_reward += reward
                if done:
                    break
            self.last_reward = ep_reward
            self.last_steps = step + 1
            self.episode += 1
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.reset_episode()

    def draw_arrow(self, r, c, action):
        """رسم سهم نظيف داخل المربع بدون الاعتماد على الخطوط"""
        cx = c * CELL_SIZE + CELL_SIZE // 2
        cy = r * CELL_SIZE + CELL_SIZE // 2
        length = 16
        head = 7

        if action == 0:  # Up
            start, end = (cx, cy + length), (cx, cy - length)
            pts = [end, (cx - head, cy - length + head), (cx + head, cy - length + head)]
        elif action == 1:  # Down
            start, end = (cx, cy - length), (cx, cy + length)
            pts = [end, (cx - head, cy + length - head), (cx + head, cy + length - head)]
        elif action == 2:  # Left
            start, end = (cx + length, cy), (cx - length, cy)
            pts = [end, (cx - length + head, cy - head), (cx - length + head, cy + head)]
        else:  # Right
            start, end = (cx - length, cy), (cx + length, cy)
            pts = [end, (cx + length - head, cy - head), (cx + length - head, cy + head)]

        pygame.draw.line(self.screen, ARROW_COLOR, start, end, 3)
        pygame.draw.polygon(self.screen, ARROW_COLOR, pts)

    def draw(self):
        self.screen.fill(BG_COLOR)

        # 1. رسم الشبكة والخلايا
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pos = (r, c)
                s = self.env.pos_to_state(pos)

                if pos == self.env.start_pos:
                    pygame.draw.rect(self.screen, START_COLOR, rect)
                    lbl = self.font_cell.render("START", True, TEXT_WHITE)
                    self.screen.blit(lbl, lbl.get_rect(center=rect.center))
                elif pos == self.env.goal_pos:
                    pygame.draw.rect(self.screen, GOAL_COLOR, rect)
                    lbl = self.font_cell.render("GOAL", True, TEXT_WHITE)
                    self.screen.blit(lbl, lbl.get_rect(center=rect.center))
                elif pos in self.env.obstacles:
                    pygame.draw.rect(self.screen, OBSTACLE_COLOR, rect)
                    lbl = self.font_cell.render("X", True, TEXT_WHITE)
                    self.screen.blit(lbl, lbl.get_rect(center=rect.center))
                else:
                    pygame.draw.rect(self.screen, BG_COLOR, rect)
                    # رسم أسهم السياسة الحالية
                    if self.show_arrows:
                        if self.mode in ["Value Iteration", "Policy Iteration"]:
                            self.draw_arrow(r, c, self.policy[s])
                        elif np.any(self.Q[s] != 0):
                            self.draw_arrow(r, c, int(np.argmax(self.Q[s])))

                pygame.draw.rect(self.screen, GRID_LINE, rect, 1)

        # 2. رسم أثر مسار الروبوت بالجولة الحالية
        if len(self.trail) > 1:
            points = [(c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2) for r, c in self.trail]
            pygame.draw.lines(self.screen, TRAIL_COLOR, False, points, 4)

        # 3. رسم الروبوت
        rr, rc = self.env.state_to_pos(self.state)
        center = (rc * CELL_SIZE + CELL_SIZE // 2, rr * CELL_SIZE + CELL_SIZE // 2)
        if self.collision_flash > 0:
            pygame.draw.circle(self.screen, (255, 50, 50), center, 30, 4)
            self.collision_flash -= 1
        pygame.draw.circle(self.screen, ROBOT_COLOR, center, 22)
        pygame.draw.circle(self.screen, TEXT_WHITE, center, 22, 2)
        r_lbl = self.font_small.render("BOT", True, TEXT_WHITE)
        self.screen.blit(r_lbl, r_lbl.get_rect(center=center))

        # 4. رسم لوحة المعلومات والتحكم على اليمين
        panel_rect = pygame.Rect(GRID_PX, 0, PANEL_WIDTH, HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)

        x = GRID_PX + 20
        y = 20

        def put_text(txt, font, color, dy=26):
            nonlocal y
            surf = font.render(txt, True, color)
            self.screen.blit(surf, (x, y))
            y += dy

        put_text("AI630 Robot Simulator", self.font_title, ACCENT_COLOR, 36)
        put_text(f"Mode: {self.mode}", self.font_text, TEXT_WHITE, 26)
        put_text(f"Maze: {'Complex (18 obs)' if self.is_complex else 'Simple (11 obs)'}", self.font_text, TEXT_WHITE, 26)
        put_text(f"Dynamics: {'Stochastic (20%)' if self.is_stochastic else 'Deterministic'}", self.font_text, TEXT_WHITE, 26)
        put_text(f"Speed: {SPEED_LEVELS[self.speed_idx]} FPS {'[PAUSED]' if self.paused else ''}", self.font_text, TRAIL_COLOR, 36)

        pygame.draw.line(self.screen, (70, 78, 90), (x, y - 10), (WIDTH - 20, y - 10), 1)

        put_text("Live Metrics", self.font_title, ACCENT_COLOR, 30)
        put_text(f"Episode: {self.episode}", self.font_text, TEXT_WHITE, 24)
        if self.mode in ["Q-Learning", "SARSA"]:
            put_text(f"Epsilon: {self.epsilon:.3f}", self.font_text, TEXT_WHITE, 24)
        else:
            put_text("Epsilon: N/A (Planned)", self.font_text, TEXT_MUTED, 24)
        put_text(f"Current Steps: {self.ep_steps}", self.font_text, TEXT_WHITE, 24)
        put_text(f"Current Reward: {self.ep_reward:.0f}", self.font_text, TEXT_WHITE, 24)
        put_text(f"Last Ep Reward: {self.last_reward:.0f}", self.font_text, START_COLOR, 24)
        put_text(f"Last Ep Steps: {self.last_steps}", self.font_text, START_COLOR, 36)

        pygame.draw.line(self.screen, (70, 78, 90), (x, y - 10), (WIDTH - 20, y - 10), 1)

        put_text("Controls", self.font_title, ACCENT_COLOR, 28)
        controls = [
            "[1] Q-Learning (Live)",
            "[2] SARSA (Live)",
            "[3] Value Iteration",
            "[4] Policy Iteration",
            "[F] Fast-Forward +50 Eps",
            "[UP/DN] Change Speed",
            "[SPACE] Pause / Resume",
            "[M] Simple / Complex Maze",
            "[S] Determ. / Stochastic",
            "[A] Toggle Policy Arrows",
            "[R] Reset Training",
        ]
        for line in controls:
            put_text(line, self.font_small, TEXT_MUTED, 20)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_1:
                        self.mode = "Q-Learning"
                        self.build_env_and_reset()
                    elif event.key == pygame.K_2:
                        self.mode = "SARSA"
                        self.build_env_and_reset()
                    elif event.key == pygame.K_3:
                        self.mode = "Value Iteration"
                        self.build_env_and_reset()
                    elif event.key == pygame.K_4:
                        self.mode = "Policy Iteration"
                        self.build_env_and_reset()
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_UP:
                        self.speed_idx = min(len(SPEED_LEVELS) - 1, self.speed_idx + 1)
                    elif event.key == pygame.K_DOWN:
                        self.speed_idx = max(0, self.speed_idx - 1)
                    elif event.key == pygame.K_f:
                        self.fast_forward_training(50)
                    elif event.key == pygame.K_m:
                        self.is_complex = not self.is_complex
                        self.build_env_and_reset()
                    elif event.key == pygame.K_s:
                        self.is_stochastic = not self.is_stochastic
                        self.build_env_and_reset()
                    elif event.key == pygame.K_a:
                        self.show_arrows = not self.show_arrows
                    elif event.key == pygame.K_r:
                        self.build_env_and_reset()

            if not self.paused:
                self.step_once()

            self.draw()
            self.clock.tick(SPEED_LEVELS[self.speed_idx])

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    sim = InteractiveSimulator()
    sim.run()