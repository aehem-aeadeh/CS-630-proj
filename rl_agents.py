import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv
from planning import plot_values_and_policy, extract_optimal_path

def choose_action(Q, state, epsilon, n_actions):
    """استراتيجية Epsilon-Greedy: يا إما استكشاف عشوائي أو استغلال أفضل حركة"""
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)  # Exploration (حركة عشوائية)
    else:
        # Exploitation (أفضل حركة، مع كسر التعادل عشوائياً لو القيم متساوية)
        max_q = np.max(Q[state])
        best_actions = np.where(Q[state] == max_q)[0]
        return np.random.choice(best_actions)


def q_learning(env, episodes=500, alpha=0.1, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05, max_steps=200, seed=42):
    """
    خوارزمية Q-Learning (Off-Policy):
    تحدث قيمة الحركة بناءً على أعلى قيمة مستقبلية ممكنة max Q(s', a').
    """
    np.random.seed(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    
    rewards_history = []
    steps_history = []
    collisions_history = []
    success_history = []
    
    curr_epsilon = epsilon

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        collisions = 0
        done = False
        
        for step in range(max_steps):
            action = choose_action(Q, state, curr_epsilon, env.n_actions)
            next_state, reward, done = env.step(action)
            
            if reward == env.collision_penalty:
                collisions += 1
                
            # معادلة تحديث Q-Learning
            best_next_q = 0.0 if done else np.max(Q[next_state])
            td_target = reward + env.gamma * best_next_q
            Q[state, action] += alpha * (td_target - Q[state, action])
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        # تقليل إبسيلون تدريجياً إذا كان مفعّل (Decreasing schedule)
        if epsilon_decay is not None:
            curr_epsilon = max(min_epsilon, curr_epsilon * epsilon_decay)
            
        rewards_history.append(total_reward)
        steps_history.append(step + 1)
        collisions_history.append(collisions)
        success_history.append(1 if done else 0)

    metrics = {
        "rewards": rewards_history,
        "steps": steps_history,
        "collisions": collisions_history,
        "success": success_history
    }
    return Q, metrics


def sarsa(env, episodes=500, alpha=0.1, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05, max_steps=200, seed=42):
    """
    خوارزمية SARSA (On-Policy):
    تحدث قيمة الحركة بناءً على الحركة الفعلية التالية Q(s', a').
    """
    np.random.seed(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    
    rewards_history = []
    steps_history = []
    collisions_history = []
    success_history = []
    
    curr_epsilon = epsilon

    for ep in range(episodes):
        state = env.reset()
        action = choose_action(Q, state, curr_epsilon, env.n_actions)
        total_reward = 0
        collisions = 0
        done = False
        
        for step in range(max_steps):
            next_state, reward, done = env.step(action)
            
            if reward == env.collision_penalty:
                collisions += 1
                
            next_action = choose_action(Q, next_state, curr_epsilon, env.n_actions)
            
            # معادلة تحديث SARSA
            next_q = 0.0 if done else Q[next_state, next_action]
            td_target = reward + env.gamma * next_q
            Q[state, action] += alpha * (td_target - Q[state, action])
            
            state = next_state
            action = next_action
            total_reward += reward
            
            if done:
                break
                
        if epsilon_decay is not None:
            curr_epsilon = max(min_epsilon, curr_epsilon * epsilon_decay)
            
        rewards_history.append(total_reward)
        steps_history.append(step + 1)
        collisions_history.append(collisions)
        success_history.append(1 if done else 0)

    metrics = {
        "rewards": rewards_history,
        "steps": steps_history,
        "collisions": collisions_history,
        "success": success_history
    }
    return Q, metrics


def extract_policy_and_values(Q):
    """استخراج جدول القيم V(s) وأفضل سياسة (الأسهم) من الـ Q-Table"""
    V = np.max(Q, axis=1)
    policy = np.argmax(Q, axis=1)
    return V, policy


def print_and_save_q_table(Q, filename="q_table.txt"):
    """طباعة عينة من الـ Q-table وحفظه كاملاً بملف نصي للتقرير"""
    header = f"{'State':<8}{'Up':>10}{'Down':>10}{'Left':>10}{'Right':>10}\n" + "-"*48
    lines = [header]
    for s in range(len(Q)):
        lines.append(f"{s:<8}{Q[s,0]:>10.2f}{Q[s,1]:>10.2f}{Q[s,2]:>10.2f}{Q[s,3]:>10.2f}")
    
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"📄 تم حفظ جدول Q-Table كاملاً في: {filename}")
    print("\n".join(lines[:11]))  # طباعة أول 10 حالات فقط عشان ما يتعبى التيرمينال
    print("...\n")


def plot_learning_curve(q_rewards, sarsa_rewards, window=20, save_path="task4_learning_curves.png"):
    """رسم منحنى المكافأة التراكمية مع تنعيم (Moving Average) ليكون واضح بالتقرير"""
    def moving_avg(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    plt.figure(figsize=(9, 5))
    plt.plot(moving_avg(q_rewards, window), label="Q-Learning", color="#1976D2", linewidth=2)
    plt.plot(moving_avg(sarsa_rewards, window), label="SARSA", color="#E64A19", linewidth=2)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel(f"Cumulative Reward (Moving Avg, w={window})", fontsize=12)
    plt.title("Q-Learning vs. SARSA: Learning Progress (8x8 Grid)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📈 تم حفظ منحنى التعلم في: {save_path}")
    plt.show()


# --- تشغيل الخوارزميات وحفظ المخرجات المطلوبة لـ Task 4 ---
if __name__ == "__main__":
    env = GridWorldEnv(size=8, is_stochastic=False)
    
    # 1. تدريب Q-Learning
    Q_ql, metrics_ql = q_learning(env, episodes=500, alpha=0.1, epsilon=1.0, epsilon_decay=0.992)
    V_ql, policy_ql = extract_policy_and_values(Q_ql)
    print(f"✅ Q-Learning انتهى | نسبة النجاح آخر 100 جولة: {np.mean(metrics_ql['success'][-100:])*100:.1f}%")
    print_and_save_q_table(Q_ql, filename="task4_qlearning_qtable.txt")
    
    # 2. تدريب SARSA
    Q_sarsa, metrics_sarsa = sarsa(env, episodes=500, alpha=0.1, epsilon=1.0, epsilon_decay=0.992)
    V_sarsa, policy_sarsa = extract_policy_and_values(Q_sarsa)
    print(f"✅ SARSA انتهى | نسبة النجاح آخر 100 جولة: {np.mean(metrics_sarsa['success'][-100:])*100:.1f}%")
    print_and_save_q_table(Q_sarsa, filename="task4_sarsa_qtable.txt")
    
    # 3. رسم السياسات المتعلمة
    plot_values_and_policy(
        env, V_ql, policy_ql,
        title="Q-Learning: Learned State Values & Policy (500 Episodes)",
        save_path="task4_qlearning_policy.png"
    )
    plot_values_and_policy(
        env, V_sarsa, policy_sarsa,
        title="SARSA: Learned State Values & Policy (500 Episodes)",
        save_path="task4_sarsa_policy.png"
    )
    
    # 4. رسم منحنى المكافآت
    plot_learning_curve(metrics_ql["rewards"], metrics_sarsa["rewards"])
    
    # 5. التحقق من طول المسار اللي تعلمه Q-Learning
    ql_path = extract_optimal_path(env, policy_ql)
    print(f"🚀 طول المسار المتعلم بـ Q-Learning: {len(ql_path) - 1} خطوة.")