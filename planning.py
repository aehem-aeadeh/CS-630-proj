import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from grid_env import GridWorldEnv

def value_iteration(env, theta=1e-6):

    V = np.zeros(env.n_states)
    policy = np.zeros(env.n_states, dtype=int)
    
    history = {}
    iteration = 0
    
    while True:
        delta = 0
        new_V = np.copy(V)
        
        for s in range(env.n_states):
            pos = env.state_to_pos(s)
            if pos == env.goal_pos or pos in env.obstacles:
                continue
                
            action_values = []
            for a in range(env.n_actions):
                q_sa = 0
                for prob, next_s, reward, _ in env.get_transitions(s, a):
                    q_sa += prob * (reward + env.gamma * V[next_s])
                action_values.append(q_sa)
                
            best_value = max(action_values)
            delta = max(delta, abs(best_value - V[s]))
            new_V[s] = best_value
            policy[s] = np.argmax(action_values)
            
        V = new_V
        iteration += 1
        
        if iteration in [1, 5]:
            history[iteration] = (np.copy(V), np.copy(policy))
            
        if delta < theta:
            break
            
    history[iteration] = (np.copy(V), np.copy(policy)) # حفظ النتيجة النهائية عند الاستقرار
    print(f"✅ Value Iteration استقرت بعد {iteration} تكرارات.")
    return V, policy, history, iteration


def policy_iteration(env, theta=1e-6):

    policy = np.zeros(env.n_states, dtype=int)
    V = np.zeros(env.n_states)
    iteration = 0
    
    while True:
        iteration += 1
        # 1. Policy Evaluation 
        while True:
            delta = 0
            for s in range(env.n_states):
                pos = env.state_to_pos(s)
                if pos == env.goal_pos or pos in env.obstacles:
                    continue
                
                a = policy[s] #هون إحنا ما بنجرب 4 حركات! إحنا بنوخذ الحركة الحالية المخزنة بالخطة
                v_s = 0
                for prob, next_s, reward, _ in env.get_transitions(s, a):
                    v_s += prob * (reward + env.gamma * V[next_s])
                    
                delta = max(delta, abs(v_s - V[s]))
                V[s] = v_s
            if delta < theta:
                break
                
        # 2. Policy Improvement 
        policy_stable = True
        for s in range(env.n_states):
            pos = env.state_to_pos(s)
            if pos == env.goal_pos or pos in env.obstacles:
                continue
                
            old_action = policy[s]
            action_values = []
            for a in range(env.n_actions):
                q_sa = 0
                for prob, next_s, reward, _ in env.get_transitions(s, a):
                    q_sa += prob * (reward + env.gamma * V[next_s])
                action_values.append(q_sa)
                
            best_action = np.argmax(action_values)
            policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
                
        if policy_stable:
            break
            
    print(f"✅ Policy Iteration استقرت بعد {iteration} دورات تحسين.")
    return V, policy, iteration


def extract_optimal_path(env, policy, max_steps=50):
    state = env.reset()
    path = [state]
    for _ in range(max_steps):
        action = policy[state]
        transitions = env.get_transitions(state, action)
        next_state = max(transitions, key=lambda x: x[0])[1]
        path.append(next_state)
        state = next_state
        if env.state_to_pos(state) == env.goal_pos:
            break
    return path


def plot_values_and_policy(env, V, policy, title, save_path=None):

    fig, ax = plt.subplots(figsize=(8, 8))
    grid_V = V.reshape((env.size, env.size))
    
    im = ax.imshow(grid_V, cmap='YlGnBu', origin='upper')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="State Value V(s)")
    
    # رموز الأسهم: 0: Up, 1: Down, 2: Left, 3: Right
    arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    for r in range(env.size):
        for c in range(env.size):
            s = env.pos_to_state((r, c))
            if (r, c) == env.start_pos:
                ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#4CAF50', alpha=0.5))
                ax.text(c, r, f"S\n{V[s]:.1f}\n{arrows[policy[s]]}", ha='center', va='center', fontweight='bold', color='black')
            elif (r, c) == env.goal_pos:
                ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#2196F3', alpha=0.8))
                ax.text(c, r, "GOAL\n+100", ha='center', va='center', fontweight='bold', color='white')
            elif (r, c) in env.obstacles:
                ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#424242'))
                ax.text(c, r, "X", ha='center', va='center', fontweight='bold', color='white', fontsize=14)
            else:
                ax.text(c, r - 0.15, f"{V[s]:.1f}", ha='center', va='center', fontsize=9, color='black')
                ax.text(c, r + 0.2, arrows[policy[s]], ha='center', va='center', fontsize=14, fontweight='bold', color='#D32F2F')

    ax.set_xticks(np.arange(env.size))
    ax.set_yticks(np.arange(env.size))
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1.2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📊 تم حفظ الرسمة في: {save_path}")
    plt.show()


if __name__ == "__main__":
    env = GridWorldEnv(size=8, is_stochastic=False)
    
    # 1. تشغيل Value Iteration
    V_vi, policy_vi, history_vi, iters_vi = value_iteration(env)
    
    # رسم وحفظ القيم والسياسة النهائية لـ Value Iteration
    plot_values_and_policy(
        env, V_vi, policy_vi,
        title=f"Value Iteration: Final State Values & Policy (Converged in {iters_vi} iters)",
        save_path="task3_value_iteration_policy.png"
    )
    
    # 2. تشغيل Policy Iteration
    V_pi, policy_pi, iters_pi = policy_iteration(env)
    
    # رسم وحفظ القيم والسياسة النهائية لـ Policy Iteration
    plot_values_and_policy(
        env, V_pi, policy_pi,
        title=f"Policy Iteration: Final State Values & Policy (Converged in {iters_pi} iters)",
        save_path="task3_policy_iteration_policy.png"
    )
    
    # 3. استخراج المسار المثالي ورسمه على الشبكة
    optimal_path = extract_optimal_path(env, policy_vi)
    print(f"🚀 طول المسار المثالي من البداية للهدف: {len(optimal_path) - 1} خطوة.")
    env.render(
        path=optimal_path,
        save_path="task3_optimal_planned_path.png",
        title=f"Optimal Path using Planning ({len(optimal_path)-1} Steps)"
    )