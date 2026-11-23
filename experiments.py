import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv
from planning import value_iteration, policy_iteration
from rl_agents import q_learning, sarsa, extract_policy_and_values

SEEDS = [0, 1, 2, 3, 4]  # تكرار التجارب بـ 5 بذور عشوائية مختلفة (Task 6)

def moving_avg(data, w=25):
    return np.convolve(data, np.ones(w) / w, mode='valid')

def evaluate_policy(env, policy, n_episodes=50, max_steps=100, seed=42):
    """تقييم أي سياسة (من التخطيط أو التعلم) على نفس المقاييس"""
    np.random.seed(seed)
    successes, steps_list, collisions_list, rewards_list = [], [], [], []
    
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        collisions = 0
        done = False
        
        for step in range(max_steps):
            action = policy[state]
            next_state, reward, done = env.step(action)
            if reward == env.collision_penalty:
                collisions += 1
            total_reward += reward
            state = next_state
            if done:
                break
                
        successes.append(1 if done else 0)
        steps_list.append(step + 1)
        collisions_list.append(collisions)
        rewards_list.append(total_reward)
        
    return {
        "success_rate": np.mean(successes) * 100,
        "avg_steps": np.mean(steps_list),
        "avg_collisions": np.mean(collisions_list),
        "avg_reward": np.mean(rewards_list)
    }

def run_multi_seed_rl(env, algo_fn, episodes=500, **kwargs):
    """تشغيل Q-Learning أو SARSA على عدة Random Seeds وحساب المتوسط"""
    all_rewards = []
    eval_results = []
    
    for s in SEEDS:
        Q, metrics = algo_fn(env, episodes=episodes, seed=s, **kwargs)
        all_rewards.append(metrics["rewards"])
        _, policy = extract_policy_and_values(Q)
        eval_results.append(evaluate_policy(env, policy, n_episodes=30, seed=s))
        
    avg_curve = np.mean(all_rewards, axis=0)
    summary = {
        "success_rate": np.mean([r["success_rate"] for r in eval_results]),
        "avg_steps": np.mean([r["avg_steps"] for r in eval_results]),
        "avg_collisions": np.mean([r["avg_collisions"] for r in eval_results]),
        "avg_reward": np.mean([r["avg_reward"] for r in eval_results]),
        "train_last100_reward": np.mean([np.mean(r[-100:]) for r in all_rewards])
    }
    return avg_curve, summary


# ==============================================================================
# EXPERIMENT A: Planning vs. Learning
# ==============================================================================
def experiment_a(log):
    log.append("\n=== EXPERIMENT A: Planning vs. Learning (Simple Deterministic 8x8) ===")
    env = GridWorldEnv(size=8, is_stochastic=False)
    
    # 1. Value Iteration
    _, pol_vi, _, iters_vi = value_iteration(env)
    res_vi = evaluate_policy(env, pol_vi, n_episodes=30)
    
    # 2. Policy Iteration
    _, pol_pi, iters_pi = policy_iteration(env)
    res_pi = evaluate_policy(env, pol_pi, n_episodes=30)
    
    # 3. Q-Learning (5 seeds)
    curve_ql, res_ql = run_multi_seed_rl(env, q_learning, episodes=500, alpha=0.1, epsilon=1.0, epsilon_decay=0.992)
    
    # 4. SARSA (5 seeds)
    curve_sarsa, res_sarsa = run_multi_seed_rl(env, sarsa, episodes=500, alpha=0.1, epsilon=1.0, epsilon_decay=0.992)
    
    header = f"{'Algorithm':<20}{'Success (%)':>13}{'Avg Steps':>12}{'Collisions':>12}{'Eval Reward':>14}"
    log.append(header)
    log.append("-" * len(header))
    for name, res in [("Value Iteration", res_vi), ("Policy Iteration", res_pi),
                      ("Q-Learning", res_ql), ("SARSA", res_sarsa)]:
        log.append(f"{name:<20}{res['success_rate']:>12.1f}%{res['avg_steps']:>12.2f}{res['avg_collisions']:>12.2f}{res['avg_reward']:>14.2f}")

    # رسم مقارنة منحنيات التعلم مع الخط المثالي للتخطيط
    plt.figure(figsize=(9, 5))
    plt.plot(moving_avg(curve_ql), label="Q-Learning (5-seed avg)", color="#1976D2", linewidth=2)
    plt.plot(moving_avg(curve_sarsa), label="SARSA (5-seed avg)", color="#E64A19", linewidth=2)
    plt.axhline(y=res_vi["avg_reward"], color="#2E7D32", linestyle="--", linewidth=2,
                label=f"Optimal Planning (VI/PI = {res_vi['avg_reward']:.1f})")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward (Moving Avg)")
    plt.title("Experiment A: Model-Based Planning vs. Model-Free Learning", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("expA_planning_vs_learning.png", dpi=300)
    plt.show()


# ==============================================================================
# EXPERIMENT B: Exploration Strategies + Alpha & Gamma Effects (Task 5)
# ==============================================================================
def experiment_b(log):
    log.append("\n=== EXPERIMENT B1: Exploration Strategies (Q-Learning, 5 seeds) ===")
    env = GridWorldEnv(size=8, is_stochastic=False)
    
    configs = [
        ("Constant eps=0.9 (High)",    dict(epsilon=0.9,  epsilon_decay=None)),
        ("Constant eps=0.3 (Moderate)",dict(epsilon=0.3,  epsilon_decay=None)),
        ("Constant eps=0.05 (Low)",    dict(epsilon=0.05, epsilon_decay=None)),
        ("Decreasing eps (1.0->0.05)", dict(epsilon=1.0,  epsilon_decay=0.992, min_epsilon=0.05)),
    ]
    
    header = f"{'Strategy':<28}{'Success (%)':>12}{'Avg Steps':>11}{'Collisions':>12}{'Train Last100':>15}"
    log.append(header)
    log.append("-" * len(header))
    
    plt.figure(figsize=(9, 5))
    for label, params in configs:
        curve, res = run_multi_seed_rl(env, q_learning, episodes=500, alpha=0.1, **params)
        log.append(f"{label:<28}{res['success_rate']:>11.1f}%{res['avg_steps']:>11.2f}{res['avg_collisions']:>12.2f}{res['train_last100_reward']:>15.2f}")
        plt.plot(moving_avg(curve), label=label, linewidth=2)
        
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward (Moving Avg)")
    plt.title("Experiment B: Effect of Exploration Strategies (Epsilon-Greedy)", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("expB_exploration_strategies.png", dpi=300)
    plt.show()

    # تأثير Learning Rate (alpha) و Discount Factor (gamma) - مطلوب في Task 5
    log.append("\n=== EXPERIMENT B2: Learning Rate (alpha) & Discount Factor (gamma) ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # 1. Alpha
    for a in [0.05, 0.1, 0.5]:
        env_a = GridWorldEnv(size=8, is_stochastic=False)
        curve, res = run_multi_seed_rl(env_a, q_learning, episodes=500, alpha=a, epsilon=1.0, epsilon_decay=0.992)
        axes[0].plot(moving_avg(curve), label=f"alpha = {a}", linewidth=2)
        log.append(f"Alpha={a:<5} | Success={res['success_rate']:.1f}%, Steps={res['avg_steps']:.1f}, EvalReward={res['avg_reward']:.1f}")
    axes[0].set_title("Effect of Learning Rate (alpha)", fontweight="bold")
    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # 2. Gamma
    for g in [0.5, 0.95]:
        env_g = GridWorldEnv(size=8, is_stochastic=False)
        env_g.gamma = g
        curve, res = run_multi_seed_rl(env_g, q_learning, episodes=500, alpha=0.1, epsilon=1.0, epsilon_decay=0.992)
        axes[1].plot(moving_avg(curve), label=f"gamma = {g}", linewidth=2)
        log.append(f"Gamma={g:<5} | Success={res['success_rate']:.1f}%, Steps={res['avg_steps']:.1f}, EvalReward={res['avg_reward']:.1f}")
    axes[1].set_title("Effect of Discount Factor (gamma)", fontweight="bold")
    axes[1].set_xlabel("Episodes")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("expB_alpha_gamma_effects.png", dpi=300)
    plt.show()


# ==============================================================================
# EXPERIMENT C: Environment Complexity (Simple vs. Complex Maze vs. Stochastic)
# ==============================================================================
def experiment_c(log):
    log.append("\n=== EXPERIMENT C: Environment Complexity (5 seeds) ===")
    
    # متاهة معقدة (18 عائق وممرات ضيقة مع الحفاظ على وجود طريق للهدف)
    complex_obstacles = [
        (0, 4),
        (1, 2), (1, 3), (1, 6),
        (2, 2), (2, 4), (2, 6),
        (3, 1), (3, 2), (3, 5),
        (4, 5),
        (5, 0), (5, 1), (5, 2), (5, 5),
        (6, 2), (6, 4), (6, 6)
    ]
    
    envs = [
        ("Simple (Det.)",   GridWorldEnv(size=8, is_stochastic=False)),
        ("Complex (Det.)",  GridWorldEnv(size=8, is_stochastic=False, custom_obstacles=complex_obstacles)),
        ("Stochastic (20%)",GridWorldEnv(size=8, is_stochastic=True)),
    ]
    
    # حفظ صورة المتاهة المعقدة للتقرير
    envs[1][1].render(save_path="expC_complex_maze_map.png", title="Experiment C: Complex 8x8 Maze (18 Obstacles)")
    
    header = f"{'Environment':<18}{'Algo':<14}{'Success (%)':>12}{'Avg Steps':>11}{'Collisions':>12}{'Eval Reward':>13}"
    log.append(header)
    log.append("-" * len(header))
    
    plt.figure(figsize=(10, 5))
    for env_name, env in envs:
        # Planning (VI)
        _, pol_vi, _, _ = value_iteration(env)
        res_vi = evaluate_policy(env, pol_vi, n_episodes=50)
        log.append(f"{env_name:<18}{'Value Iter.':<14}{res_vi['success_rate']:>11.1f}%{res_vi['avg_steps']:>11.2f}{res_vi['avg_collisions']:>12.2f}{res_vi['avg_reward']:>13.2f}")
        
        # Q-Learning
        c_ql, res_ql = run_multi_seed_rl(env, q_learning, episodes=700, alpha=0.1, epsilon=1.0, epsilon_decay=0.994)
        log.append(f"{env_name:<18}{'Q-Learning':<14}{res_ql['success_rate']:>11.1f}%{res_ql['avg_steps']:>11.2f}{res_ql['avg_collisions']:>12.2f}{res_ql['avg_reward']:>13.2f}")
        plt.plot(moving_avg(c_ql), label=f"Q-Learning - {env_name}", linewidth=2)
        
        # SARSA
        c_sa, res_sa = run_multi_seed_rl(env, sarsa, episodes=700, alpha=0.1, epsilon=1.0, epsilon_decay=0.994)
        log.append(f"{env_name:<18}{'SARSA':<14}{res_sa['success_rate']:>11.1f}%{res_sa['avg_steps']:>11.2f}{res_sa['avg_collisions']:>12.2f}{res_sa['avg_reward']:>13.2f}")
        plt.plot(moving_avg(c_sa), linestyle="--", label=f"SARSA - {env_name}", linewidth=1.8)
        
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward (Moving Avg)")
    plt.title("Experiment C: Simple vs. Complex vs. Stochastic (20% Slip)", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("expC_environment_complexity.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    summary_log = []
    experiment_a(summary_log)
    experiment_b(summary_log)
    experiment_c(summary_log)
    
    full_report = "\n".join(summary_log)
    print(full_report)
    with open("experiments_summary.txt", "w") as f:
        f.write(full_report)
    print("\n✅ تم حفظ ملخص جداول التجارب كلها في: experiments_summary.txt")