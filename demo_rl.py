"""
Sentinel RL — Pitch Demo
=========================
Side-by-side comparison on CartPole:
  1. BASELINE: Standard PPO with conservative learning rate
  2. SENTINEL: Aggressive PPO + AutoGuardian (revert on collapse)

Demonstrates the core thesis: "Safety enables Speed"

Requirements: pip install gymnasium torch numpy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import time
import sys
import os

# Add parent to path for pkg imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pkg import ModelHistory, AutoGuardian


# ======================================================================
#  SIMPLE POLICY NETWORK (Actor-Critic for CartPole)
# ======================================================================
class PolicyNet(nn.Module):
    """Small actor-critic network for CartPole."""

    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def get_action(self, obs):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            logits, value = self(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action), value, dist.entropy()


# ======================================================================
#  SIMPLE PPO UPDATE
# ======================================================================
def ppo_update(policy, optimizer, trajectories, clip_eps=0.2,
               value_coef=0.5, entropy_coef=0.01, epochs=4):
    """
    Minimal PPO update. Returns loss, policy entropy, and gradient norm.
    """
    states = torch.FloatTensor(np.array(trajectories["states"]))
    actions = torch.LongTensor(trajectories["actions"])
    old_log_probs = torch.stack(trajectories["log_probs"])
    returns = torch.FloatTensor(trajectories["returns"])
    advantages = returns - torch.FloatTensor(trajectories["values"])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_loss = 0
    total_entropy = 0
    total_grad_norm = 0
    n_updates = 0

    for _ in range(epochs):
        logits, values = policy(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(values.squeeze(), returns)
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()

        # Measure gradient norm before clipping
        grad_norm = 0
        for p in policy.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        total_entropy += entropy.item()
        total_grad_norm += grad_norm
        n_updates += 1

    return (total_loss / n_updates, total_entropy / n_updates,
            total_grad_norm / n_updates)


# ======================================================================
#  COLLECT TRAJECTORIES
# ======================================================================
def collect_trajectories(env, policy, n_steps=256, gamma=0.99):
    """Collect a batch of experience from the environment."""
    trajectories = {
        "states": [], "actions": [], "rewards": [],
        "log_probs": [], "values": [], "returns": [],
    }
    obs, _ = env.reset()
    episode_rewards = []
    current_ep_reward = 0

    for _ in range(n_steps):
        action, log_prob, value, _ = policy.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        trajectories["states"].append(obs)
        trajectories["actions"].append(action)
        trajectories["rewards"].append(reward)
        trajectories["log_probs"].append(log_prob)
        trajectories["values"].append(value.item())

        current_ep_reward += reward
        obs = next_obs

        if done:
            episode_rewards.append(current_ep_reward)
            current_ep_reward = 0
            obs, _ = env.reset()

    # Compute discounted returns
    returns = []
    R = 0
    for r in reversed(trajectories["rewards"]):
        R = r + gamma * R
        returns.insert(0, R)
    trajectories["returns"] = returns

    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    return trajectories, avg_reward


# ======================================================================
#  POISON INJECTION (Simulates environmental shift / reward hacking)
# ======================================================================
def inject_poison(trajectories, intensity=5.0):
    """
    Corrupt the reward signal to simulate:
    - Reward hacking / adversarial attack
    - Sudden environment distribution shift
    """
    n = len(trajectories["rewards"])
    for i in range(n):
        # Flip rewards: good actions get punished, bad get rewarded
        trajectories["rewards"][i] = -trajectories["rewards"][i] * intensity

    # Recompute returns with poisoned rewards
    returns = []
    R = 0
    for r in reversed(trajectories["rewards"]):
        R = r + 0.99 * R
        returns.insert(0, R)
    trajectories["returns"] = returns
    return trajectories


# ======================================================================
#  TRAINING LOOP — BASELINE (No Protection)
# ======================================================================
def train_baseline(total_updates=80, lr=0.0003, poison_range=(30, 45),
                   seed=42):
    """Standard conservative PPO — no guardian."""
    print("\n" + "=" * 60)
    print("  BASELINE: Standard PPO (lr={})".format(lr))
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    history = []  # Just for logging

    for update in range(total_updates):
        # Collect experience
        trajectories, avg_reward = collect_trajectories(env, policy)

        # Poison injection window
        is_poisoned = poison_range[0] <= update <= poison_range[1]
        if is_poisoned:
            trajectories = inject_poison(trajectories)
            tag = "☣️ POISON"
        else:
            tag = "Normal  "

        # PPO update (always learns — no protection)
        loss, entropy, grad_norm = ppo_update(policy, optimizer, trajectories)

        print(f"  [{tag}] Update {update:3d} | "
              f"Reward: {avg_reward:7.1f} | Loss: {loss:7.3f} | "
              f"Entropy: {entropy:.3f} | GradNorm: {grad_norm:.2f}")

        history.append({
            "update": update,
            "reward": avg_reward,
            "loss": loss,
            "entropy": entropy,
            "grad_norm": grad_norm,
            "poisoned": is_poisoned,
        })

    env.close()
    return history


# ======================================================================
#  TRAINING LOOP — SENTINEL (With Guardian Protection)
# ======================================================================
def train_sentinel(total_updates=80, lr=0.003, poison_range=(30, 45),
                   seed=42):
    """
    Aggressive PPO + Sentinel Guardian.
    NOTE: Learning rate is 10x higher than baseline!
    """
    print("\n" + "=" * 60)
    print("  SENTINEL: Aggressive PPO + Guardian (lr={})".format(lr))
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # --- SENTINEL SETUP ---
    history = ModelHistory(policy, max_fine_slots=200)
    guardian = AutoGuardian(
        history,
        metrics_config={
            "loss": {"sensitivity": 2.0, "direction": "lower_is_better"},
            "reward": {"sensitivity": 2.0, "direction": "higher_is_better",
                       "window": 20},
            "entropy": {"sensitivity": 3.0, "direction": "higher_is_better"},
            "grad_norm": {"sensitivity": 2.5, "direction": "lower_is_better"},
        },
        patience=3,
        cooldown=5,
        enable_prediction=True,
        prediction_horizon=5,
        total_steps=total_updates,
    )

    log = []

    for update in range(total_updates):
        # Collect experience
        trajectories, avg_reward = collect_trajectories(env, policy)

        # Poison injection window
        is_poisoned = poison_range[0] <= update <= poison_range[1]
        if is_poisoned:
            trajectories = inject_poison(trajectories)
            tag = "☣️ POISON"
        else:
            tag = "Normal  "

        # Forward pass to get metrics (before deciding whether to learn)
        loss, entropy, grad_norm = ppo_update(policy, optimizer, trajectories)

        metrics = {
            "loss": loss,
            "reward": avg_reward,
            "entropy": entropy,
            "grad_norm": grad_norm,
        }

        # --- GUARDIAN CHECK ---
        status = guardian.step(update, metrics)

        if status == "safe" or status == "warmup":
            # Already updated above — keep it
            symbol = "✅"
        elif status == "pre_warning":
            # Predictive warning but still learning
            symbol = "🔮"
        elif status == "warning":
            # Learning but guardian is watching
            symbol = "⚠️"
        elif status == "reverted":
            # Guardian reverted the model — the PPO update we did is undone
            # We need to also reset the optimizer state
            optimizer = optim.Adam(policy.parameters(), lr=lr)
            symbol = "🛑"
        elif status == "cooldown":
            # Don't learn — reload last safe weights
            history.checkout(relative_back=0)
            optimizer = optim.Adam(policy.parameters(), lr=lr)
            symbol = "🧊"
        else:
            symbol = "❓"

        print(f"  [{tag}] Update {update:3d} {symbol} | "
              f"Reward: {avg_reward:7.1f} | Loss: {loss:7.3f} | "
              f"Entropy: {entropy:.3f} | Status: {status}")

        log.append({
            "update": update,
            "reward": avg_reward,
            "loss": loss,
            "entropy": entropy,
            "grad_norm": grad_norm,
            "status": status,
            "poisoned": is_poisoned,
        })

    env.close()

    # --- POST-MORTEM ---
    print(f"\n📊 Sentinel Summary:")
    print(f"   Total reverts: {guardian.total_reverts}")
    print(f"   Revert reports: {len(guardian.revert_reports)}")
    for report in guardian.revert_reports:
        print(report)

    # Show diff between pre-poison and post-recovery if possible
    history.summary()

    return log


# ======================================================================
#  COMPARISON VISUALIZATION
# ======================================================================
def plot_comparison(baseline_log, sentinel_log, save_path="comparison.png"):
    """Generate side-by-side comparison plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Sentinel RL: Safety Enables Speed", fontsize=16,
                 fontweight="bold")

    baseline_updates = [d["update"] for d in baseline_log]
    sentinel_updates = [d["update"] for d in sentinel_log]

    # --- Reward ---
    ax = axes[0, 0]
    ax.plot(baseline_updates, [d["reward"] for d in baseline_log],
            label="Baseline (lr=0.0003)", color="#888888", linewidth=1.5)
    ax.plot(sentinel_updates, [d["reward"] for d in sentinel_log],
            label="Sentinel (lr=0.003)", color="#00CC96", linewidth=2)

    # Shade poison zone
    poison_start = min(d["update"] for d in baseline_log if d["poisoned"])
    poison_end = max(d["update"] for d in baseline_log if d["poisoned"])
    ax.axvspan(poison_start, poison_end, alpha=0.15, color="red",
               label="Poison Window")

    # Mark revert events
    for d in sentinel_log:
        if d["status"] == "reverted":
            ax.axvline(d["update"], color="red", linestyle="--",
                       alpha=0.5, linewidth=1)

    ax.set_title("Episode Reward")
    ax.set_xlabel("Update")
    ax.set_ylabel("Avg Reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Loss ---
    ax = axes[0, 1]
    ax.plot(baseline_updates, [d["loss"] for d in baseline_log],
            label="Baseline", color="#888888", linewidth=1.5)
    ax.plot(sentinel_updates, [d["loss"] for d in sentinel_log],
            label="Sentinel", color="#00CC96", linewidth=2)
    ax.axvspan(poison_start, poison_end, alpha=0.15, color="red")
    ax.set_title("Training Loss")
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Entropy ---
    ax = axes[1, 0]
    ax.plot(baseline_updates, [d["entropy"] for d in baseline_log],
            label="Baseline", color="#888888", linewidth=1.5)
    ax.plot(sentinel_updates, [d["entropy"] for d in sentinel_log],
            label="Sentinel", color="#00CC96", linewidth=2)
    ax.axvspan(poison_start, poison_end, alpha=0.15, color="red")
    ax.set_title("Policy Entropy")
    ax.set_xlabel("Update")
    ax.set_ylabel("Entropy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Gradient Norm ---
    ax = axes[1, 1]
    ax.plot(baseline_updates, [d["grad_norm"] for d in baseline_log],
            label="Baseline", color="#888888", linewidth=1.5)
    ax.plot(sentinel_updates, [d["grad_norm"] for d in sentinel_log],
            label="Sentinel", color="#00CC96", linewidth=2)
    ax.axvspan(poison_start, poison_end, alpha=0.15, color="red")
    ax.set_title("Gradient Norm")
    ax.set_xlabel("Update")
    ax.set_ylabel("Grad Norm")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 Comparison plot saved to: {save_path}")
    return save_path


# ======================================================================
#  DEMO: Weight Diff Analysis
# ======================================================================
def demo_diff_analysis():
    """Show the commit diff feature — which layers collapsed?"""
    print("\n" + "=" * 60)
    print("  DEMO: Weight Diff Analysis")
    print("=" * 60)

    policy = PolicyNet()
    history = ModelHistory(policy)

    # Commit initial state
    history.commit(0, {"loss": 1.0}, tag="initial")

    # Simulate some training that modifies policy_head heavily
    with torch.no_grad():
        policy.policy_head.weight += torch.randn_like(
            policy.policy_head.weight) * 5.0
        policy.value_head.weight += torch.randn_like(
            policy.value_head.weight) * 0.1

    history.commit(10, {"loss": 5.0}, tag="corrupted")

    # Show diff
    history.print_diff(0, 10)

    # Selective revert — only fix the policy head
    print("\n⚡ Selective Revert: Only reverting policy_head...")
    history.checkout(step=0, layers=["policy_head"])

    # Show diff after selective revert
    history.commit(11, {"loss": 1.2}, tag="partial_fix")
    history.print_diff(0, 11)


# ======================================================================
#  DEMO: Branch Management
# ======================================================================
def demo_branching():
    """Show the branching feature — experiment with hyperparams safely."""
    print("\n" + "=" * 60)
    print("  DEMO: Branch Management")
    print("=" * 60)

    policy = PolicyNet()
    history = ModelHistory(policy)

    # Train on main for a bit
    for step in range(5):
        with torch.no_grad():
            policy.shared[0].weight += torch.randn_like(
                policy.shared[0].weight) * 0.01
        history.commit(step, {"loss": 1.0 - step * 0.1})

    # Branch: try aggressive experiment
    history.create_branch("aggressive_lr")
    history.tag_milestone("pre_experiment")

    for step in range(5, 10):
        with torch.no_grad():
            policy.shared[0].weight += torch.randn_like(
                policy.shared[0].weight) * 0.5  # Large updates
        history.commit(step, {"loss": 0.5 + step * 0.3},
                       branch="aggressive_lr")

    print("\n📋 Branch state:")
    history.list_branches()

    # Experiment failed — switch back to main
    print("\n⏪ Experiment failed, switching back to main...")
    history.switch_branch("main")
    history.summary()


# ======================================================================
#  MAIN
# ======================================================================
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         SENTINEL RL — Pitch Demo Suite                  ║")
    print("║         'Safety Enables Speed'                          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. Feature Demos
    demo_diff_analysis()
    demo_branching()

    # 2. The Main Event: Side-by-side comparison
    print("\n\n" + "🏁 " * 20)
    print("  MAIN EVENT: Baseline vs Sentinel on CartPole + Poison")
    print("🏁 " * 20)

    baseline_log = train_baseline(
        total_updates=80, lr=0.0003,
        poison_range=(30, 45), seed=42,
    )
    sentinel_log = train_sentinel(
        total_updates=80, lr=0.003,  # 10x higher!
        poison_range=(30, 45), seed=42,
    )

    # 3. Results Summary
    print("\n\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)

    # Compare final 10 updates (post-recovery)
    baseline_final = np.mean([d["reward"] for d in baseline_log[-10:]])
    sentinel_final = np.mean([d["reward"] for d in sentinel_log[-10:]])

    # Compare peak performance
    baseline_peak = max(d["reward"] for d in baseline_log)
    sentinel_peak = max(d["reward"] for d in sentinel_log)

    # Time to first "good" performance (reward > 150)
    baseline_ttg = next(
        (d["update"] for d in baseline_log if d["reward"] > 150), "Never")
    sentinel_ttg = next(
        (d["update"] for d in sentinel_log if d["reward"] > 150), "Never")

    print(f"\n  {'Metric':<30} {'Baseline':>12} {'Sentinel':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Learning Rate':<30} {'0.0003':>12} {'0.003':>12}")
    print(f"  {'Final Avg Reward (last 10)':<30} {baseline_final:>12.1f} "
          f"{sentinel_final:>12.1f}")
    print(f"  {'Peak Reward':<30} {baseline_peak:>12.1f} "
          f"{sentinel_peak:>12.1f}")
    print(f"  {'Updates to Reward > 150':<30} {str(baseline_ttg):>12} "
          f"{str(sentinel_ttg):>12}")

    # 4. Plot
    plot_path = plot_comparison(baseline_log, sentinel_log,
                                save_path="comparison.png")

    print("\n✅ Demo complete!")
    return baseline_log, sentinel_log


if __name__ == "__main__":
    main()