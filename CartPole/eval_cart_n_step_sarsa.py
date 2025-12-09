import torch
import numpy as np
from models import QNetwork
from cartpole_env import CartPoleEnv
import matplotlib.pyplot as plt
import os


def eval_cartpole_env_sarsa(env: CartPoleEnv, q_net: QNetwork, device: torch.device, num_episodes: int = 1000) -> dict:
    episode_rewards = []
    episode_steps = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            s_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                q_vals = q_net(s_tensor)
                action = torch.argmax(q_vals).item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_steps': np.mean(episode_steps),
        'std_steps': np.std(episode_steps),
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
    }


def load_q_network():
    q_net = QNetwork(state_dim=4, action_dim=2, hidden_dim=128)
    checkpoint_dir = "checkpoints/sarsa_cartpole"
    checkpoint_path = os.path.join(checkpoint_dir, "q_net_cartpole_sarsa.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Q-network checkpoint not found at {checkpoint_path}")

    q_net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return q_net


def test_q_network():
    q_net = load_q_network()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net.to(device)

    env = CartPoleEnv(max_episode_steps=500, seed=0)
    eval_results = eval_cartpole_env_sarsa(env, q_net, device, num_episodes=100)

    print(f"\nSARSA n-step Evaluation Results for CartPole (over {len(eval_results['episode_rewards'])} episodes):")
    print(f"  Average Reward: {eval_results['avg_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    print(f"  Average Steps:  {eval_results['avg_steps']:.4f} ± {eval_results['std_steps']:.4f}")
    print(f"  Min Reward: {np.min(eval_results['episode_rewards']):.4f}")
    print(f"  Max Reward: {np.max(eval_results['episode_rewards']):.4f}")
    print(f"  Min Steps: {np.min(eval_results['episode_steps']):.0f}")
    print(f"  Max Steps: {np.max(eval_results['episode_steps']):.0f}")

    os.makedirs("results/eval", exist_ok=True)

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.plot(eval_results['episode_rewards'], alpha=0.6, color='blue')
    plt.axhline(eval_results['avg_reward'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {eval_results['avg_reward']:.2f}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards (SARSA n-step - CartPole)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(eval_results['episode_steps'], alpha=0.6, color='green')
    plt.axhline(eval_results['avg_steps'], color='orange', linestyle='--', linewidth=2,
                label=f"Mean: {eval_results['avg_steps']:.2f}")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Steps (SARSA n-step - CartPole)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.hist(eval_results['episode_rewards'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(eval_results['avg_reward'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {eval_results['avg_reward']:.2f}")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution (SARSA n-step - CartPole)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.hist(eval_results['episode_steps'], bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(eval_results['avg_steps'], color='orange', linestyle='--', linewidth=2,
                label=f"Mean: {eval_results['avg_steps']:.2f}")
    plt.xlabel("Steps")
    plt.ylabel("Frequency")
    plt.title("Steps Distribution (SARSA n-step - CartPole)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = "eval_results_sarsa_cartpole.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\nEvaluation plot saved to: {save_path}")

    return eval_results


if __name__ == "__main__":
    test_q_network()
