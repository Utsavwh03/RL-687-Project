import os
import argparse
import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from cartpole_env import CartPoleEnv
from models import QNetwork
from semi_gradient_n_step_sarsa import (
    train_sarsa_n_step)


def plot_curve(values, save_path, title):
    plt.figure(figsize=(8,4))
    plt.plot(values, alpha=0.5, label="raw")
    if len(values) >= 50:
        kernel = np.ones(50)/50
        smoothed = np.convolve(values, kernel, mode="valid")
        plt.plot(range(49, 49+len(smoothed)), smoothed, linewidth=2, label="smoothed")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def policy_from_q(q_net):
    def _policy_fn(state):
        return q_net(state)
    return _policy_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=3000)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = CartPoleEnv(max_episode_steps=500, seed=args.seed)

    state_dim = 4
    action_dim = 2

    q_net = QNetwork(state_dim, action_dim)
    optimizer = Adam(q_net.parameters(), lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net.to(device)

    print("="*60)
    print("TRAINING N-STEP SARSA ON CARTPOLE")
    print("="*60)
    print(f"Episodes: {args.num_episodes}")
    print(f"n = {args.n}")
    print(f"gamma = {args.gamma}")
    print(f"LR = {args.lr}")
    print(f"Device = {device}")
    print("="*60)

    policy_fn = policy_from_q(q_net)

    logs = train_sarsa_n_step(
        env=env,
        q_net=q_net,
        optimizer=optimizer,
        policy_fn=policy_fn,
        device=device,
        n=args.n,
        num_episodes=args.num_episodes,
        gamma=args.gamma,
        normalize_fn=None,
        verbose=True,
        print_every=100
    )

    os.makedirs("plots/sarsa_cartpole/", exist_ok=True)
    os.makedirs("checkpoints/sarsa_cartpole/", exist_ok=True)

    plot_curve(logs["rewards"], "plots/sarsa_cartpole/rewards.png", "Episode Rewards")
    plot_curve(logs["lengths"], "plots/sarsa_cartpole/lengths.png", "Episode Lengths")
    plot_curve(logs["loss"], "plots/sarsa_cartpole/loss.png", "TD Loss")

    torch.save(q_net.state_dict(), "checkpoints/sarsa_cartpole/q_net_cartpole_sarsa.pth")

    print("\nTraining finished. Model saved!")

if __name__ == "__main__":
    main()
