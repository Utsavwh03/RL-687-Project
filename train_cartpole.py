import numpy as np
import torch
from torch.optim import Adam
import os
import argparse

from cartpole_env import CartPoleEnv
from models import PolicyNetwork, ValueNetwork
from actor_critic import train_actor_critic
from utils_plotting import plot_curve
from evaluate_cartpole import evaluate_policy, print_evaluation_results


def main():
    # Configuration
    parser = argparse.ArgumentParser(description='Train Actor-Critic on CartPole')
    parser.add_argument('--num_episodes', type=int, default=1000, 
                        help='Number of training episodes (default: 1000)')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='Number of episodes for evaluation (default: 100)')
    parser.add_argument('--no_eval', action='store_true',
                        help='Skip evaluation after training')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # CartPole environment setup
    max_episode_steps = 500
    env = CartPoleEnv(max_episode_steps=max_episode_steps)

    # State and action dimensions for CartPole
    state_dim = 4  # [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    action_dim = 2  # [push left, push right]

    # Initialize networks
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim=128).to(device)
    value_net = ValueNetwork(state_dim, hidden_dim=128).to(device)

    optimizer_policy = Adam(policy_net.parameters(), lr=3e-4)
    optimizer_value = Adam(value_net.parameters(), lr=3e-4)

    # Create directories for results
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results/evaluation", exist_ok=True)

    print("\n" + "="*60)
    print("TRAINING ACTOR-CRITIC ON CARTPOLE")
    print("="*60)
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max episode steps: {max_episode_steps}")
    print(f"Number of training episodes: {args.num_episodes}")
    print("="*60 + "\n")

    logs = train_actor_critic(
        env,
        policy_net,
        value_net,
        optimizer_policy,
        optimizer_value,
        num_episodes=args.num_episodes,
        gamma=0.99,
        device=device,
        verbose=True
    )

    # ------------ Save logs ------------
    print("\nSaving logs...")
    for name, values in logs.items():
        # Filter out None values for hitrates if not applicable
        if name == "hitrates" and all(v is None for v in values):
            continue
        np.save(f"results/logs/cartpole_{name}.npy", np.array(values))

    # ------------ Plot curves ------------
    print("Generating plots...")
    plot_curve(logs["rewards"], "results/plots/cartpole_reward_curve.png", "Episode Reward")
    plot_curve(logs["episode_lengths"], "results/plots/cartpole_episode_lengths.png", "Episode Length")
    plot_curve(logs["actor_loss"], "results/plots/cartpole_actor_loss.png", "Actor Loss")
    plot_curve(logs["critic_loss"], "results/plots/cartpole_critic_loss.png", "Critic Loss")
    plot_curve(logs["td_error"], "results/plots/cartpole_td_error.png", "TD Error")

    # ------------ Save checkpoints ------------
    print("Saving checkpoints...")
    torch.save(policy_net.state_dict(), "checkpoints/cartpole_policy_final.pth")
    torch.save(value_net.state_dict(), "checkpoints/cartpole_value_final.pth")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("Logs saved to: results/logs/")
    print("Plots saved to: results/plots/")
    print("Checkpoints saved to: checkpoints/")
    print("="*60)

    # ------------ Evaluate trained policy ------------
    if not args.no_eval:
        print("\n" + "="*60)
        print("EVALUATING TRAINED POLICY")
        print("="*60)
        print(f"Running evaluation for {args.eval_episodes} episodes...")
        
        metrics = evaluate_policy(env, policy_net, num_episodes=args.eval_episodes, device=device, render=False)
        metrics['success_count'] = int(metrics['success_rate'] * metrics['num_episodes'])
        
        print_evaluation_results(metrics)
        
        # Save evaluation results
        np.save("results/evaluation/cartpole_eval_rewards.npy", np.array(metrics['episode_rewards']))
        np.save("results/evaluation/cartpole_eval_lengths.npy", np.array(metrics['episode_lengths']))
        print("Evaluation results saved to results/evaluation/")
    else:
        print("\nSkipping evaluation (use --no_eval to suppress this message)")


if __name__ == "__main__":
    main()
