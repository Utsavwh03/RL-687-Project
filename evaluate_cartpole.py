import numpy as np
import torch
from torch.distributions import Categorical
import os

from cartpole_env import CartPoleEnv
from models import PolicyNetwork, ValueNetwork


def evaluate_policy(env, policy_net, num_episodes=100, device="cpu", render=False):
    """
    Evaluate a trained policy on the environment.
    
    Args:
        env: The environment to evaluate on
        policy_net: Trained policy network
        num_episodes: Number of episodes to run
        device: Device to run on
        render: Whether to print step-by-step info (for debugging)
    
    Returns:
        Dictionary with evaluation metrics
    """
    policy_net.eval()
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0  # Episodes that reached max steps
    
    for ep in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        
        done = False
        ep_reward = 0
        ep_steps = 0
        
        while not done:
            # Sample action from policy
            with torch.no_grad():
                logits = policy_net(state)
                dist = Categorical(logits=logits)
                action = dist.sample()
            
            # Step environment
            next_state, reward, done, info = env.step(action.item())
            
            ep_reward += reward
            ep_steps += 1
            
            if render and ep_steps % 50 == 0:
                print(f"  Step {ep_steps}: action={action.item()}, reward={reward:.1f}, "
                      f"cart_pos={next_state[0]:.3f}, pole_angle={next_state[2]:.3f}")
            
            state = torch.tensor(next_state, dtype=torch.float32, device=device)
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        
        # Check if episode was successful (reached max steps)
        if ep_steps >= env.max_episode_steps:
            success_count += 1
    
    # Calculate statistics
    metrics = {
        "num_episodes": num_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
        "success_rate": success_count / num_episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }
    
    return metrics


def print_evaluation_results(metrics):
    """Print evaluation results in a readable format."""
    print("\n" + "="*60)
    print("CARTPOLE EVALUATION RESULTS")
    print("="*60)
    print(f"Number of episodes: {metrics['num_episodes']}")
    print(f"\n📊 Reward Statistics:")
    print(f"  Mean reward:     {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Min reward:      {metrics['min_reward']:.2f}")
    print(f"  Max reward:      {metrics['max_reward']:.2f}")
    print(f"\n📏 Episode Length Statistics:")
    print(f"  Mean length:     {metrics['mean_episode_length']:.2f} ± {metrics['std_episode_length']:.2f}")
    print(f"  Min length:      {np.min(metrics['episode_lengths'])}")
    print(f"  Max length:      {np.max(metrics['episode_lengths'])}")
    print(f"\n✅ Success Rate:")
    print(f"  Episodes reaching max steps: {metrics['success_count']}/{metrics['num_episodes']}")
    print(f"  Success rate:    {metrics['success_rate']*100:.2f}%")
    print("="*60 + "\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Environment setup
    max_episode_steps = 500
    env = CartPoleEnv(max_episode_steps=max_episode_steps)
    
    # Network dimensions
    state_dim = 4
    action_dim = 2
    
    # Load trained models
    policy_path = "checkpoints/cartpole_policy_final.pth"
    
    if not os.path.exists(policy_path):
        print(f"❌ Error: Policy checkpoint not found at {policy_path}")
        print("Please train the model first using train_cartpole.py")
        return
    
    print(f"Loading policy from {policy_path}...")
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim=128).to(device)
    policy_net.load_state_dict(torch.load(policy_path, map_location=device))
    policy_net.eval()
    
    print("Evaluating policy...")
    metrics = evaluate_policy(env, policy_net, num_episodes=100, device=device, render=False)
    
    # Add success_count for printing
    metrics['success_count'] = int(metrics['success_rate'] * metrics['num_episodes'])
    
    print_evaluation_results(metrics)
    
    # Save evaluation results
    os.makedirs("results/evaluation", exist_ok=True)
    np.save("results/evaluation/cartpole_eval_rewards.npy", np.array(metrics['episode_rewards']))
    np.save("results/evaluation/cartpole_eval_lengths.npy", np.array(metrics['episode_lengths']))
    
    print("Evaluation results saved to results/evaluation/")


if __name__ == "__main__":
    main()
