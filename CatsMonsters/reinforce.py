# Implementing the REINFORCE WITH BASELINE algorithm

import torch 
import numpy as np
from torch.distributions import Categorical
from models import PolicyNetwork, ValueNetwork
from env_cat_monsters import CatMonstersEnv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def sample_episode(env, policy_net: PolicyNetwork, device: torch.device, gamma: float=0.99) -> tuple:
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    while not done:
        s_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        action_logits = policy_net(s_tensor)
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        next_state, reward, done, _ = env.step(action.item())
        states.append(s_tensor)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    
    total_reward = sum(rewards)
    avg_reward = np.mean(rewards) if rewards else 0.0
    
    episode_stats = {
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'num_steps': len(rewards)
    }
    
    return states, actions, rewards, episode_stats


def compute_returns(rewards: list, gamma: float=0.99) -> list:
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


def reinforce_update(policy_net: PolicyNetwork, value_net: ValueNetwork, optimiser_policy: torch.optim.Optimizer,
                     optimiser_value: torch.optim.Optimizer, states: list, actions: list, rewards: list, gamma: float=0.99,
                     device: torch.device=torch.device('cpu'), normalize_advantages: bool=True,
                     entropy_coef: float=0.01, max_grad_norm: float=0.5) -> tuple:
  
    returns = torch.tensor(compute_returns(rewards, gamma),
                           dtype=torch.float32, device=device)
    states_tensor = torch.stack(states).to(device)
    actions_tensor = torch.tensor([a.item() for a in actions],
                                  dtype=torch.long, device=device)

    values = value_net(states_tensor)
    if len(values.shape) == 0:
        values = values.unsqueeze(0)
    elif len(values.shape) > 1:
        values = values.squeeze()
    critic_loss = F.mse_loss(values, returns)

    logits = policy_net(states_tensor)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    chosen_log_probs = log_probs[torch.arange(len(actions_tensor), device=device), actions_tensor]

    advantages = (returns - values.detach())
    
    if normalize_advantages and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    actor_loss = -(chosen_log_probs * advantages).mean() - entropy_coef * entropy

    optimiser_policy.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
    optimiser_policy.step()

    optimiser_value.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
    optimiser_value.step()

    return actor_loss.item(), critic_loss.item(), entropy.item()


def train_reinforce(env, policy_net: PolicyNetwork, value_net: ValueNetwork, optimiser_policy: torch.optim.Optimizer,
                    optimiser_value: torch.optim.Optimizer, device: torch.device=torch.device('cpu'), num_episodes: int=1000,
                    gamma: float=0.99, normalize_advantages: bool=True,
                    entropy_coef: float=0.01, max_grad_norm: float=0.5):

    episode_rewards = []
    episode_losses = []
    
    for _ in range(num_episodes):
        states, actions, rewards, stats = sample_episode(env, policy_net, device, gamma)
        actor_loss, critic_loss, entropy = reinforce_update(policy_net, value_net, optimiser_policy, optimiser_value,
                                                               states, actions, rewards, gamma, device, normalize_advantages,
                                                               entropy_coef, max_grad_norm)
        
        episode_rewards.append(stats['total_reward'])
        episode_losses.append((actor_loss, critic_loss, entropy))
        
    return episode_rewards, episode_losses


def eval_cat_monsters_env(env: CatMonstersEnv, policy_net: PolicyNetwork, device: torch.device, num_episodes: int=100) -> dict:
   
    episode_rewards = []
    episode_steps = []

    for _ in range(num_episodes):
        _, _, _, stats = sample_episode(env, policy_net, device)
        episode_rewards.append(stats['total_reward'])
        episode_steps.append(stats['num_steps'])
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_steps': np.mean(episode_steps),
        'std_steps': np.std(episode_steps),
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,

    }


def load_policy_and_value_networks():
    policy_net = PolicyNetwork(state_dim=25, action_dim=4, hidden_dim=128)
    value_net = ValueNetwork(state_dim=25, hidden_dim=128)
    policy_net.load_state_dict(torch.load("policy_net_cat_monsters.pth"))
    value_net.load_state_dict(torch.load("value_net_cat_monsters.pth"))
    return policy_net, value_net


def test_policy_and_value_networks():
    """Test loaded policy and value networks on CatMonstersEnv."""
    policy_net, value_net = load_policy_and_value_networks()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    value_net.to(device)
    env = CatMonstersEnv(seed=42)
    
    eval_results = eval_cat_monsters_env(env, policy_net, device, num_episodes=100)
    return eval_results


if __name__ == "__main__":
    # test_sample_episode()
    # smoke_test()  # Uncomment to test cache environment
    test_policy_and_value_networks()

