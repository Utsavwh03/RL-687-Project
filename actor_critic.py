import torch
from torch.distributions import Categorical

def train_actor_critic(
    env,
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    num_episodes=5000,
    gamma=0.99,
    device="cpu",
    verbose=False
):
    # Episode logs
    rewards_log = []
    episode_lengths_log = []
    hitrate_log = []  # Optional, for environments that provide "hit" in info

    # Step-wise logs
    actor_loss_log = []
    critic_loss_log = []
    td_error_log = []

    for ep in range(num_episodes):

        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)

        done = False
        ep_reward = 0
        ep_hits = 0
        ep_steps = 0
        tracking_hits = False  # Track if environment provides "hit" in info

        while not done:

            # -------- 1. Sample action from policy --------
            logits = policy_net(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # -------- 2. Environment step --------
            next_state_np, reward, done, info = env.step(action.item())
            next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device)

            # -------- 3. Compute TD(0) advantage --------
            value_s = value_net(state)
            value_next = value_net(next_state).detach() if not done else 0.0

            td_target = reward + gamma * value_next
            delta = td_target - value_s      # TD-error (Advantage estimator)

            # -------- 4. Critic update --------
            critic_loss = delta.pow(2)
            optimizer_value.zero_grad()
            critic_loss.backward()
            optimizer_value.step()

            # -------- 5. Actor update --------
            actor_loss = -log_prob * delta.detach()
            optimizer_policy.zero_grad()
            actor_loss.backward()
            optimizer_policy.step()

            # -------- Advance to next state --------
            state = next_state

            ep_reward += reward
            ep_steps += 1

        # ---------- End-of-episode logs ----------
        episode_lengths_log.append(ep_steps)
        rewards_log.append(ep_reward)
        
        # Calculate hit rate if hits were tracked
        if tracking_hits and ep_steps > 0:
            hitrate = ep_hits / ep_steps
            hitrate_log.append(hitrate)
        else:
            hitrate_log.append(None)

        if verbose and ep % 100 == 0:
            if hitrate_log[-1] is not None:
                print(f"[Episode {ep}] Reward={ep_reward:.1f}, Length={ep_steps}, HitRate={hitrate_log[-1]:.3f}")
            else:
                print(f"[Episode {ep}] Reward={ep_reward:.1f}, Length={ep_steps}")

    return {
        "rewards": rewards_log,
        "episode_lengths": episode_lengths_log,
        "hitrates": hitrate_log,  # May contain None values for environments without hit tracking
        "actor_loss": actor_loss_log,
        "critic_loss": critic_loss_log,
        "td_error": td_error_log
    }
