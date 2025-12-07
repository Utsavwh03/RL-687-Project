import numpy as np

class CartPoleEnv:
    """
    CartPole environment implementation.
    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Actions: 0 (push left) or 1 (push right)
    Reward: +1 for each step the pole stays upright
    """
    
    def __init__(self, max_episode_steps=500):
        self.max_episode_steps = max_episode_steps
        
        # Physical constants
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        self.length = 0.5  # half the pole's length
        self.polemass_length = self.pole_mass * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        
        # State bounds for normalization (optional, but helpful)
        self.state_bounds = np.array([
            [self.x_threshold * 2, np.finfo(np.float32).max],
            [self.theta_threshold_radians * 2, np.finfo(np.float32).max],
        ])
        
        self.t = 0
        self.state = None
        
    def reset(self):
        """Reset the environment to initial state."""
        self.t = 0
        # Initialize state with small random values
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state.copy()
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: 0 (push left) or 1 (push right)
        
        Returns:
            next_state, reward, done, info
        """
        assert action in [0, 1], f"Action must be 0 or 1, got {action}"
        
        x, x_dot, theta, theta_dot = self.state
        
        # Apply force based on action
        force = self.force_mag if action == 1 else -self.force_mag
        
        # Calculate derivatives using physics equations
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        temp = (force + self.polemass_length * theta_dot ** 2 * sin_theta) / self.total_mass
        thetaacc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0 / 3.0 - self.polemass_length * cos_theta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * cos_theta / self.total_mass
        
        # Update state using Euler's method
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        # Check if episode is done
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.t >= self.max_episode_steps - 1
        )
        
        # Reward is +1 for each step the pole stays upright
        reward = 1.0
        
        self.t += 1
        
        # Info dict - no "hit" key needed for CartPole
        info = {}
        
        return self.state.copy(), reward, done, info


# Smoke test the environment
if __name__ == "__main__":
    env = CartPoleEnv(max_episode_steps=500)
    
    print("="*60)
    print("CARTPOLE ENVIRONMENT TEST")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Max episode steps: {env.max_episode_steps}")
    print(f"  - State dimension: 4")
    print(f"  - Action dimension: 2")
    print("="*60)
    
    state = env.reset()
    print(f"\n🔵 INITIAL STATE: {state}")
    print(f"  [cart_pos, cart_vel, pole_angle, pole_angular_vel]")
    
    total_reward = 0
    steps = 0
    
    print("\n" + "="*60)
    print("EPISODE EXECUTION")
    print("="*60)
    
    while True:
        action = np.random.randint(0, 2)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if steps % 50 == 0:
            print(f"Step {steps}: action={action}, reward={reward}, done={done}, "
                  f"cart_pos={next_state[0]:.3f}, pole_angle={next_state[2]:.3f}")
        
        if done:
            break
    
    print("\n" + "="*60)
    print("EPISODE SUMMARY")
    print("="*60)
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward}")
    print(f"  Average reward per step: {total_reward/steps:.3f}")
