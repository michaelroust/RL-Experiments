import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = []
        self.max_size = int(max_size)
        self.position = 0

    def add(self, transition):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        return zip(*[self.buffer[i] for i in indices])

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.to(device)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        self.to(device)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.max_action = max_action
        self.replay_buffer = ReplayBuffer()
        
        self.total_it = 0
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005  # Target network update rate
        self.batch_size = 256
        self.policy_delay = 2  # Delayed policy updates
        self.noise_scale = 0.2  # Noise scale for exploration
        self.noise_clip = 0.5  # Noise clipping range

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        # Compute target Q value
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.noise_scale).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def evaluate_agent(env, agent, num_episodes=10):
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        rewards.append(episode_reward)
    return rewards

def train_td3(env, num_episodes=1_000_000, max_steps=1000, eval_interval=10_000, eval_episodes=10):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    rewards = []
    best_eval_reward = -np.inf
    
    for episode in tqdm(range(num_episodes), desc="Training TD3"):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            # Clip action to valid range
            action = np.clip(action, -max_action, max_action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            if done:
                break
            agent.train(replay_buffer, batch_size=256)
        rewards.append(episode_reward)
        
        # Evaluate every eval_interval episodes
        if (episode + 1) % eval_interval == 0:
            eval_rewards = evaluate_agent(env, agent, eval_episodes)
            avg_eval_reward = np.mean(eval_rewards)
            print(f"Evaluation at episode {episode + 1}, Average Reward: {avg_eval_reward:.2f}")
            
            # Save the best model
            if avg_eval_reward > best_eval_reward:
                best_eval_reward = avg_eval_reward
                torch.save(agent.actor.state_dict(), 'best_walker2d_td3_model.pth')
                print(f"New best model saved with reward: {best_eval_reward:.2f}")
        
        # Display current performance
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Current Reward: {episode_reward:.2f}")
    
    # Save the final model
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_target_state_dict': agent.actor_target.state_dict(),
        'critic_target_state_dict': agent.critic_target.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        'total_it': agent.total_it
    }, 'final_walker2d_td3_model.pth')
    print("Final model saved as 'final_walker2d_td3_model.pth'")
    
    return rewards, agent

def plot_rewards(rewards):
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('TD3 Learning Curve (Walker2d)')
    plt.savefig('walker2d_td3_learning_curve.png')
    plt.close()

if __name__ == "__main__":
    # Create environment for continuous control
    env = gym.make('Walker2d-v4')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    num_episodes = 1_000_000  # 1M episodes
    max_steps = 1000          # 1000 steps per episode
    eval_interval = 10_000    # Evaluate every 10K episodes
    eval_episodes = 10        # 10 episodes for evaluation
    
    print(f"Training TD3 for {num_episodes} episodes...")
    rewards, agent = train_td3(env, num_episodes, max_steps, eval_interval, eval_episodes)
    plot_rewards(rewards)
    
    print("Training complete. Learning curve saved as walker2d_td3_learning_curve.png") 