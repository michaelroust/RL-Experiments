import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
        
        for i in ind:
            state, action, reward, next_state, done = self.storage[i]
            batch_states.append(np.asarray(state))
            batch_actions.append(np.asarray(action))
            batch_rewards.append(np.asarray(reward))
            batch_next_states.append(np.asarray(next_state))
            batch_dones.append(np.asarray(done))

        return np.array(batch_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), \
               np.array(batch_next_states), np.array(batch_dones).reshape(-1, 1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 2 * action_dim)  # Mean and log_std
        self.to(device)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean, log_std = self.l3(a).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action * self.max_action, log_prob

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

class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.max_action = max_action
        self.replay_buffer = ReplayBuffer()
        
        self.total_it = 0
        self.alpha = 0.2  # Temperature parameter
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005  # Target network update rate
        self.batch_size = 256

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

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
            next_action, next_log_pi = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * (target_Q - self.alpha * next_log_pi)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        action, log_pi = self.actor.sample(state)
        actor_Q1, actor_Q2 = self.critic(state, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha * log_pi - actor_Q).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def train_sac(env, num_episodes=1000, max_steps=1000):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = SAC(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    rewards = []
    
    for episode in range(num_episodes):
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
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")
    return rewards, agent

def plot_rewards(rewards):
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('SAC Learning Curve (Lunar Lander)')
    plt.savefig('lunar_lander_sac_learning_curve.png')
    plt.close()

if __name__ == "__main__":
    # Create environment for continuous control
    env = gym.make('LunarLanderContinuous-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    num_episodes = 1000
    max_steps = 1000
    
    print(f"Training SAC for {num_episodes} episodes...")
    rewards, agent = train_sac(env, num_episodes, max_steps)
    plot_rewards(rewards)
    
    # Save the trained model
    torch.save(agent.actor.state_dict(), 'best_lunar_lander_sac_model.pth')
    print("Training complete. Learning curve saved as lunar_lander_sac_learning_curve.png") 