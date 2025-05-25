import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        logits = self.actor(state)
        value = self.critic(state)
        return logits, value

def compute_gae(rewards, values, next_value, gamma=0.99, lambda_=0.95):
    advantages = []
    gae = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_value - v
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
        next_value = v
    return torch.tensor(advantages, dtype=torch.float32)

def ppo(env, num_episodes, max_steps, gamma=0.99, lambda_=0.95, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, lr=3e-4, epochs=4):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rewards_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        states, actions, rewards, values, log_probs = [], [], [], [], []

        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state)
            logits, value = model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())

            state = next_state
            episode_reward += reward

            if done:
                break

        next_value = model(torch.FloatTensor(next_state))[1].item()
        advantages = compute_gae(rewards, values, next_value, gamma, lambda_)
        returns = advantages + torch.tensor(values)

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.stack(actions)
        old_log_probs_tensor = torch.tensor(log_probs)
        returns_tensor = returns.detach()
        advantages_tensor = advantages.detach()

        for _ in range(epochs):
            for idx in range(len(states)):
                state_tensor = states_tensor[idx]
                action = actions_tensor[idx]
                old_log_prob = old_log_probs_tensor[idx]
                advantage = advantages_tensor[idx]
                ret = returns_tensor[idx]

                logits, value = model(state_tensor)
                dist = Categorical(logits=logits)
                new_log_prob = dist.log_prob(action)
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
                actor_loss = -torch.min(surr1, surr2)
                critic_loss = (value - ret).pow(2)
                entropy = dist.entropy()
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards_history.append(episode_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

    return rewards_history, model

def plot_rewards(rewards):
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('PPO Learning Curve (Lunar Lander)')
    plt.savefig('lunar_lander_ppo_learning_curve.png')
    plt.close()

def visualize_model(env, model, num_episodes=5, max_steps=1000):
    """Visualize the trained model's performance."""
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state)
            logits, _ = model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        print(f"Visualization Episode {episode + 1}, Total Reward: {episode_reward}")
    
    env.close()

if __name__ == "__main__":
    # Training phase
    env = gym.make('LunarLander-v3')
    num_episodes = 5000
    max_steps = 1000
    print(f"Training PPO for {num_episodes} episodes...")
    rewards, model = ppo(env, num_episodes, max_steps, 
                        lr=1e-4,
                        epochs=8,
                        clip_ratio=0.15,
                        value_coef=0.7,
                        entropy_coef=0.02)
    plot_rewards(rewards)
    print("Training complete. Learning curve saved as lunar_lander_ppo_learning_curve.png")
    
    # Save the trained model
    torch.save(model.state_dict(), 'best_lunar_lander_model.pth')
    
    # Wait for user input before visualization
    input("\nPress Enter to visualize the trained model...")
    
    # Visualization phase
    print("\nVisualizing trained model...")
    env = gym.make('LunarLander-v3', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(state_dim, action_dim)
    model.load_state_dict(torch.load('best_lunar_lander_model.pth'))
    model.eval()
    visualize_model(env, model) 