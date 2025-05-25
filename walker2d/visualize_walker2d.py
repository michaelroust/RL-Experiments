import gymnasium as gym
import torch
import numpy as np
from walker2d import Actor, TD3

def visualize_model(env, agent, num_episodes=5, max_steps=1000):
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            if done:
                break
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Total Reward: {episode_reward:.2f}")
    
    print(f"\nAverage Reward over {num_episodes} episodes: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")

if __name__ == "__main__":
    # Create environment for continuous control
    env = gym.make('Walker2d-v4', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize TD3 agent
    agent = TD3(state_dim, action_dim, max_action)
    
    # Load the trained model
    try:
        # Load the full model checkpoint
        checkpoint = torch.load('final_walker2d_td3_model.pth')
        
        # Load all components
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        agent.total_it = checkpoint['total_it']
        
        print("Final model loaded successfully.")
    except Exception as e:
        print(f"Error loading final model: {e}")
        print("Trying to load best model instead...")
        try:
            state_dict = torch.load('best_walker2d_td3_model.pth')
            agent.actor.load_state_dict(state_dict)
            print("Best model loaded successfully.")
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Please ensure you have trained the model first")
            exit(1)
    
    # Visualize the model
    print("\nRunning visualization...")
    visualize_model(env, agent, num_episodes=5) 