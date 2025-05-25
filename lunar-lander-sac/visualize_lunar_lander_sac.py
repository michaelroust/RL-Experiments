import gymnasium as gym
import torch
from lunar_lander_sac import Actor, SAC
import os
from datetime import datetime

def visualize_model(env, agent, num_episodes=5, max_steps=1000, record_video=True):
    """Visualize the trained model's performance."""
    # Create videos directory if it doesn't exist
    if record_video:
        videos_dir = "videos"
        os.makedirs(videos_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for episode in range(num_episodes):
        if record_video:
            # Create a new environment for each episode to record
            env = gym.make('LunarLanderContinuous-v3', render_mode='rgb_array')
            env = gym.wrappers.RecordVideo(env, videos_dir, 
                                         episode_trigger=lambda x: True,
                                         name_prefix=f"episode_{episode+1}_{timestamp}")
        
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
        
        print(f"Visualization Episode {episode + 1}, Total Reward: {episode_reward:.2f}")
        env.close()
    
    if record_video:
        print(f"\nVideos saved in {videos_dir} directory")

if __name__ == "__main__":
    # Create environment with human rendering
    env = gym.make('LunarLanderContinuous-v3', render_mode='human')
    
    # Initialize model with correct dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Create SAC agent
    agent = SAC(state_dim, action_dim, max_action)
    
    # Load the trained model
    try:
        agent.actor.load_state_dict(torch.load('best_lunar_lander_sac_model.pth'))
        agent.actor.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Could not find the model file 'best_lunar_lander_sac_model.pth'")
        print("Please make sure the model file exists in the current directory.")
        exit(1)
    
    # Visualize the model
    print("\nVisualizing trained model...")
    visualize_model(env, agent, num_episodes=5, record_video=True) 