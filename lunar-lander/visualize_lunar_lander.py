import gymnasium as gym
import torch
from lunar_lander_ppo import ActorCritic

def visualize_model(env, model, num_episodes=5, max_steps=1000):
    """Visualize the trained model's performance."""
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state)
            logits, _ = model(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
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
    # Create environment with human rendering
    env = gym.make('LunarLander-v3', render_mode='human')
    
    # Initialize model with correct dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(state_dim, action_dim)
    
    # Load the trained model
    try:
        model.load_state_dict(torch.load('best_lunar_lander_model.pth'))
        model.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Could not find the model file 'best_lunar_lander_model.pth'")
        print("Please make sure the model file exists in the current directory.")
        exit(1)
    
    # Visualize the model
    print("\nVisualizing trained model...")
    visualize_model(env, model, num_episodes=5) 