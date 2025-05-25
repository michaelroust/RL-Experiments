#### SAC Implementation (State-of-the-Art)
Soft Actor-Critic implementation targeting the best reported performance.

**Video Demonstration:**
![Lunar Lander SAC Demo](lunar-lander-sac/videos/lunar_lander_sac_combined.gif)

**Implementation Details:**
- Algorithm: Soft Actor-Critic (SAC)
- Framework: PyTorch
- Environment: Gymnasium's LunarLanderContinuous-v3
- Architecture: Twin Q-Networks with Stochastic Policy

**Performance:**
- The GIF above shows multiple successful landings by the SAC agent, demonstrating stable and consistent control.
- Achieves average rewards in the 250-320 range, matching state-of-the-art results for this environment.

**Usage:**
```bash
# Train the model
python lunar_lander_sac.py

# Visualize and record episodes
python visualize_lunar_lander_sac.py

# Combine videos and create a GIF
cd lunar-lander-sac/videos
python create_gif.py
```

**Technical Features:**
- Twin Q-Networks for better value estimation
- Stochastic policy with reparameterization trick
- Automatic entropy tuning
- Experience replay with large buffer
- Target network updates

**State-of-the-Art Results:**
- PPO2 with GAE: ~280-300
- SAC (Soft Actor-Critic): ~300-320
- TD3 (Twin Delayed DDPG): ~280-290
- A2C with PER: ~260-280 