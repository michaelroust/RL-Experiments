# Reinforcement Learning Experiments

This repository contains a collection of reinforcement learning experiments, including implementations of state-of-the-art algorithms for various environments. The current focus is on the Lunar Lander environment, where our Soft Actor-Critic (SAC) implementation achieves state-of-the-art results.

#### SAC Implementation (State-of-the-Art)
Soft Actor-Critic implementation targeting the best reported performance.

**5 Trials (Average Score: 290.43)** - ~2-3 hours on MPS
![Lunar Lander SAC Demo](lunar-lander-sac/videos/lunar_lander_sac_combined.gif?loop=1)

**Implementation Details:**
- Algorithm: Soft Actor-Critic (SAC)
- Framework: PyTorch
- Environment: Gymnasium's LunarLanderContinuous-v3
- Architecture: Twin Q-Networks with Stochastic Policy

**Performance:**
- The GIF above shows multiple successful landings by the SAC agent, demonstrating stable and consistent control.
- Achieves average rewards in the 250-320 range, matching state-of-the-art results for this environment.
- State-of-the-art performance with average score of 290.43 over 5 trials.

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

**Training Time:**
- CPU: ~12-15 hours for 100K episodes
- A100 GPU: ~3-4 hours for 100K episodes
- Optimized for A100 with:
  - 256-256 unit networks
  - Batch size of 256
  - 1M replay buffer
  - 3e-4 learning rate

**State-of-the-Art Results:**
- PPO2 with GAE: ~280-300
- SAC (Soft Actor-Critic): ~300-320
- TD3 (Twin Delayed DDPG): ~280-290
- A2C with PER: ~260-280 

#### Walker2d Implementation
TD3 (Twin Delayed DDPG) implementation for the Walker2d environment, with both PyTorch and JAX versions.

**Performance Demo (Average Score: 5000+)** - ~2-3 days on A100

![Walker2d TD3 Demo](walker2d/videos/replay.gif?loop=1)

**Implementation Details:**
- Algorithm: Twin Delayed DDPG (TD3)
- Frameworks: PyTorch and JAX
- Environment: Gymnasium's Walker2d-v4
- Architecture: Twin Q-Networks with Delayed Policy Updates

**Features:**
- PyTorch implementation with full training pipeline
- JAX implementation for faster training
- Model visualization and evaluation tools
- Pre-trained model download functionality

**Usage:**
```bash
# PyTorch Version
# Train the model
python walker2d/walker2d.py

# Visualize the trained model
python walker2d/visualize_walker2d.py

# JAX Version
# Download pre-trained model
python walker2d-hf/download_model.py

# Visualize the model
python walker2d-hf/visualise_walker2d_hf.py
```

**Technical Features:**
- Twin Q-Networks for stable value estimation
- Delayed policy updates to prevent overestimation
- Target policy smoothing for regularization
- Experience replay with large buffer
- Target network updates with soft updates 

**Training Time:**
- CPU: ~5-7 days for 1M episodes
- A100 GPU: ~2-3 days for 1M episodes
- Optimized for A100 with:
  - 128-128 unit networks
  - Layer normalization
  - 1024 batch size
  - 2M replay buffer
  - 1e-4 learning rate 