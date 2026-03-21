"""
50k Baseline Training (v2.0 with realistic IC)

Author: 小A 🤖
Date: 2026-03-20

YZ指令: 长任务 + 监控 + 日志
"""

import numpy as np
import time
from mhd_elsasser_env import MHDElsasserEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os

class MonitoringCallback(BaseCallback):
    """Custom callback for monitoring long training"""
    
    def __init__(self, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.start_time = None
        
    def _on_training_start(self):
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            steps_per_sec = self.n_calls / elapsed
            eta = (50000 - self.n_calls) / steps_per_sec
            
            print(f"\n{'='*60}")
            print(f"Checkpoint @ {self.n_calls} steps")
            print(f"  Elapsed: {elapsed/60:.1f} min")
            print(f"  Speed: {steps_per_sec:.1f} steps/sec")
            print(f"  ETA: {eta/60:.1f} min")
            print(f"{'='*60}\n")
            
        return True

print('='*60)
print('v2.0 Phase 4.5: 50k Baseline Training')
print('='*60)
print(f'\nConfig:')
print(f'  IC: ballooning_ic_v2 (realistic β~0.17)')
print(f'  Grid: 16×32×16')
print(f'  Total steps: 50,000')
print(f'  Eval frequency: 5,000')
print(f'  Monitoring: Every 5,000 steps')
print('')

# Create environment
def make_env():
    return MHDElsasserEnv(grid_shape=(16,32,16), max_episode_steps=100)

env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])

# PPO config
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log='./logs/v2_50k_baseline/'
)

# Callbacks
os.makedirs('./logs/v2_50k_baseline/', exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./logs/v2_50k_baseline/',
    log_path='./logs/v2_50k_baseline/',
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
    verbose=1
)

monitor_callback = MonitoringCallback(check_freq=5000)

print('Starting 50k baseline training...\n')

try:
    model.learn(
        total_timesteps=50000,
        callback=[eval_callback, monitor_callback],
        progress_bar=True
    )
    
    print('\n' + '='*60)
    print('✅ Training Complete!')
    print('='*60)
    
    # Save final model
    model.save('./logs/v2_50k_baseline/final_model_50k')
    print('Model saved to ./logs/v2_50k_baseline/final_model_50k')
    
    # Load and print eval results
    data = np.load('./logs/v2_50k_baseline/evaluations.npz')
    timesteps = data['timesteps']
    results = data['results']
    ep_lengths = data['ep_lengths']
    
    print('\n' + '='*60)
    print('Results Summary')
    print('='*60)
    print('Timestep | Mean Reward | Mean Length')
    print('---------|-------------|-------------')
    for i in range(len(timesteps)):
        t = timesteps[i]
        r = results[i].mean()
        l = ep_lengths[i].mean()
        marker = '✅' if l >= 100 else '⚠️'
        print(f'{marker} {t:5d}  | {r:11.2f} | {l:11.1f}')
    
    improvement = results.mean(axis=1).max() - results.mean(axis=1)[0]
    
    print('\n' + '='*60)
    print(f'First reward:       {results[0].mean():.2f}')
    print(f'Best reward:        {results.mean(axis=1).max():.2f}')
    print(f'Final reward:       {results[-1].mean():.2f}')
    print(f'Improvement:        {improvement:.2f}')
    print(f'Avg episode length: {ep_lengths.mean():.1f} steps')
    
    if ep_lengths.mean() >= 95:
        print('\n✅✅ Stability maintained throughout training!')
    else:
        print('\n⚠️  Some episodes terminated early')
    
    print('='*60)

except KeyboardInterrupt:
    print('\n⚠️  Training interrupted by user')
    model.save('./logs/v2_50k_baseline/interrupted_model')
    print('Partial model saved')
    
except Exception as e:
    print(f'\n❌ Training failed: {e}')
    import traceback
    traceback.print_exc()
